"""Frozen offline Laya resident: local senses and bounded consequence context.

No neural updates, predictor, privileged state or scripted action fallback.
Tournament selection is a capacity workaround, not an equivalent full softmax.
"""
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import random

import torch

from agents.affordances import action_mask
from agents.network import ACTIONS

ADAPTER = Path.home() / 'Projects/laya-lab/npu/laya_lite.py'
INSTRUCTION = 'Choose the next action using local senses and recent observed consequences.'


@dataclass(frozen=True)
class Config:
    adapter_path: str = str(ADAPTER)
    history_capacity: int = 3
    journal_capacity: int = 128
    group_size: int = 8
    max_tokens: int = 512
    head_tokens: int = 192

    def __post_init__(self):
        if not 2 <= self.group_size <= 12 or not 0 <= self.history_capacity <= 3:
            raise ValueError('Laya supports groups of 2..12 and history of 0..3')
        if self.journal_capacity <= 0 or self.max_tokens != 512 or self.head_tokens != 192:
            raise ValueError('Use the 512-token IR and conservative 192-token question budget')


def _rounded(value):
    return round(float(value), 2)


def body_values(observation):
    return {**dict(zip(('food', 'water', 'energy', 'warmth'), map(float, observation['needs']))),
            **{k: float(observation['body'].get(k, 0)) for k in ('health', 'pain')},
            **{f'carried_{k}': int(v) for k, v in observation['inventory'].items()}}


def compact_observation(observation):
    """Intentional lossy local projection; relative offsets only, no hidden data."""
    resources = []
    for tile in observation['tiles']:
        if tile.get('terrain', -1) < 0:
            continue
        resource = tile.get('resource', {})
        if resource:
            resources.append([tile['dx'], tile['dy'], resource.get('kind'),
                              resource.get('amount', 0), resource.get('stage', 0)])
        if tile.get('cache', {}).get('food', 0):
            resources.append([tile['dx'], tile['dy'], 'cache', tile['cache']['food'], 0])
    resources.sort(key=lambda r: abs(r[0])+abs(r[1]))
    # One nearest example per kind, at most six. This is sensing compression,
    # never a movement target, plan or safe-food classification.
    distinct = {}
    for row in resources:
        distinct.setdefault(row[2], row)
    return dict(body={k: _rounded(v) for k, v in body_values(observation).items()}, facing=observation.get('facing'),
                unconscious=bool(observation['body'].get('unconscious')),
                touch=[[t['direction'], bool(t['blocked'])] for t in observation['touch']],
                resources=list(distinct.values())[:6],
                neighbors=[[o['dx'], o['dy'], bool(o.get('unconscious'))]
                           for o in observation.get('others', ())[:3]],
                hearing=[[h.get('kind'), h.get('bearing'), _rounded(h.get('strength', 0))]
                         for h in observation.get('hearing', ())[:3]],
                tones=[[e.get('tone'), e.get('bearing')] for e in observation.get('auditory_memory', ())[-5:]],
                water_carried=observation.get('ecology', {}).get('water_carried', 0))


class LayaBrain:
    backend = 'laya-npu'

    def __init__(self, seed, config=None, *, engine=None):
        self.seed = seed
        self.config = config or Config()
        self.rng = random.Random(seed)
        self._engine = engine
        self.decisions = self.updates = self.policy_version = 0
        self.masked = True
        self.training = False
        self.buffer = []
        self.pending = None
        self.journal = []
        self.reward_parts = {}
        self.prediction_error_total = 0.
        self.observed_transitions = 0
        self.preferences = []
        self.diagnostics = {}

    def _load_engine(self):
        if self._engine is None:
            path = Path(self.config.adapter_path)
            spec = importlib.util.spec_from_file_location('godhood_laya_offline', path)
            if spec is None or spec.loader is None:
                raise RuntimeError('Existing offline Laya adapter is unavailable')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._engine = module.LayaLite('NPU')
        return self._engine

    def add_reward(self, parts):
        # Simulator-measured feedback is context only; weights never update.
        for key, value in parts.items():
            self.reward_parts[key] = self.reward_parts.get(key, 0.) + float(value)

    def _complete(self, observation):
        if self.pending is None:
            return
        current = body_values(observation)
        previous = self.pending['body']
        event = dict(decision=self.decisions, action=dict(ACTIONS[self.pending['chosen']]),
                     observed={k: round(v-previous.get(k, 0), 6) for k, v in current.items()},
                     reward=dict(self.reward_parts))
        self.journal.append(event)
        self.journal = self.journal[-self.config.journal_capacity:]
        self.observed_transitions += 1
        self.pending = None
        self.reward_parts = {}

    def _choice(self, candidates, senses, history):
        engine = self._load_engine()
        if getattr(engine, 'cfg', {}).get('head_max_len', 192) < self.config.head_tokens:
            raise ValueError('Laya adapter head is smaller than the checked question budget')
        ordered = list(candidates)
        self.rng.shuffle(ordered)
        criteria = {f'c{i}': ' '.join(str(v) for v in ACTIONS[index].values())
                    for i, index in enumerate(ordered)}
        question = dict(type='choice', instructions=INSTRUCTION, criteria=criteria)
        tok = engine.tok
        token_ids = lambda s: tok(s, add_special_tokens=False)['input_ids']
        mask_token = getattr(tok, 'mask_token', '[MASK]')
        head = token_ids('choice question: '+INSTRUCTION)
        option_ids = [token_ids(' '+key+': '+value) for key, value in criteria.items()]
        if any(len(ids) > 48 for ids in option_ids):
            raise ValueError('Laya action option would be truncated')
        option_total = sum(1+len(ids) for ids in option_ids)
        if len(head) > self.config.head_tokens-option_total or self.config.head_tokens-option_total < 16:
            raise ValueError('Laya question head would be truncated')
        # Four special tokens: CLS, separator after question, after options,
        # and after state. Mirrors existing build_sequence exactly.
        room = self.config.max_tokens-(len(head)+option_total+4)
        state = dict(senses=senses, history=list(history))
        omitted = []
        def length():
            return len(token_ids(json.dumps(state, ensure_ascii=False).replace(mask_token, ' ')))
        while length() > room:
            if state['history']:
                state['history'].pop(0); omitted.append('oldest_history')
            else:
                field = next((k for k in ('hearing', 'neighbors', 'resources') if state['senses'][k]), None)
                if field is None:
                    raise ValueError('Laya core local state exceeds token budget')
                state['senses'] = dict(state['senses'])
                state['senses'][field] = list(state['senses'][field][:-1])
                omitted.append(field)
        answer = engine.system_one(state, {'action': question})['answers']['action']
        label = answer['choice']
        if label not in criteria:
            raise ValueError('Laya returned an action outside the offered options')
        self.diagnostics['calls'] += 1
        self.diagnostics['context_omissions'].extend(omitted)
        self.diagnostics['max_input_tokens'] = max(self.diagnostics['max_input_tokens'],
                                                  len(head)+option_total+4+length())
        return ordered[list(criteria).index(label)]

    def decide(self, observation):
        self._complete(observation)
        mask = torch.tensor([action_mask(observation, ACTIONS)], dtype=torch.bool)
        feasible = [i for i, allowed in enumerate(mask[0].tolist()) if allowed]
        if not feasible:
            raise ValueError('No physically possible Laya action')
        senses = compact_observation(observation)
        history = self.journal[-self.config.history_capacity:] if self.config.history_capacity else []
        self.diagnostics = dict(calls=0, context_omissions=[], max_input_tokens=0,
                                feasible_actions=len(feasible), selection='model tournament')
        self.rng.shuffle(feasible)
        candidates = feasible
        while len(candidates) > self.config.group_size:
            candidates = [self._choice(candidates[i:i+self.config.group_size], senses, history)
                          for i in range(0, len(candidates), self.config.group_size)]
        chosen = self._choice(candidates, senses, history)
        self.pending = dict(chosen=chosen, mask=mask, body=body_values(observation))
        self.decisions += 1
        return dict(ACTIONS[chosen])

    def inspector(self):
        return dict(backend=self.backend, masked=True, training=False, updates=0,
                    decisions=self.decisions, policy_version=0, rollout_fill=0,
                    rollout_capacity=0, diagnostics=self.diagnostics, preferences=[],
                    recent=self.journal[-5:], observed_transitions=self.observed_transitions,
                    weights='frozen existing typed-decisions checkpoint',
                    memory='own bounded observed consequence context; no weight learning',
                    history_capacity=self.config.history_capacity, predictor=False,
                    compressed_fields=['body', 'facing', 'touch', 'nearest resource per kind (max6)',
                                       'neighbors(max3)', 'hearing(max3)', 'recent tones(max5)', 'water_carried'],
                    omitted_senses=['full tile grid', 'visual recall sketches',
                                    'soil', 'structures', 'neighbor colors', 'attention', 'carried cache'])

    def state(self):
        return dict(backend=self.backend, seed=self.seed, config=asdict(self.config),
                    rng=self.rng.getstate(), **{k: getattr(self, k) for k in
                    ('decisions', 'pending', 'journal', 'reward_parts', 'observed_transitions', 'diagnostics')})

    def restore(self, data):
        if data['backend'] != self.backend or data['config'] != asdict(self.config) or data['seed'] != self.seed:
            raise ValueError('Incompatible Laya resident checkpoint')
        self.rng.setstate(data['rng'])
        for key in ('decisions', 'pending', 'journal', 'reward_parts', 'observed_transitions', 'diagnostics'):
            setattr(self, key, data[key])
