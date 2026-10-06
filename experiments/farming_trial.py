"""Controlled delayed-harvest experiment. Local disposable worlds only; not navigation/PPO."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.world import World, SIZE

SEEDS = (4813, 5824, 6835)
TRAIN_EPISODES = 12
EVAL_EPISODES = 3
DECISIONS = 80
STRIDE = 120
VERBS = ('rest', 'eat', 'make_seeds', 'plant', 'gather')


def arena(seed, competition):
    w = World(seed, memory_capacity=1)
    a = w.residents['player']
    a.x = a.y = 32
    a.food = 40
    a.inventory['food'] = 2
    w.residents = {'player': a, **({'r0': w.residents['r0']} if competition else {})}
    if competition:
        w.residents['r0'].x, w.residents['r0'].y = 33, 32
    w.resources.clear()
    w.structures.clear()
    w.terrain = [[0] * SIZE for _ in range(SIZE)]
    return w


def sense(w):
    o = w.observe('player', include_memory=False)
    tile = next(t for t in o['tiles'] if t['dx'] == t['dy'] == 0)
    r = tile.get('resource', {})
    inv = o['inventory']
    food, seeds = inv['food'], inv['seed']
    phase = 1 + r.get('stage', 0) if r.get('kind') == 'crop' else 4 if r.get('kind') == 'berry' else 0
    mask = [True, food > 0 and o['needs'][0] <= 95,
            food > 0 and sum(inv.values()) < 12, seeds > 0 and not r,
            r.get('kind') == 'berry' and r.get('amount', 0) > 0 and sum(inv.values()) < 12]
    key = (int(o['needs'][0] // 20), food, seeds, phase, r.get('amount', 0), bool(o['others']))
    brief = {'fullness': round(o['needs'][0]), 'food': food, 'seeds': seeds,
             'plot_stage': phase, 'fruit_on_plot': r.get('amount', 0), 'neighbor_visible': bool(o['others'])}
    return key, mask, brief


def competitor(w):
    if 'r0' not in w.residents:
        return {}
    o = w.observe('r0', include_memory=False)
    if o['inventory']['food'] and o['needs'][0] <= 95:
        return {'r0': {'verb': 'eat'}}
    fruit = any(t.get('resource', {}).get('kind') == 'berry' and t['resource']['amount'] > 0
                and abs(t['dx']) + abs(t['dy']) <= 1 for t in o['tiles'])
    return {'r0': {'verb': 'gather' if fruit else 'rest'}}


class Learner:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.] * len(VERBS))
        self.trace = {}

    def choose(self, key, mask, train):
        legal = [i for i, yes in enumerate(mask) if yes]
        best = max(self.q[key][i] for i in legal)
        greedy = [i for i in legal if self.q[key][i] == best]
        action = self.rng.choice(legal if train and self.rng.random() < .2 else greedy)
        # Watkins Q(lambda): clear old eligibility when taking a non-greedy action.
        if action not in greedy:
            self.trace.clear()
        return action

    def update(self, key, action, reward, next_key, next_mask, terminal):
        future = 0 if terminal else max(v for v, yes in zip(self.q[next_key], next_mask) if yes)
        error = reward + .995 * future - self.q[key][action]
        self.trace[(key, action)] = 1.
        for pair, eligibility in list(self.trace.items()):
            state, act = pair
            self.q[state][act] += .15 * error * eligibility
            eligibility *= .995 * .9
            if eligibility < .001:
                del self.trace[pair]
            else:
                self.trace[pair] = eligibility


def laya_choice(engine, adapter, brief, recent, mask, seed):
    names = [v for i, v in enumerate(VERBS) if mask[i]]
    offset = seed % len(names)
    names = names[offset:] + names[:offset]
    state = {'senses': brief, 'recent': recent[-3:]}
    q = {'action': {'type': 'choice', 'instructions':
         'Choose to gain fullness from eating across a long life. Plot stages: 0 empty, 1-3 growing, 4 ripe. Actions use normal food, seeds, planting and harvest. Neighbors may harvest too.',
         'criteria': {name: None for name in names if mask[VERBS.index(name)]}}}
    seq, _ = adapter.build_sequence(engine.tok, state, adapter.to_internal(q['action']), 4096,
                                     engine.cfg.get('head_max_len', 192))
    if len(seq) > adapter.L:
        raise ValueError('Laya input would truncate')
    answer = engine.system_one(state, q)['answers']['action']['choice']
    return VERBS.index(answer)


def episode(kind, seed, competition, learner, training=False, engine=None, adapter=None):
    w = arena(seed, competition)
    a = w.residents['player']
    rng = random.Random(seed)
    learner.trace.clear()
    learner.rng = random.Random(seed ^ 0xACE)
    row = {'seed': seed, 'competition': competition, 'training': training, 'nutrition': 0.,
           'meals': 0, 'conversions': 0, 'planted': 0, 'harvested': 0, 'invalid': 0, 'low_food_ticks': 0, 'zero_food_ticks': 0, 'fullness_sum': 0.}
    recent, latencies = [], []
    history = []
    for decision in range(DECISIONS):
        key, mask, brief = sense(w)
        start = time.perf_counter()
        if kind == 'laya':
            action = laya_choice(engine, adapter, brief, recent, mask, seed + decision)
        elif kind == 'random':
            action = rng.choice([i for i, yes in enumerate(mask) if yes])
        elif kind == 'scripted':
            # Recipe-informed positive control, not learned behavior.
            if a.food < 25 and mask[1]:
                action = 1
            elif mask[4]:
                action = 4
            elif mask[3]:
                action = 3
            elif mask[2] and not a.inventory['seed'] and not w.resources:
                action = 2
            elif mask[1] and a.food < 70:
                action = 1
            else:
                action = 0
        else:
            action = learner.choose(key, mask, training)
        latencies.append((time.perf_counter() - start) * 1000)
        food = a.food
        command = {'verb': VERBS[action]}
        result = w.step({'player': command, **competitor(w)})['player']
        nutrition = max(0., a.food - food + .008) if action == 1 and result[0] else 0.
        row['fullness_sum'] += a.food
        row['low_food_ticks'] += int(a.food < 20)
        row['zero_food_ticks'] += int(a.food == 0)
        for _ in range(STRIDE - 1):
            w.step({rid: {'verb': 'wait'} for rid in w.residents})
            row['low_food_ticks'] += int(a.food < 20)
            row['zero_food_ticks'] += int(a.food == 0)
            row['fullness_sum'] += a.food
        reward = nutrition / 25
        next_key, next_mask, _ = sense(w)
        if training:
            learner.update(key, action, reward, next_key, next_mask, decision == DECISIONS - 1)
        row['nutrition'] += nutrition
        row['invalid'] += int(not result[0])
        if result[0]:
            for verb, label in [('eat', 'meals'), ('make_seeds', 'conversions'), ('plant', 'planted'), ('gather', 'harvested')]:
                row[label] += int(VERBS[action] == verb)
        recent.append({'action': VERBS[action], 'reward': round(reward, 3)})
        history.append({'decision': decision, 'observation': brief, 'action': VERBS[action], 'reward': reward})
    row.update(nutrition=round(row['nutrition'], 3), fullness=round(a.food, 3),
               rival_harvested=w.residents['r0'].gathered if competition else 0,
               median_ms=statistics.median(latencies), ticks=w.tick,
               mean_fullness=round(row.pop('fullness_sum') / w.tick, 3),
               final_inventory=dict(a.inventory), visited_states=len(learner.q))
    return row, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--laya-adapter')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    prereg = {'seeds': SEEDS, 'train_episodes': TRAIN_EPISODES, 'evaluation_episodes': EVAL_EPISODES,
              'laya_eval_episodes': 1, 'decisions': DECISIONS, 'stride': STRIDE,
              'hypothesis': 'Trained Q(lambda) exceeds paired untrained control nutrition in both solo and competition on all three seeds, with >2 meals and at least one harvest.',
              'q_parameters': {'alpha': .15, 'gamma': .995, 'lambda': .9, 'epsilon_train': .2, 'epsilon_eval': 0, 'initial_q': 0, 'traces': 'Watkins replacing, clear on non-greedy, reset each episode', 'bins': 'fullness floor/20, exact inventory, visible crop stage/amount, neighbor presence'},
              'reward': 'Actual useful fullness gained from eating /25 only; no farming bonuses.',
              'limits': 'Fixed position, one shared plot; known legal action masks; repeated independent training episodes; no navigation, reproduction, mortality or live-game integration. Frozen pretrained Laya is a different prior, not a blank-slate learner.',
              'hashes': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['experiments/farming_trial.py', 'sim/world.py']}}
    (out/'preregistration.json').write_text(json.dumps(prereg, indent=2))
    rows = []
    for competition in (False, True):
        for seed in SEEDS:
            trained = Learner(seed)
            for ep in range(TRAIN_EPISODES):
                row, _ = episode('q_lambda', seed + ep*10000, competition, trained, training=True)
                row.update(kind='q_lambda', base_seed=seed, episode=ep)
                rows.append(row)
            for kind in ('q_lambda', 'frozen', 'random', 'scripted'):
                policy = trained if kind == 'q_lambda' else Learner(seed)
                for ep in range(EVAL_EPISODES):
                    row, history = episode(kind, seed + 1000000 + ep*10000, competition, policy)
                    row.update(kind=kind, base_seed=seed, episode=ep)
                    rows.append(row)
                    (out/f'{kind}-{competition}-{seed}-{ep}.json').write_text(json.dumps(history))
            print(f'Completed small policies: competition={competition}, seed={seed}', flush=True)
    (out/'results.json').write_text(json.dumps(rows, indent=2))
    if args.laya_adapter:
        os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
        spec = importlib.util.spec_from_file_location('local_laya', args.laya_adapter)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        engine = adapter.LayaLite('NPU')
        for competition in (False, True):
            for seed in SEEDS:
                row, history = episode('laya', seed + 1000000, competition, Learner(seed), engine=engine, adapter=adapter)
                row.update(kind='laya', base_seed=seed, episode=0)
                rows.append(row)
                (out/f'laya-{competition}-{seed}.json').write_text(json.dumps(history))
                (out/'results.json').write_text(json.dumps(rows, indent=2))
                print(f'Completed Laya: competition={competition}, seed={seed}', flush=True)
    (out/'complete.json').write_text(json.dumps({'rows': len(rows), 'complete': True}))


if __name__ == '__main__':
    main()
