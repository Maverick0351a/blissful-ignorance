"""Independent masked recurrent PPO, with contiguous on-policy sequences.

This is our PyTorch implementation, not an SB3 wrapper. Rollout boundaries
deliberately reset recurrent state after updates. Pauses and fainting do not.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from torch import nn

from agents.affordances import action_mask
from agents.physiology import food_relief
from agents.preservation import decision_transaction, require_consistent, validate_roster
from agents.network import ACTIONS, FEATURES, PATCH_CHANNELS, ResidentNetwork, encode_observation
from sim.world import World

torch.set_num_threads(1)


@dataclass(frozen=True)
class Config:
    hidden_size: int = 64
    rollout: int = 128
    epochs: int = 4
    learning_rate: float = .0003
    gamma: float = .997
    gae_lambda: float = .98
    clip: float = .2
    entropy: float = .01
    target_kl: float = .03
    journal_capacity: int = 128
    curiosity_coefficient: float = 0.

    def __post_init__(self):
        if min(self.hidden_size, self.rollout, self.epochs, self.journal_capacity) <= 0:
            raise ValueError('Positive capacity and update budgets are required')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('Invalid discount settings')
        if not 0 <= self.curiosity_coefficient <= .01:raise ValueError('Curiosity budget must be in [0, .01]')


def advantages(rewards, values, bootstrap, terminals, gamma, lam):
    """Terminal cuts credit; a finite rollout cutoff bootstraps normally."""
    result = torch.zeros_like(values)
    carry = values.new_tensor(0.)
    future = values.new_tensor(bootstrap)
    for i in reversed(range(len(rewards))):
        continuing = 1. - float(terminals[i])
        delta = rewards[i] + gamma * future * continuing - values[i]
        carry = delta + gamma * lam * continuing * carry
        result[i] = carry
        future = values[i]
    return result, result + values


OUTCOMES = ('fullness', 'hydration', 'health', 'pain', 'food_carried', 'wood_carried', 'seeds_carried')
OUTCOME_SCALES = (25,35,20,20,1,1,1)


def body_values(observation):
    inv = observation['inventory']; body = observation['body']
    return torch.tensor([observation['needs'][0]/25, observation['needs'][1]/35,
                         body['health']/20, body['pain']/20,
                         inv.get('food', 0), inv.get('wood', 0), inv.get('seed', 0)])


def consequence_input(patch, features, chosen):
    symbol = torch.zeros((1, len(ACTIONS))); symbol[0, chosen] = 1
    return torch.cat((patch.mean((2, 3)), features, symbol), 1)


class Brain:
    backend = 'recurrent-ppo'
    network_class = ResidentNetwork

    def __init__(self, seed, config=None):
        self.seed = seed; self.config = config or Config()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = self.network_class(hidden_size=self.config.hidden_size)
            self.predictor = nn.Sequential(nn.Linear(PATCH_CHANNELS+FEATURES+len(ACTIONS), 64),
                                           nn.Tanh(), nn.Linear(64, len(OUTCOMES)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=.001)
        self.rng = random.Random(seed)
        self.hidden = self.model.initial_state()
        self.masked = True; self.training = True
        self.pending = None; self.reward = 0.; self.reward_parts = {}
        self.buffer = []; self.journal = []
        self.decisions = 0; self.updates = 0; self.policy_version = 0
        self.diagnostics = {}; self.preferences = []
        self.prediction_errors = []; self.observed_transitions = 0; self.prediction_error_total = 0.
        self.progress_fast = None; self.progress_slow = None

    def add_reward(self, parts):
        if not self.training: return
        for key, value in parts.items():
            self.reward_parts[key] = self.reward_parts.get(key, 0.) + float(value)
            self.reward += float(value)

    def _complete(self, observation, terminal=False):
        if self.pending is None: return
        transition = self.pending
        actual = body_values(observation) - transition['body']
        error = float((actual-transition['prediction']).abs().mean())
        self.observed_transitions += 1; self.prediction_error_total += error
        if self.training:
            if self.progress_fast is None:self.progress_fast=self.progress_slow=error
            else:
                self.progress_fast += .05*(error-self.progress_fast)
                self.progress_slow += .005*(error-self.progress_slow)
            # Experimental learning-progress proxy. Raw surprise earns nothing.
            progress = max(0.,self.progress_slow-self.progress_fast)
            bonus = self.config.curiosity_coefficient*min(1.,progress)
            if bonus:self.reward+=bonus;self.reward_parts['curiosity']=bonus
        event = {'decision': self.decisions, 'policy_version': transition['version'],
                 'action': dict(ACTIONS[transition['chosen']]),
                 'predicted': {k: round(float(v)*scale, 4) for k,v,scale in zip(OUTCOMES,transition['prediction'],OUTCOME_SCALES)},
                 'observed': {k: round(float(v)*scale, 4) for k,v,scale in zip(OUTCOMES,actual,OUTCOME_SCALES)},
                 'prediction_mae': error, 'reward': dict(self.reward_parts)}
        if self.training:
            self.journal.append(event); self.journal = self.journal[-self.config.journal_capacity:]
            self.prediction_errors.append(error); self.prediction_errors = self.prediction_errors[-128:]
            transition.update(reward=self.reward, terminal=terminal, outcome=actual)
            self.buffer.append(transition)
        self.pending = None; self.reward = 0.; self.reward_parts = {}

    def distribution(self, logits, mask):
        """The flat production policy; experiments may override only this head."""
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))

    def decide(self, observation):
        self._complete(observation)
        patch, features = encode_observation(observation)
        if self.training and len(self.buffer) >= self.config.rollout:
            with torch.no_grad(): _, future, _ = self.model(patch, features, self.hidden)
            self._learn(float(future))
        mask = torch.tensor([action_mask(observation, ACTIONS)])
        with torch.no_grad():
            logits, value, hidden = self.model(patch, features, self.hidden)
            distribution = self.distribution(logits, mask)
            probabilities = distribution.probs[0].tolist()
            chosen = self.rng.choices(range(len(ACTIONS)), weights=probabilities)[0]
            predictor_input = consequence_input(patch, features, chosen)
            prediction = self.predictor(predictor_input)[0]
        self.preferences = [{'action': dict(ACTIONS[i]), 'probability': round(probabilities[i], 4)}
                            for i in sorted(range(len(ACTIONS)), key=lambda i: probabilities[i], reverse=True)[:3]]
        self.pending = dict(patch=patch, features=features, hidden=self.hidden, chosen=chosen,
                            mask=mask, logp=float(distribution.log_prob(torch.tensor(chosen))),
                            value=float(value), version=self.policy_version, body=body_values(observation),
                            predictor_input=predictor_input, prediction=prediction)
        self.hidden = tuple(t.detach() for t in hidden)
        self.decisions += 1
        return dict(ACTIONS[chosen])

    def _sequence(self, patch, features, hidden):
        # The CNN is batched; temporal recurrence stays contiguous for BPTT.
        encoded = torch.cat((self.model.patch_encoder(patch), self.model.feature_encoder(features)), -1)
        logits = []; values = []
        for entry in encoded:
            hidden = self.model.recurrent(entry[None], hidden)
            logits.append(self.model.actor(hidden[0])); values.append(self.model.value(hidden[0]).squeeze(-1))
        return torch.cat(logits), torch.cat(values)

    def _learn(self, bootstrap):
        rows = self.buffer
        if not rows: return
        if len({t['version'] for t in rows}) != 1 or rows[0]['version'] != self.policy_version:
            raise RuntimeError('Rollout contains stale behavior policies')
        patch = torch.cat([t['patch'] for t in rows]); features = torch.cat([t['features'] for t in rows])
        masks = torch.cat([t['mask'] for t in rows]); selected = torch.tensor([t['chosen'] for t in rows])
        old_logp = torch.tensor([t['logp'] for t in rows]); values = torch.tensor([t['value'] for t in rows])
        adv, returns = advantages([t['reward'] for t in rows], values, bootstrap,
                                  [t['terminal'] for t in rows], self.config.gamma, self.config.gae_lambda)
        normalized = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
        epochs = 0; kl = 0.; clip_fraction = 0.
        for _ in range(self.config.epochs):
            logits, predictions = self._sequence(patch, features, rows[0]['hidden'])
            distribution = self.distribution(logits, masks)
            logp = distribution.log_prob(selected); delta = logp-old_logp; ratio = delta.exp()
            kl = float(((ratio-1)-delta).mean().detach())
            if kl > self.config.target_kl * 1.5: break
            actor_loss = -torch.minimum(ratio*normalized,
                                       ratio.clamp(1-self.config.clip, 1+self.config.clip)*normalized).mean()
            value_loss = .5*(predictions-returns).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + value_loss - self.config.entropy*entropy
            if not torch.isfinite(loss): raise RuntimeError('Non-finite PPO loss')
            self.optimizer.zero_grad(); loss.backward()
            gradient = nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step(); epochs += 1
            clip_fraction = float(((ratio-1).abs() > self.config.clip).float().mean())
        # A separate predictor learns only this resident's observed consequences.
        inputs = torch.cat([t['predictor_input'] for t in rows]); outcomes = torch.stack([t['outcome'] for t in rows])
        prediction_loss = nn.functional.smooth_l1_loss(self.predictor(inputs), outcomes)
        self.predictor_optimizer.zero_grad(); prediction_loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.); self.predictor_optimizer.step()
        self.updates += 1; self.policy_version += 1
        self.diagnostics = dict(sequence_length=len(rows), epochs=epochs, approximate_kl=kl,
                                clip_fraction=clip_fraction, entropy=float(entropy.detach()),
                                gradient_norm=float(gradient), mean_return=float(returns.mean()),
                                predictor_loss=float(prediction_loss.detach()), recurrent_boundary='reset_after_update')
        self.buffer = []; self.hidden = self.model.initial_state()

    def finish(self, observation, terminal=False):
        self._complete(observation, terminal)
        if self.training and self.buffer:
            patch, features = encode_observation(observation)
            with torch.no_grad(): _, future, _ = self.model(patch, features, self.hidden)
            self._learn(0. if terminal else float(future))
        self.pending = None; self.hidden = self.model.initial_state()

    def freeze(self):
        self.training = False; self.buffer = []; self.pending = None
        self.reward = 0.; self.reward_parts = {}; self.hidden = self.model.initial_state()

    def inspector(self):
        return dict(backend=self.backend, masked=True, training=self.training, updates=self.updates,
                    decisions=self.decisions, policy_version=self.policy_version, rollout_fill=len(self.buffer),
                    rollout_capacity=self.config.rollout, diagnostics=self.diagnostics,
                    preferences=self.preferences, recent=self.journal[-5:],
                    prediction_mae=sum(self.prediction_errors)/max(1,len(self.prediction_errors)),
                    curiosity_coefficient=self.config.curiosity_coefficient,
                    learning_progress=max(0.,(self.progress_slow or 0.)-(self.progress_fast or 0.)))

    def state(self):
        return dict(backend=self.backend, config=asdict(self.config), seed=self.seed,
                    model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    predictor=self.predictor.state_dict(), predictor_optimizer=self.predictor_optimizer.state_dict(),
                    rng=self.rng.getstate(), **{k:getattr(self,k) for k in ('hidden','training','pending','reward',
                    'reward_parts','buffer','journal','decisions','updates','policy_version','diagnostics',
                    'preferences','prediction_errors','observed_transitions','prediction_error_total',
                    'progress_fast','progress_slow')})

    def restore(self, data):
        if data['backend'] != self.backend or data['config'] != asdict(self.config):
            raise ValueError('Incompatible PPO checkpoint')
        for key in ('model','optimizer','predictor','predictor_optimizer'):getattr(self,key).load_state_dict(data[key])
        self.rng.setstate(data['rng'])
        for key in ('hidden','training','pending','reward','reward_parts','buffer','journal','decisions','updates',
                    'policy_version','diagnostics','preferences','prediction_errors','observed_transitions',
                    'prediction_error_total','progress_fast','progress_slow'):
            setattr(self,key,data[key])


class Population:
    backend = 'recurrent-ppo'

    @staticmethod
    def new_metrics():
        return dict(invalid=0, nutrition=0., zero_food_ticks=0, unconscious_ticks=0,
                                conscious_decisions=0, conscious_invalid=0, damage=0., safe_eaten=0, amber_eaten=0,
                                prediction_error_sum=0., prediction_events=0, thorn_contacts=0)

    def __init__(self, seed=5729, config=None, *, world=None, resident_ids=('r0','r1')):
        self.world = world if world is not None else World(seed)
        if world is None:self.world.ecology_enabled = True
        ids = tuple(resident_ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(self.world.residents)-{'player'}:
            raise ValueError('Learner identities must be unique autonomous residents')
        self.brains = {rid:Brain(seed+i,config) for i,rid in enumerate(ids)}
        self.metrics = {rid:self.new_metrics() for rid in self.brains}
        self.neighbors_scripted = True
        if world is None:
            for rid in self.brains:
                self.world.residents[rid].food = 50
                self.world.residents[rid].inventory.update(food=2,wood=3)

    @classmethod
    def from_world(cls, world, config=None):
        """Attach fresh brains without changing a saved body's resources or memory."""
        result = cls(world.seed, config, world=world,
                     resident_ids=tuple(rid for rid in world.residents if rid != 'player'))
        result.neighbors_scripted = False
        return result

    def enable_all_residents(self):
        """Migrate scripted neighbors; never repair a missing saved controller."""
        require_consistent(self)
        validate_roster(self.world, self.brains, self.metrics, self.neighbors_scripted, require_all=True)
        config = next(iter(self.brains.values())).config
        for i, rid in enumerate(rid for rid in self.world.residents if rid != 'player'):
            if rid not in self.brains:
                self.brains[rid] = Brain(self.world.seed+i, config)
                self.metrics[rid] = self.new_metrics()
        self.neighbors_scripted = False

    def step(self, player_command=None):
        require_consistent(self)
        deciding = self.world.tick % 4 == 0
        commands = {}
        # Keep external inference first, but roll back every brain if any decision
        # fails, including a later PPO update after successful Laya inference.
        local_backends = ('recurrent-ppo', 'feedforward-replay', 'recurrent-replay',
                          'experimental-retained-replay', 'experimental-category-ppo')
        external = {rid:b for rid,b in self.brains.items() if b.backend not in local_backends}
        if deciding:
            with decision_transaction(self):
                commands.update({rid:b.decide(self.world.observe(rid)) for rid,b in external.items()})
                commands.update({rid:brain.decide(self.world.observe(rid))
                                 for rid,brain in self.brains.items() if rid not in commands})
        else:
            commands = {rid:{'verb':'wait'} for rid in self.brains}
        try:
            return self._advance(commands, deciding, player_command)
        except BaseException:
            # A failure after applying world actions cannot safely retry or save
            # partial physics/reward state. Preserve the last complete disk save.
            self.recovery_required = 'World transition failed. Saving and advancing are blocked; reload a complete checkpoint or restart.'
            raise

    def _advance(self, commands, deciding, player_command):
        if not self.neighbors_scripted:
            commands.update({rid:{'verb':'rest'} for rid in self.world.residents if rid not in self.brains})
        if player_command: commands['player'] = player_command
        before = {rid:(a.food,a.health,a.pain,a.unconscious)
                  for rid in self.brains for a in (self.world.residents[rid],)}
        results = self.world.step(commands, scripted=self.neighbors_scripted)
        for rid, brain in self.brains.items():
            a = self.world.residents[rid]; food,health,pain,unconscious = before[rid]; m = self.metrics[rid]
            relief = food_relief(food,a.food)
            # Pain rise remains observable at the health floor; recovery earns nothing.
            injury = max(0., health-a.health, a.pain-pain)
            parts = dict(food=relief/25, hunger=-min(1.,max(0.,(60-a.food)/60))*.005,
                         injury=-injury/20)
            brain.add_reward(parts)
            m['nutrition'] += relief; m['zero_food_ticks'] += a.food <= 0
            m['unconscious_ticks'] += a.unconscious; m['damage'] += injury
            if deciding:
                success = results[rid][0]; command = commands[rid]
                m['invalid'] += not success
                if not unconscious:
                    m['conscious_decisions'] += 1; m['conscious_invalid'] += not success
                if success and command['verb'] == 'eat':
                    m['amber_eaten' if command.get('item') == 'amber_fruit' else 'safe_eaten'] += 1
                m['prediction_error_sum'] = brain.prediction_error_total
                m['prediction_events'] = brain.observed_transitions
            m['thorn_contacts'] = a.thorn_contacts
                # Diagnostic prediction errors are evaluated on actual 4-tick transitions
                # by the experiment; this per-tick code doesn't fabricate them.
        return results

    def save(self, path):
        require_consistent(self)
        validate_roster(self.world, self.brains, self.metrics, self.neighbors_scripted)
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary = path.with_suffix('.tmp')
        torch.save(dict(schema=2,backend=self.backend,world=self.world.to_dict(),neighbors_scripted=self.neighbors_scripted,
                        brains={rid:b.state() for rid,b in self.brains.items()},metrics=self.metrics),temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, require_all=False):
        data = torch.load(path,map_location='cpu',weights_only=True)
        if data['schema'] != 2 or data['backend'] != cls.backend:raise ValueError('Unsupported PPO population')
        world = World.from_dict(data['world'])
        validate_roster(world, data['brains'], data['metrics'], data['neighbors_scripted'], require_all=require_all)
        result = cls(world.seed, world=world, resident_ids=tuple(data['brains']))
        for rid, saved in data['brains'].items():
            if saved.get('backend') == 'laya-npu':
                from agents.laya_resident import LayaBrain, Config as LayaConfig
                brain = LayaBrain(saved['seed'], LayaConfig(**saved['config']))
            elif saved.get('backend') == 'recurrent-ppo':
                brain = Brain(saved['seed'], Config(**saved['config']))
            elif saved.get('backend') in ('feedforward-replay', 'recurrent-replay'):
                from agents.sequence_replay import Brain as ReplayBrain, Config as ReplayConfig
                brain = ReplayBrain(saved['seed'], ReplayConfig(**saved['config']))
            elif saved.get('backend') == 'experimental-retained-replay':
                from agents.retained_replay import Brain as RetainedBrain, Config as RetainedConfig
                brain = RetainedBrain(saved['seed'], RetainedConfig(**saved['config']))
            else:raise ValueError('Unsupported resident model; no replacement controller was created')
            brain.restore(saved);result.brains[rid] = brain
        result.metrics = data['metrics']; result.neighbors_scripted = data['neighbors_scripted']
        return result
