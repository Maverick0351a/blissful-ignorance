"""Private visual Double-DQN learners, with optional recurrent sequence replay.

This is a small R2D2-inspired experiment, not a reproduction of distributed
R2D2. Both variants use one-step masked Double-DQN targets, prioritized chunks
and importance weighting. Recurrent replay restores a stored behavior carry,
then burns in context separately under current online and target weights.
"""
from dataclasses import asdict, dataclass
import copy
import random

import torch
from torch import nn

from agents.affordances import action_mask
from agents.network import ACTIONS, ResidentNetwork, encode_observation


@dataclass(frozen=True)
class Config:
    recurrent: bool = True
    hidden_size: int = 64
    unroll: int = 16
    burn_in: int = 8
    capacity: int = 64  # Sequence chunks, private to each resident.
    batch_size: int = 4
    warmup: int = 4
    learning_rate: float = .0003
    gamma: float = .997
    epsilon: float = .25
    eval_epsilon: float = .05
    priority_alpha: float = .6
    target_interval: int = 32

    def __post_init__(self):
        if min(self.hidden_size, self.unroll, self.capacity, self.batch_size,
               self.warmup, self.target_interval) <= 0:
            raise ValueError('Positive replay and model budgets are required')
        if self.burn_in < 0 or self.warmup > self.capacity:
            raise ValueError('Invalid replay context or warmup')
        if not all(0 <= x <= 1 for x in (self.gamma, self.epsilon,
                                        self.eval_epsilon, self.priority_alpha)):
            raise ValueError('Invalid probability or discount')
        if self.learning_rate <= 0:
            raise ValueError('Learning rate must be positive')


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Identical sensory encoders to PPO. No privileged features are added.
        base = ResidentNetwork(config.hidden_size)
        self.patch_encoder = base.patch_encoder
        self.feature_encoder = base.feature_encoder
        self.recurrent = base.recurrent if config.recurrent else None
        self.core = None if config.recurrent else nn.Sequential(
            nn.Linear(80, config.hidden_size), nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU())
        self.q = nn.Linear(config.hidden_size, len(ACTIONS))
        self.hidden_size = config.hidden_size

    def initial_state(self, batch_size=1):
        zeros = self.q.weight.new_zeros(batch_size, self.hidden_size)
        return zeros, zeros.clone()

    def encode(self, patch, features):
        return torch.cat((self.patch_encoder(patch), self.feature_encoder(features)), -1)

    def step(self, encoded, hidden):
        if self.recurrent is None:
            return self.q(self.core(encoded)), hidden
        hidden = self.recurrent(encoded, hidden)
        return self.q(hidden[0]), hidden

    def forward(self, patch, features, hidden):
        return self.step(self.encode(patch, features), hidden)

    def sequence(self, patch, features, hidden, burn_in=0):
        """One contiguous sequence; burn-in reconstructs carry without gradients."""
        if not 0 <= burn_in < len(patch):
            raise ValueError('Burn-in must leave a learning observation')
        if burn_in:
            with torch.no_grad():
                context = self.encode(patch[:burn_in], features[:burn_in])
                for encoded in context:
                    _, hidden = self.step(encoded[None], hidden)
            hidden = tuple(x.detach() for x in hidden)
        encoded = self.encode(patch[burn_in:], features[burn_in:])
        values = []
        for entry in encoded:
            q, hidden = self.step(entry[None], hidden)
            values.append(q)
        return torch.cat(values), hidden


def double_targets(rewards, online_next, target_next, masks, terminals, gamma):
    """Only feasible next actions compete. True terminals never bootstrap."""
    if not bool(masks.any(-1).all()):
        raise ValueError('Every observation must have a feasible action')
    with torch.no_grad():
        selected = online_next.masked_fill(~masks, -torch.inf).argmax(-1)
        future = target_next.gather(-1, selected[:, None]).squeeze(-1)
        return rewards + gamma * torch.where(terminals, torch.zeros_like(future), future)


class Brain:
    def __init__(self, seed, config=None):
        self.seed = seed
        self.config = config or Config()
        self.backend = 'recurrent-replay' if self.config.recurrent else 'feedforward-replay'
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = QNetwork(self.config)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.rng = random.Random(seed)
        self.replay_rng = random.Random(seed ^ 0xDB12)
        self.hidden = self.model.initial_state()
        self.training = True
        self.masked = True
        self.pending = None
        self.reward = 0.
        self.context = []
        self.segment = []
        self.replay = []
        self.priorities = []
        self.decisions = self.updates = self.transitions = 0
        self.observed_transitions = 0  # Population's consequence-prediction event count.
        self.prediction_error_total = 0.  # No consequence predictor in these variants.
        self.diagnostics = {}

    def add_reward(self, parts):
        if self.training:
            self.reward += sum(float(x) for x in parts.values())

    def _complete(self, patch, features, mask, terminal=False):
        if self.pending is None:
            return
        if self.training:
            row = self.pending
            row.update(reward=self.reward, next_patch=patch, next_features=features,
                       next_mask=mask, terminal=terminal)
            self.segment.append(row)
            self.transitions += 1
            if len(self.segment) >= self.config.unroll or terminal:
                self._emit()
        self.pending = None
        self.reward = 0.

    def _emit(self):
        if not self.segment:
            return
        rows = self.context + self.segment
        self.store_chunk(dict(rows=rows, burn_in=len(self.context)))
        self.context = rows[-self.config.burn_in:] if self.config.burn_in else []
        if rows[-1]['terminal']:
            self.context = []
        self.segment = []
        if len(self.replay) >= self.config.warmup:
            self.train_batch()

    def store_chunk(self, chunk):
        self.replay.append(chunk)
        self.priorities.append(max(self.priorities, default=1.))
        if len(self.replay) > self.config.capacity:
            self.replay.pop(0)
            self.priorities.pop(0)

    def explore(self, allowed):
        return self.rng.choice(allowed)

    def sample_indices(self, weights):
        return self.replay_rng.choices(range(len(self.replay)), weights=weights,
                                       k=self.config.batch_size)

    def decide(self, observation):
        patch, features = encode_observation(observation)
        mask = torch.tensor([action_mask(observation, ACTIONS)])
        self._complete(patch, features, mask)
        with torch.no_grad():
            scores, hidden = self.model(patch, features, self.hidden)
        allowed = mask[0].nonzero().flatten().tolist()
        epsilon = self.config.epsilon if self.training else self.config.eval_epsilon
        if self.rng.random() < epsilon:
            chosen = self.explore(allowed)
        else:
            chosen = int(scores.masked_fill(~mask, -torch.inf).argmax(-1))
        if self.training:
            self.pending = dict(patch=patch, features=features, mask=mask, chosen=chosen,
                                hidden=tuple(x.detach().clone() for x in self.hidden),
                                version=self.updates)
        self.hidden = tuple(x.detach() for x in hidden)
        self.decisions += 1
        return dict(ACTIONS[chosen])

    def train_batch(self):
        weights = [max(p, 1e-4) ** self.config.priority_alpha for p in self.priorities]
        total = sum(weights)
        indices = self.sample_indices(weights)
        # beta=1 corrects prioritized sampling to a uniform chunk objective.
        importance = [total / (len(weights) * weights[i]) for i in indices]
        losses = []
        priority_updates = {}
        ages = []
        td_errors = []
        for index, weight in zip(indices, importance):
            chunk = self.replay[index]
            rows = chunk['rows']
            burn = chunk['burn_in']
            tail = rows[burn:]
            patch = torch.cat([r['patch'] for r in rows] + [rows[-1]['next_patch']])
            features = torch.cat([r['features'] for r in rows] + [rows[-1]['next_features']])
            # Both carries are reconstructed independently, never target <- online carry.
            # The starting carry is an approximation from collection; burn-in reduces staleness.
            online, _ = self.model.sequence(patch, features, rows[0]['hidden'], burn)
            with torch.no_grad():
                target, _ = self.target.sequence(patch, features, rows[0]['hidden'], burn)
            targets = double_targets(torch.tensor([r['reward'] for r in tail]),
                                     online[1:].detach(), target[1:],
                                     torch.cat([r['next_mask'] for r in tail]),
                                     torch.tensor([r['terminal'] for r in tail]), self.config.gamma)
            actions = torch.tensor([r['chosen'] for r in tail])
            predictions = online[:-1].gather(-1, actions[:, None]).squeeze(-1)
            error = (predictions.detach() - targets).abs()
            losses.append(weight * nn.functional.smooth_l1_loss(predictions, targets))
            priority_updates[index] = .9 * float(error.max()) + .1 * float(error.mean()) + 1e-4
            ages.append(self.updates - rows[0]['version'])
            td_errors.append(float(error.mean()))
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite replay loss')
        self.optimizer.zero_grad()
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        self.updates += 1
        for index, priority in priority_updates.items():
            self.priorities[index] = priority
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.diagnostics = dict(loss=float(loss.detach()), mean_abs_td=sum(td_errors)/len(td_errors),
                                gradient_norm=float(gradient), replay_chunks=len(self.replay),
                                max_carry_age_updates=max(ages), importance_beta=1.)

    def finish(self, observation, terminal=False):
        patch, features = encode_observation(observation)
        mask = torch.tensor([action_mask(observation, ACTIONS)])
        self._complete(patch, features, mask, terminal)
        if self.training:
            self._emit()
        # A new experimental life starts fresh; its replay retains episode boundaries.
        self.context = []
        self.hidden = self.model.initial_state()

    def freeze(self):
        self.training = False
        self.pending = None
        self.reward = 0.
        self.context = []
        self.segment = []
        self.hidden = self.model.initial_state()

    def inspector(self):
        return dict(backend=self.backend, training=self.training, decisions=self.decisions,
                    updates=self.updates, replay_chunks=len(self.replay), diagnostics=self.diagnostics,
                    parameters=sum(p.numel() for p in self.model.parameters()))

    def state(self):
        return dict(backend=self.backend, config=asdict(self.config), seed=self.seed,
                    model=self.model.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), rng=self.rng.getstate(),
                    replay_rng=self.replay_rng.getstate(),
                    **{k: getattr(self, k) for k in ('hidden', 'training', 'pending', 'reward',
                       'context', 'segment', 'replay', 'priorities', 'decisions', 'updates',
                       'transitions', 'diagnostics')})

    def restore(self, data):
        if (data['backend'] != self.backend or data['config'] != asdict(self.config)
                or data['seed'] != self.seed):
            raise ValueError('Incompatible replay checkpoint')
        for key in ('model', 'target', 'optimizer'):
            getattr(self, key).load_state_dict(data[key])
        self.rng.setstate(data['rng'])
        self.replay_rng.setstate(data['replay_rng'])
        for key in ('hidden', 'training', 'pending', 'reward', 'context', 'segment', 'replay',
                    'priorities', 'decisions', 'updates', 'transitions', 'diagnostics'):
            setattr(self, key, copy.deepcopy(data[key]))
