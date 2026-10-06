"""Isolated retention/exploration experiments on the existing recurrent learner.

The reservation is a declared memory-selection bias toward the resident's own
nutrition and injury consequences. It never supplies an action or experience.
All priorities are sampled together with the base learner's beta=1 correction.
"""
from dataclasses import dataclass
import copy
import random

from agents.network import ACTIONS
from agents.sequence_replay import Brain as ReplayBrain, Config as ReplayConfig


@dataclass(frozen=True)
class Config(ReplayConfig):
    retain_consequences: bool = False
    balanced_exploration: bool = False
    reserved_chunks: int = 16

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.reserved_chunks < self.capacity:
            raise ValueError('Reserve must leave space for recent experience')


def exploration_groups(allowed):
    """One category per verb; parameter variants share that category's mass."""
    groups = {}
    for index in allowed:
        groups.setdefault(ACTIONS[index]['verb'], []).append(index)
    return list(groups.values())


class Brain(ReplayBrain):
    def __init__(self, seed, config=None):
        super().__init__(seed, config or Config())
        self.backend = 'experimental-retained-replay'
        self.retention_rng = random.Random(seed ^ 0x71A9)
        self.experiment = dict(emitted_chunks=0, salient_chunks_seen=0,
                              sampled_chunks=0, sampled_eating=0,
                              sampled_nutrition=0, sampled_injury=0)

    def add_reward(self, parts):
        super().add_reward(parts)
        if self.training and self.pending is not None:
            # These are the same locally observed body changes used by reward.
            for key, occurred in (('nutrition', parts.get('food', 0) > 1e-7),
                                  ('injury', parts.get('injury', 0) < -1e-7)):
                self.pending[key] = bool(self.pending.get(key, False) or occurred)

    def store_chunk(self, chunk):
        tail = chunk['rows'][chunk['burn_in']:]
        salient = any(row.get('nutrition') or row.get('injury') for row in tail)
        self.experiment['emitted_chunks'] += 1
        self.experiment['salient_chunks_seen'] += bool(salient)
        chunk.update(serial=self.experiment['emitted_chunks'], protected=False)
        if self.config.retain_consequences and salient:
            protected = [c for c in self.replay if c['protected']]
            if len(protected) < self.config.reserved_chunks:
                chunk['protected'] = True
            else:
                # Uniform reservoir over all consequence-bearing chunks seen.
                slot = self.retention_rng.randrange(self.experiment['salient_chunks_seen'])
                if slot < self.config.reserved_chunks:
                    protected[slot]['protected'] = False
                    chunk['protected'] = True
        self.replay.append(chunk)
        self.priorities.append(max(self.priorities, default=1.))
        if len(self.replay) > self.config.capacity:
            # Chunks remain chronological; no duplicate recent/reserved copies.
            index = next(i for i, c in enumerate(self.replay) if not c['protected'])
            self.replay.pop(index)
            self.priorities.pop(index)

    def explore(self, allowed):
        if self.training and self.config.balanced_exploration:
            return self.rng.choice(self.rng.choice(exploration_groups(allowed)))
        # Identical flat 5% epsilon exploration in every frozen evaluation.
        return super().explore(allowed)

    def sample_indices(self, weights):
        indices = super().sample_indices(weights)
        self.experiment['sampled_chunks'] += len(indices)
        for index in indices:
            chunk = self.replay[index]
            for row in chunk['rows'][chunk['burn_in']:]:
                self.experiment['sampled_eating'] += ACTIONS[row['chosen']]['verb'] == 'eat'
                for key in ('nutrition', 'injury'):
                    self.experiment['sampled_' + key] += bool(row.get(key))
        return indices

    def inspector(self):
        tails = [row for c in self.replay for row in c['rows'][c['burn_in']:]]
        return dict(super().inspector(), experiment=dict(self.experiment),
                    retained_transitions=len(tails),
                    retained_eating=sum(ACTIONS[r['chosen']]['verb'] == 'eat' for r in tails),
                    retained_nutrition=sum(bool(r.get('nutrition')) for r in tails),
                    retained_injury=sum(bool(r.get('injury')) for r in tails),
                    protected_chunks=sum(c['protected'] for c in self.replay))

    def state(self):
        return dict(super().state(), retention_rng=self.retention_rng.getstate(),
                    experiment=copy.deepcopy(self.experiment))

    def restore(self, data):
        super().restore(data)
        self.retention_rng.setstate(data['retention_rng'])
        self.experiment = copy.deepcopy(data['experiment'])
