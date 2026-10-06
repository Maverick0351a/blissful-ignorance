"""Model-free parallel environment boundary over the existing world physics.

One external action advances four physical ticks, exactly as Population.step.
No brain, recipe policy, global observation or live save is created or loaded.
"""
import copy
import operator

import numpy as np

from agents.affordances import action_mask
from agents.network import ACTIONS, encode_observation
from agents.physiology import food_relief
from sim.world import World


class LearningEnv:
    def __init__(self, resident_ids=('r0', 'r1', 'r2', 'r3'), max_cycles=4096,
                 world_factory=None):
        self.possible_agents = list(resident_ids)
        if (not self.possible_agents or len(set(self.possible_agents)) != len(self.possible_agents)
                or 'player' in self.possible_agents or max_cycles <= 0):
            raise ValueError('Use distinct resident IDs and a positive episode budget')
        self.max_cycles = operator.index(max_cycles)
        self.world_factory = world_factory or (lambda seed, options: World(seed))
        self.agents = []
        self.world = None
        self.cycles = 0
        self._raw = {}

    def _observations(self, ids):
        observations = {}
        for rid in ids:
            raw = self.world.observe(rid)
            self._raw[rid] = raw
            patch, features = encode_observation(raw)
            observations[rid] = dict(patch=patch[0].numpy().copy(),
                features=features[0].numpy().copy(),
                action_mask=np.asarray(action_mask(raw, ACTIONS), dtype=np.int8))
        return observations

    def reset(self, seed=None, options=None):
        self.world = self.world_factory(1729 if seed is None else operator.index(seed),
                                        copy.deepcopy(options or {}))
        if not set(self.possible_agents) <= set(self.world.residents) or self.world.tick % 4:
            raise ValueError('Scenario must contain the configured residents at a decision boundary')
        # Disposable experiments contain precisely their declared residents.
        self.world.residents = {rid: self.world.residents[rid] for rid in self.possible_agents}
        self.agents = list(self.possible_agents)
        self.cycles = 0
        self._raw = {}
        return self._observations(self.agents), {rid: {} for rid in self.agents}

    def raw_observation(self, rid):
        """Compatibility bridge for existing learners; still local senses only."""
        if rid not in self._raw:
            raise KeyError(rid)
        return copy.deepcopy(self._raw[rid])

    def step(self, actions):
        if not self.agents:
            if actions:
                raise ValueError('Reset before stepping an inactive episode')
            return {}, {}, {}, {}, {}
        ids = list(self.agents)
        if set(actions) != set(ids):
            raise ValueError('Provide exactly one action for every active resident')
        commands = {}
        # Validate the entire batch before advancing any part of the world.
        for rid in ids:
            if isinstance(actions[rid], (bool, np.bool_)):
                raise ValueError('Actions must be integer indices')
            index = operator.index(actions[rid])
            if not 0 <= index < len(ACTIONS):
                raise ValueError('Action index outside the primitive vocabulary')
            commands[rid] = dict(ACTIONS[index])
        rewards = {rid: 0. for rid in ids}
        infos = {rid: dict(reward_parts_by_tick=[], nutrition=0., injury=0.,
                          zero_food_ticks=0, unconscious_ticks=0, fullness_sum=0.) for rid in ids}
        for offset in range(4):
            before = {rid: (a.food, a.health, a.pain) for rid in ids
                      for a in (self.world.residents[rid],)}
            applied = commands if offset == 0 else {rid: {'verb': 'wait'} for rid in ids}
            results = self.world.step(applied, scripted=False)
            for rid in ids:
                a = self.world.residents[rid]
                food, health, pain = before[rid]
                nutrition = food_relief(food, a.food)
                injury = max(0., health-a.health, a.pain-pain)
                parts = dict(food=nutrition/25, hunger=-min(1., max(0., (60-a.food)/60))*.005,
                             injury=-injury/20)
                rewards[rid] += sum(parts.values())
                info = infos[rid]
                info['reward_parts_by_tick'].append(parts)
                info['nutrition'] += nutrition
                info['injury'] += injury
                info['zero_food_ticks'] += a.food <= 0
                info['unconscious_ticks'] += a.unconscious
                info['fullness_sum'] += a.food
                if offset == 0:
                    info['action_success'] = bool(results[rid][0])
        self.cycles += 1
        observations = self._observations(ids)
        terminated = {rid: False for rid in ids}  # Mortality remains off.
        truncated = {rid: self.cycles >= self.max_cycles for rid in ids}
        if all(truncated.values()):
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    def state(self):
        if self.world is None:
            raise ValueError('No episode exists')
        return dict(schema=1, possible_agents=list(self.possible_agents),
                    agents=list(self.agents), max_cycles=self.max_cycles,
                    cycles=self.cycles, world=copy.deepcopy(self.world.to_dict()))

    def restore(self, data):
        if (data['schema'] != 1 or data['possible_agents'] != self.possible_agents
                or data['max_cycles'] != self.max_cycles
                or not 0 <= data['cycles'] <= self.max_cycles
                or data['agents'] != ([] if data['cycles'] == self.max_cycles else self.possible_agents)):
            raise ValueError('Incompatible environment checkpoint')
        world = World.from_dict(copy.deepcopy(data['world']))
        if set(world.residents) != set(self.possible_agents) or world.tick % 4:
            raise ValueError('Checkpoint resident set or decision boundary changed')
        self.world = world
        self.cycles = data['cycles']
        self.agents = list(data['agents'])
        self._raw = {}
        return self._observations(self.agents)

    def close(self):
        self.agents = []
        self._raw = {}
        self.world = None
