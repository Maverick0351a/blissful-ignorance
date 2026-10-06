"""Optional PettingZoo ParallelEnv wrapper, with private observations only."""
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from agents.environment import LearningEnv
from agents.network import ACTIONS, FEATURES, PATCH_CHANNELS


class GodhoodParallelEnv(ParallelEnv):
    metadata = {'name': 'godhood_trials_v1', 'render_modes': ['ansi'],
                'is_parallelizable': True}

    def __init__(self, resident_ids=('r0', 'r1', 'r2', 'r3'), max_cycles=4096,
                 world_factory=None, render_mode=None):
        if render_mode not in (None, 'ansi'):
            raise ValueError('This headless adapter supports only ansi rendering')
        self.core = LearningEnv(resident_ids, max_cycles, world_factory)
        self.render_mode = render_mode
        self.possible_agents = list(self.core.possible_agents)
        self.agents = []
        self.observation_spaces = {rid: spaces.Dict({
            'patch': spaces.Box(0., 1., (PATCH_CHANNELS, 9, 9), np.float32),
            'features': spaces.Box(0., 1., (FEATURES,), np.float32),
            'action_mask': spaces.MultiBinary(len(ACTIONS))}) for rid in self.possible_agents}
        self.action_spaces = {rid: spaces.Discrete(len(ACTIONS)) for rid in self.possible_agents}

    @property
    def max_cycles(self):
        return self.core.max_cycles

    @max_cycles.setter
    def max_cycles(self, value):
        import operator
        value = operator.index(value)
        if value <= 0:
            raise ValueError('Episode budget must be positive')
        self.core.max_cycles = value

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        observations, infos = self.core.reset(seed, options)
        self.agents = list(self.core.agents)
        if seed is not None:
            for i, rid in enumerate(self.possible_agents):
                self.action_spaces[rid].seed(seed+i)
                self.observation_spaces[rid].seed(seed+1000+i)
        return observations, infos

    def step(self, actions):
        result = self.core.step(actions)
        self.agents = list(self.core.agents)
        return result

    def render(self):
        if self.render_mode == 'ansi' and self.core.world is not None:
            return '\n'.join(f'{rid}: fullness {self.core.world.residents[rid].food:.1f}'
                             for rid in self.possible_agents)
        return None

    def state(self):
        raise NotImplementedError('Global world state is excluded from the private learning interface')

    def close(self):
        self.core.close()
        self.agents = []
