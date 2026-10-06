"""Experimental PPO boundary handling; never selected by the live server.

The rebuild arm re-encodes a bounded private observation prefix with current
weights after each update. It adds no teacher, action rule or gradient through
the prefix. This is finite working context, not unlimited episodic memory.
"""
import torch

from agents.sequence_ppo import Brain as BaseBrain


class Brain(BaseBrain):
    backend = 'experimental-memory-ppo'

    def __init__(self, seed, config=None, mode='reset', capacity=128):
        if mode not in ('reset', 'rebuild') or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('Use reset/rebuild and a positive private context capacity')
        super().__init__(seed, config)
        self.memory_mode = mode
        self.memory_capacity = capacity
        self.history = []
        self.boundaries = 0
        self.rebuilds = 0
        self.context_forward_steps = 0
        self.evaluation_boundary = 'continuous'
        self.episode_decisions = 0
        self.last_boundary = False

    @torch.no_grad()
    def rebuild(self):
        hidden = self.model.initial_state()
        if self.history:
            patches = torch.cat([p for p, _ in self.history])
            features = torch.cat([f for _, f in self.history])
            encoded = torch.cat((self.model.patch_encoder(patches),
                                 self.model.feature_encoder(features)), -1)
            for item in encoded:
                hidden = self.model.recurrent(item[None], hidden)
        self.hidden = tuple(t.detach() for t in hidden)
        self.rebuilds += 1
        self.context_forward_steps += len(self.history)

    def _learn(self, bootstrap):
        if not self.buffer:
            return
        super()._learn(bootstrap)
        if self.memory_mode == 'rebuild':
            self.rebuild()
        self.boundaries += 1
        self.last_boundary = True
        self.diagnostics['recurrent_boundary'] = self.memory_mode + '_after_update'
        self.diagnostics['context_length'] = len(self.history) if self.memory_mode == 'rebuild' else 0

    def decide(self, observation):
        self.last_boundary = False
        if (not self.training and self.evaluation_boundary == 'reset'
                and self.episode_decisions and self.episode_decisions % self.config.rollout == 0):
            self.hidden = self.model.initial_state()
            self.boundaries += 1
            self.last_boundary = True
        command = super().decide(observation)
        self.history.append((self.pending['patch'].detach().clone(),
                             self.pending['features'].detach().clone()))
        self.history = self.history[-self.memory_capacity:]
        self.episode_decisions += 1
        return command

    def finish(self, observation, terminal=False):
        super().finish(observation, terminal)
        self.history = []
        self.episode_decisions = 0

    def freeze(self):
        super().freeze()
        self.history = []
        self.episode_decisions = 0
        self.last_boundary = False

    def state(self):
        result = super().state()
        result.update(memory_mode=self.memory_mode, memory_capacity=self.memory_capacity,
                      history=self.history, boundaries=self.boundaries, rebuilds=self.rebuilds,
                      context_forward_steps=self.context_forward_steps,
                      evaluation_boundary=self.evaluation_boundary,
                      episode_decisions=self.episode_decisions, last_boundary=self.last_boundary)
        return result

    def restore(self, data):
        if (data['memory_mode'] != self.memory_mode or data['memory_capacity'] != self.memory_capacity
                or len(data['history']) > self.memory_capacity
                or data['evaluation_boundary'] not in ('continuous', 'reset')):
            raise ValueError('Incompatible private memory checkpoint')
        super().restore(data)
        for key in ('history', 'boundaries', 'rebuilds', 'context_forward_steps',
                    'evaluation_boundary', 'episode_decisions', 'last_boundary'):
            setattr(self, key, data[key])
