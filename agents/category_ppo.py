"""Experimental verb/argument PPO head. No routes, needs rules or assigned meanings.

Every existing primitive remains available under the same physical mask.
Sampling the joint categorical is equivalent to first sampling a verb, then
its argument. PPO clipping, KL and entropy all use that JOINT distribution.
This backend is deliberately not supported by the live population loader.
"""
import copy

import torch
from torch import nn

from agents.network import ACTIONS, ResidentNetwork
from agents.sequence_ppo import Brain as FlatBrain

CATEGORIES = tuple(dict.fromkeys(a['verb'] for a in ACTIONS))
ACTION_CATEGORY = tuple(CATEGORIES.index(a['verb']) for a in ACTIONS)
MEMBERSHIP = torch.tensor([[c == k for c in ACTION_CATEGORY] for k in range(len(CATEGORIES))])


def joint_distribution(logits, mask):
    """P(primitive) = P(feasible verb) * P(feasible argument | verb)."""
    if logits.shape[-1] != len(CATEGORIES) + len(ACTIONS) or mask.shape != logits.shape[:-1] + (len(ACTIONS),):
        raise ValueError('Category head dimensions do not match the action schema')
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Each policy row needs at least one physically feasible action')
    members = MEMBERSHIP.to(logits.device)
    legal = mask.unsqueeze(-2) & members
    available = legal.any(-1)
    category_logp = torch.log_softmax(logits[..., :len(CATEGORIES)].masked_fill(~available, -1e9), -1)
    arguments = logits[..., len(CATEGORIES):]
    normalizers = torch.logsumexp(arguments.unsqueeze(-2).masked_fill(~legal, -1e9), -1)
    joint = category_logp[..., ACTION_CATEGORY] + arguments - normalizers[..., ACTION_CATEGORY]
    return torch.distributions.Categorical(logits=joint.masked_fill(~mask, -1e9))


class CategoryHead(nn.Module):
    def __init__(self, argument_head, hidden_size):
        super().__init__()
        self.arguments = argument_head
        # A separate initialization stream preserves the matched trunk, value,
        # argument head AND consequence predictor initialization in the flat arm.
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed() + 938719)
            self.category = nn.Linear(hidden_size, len(CATEGORIES))

    def forward(self, hidden):
        return torch.cat((self.category(hidden), self.arguments(hidden)), -1)


class CategoryNetwork(ResidentNetwork):
    def __init__(self, hidden_size=64):
        super().__init__(hidden_size=hidden_size)
        self.actor = CategoryHead(self.actor, hidden_size)


class Brain(FlatBrain):
    backend = 'experimental-category-ppo'
    network_class = CategoryNetwork

    def distribution(self, logits, mask):
        return joint_distribution(logits, mask)

    def state(self):
        data = super().state()
        data['category_schema'] = dict(version=1, actions=copy.deepcopy(list(ACTIONS)), categories=list(CATEGORIES))
        return data

    def restore(self, data):
        expected = dict(version=1, actions=list(ACTIONS), categories=list(CATEGORIES))
        if data.get('category_schema') != expected:
            raise ValueError('Incompatible experimental category action schema')
        super().restore(data)
