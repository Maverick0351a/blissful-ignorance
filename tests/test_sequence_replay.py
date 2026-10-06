import copy
from pathlib import Path
import tempfile
import unittest

import torch

from agents.affordances import action_mask
from agents.network import ACTIONS, encode_observation
from agents.sequence_ppo import Population
from agents.sequence_replay import Brain, Config, QNetwork, double_targets
from experiments.scarcity_routes import configure


def population(seed=781, recurrent=True):
    p = Population(seed)
    c = Config(recurrent=recurrent, hidden_size=16, unroll=4, burn_in=2,
               capacity=8, batch_size=2, warmup=2, target_interval=2)
    p.brains = {rid: Brain(seed+i, c) for i, rid in enumerate(p.brains)}
    configure(p, seed, 'feeding')
    return p


def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            equal(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal(test, x, y)
    else:
        test.assertEqual(a, b)


class ReplayTests(unittest.TestCase):
    def test_targets_use_online_selection_target_value_mask_and_terminal(self):
        online = torch.tensor([[10., 5., 2.], [10., 5., 2.]])
        target = torch.tensor([[100., 3., 20.], [100., 3., 20.]])
        mask = torch.tensor([[False, True, True]] * 2)
        result = double_targets(torch.tensor([1., 1.]), online, target, mask,
                                torch.tensor([False, True]), .5)
        self.assertEqual(result.tolist(), [2.5, 1.])

    def test_burn_in_matches_current_recurrence_without_context_gradient(self):
        p = population()
        brain = p.brains['r0']
        patch, features = encode_observation(p.world.observe('r0'))
        patches = patch.repeat(5, 1, 1, 1).requires_grad_()
        features = features.repeat(5, 1)
        burned, _ = brain.model.sequence(patches, features, brain.model.initial_state(), 2)
        with torch.no_grad():
            full, _ = brain.model.sequence(patches, features, brain.model.initial_state())
        self.assertTrue(torch.allclose(burned, full[2:], atol=1e-7))
        burned[-1].sum().backward()
        self.assertEqual(float(patches.grad[:2].abs().sum()), 0.)
        self.assertGreater(float(patches.grad[2].abs().sum()), 0.)

    def test_feedforward_has_no_temporal_gradient(self):
        p = population(recurrent=False)
        b = p.brains['r0']
        patch, features = encode_observation(p.world.observe('r0'))
        patches = patch.repeat(3, 1, 1, 1).requires_grad_()
        scores, _ = b.model.sequence(patches, features.repeat(3, 1), b.model.initial_state())
        scores[-1].sum().backward()
        self.assertEqual(float(patches.grad[:-1].abs().sum()), 0.)
        self.assertGreater(float(patches.grad[-1].abs().sum()), 0.)

    def test_every_transition_learned_once_in_chunks_and_no_episode_crossing(self):
        p = population()
        b = p.brains['r0']
        for _ in range(44):
            p.step()
        b.finish(p.world.observe('r0'), terminal=True)
        self.assertEqual(b.transitions, 11)
        tails = [c['rows'][c['burn_in']:] for c in b.replay]
        self.assertEqual([len(t) for t in tails], [4, 4, 3])
        self.assertEqual([c['burn_in'] for c in b.replay], [0, 2, 2])
        self.assertEqual(sum(r['terminal'] for t in tails for r in t), 1)
        self.assertTrue(tails[-1][-1]['terminal'])
        self.assertEqual(b.context, [])
        for _ in range(16):
            p.step()
        b.finish(p.world.observe('r0'), terminal=True)
        self.assertEqual(b.replay[-1]['burn_in'], 0)
        self.assertFalse(any(r['terminal'] for r in b.replay[-1]['rows'][:-1]))

    def test_exact_resume_with_pending_reward_replay_optimizer_and_both_rngs(self):
        for recurrent in (False, True):
            p = population(recurrent=recurrent)
            for _ in range(53):
                p.step()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'checkpoint.pt'
                p.save(path)
                q = Population.load(path)
                for _ in range(47):
                    self.assertEqual(p.step(), q.step())
                self.assertEqual(p.world.to_dict(), q.world.to_dict())
                self.assertEqual(p.metrics, q.metrics)
                for rid in p.brains:
                    equal(self, p.brains[rid].state(), q.brains[rid].state())

    def test_training_updates_only_own_brain_and_freeze_prevents_learning(self):
        p = population()
        b, other = p.brains.values()
        untouched = copy.deepcopy(other.state())
        before = copy.deepcopy(b.model.state_dict())
        observation = p.world.observe('r0')
        for _ in range(17):
            b.decide(observation)
            b.add_reward({'food': 1.})
        self.assertGreater(b.updates, 0)
        self.assertTrue(any(not torch.equal(v, b.model.state_dict()[k]) for k, v in before.items()))
        equal(self, untouched, other.state())
        b.freeze()
        frozen = copy.deepcopy(b.state())
        for _ in range(20):
            b.decide(observation)
            b.add_reward({'food': 10.})
        for k in ('model', 'target', 'optimizer', 'replay', 'priorities', 'replay_rng', 'updates'):
            equal(self, frozen[k], b.state()[k])

    def test_local_masks_and_all_tones_remain_available(self):
        p = population()
        observation = p.world.observe('r0')
        observation['inventory']['amber_fruit'] = 1
        mask = action_mask(observation, ACTIONS)
        self.assertTrue(mask[ACTIONS.index({'verb': 'eat', 'item': 'amber_fruit'})])
        self.assertEqual(sum(ok and a['verb'] == 'tone' for a, ok in zip(ACTIONS, mask)), 10)
        altered = copy.deepcopy(observation)
        altered.update(global_food={'secret': 100}, god=True, tick=100000)
        for a, b in zip(encode_observation(observation), encode_observation(altered)):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
