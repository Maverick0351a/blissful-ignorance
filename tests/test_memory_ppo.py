import copy
import unittest

import torch

from agents.memory_ppo import Brain
from agents.sequence_ppo import Brain as BaseBrain, Config
from agents.network import ACTIONS, encode_observation
from experiments.learning_curve import make_env
from experiments.replay_comparison import learning_digest
from tests.test_sequence_replay import equal


class MemoryTests(unittest.TestCase):
    def test_reset_preserves_baseline_learning_and_actions(self):
        env, _ = make_env(731, 'feeding', 40, 'native')
        config = Config(rollout=8, epochs=2)
        a, b = BaseBrain(32, config), Brain(32, config, 'reset', 8)
        for _ in range(33):
            obs = env.raw_observation('r0')
            ca, cb = a.decide(obs), b.decide(obs)
            self.assertEqual(ca, cb)
            _, _, _, _, infos = env.step({'r0': ACTIONS.index(ca), 'r1': 4})
            for parts in infos['r0']['reward_parts_by_tick']:
                a.add_reward(parts)
                b.add_reward(parts)
        original, experimental = a.state(), b.state()
        for key in original:
            if key not in ('backend', 'diagnostics'):
                equal(self, original[key], experimental[key])
        self.assertEqual(a.updates, 4)

    def test_rebuild_uses_updated_weights_and_only_bounded_private_inputs(self):
        env, _ = make_env(733, 'feeding', 12, 'native')
        b = Brain(37, Config(rollout=8, epochs=2), 'rebuild', 6)
        for _ in range(8):
            action = b.decide(env.raw_observation('r0'))
            _, _, _, _, infos = env.step({'r0': ACTIONS.index(action), 'r1': 4})
            for parts in infos['r0']['reward_parts_by_tick']:
                b.add_reward(parts)
        b._complete(env.raw_observation('r0'))
        old = copy.deepcopy(b.model.state_dict())
        b._learn(0.)
        self.assertTrue(any(not torch.equal(v, b.model.state_dict()[k]) for k, v in old.items()))
        expected = b.model.initial_state()
        with torch.no_grad():
            for patch, features in b.history:
                _, _, expected = b.model(patch, features, expected)
        for actual, wanted in zip(b.hidden, expected):
            torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
            self.assertFalse(actual.requires_grad)
        self.assertEqual(len(b.history), 6)
        self.assertEqual(b.context_forward_steps, 6)
        self.assertGreater(sum(float(t.abs().sum()) for t in b.hidden), 0)
        raw = env.raw_observation('r0')
        altered = copy.deepcopy(raw)
        altered.update(secret_food_position=(0, 0), world_tick=999, player=True)
        for original, changed in zip(encode_observation(raw), encode_observation(altered)):
            self.assertTrue(torch.equal(original, changed))

    def test_exact_resume_across_updates_and_independent_storage(self):
        env, _ = make_env(734, 'feeding', 40, 'native')
        a = Brain(43, Config(rollout=8, epochs=2), 'rebuild', 8)
        obs = env.raw_observation('r0')
        for _ in range(11):
            a.decide(obs)
            a.add_reward({'test': .01})
        b = Brain(43, a.config, 'rebuild', 8)
        b.restore(copy.deepcopy(a.state()))
        self.assertNotEqual(a.history[0][0].data_ptr(), b.history[0][0].data_ptr())
        self.assertNotEqual(a.model.actor.weight.data_ptr(), b.model.actor.weight.data_ptr())
        for _ in range(20):
            self.assertEqual(a.decide(obs), b.decide(obs))
            a.add_reward({'test': .01})
            b.add_reward({'test': .01})
        equal(self, a.state(), b.state())
        with self.assertRaises(ValueError):
            Brain(43, a.config, 'reset', 8).restore(copy.deepcopy(a.state()))

    def test_frozen_reset_is_runtime_only_and_episode_finish_clears_context(self):
        env, _ = make_env(735, 'feeding', 40, 'native')
        b = Brain(45, Config(rollout=8, epochs=2), 'rebuild', 8)
        obs = env.raw_observation('r0')
        for _ in range(9):
            b.decide(obs)
        b.finish(obs, terminal=False)
        self.assertEqual(b.history, [])
        b.freeze()
        b.evaluation_boundary = 'reset'
        before, boundaries = learning_digest(b), b.boundaries
        for _ in range(25):
            b.decide(obs)
        self.assertEqual(b.boundaries-boundaries, 3)
        self.assertEqual(learning_digest(b), before)
        self.assertEqual(b.buffer, [])
        self.assertEqual(len(b.history), 8)


if __name__ == '__main__':
    unittest.main()
