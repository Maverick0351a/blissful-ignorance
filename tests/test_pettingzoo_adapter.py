"""Optional upstream compatibility and physics parity checks.

Skipped when the separately reviewed PettingZoo dependencies are unavailable.
"""
import importlib.util
import unittest

import numpy as np

from agents.environment import LearningEnv

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('pettingzoo', 'gymnasium'))
if AVAILABLE:
    from adapters.pettingzoo_env import GodhoodParallelEnv
    from pettingzoo.test import parallel_api_test, parallel_seed_test


@unittest.skipUnless(AVAILABLE, 'Optional PettingZoo dependencies are not installed')
class PettingZooAdapterTests(unittest.TestCase):
    def test_official_parallel_api(self):
        env = GodhoodParallelEnv()
        try:
            parallel_api_test(env, num_cycles=1000)
        finally:
            env.close()

    def test_official_seed_determinism(self):
        parallel_seed_test(lambda: GodhoodParallelEnv(max_cycles=500), num_cycles=500)

    def test_observation_spaces_and_exact_native_trajectory(self):
        native = LearningEnv(max_cycles=300)
        wrapped = GodhoodParallelEnv(max_cycles=300)
        left, _ = native.reset(seed=7419)
        right, _ = wrapped.reset(seed=7419)
        for _ in range(300):
            for rid in left:
                self.assertTrue(wrapped.observation_space(rid).contains(right[rid]))
                for key in left[rid]:
                    np.testing.assert_array_equal(left[rid][key], right[rid][key])
            actions = {rid: wrapped.action_space(rid).sample(mask=right[rid]['action_mask'])
                       for rid in wrapped.agents}
            a, b = native.step(actions), wrapped.step(actions)
            self.assertEqual(a[1:], b[1:])
            self.assertEqual(native.state(), wrapped.core.state())
            left, right = a[0], b[0]
        self.assertFalse(wrapped.agents)
        for rid in right:
            self.assertTrue(wrapped.observation_space(rid).contains(right[rid]))
        native.close()
        wrapped.close()

    def test_no_global_state_and_explicit_render_mode(self):
        env = GodhoodParallelEnv(render_mode='ansi')
        env.reset(seed=9419)
        with self.assertRaises(NotImplementedError):
            env.state()
        self.assertIn('r0: fullness', env.render())
        env.close()
        self.assertIsNone(env.render())
        with self.assertRaises(ValueError):
            GodhoodParallelEnv(render_mode='rgb_array')


if __name__ == '__main__':
    unittest.main()
