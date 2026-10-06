import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from agents.environment import LearningEnv
from agents.network import ACTIONS, FEATURES, PATCH_CHANNELS
from agents.sequence_ppo import Population
from experiments.scarcity_routes import configure
from sim.world import World
from tests.test_sequence_replay import equal


class EnvironmentTests(unittest.TestCase):
    def test_observations_are_bounded_local_arrays_with_all_tones(self):
        env = LearningEnv()
        observations, infos = env.reset(seed=313)
        self.assertEqual(set(observations), set(env.possible_agents))
        for rid, obs in observations.items():
            self.assertEqual(obs['patch'].shape, (PATCH_CHANNELS, 9, 9))
            self.assertEqual(obs['features'].shape, (FEATURES,))
            self.assertEqual(obs['action_mask'].dtype, np.int8)
            self.assertTrue(np.isfinite(obs['patch']).all())
            self.assertTrue(np.isfinite(obs['features']).all())
            self.assertTrue(((0 <= obs['features']) & (obs['features'] <= 1)).all())
            self.assertEqual(sum(a['verb'] == 'tone' and obs['action_mask'][i]
                                 for i, a in enumerate(ACTIONS)), 10)
            self.assertEqual(infos[rid], {})
            self.assertFalse({'position', 'id', 'tick', 'player', 'global_food'} & env.raw_observation(rid).keys())
        untouched = env.raw_observation('r0')
        env.raw_observation('r0')['needs'][0] = -999
        self.assertEqual(env.raw_observation('r0'), untouched)

    def test_invalid_batch_does_not_advance_or_consume_rng(self):
        env = LearningEnv()
        env.reset(seed=313)
        before = env.state()
        for actions in ({'r0': 0}, {rid: len(ACTIONS) for rid in env.agents},
                        {rid: True for rid in env.agents}):
            with self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(before, env.state())

    def test_episode_truncation_is_not_death_and_reset_is_deterministic(self):
        env = LearningEnv(max_cycles=2)
        initial, _ = env.reset(seed=313)
        rest = {rid: ACTIONS.index({'verb': 'rest'}) for rid in env.agents}
        env.step(rest)
        observations, rewards, terminated, truncated, infos = env.step(rest)
        self.assertEqual(env.world.tick, 8)
        self.assertEqual(env.agents, [])
        self.assertFalse(any(terminated.values()))
        self.assertTrue(all(truncated.values()))
        self.assertEqual(set(observations), set(rest))
        for rid in rest:
            self.assertEqual(len(infos[rid]['reward_parts_by_tick']), 4)
            self.assertEqual(rewards[rid], sum(sum(p.values()) for p in infos[rid]['reward_parts_by_tick']))
            self.assertIsInstance(env.raw_observation(rid), dict)
        self.assertEqual(env.step({}), ({}, {}, {}, {}, {}))
        again, _ = env.reset(seed=313)
        for rid in initial:
            for key in initial[rid]:
                np.testing.assert_array_equal(initial[rid][key], again[rid][key])

    def test_exact_world_and_brain_parity_with_existing_population(self):
        p = Population(413)
        configure(p, 413, 'feeding')
        original = copy.deepcopy(p.world.to_dict())
        env = LearningEnv(tuple(p.brains), max_cycles=200,
                          world_factory=lambda seed, options: World.from_dict(copy.deepcopy(original)))
        env.reset(seed=413)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            p.save(path)
            q = Population.load(path)
        for _ in range(145):
            commands = {rid: b.decide(env.raw_observation(rid)) for rid, b in q.brains.items()}
            _, _, _, _, infos = env.step({rid: ACTIONS.index(c) for rid, c in commands.items()})
            for rid, b in q.brains.items():
                for parts in infos[rid]['reward_parts_by_tick']:
                    b.add_reward(parts)
            for _ in range(4):
                p.step()
        self.assertEqual(p.world.to_dict(), env.world.to_dict())
        for rid in p.brains:
            self.assertGreater(p.brains[rid].updates, 0)
            equal(self, p.brains[rid].state(), q.brains[rid].state())

    def test_checkpoint_continuation_and_incompatible_restore(self):
        p = LearningEnv(max_cycles=100)
        p.reset(seed=415)
        rng = random.Random(99)
        for _ in range(7):
            p.step({rid: rng.randrange(len(ACTIONS)) for rid in p.agents})
        q = LearningEnv(max_cycles=100)
        q.restore(p.state())
        for _ in range(13):
            actions = {rid: rng.randrange(len(ACTIONS)) for rid in p.agents}
            a, b = p.step(actions), q.step(actions)
            self.assertEqual(a[1:], b[1:])
            for rid in a[0]:
                for key in a[0][rid]:
                    np.testing.assert_array_equal(a[0][rid][key], b[0][rid][key])
        self.assertEqual(p.state(), q.state())
        incompatible = LearningEnv(('r0',), max_cycles=100)
        with self.assertRaises(ValueError):
            incompatible.restore(p.state())


if __name__ == '__main__':
    unittest.main()
