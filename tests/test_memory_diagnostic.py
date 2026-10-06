import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from agents.memory_ppo import Brain
from agents.sequence_ppo import Config
from experiments.learning_curve import make_env, run_life
from experiments.memory_continuity import FoodReturns, observe_life
from experiments.audit_memory_continuity import recount_context
from tests.test_sequence_replay import equal


class MemoryDiagnosticTests(unittest.TestCase):
    def test_food_returns_require_seen_lost_and_revisited_target(self):
        tracker = FoodReturns()
        def step(i, position, visible, remaining, boundary=False):
            tracker.observe(dict(position=position, visible_food=visible,
                previous_food_remaining=remaining, boundary=boundary, conscious=True), i)
        step(0, [0, 0], [[3, 0]], [])
        step(1, [0, 1], [], [[3, 0]], True)
        step(2, [1, 1], [[3, 0]], [[3, 0]])
        self.assertEqual(tracker.events, [])
        step(3, [2, 0], [[3, 0]], [[3, 0]])
        self.assertEqual(tracker.summary()['boundary_returns'], 1)
        self.assertEqual(tracker.events[0]['cue'], 0)
        step(4, [0, 1], [], [[3, 0]])
        step(132, [0, 1], [], [[3, 0]])
        self.assertEqual(tracker.summary()['expired'], 1)
        step(133, [0, 0], [[3, 0]], [])
        step(134, [0, 1], [], [])
        self.assertIsNone(tracker.active)

    def test_observer_preserves_trajectory_and_separately_audits_boundaries(self):
        seed, decisions = 739, 145
        a = {rid: Brain(51+i, Config(), 'rebuild') for i, rid in enumerate(('r0', 'r1'))}
        b = {rid: Brain(51+i, Config(), 'rebuild') for i, rid in enumerate(('r0', 'r1'))}
        plain, arenas = make_env(seed, 'feeding', decisions, 'native')
        run_life(plain, arenas, a, time.perf_counter()+60)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            observed, row = observe_life(seed, 'feeding', b, decisions, 'policy', out, 'test', time.perf_counter()+60)
            self.assertEqual(plain.world.to_dict(), observed.world.to_dict())
            for rid in a:
                equal(self, a[rid].state(), b[rid].state())
            recount_context(out, row, decisions, True, 128)
            # A wrong boundary marker must be caught, even with unchanged totals.
            path = out/row['senses']
            lines = path.read_text().splitlines()
            altered = json.loads(lines[128])
            altered['agents']['r0']['boundary'] = False
            lines[128] = json.dumps(altered)
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(AssertionError):
                recount_context(out, row, decisions, True, 128)


if __name__ == '__main__':
    unittest.main()
