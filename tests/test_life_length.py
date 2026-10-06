import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from experiments.audit_life_length import recount_behavior
from experiments.learning_curve import make_env, run_life
from experiments.life_length import make_brains, observe_life, schedule
from tests.test_sequence_replay import equal


class LifeLengthTests(unittest.TestCase):
    def test_training_treatment_has_equal_experience_and_matched_map_exposure(self):
        short, long = schedule(91, 'short'), schedule(91, 'long')
        self.assertEqual((len(short), len(long)), (32, 4))
        self.assertEqual(sum(p['decisions'] for p in short), 16384)
        self.assertEqual(sum(p['decisions'] for p in long), 16384)
        for block in range(4):
            a = [p for p in short if p['block'] == block]
            b = long[block]
            self.assertEqual(sum(p['decisions'] for p in a), b['decisions'])
            self.assertEqual({p['world_seed'] for p in a}, {b['world_seed']})
            self.assertEqual({p['stage'] for p in a}, {b['stage']})
            self.assertTrue(all(p['decisions'] % 128 == 0 for p in a))

    def test_observer_has_exact_world_learning_parity_and_rejects_false_positions(self):
        decisions, seed = 145, 759
        plain, arenas = make_env(seed, 'feeding', decisions, 'native')
        a, b = make_brains(93), make_brains(93)
        run_life(plain, arenas, a, time.perf_counter()+60)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            observed, row = observe_life(seed, 'feeding', b, decisions, 'policy', out, 'test', time.perf_counter()+60)
            self.assertEqual(plain.world.to_dict(), observed.world.to_dict())
            for rid in a:
                equal(self, a[rid].state(), b[rid].state())
            recount_behavior(out, row)
            path = out/row['senses']
            lines = path.read_text().splitlines()
            altered = json.loads(lines[2])
            altered['agents']['r0']['position'][0] += 2
            lines[2] = json.dumps(altered)
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(AssertionError):
                recount_behavior(out, row)

    def test_time_limit_finishes_deliver_equal_full_rollout_counts(self):
        counts, starts = {}, {}
        for arm in ('short', 'long'):
            brains = make_brains(95)
            for plan in [p for p in schedule(759, arm, pilot=True) if p['block'] == 0]:
                env, arenas = make_env(plan['world_seed'], plan['stage'], plan['decisions'], 'native')
                starts.setdefault(arm, copy.deepcopy(env.world.to_dict()))
                run_life(env, arenas, brains, time.perf_counter()+60)
                for rid, brain in brains.items():
                    brain.finish(env.raw_observation(rid), terminal=False)
                    self.assertEqual(brain.buffer, [])
                    self.assertIsNone(brain.pending)
                    self.assertEqual(brain.diagnostics['sequence_length'], 128)
            counts[arm] = [(b.decisions, b.updates) for b in brains.values()]
        self.assertEqual(starts['short'], starts['long'])
        self.assertEqual(counts['short'], [(256, 2), (256, 2)])
        self.assertEqual(counts['long'], counts['short'])


if __name__ == '__main__':
    unittest.main()
