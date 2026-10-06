import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch
from agents.sequence_ppo import Config, Population
from experiments.population_watch import observe
from sim.checkpoint_lease import checkpoint_lease
from sim.world import World


class PopulationWatchTests(unittest.TestCase):
    def test_recorder_preserves_exact_continuation_across_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            population = Population.from_world(World(8723), Config(rollout=8, epochs=1))
            # Start off the decision boundary, with pending transitions and one fainted body.
            population.world.residents['r0'].food = 0
            population.world.residents['r1'].unconscious = True
            for _ in range(3):
                population.step()
            population.save(root/'start.pt')
            expected = Population.load(root/'start.pt')
            for _ in range(65):
                expected.step()
            actual = Population.load(root/'start.pt')
            result = observe(actual, root, 65, 60)
            self.assertEqual(actual.world.to_dict(), expected.world.to_dict())
            self.assertEqual(actual.metrics, expected.metrics)
            self.assertEqual(result['ticks'], 65)
            for rid, brain in actual.brains.items():
                other = expected.brains[rid]
                self.assertEqual(brain.decisions, other.decisions)
                self.assertEqual(brain.updates, other.updates)
                self.assertEqual(brain.rng.getstate(), other.rng.getstate())
                self.assertEqual(len(brain.buffer), len(other.buffer))
                self.assertIsNotNone(brain.pending)
                for key, value in brain.model.state_dict().items():
                    self.assertTrue(torch.equal(value, other.model.state_dict()[key]))
                for left, right in zip(brain.hidden, other.hidden):
                    self.assertTrue(torch.equal(left, right))
            with gzip.open(root/'trace.jsonl.gz', 'rt') as trace:
                rows = [json.loads(line) for line in trace]
            self.assertFalse(rows[0]['decisions'])
            self.assertEqual(set(rows[1]['decisions']), set(actual.brains))

    def test_rejects_scripted_population(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'independent brain'):
                observe(Population(14), Path(folder), 4, 10)

    def test_stop_preserves_pending_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = Population.from_world(World(67))
            p.step()
            (root/'STOP').touch()
            result = observe(p, root, 20, 10)
            self.assertEqual(result['ticks'], 0)
            self.assertEqual(result['stop_reason'], 'stop marker')
            self.assertTrue(all(b.pending is not None for b in p.brains.values()))

    def test_exclusive_ownership_and_release_after_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'test failure'):
                with checkpoint_lease(root):
                    with self.assertRaisesRegex(RuntimeError, 'already has an owner'):
                        with checkpoint_lease(root):
                            self.fail('Second owner must not enter')
                    raise ValueError('test failure')
            self.assertFalse((root/'population-owner.json').exists())
            with checkpoint_lease(root):
                self.assertTrue((root/'population-owner.json').exists())


if __name__ == '__main__':
    unittest.main()
