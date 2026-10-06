import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from agents.laya_resident import LayaBrain
from agents.sequence_ppo import Population, Config
from experiments.add_laya_resident import add_laya
from experiments.living_population import LearningRuntime
from sim.world import World


class MixedPopulationTests(unittest.TestCase):
    def test_arrival_preserves_existing_lives_and_checkpoints_both_models(self):
        p=Population.from_world(World(945), Config(rollout=8,epochs=1))
        for _ in range(7):p.step()
        original=copy.deepcopy(p.world.to_dict())
        weights={rid:copy.deepcopy(b.model.state_dict()) for rid,b in p.brains.items()}
        counters={rid:(b.decisions,b.updates,len(b.buffer)) for rid,b in p.brains.items()}
        self.assertTrue(add_laya(p));self.assertFalse(add_laya(p))
        self.assertEqual(p.world.tick,original['tick'])
        for rid,b in p.brains.items():
            if rid=='laya':continue
            self.assertEqual((b.decisions,b.updates,len(b.buffer)),counters[rid])
            for key,value in weights[rid].items():self.assertTrue(torch.equal(value,b.model.state_dict()[key]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'population.pt';p.save(path);q=Population.load(path)
            self.assertEqual(p.world.to_dict(),q.world.to_dict())
            self.assertEqual(p.metrics,q.metrics)
            self.assertIsInstance(q.brains['laya'],LayaBrain)
            self.assertIsNone(q.brains['laya']._engine)
            self.assertEqual(q.brains['laya'].rng.getstate(),p.brains['laya'].rng.getstate())
            self.assertEqual(set(q.brains),set(q.world.residents)-{'player'})

    def test_model_failure_does_not_advance_other_brains_or_time(self):
        p=Population.from_world(World(31));add_laya(p)
        before=copy.deepcopy(p.world.to_dict())
        rng=p.brains['laya'].rng.getstate()
        with patch.object(p.brains['laya'],'_load_engine',side_effect=RuntimeError('test NPU offline')):
            with self.assertRaisesRegex(RuntimeError,'NPU offline'):p.step()
        self.assertEqual(p.world.to_dict(),before)
        self.assertTrue(all(b.decisions==0 for b in p.brains.values()))
        self.assertEqual(rng,p.brains['laya'].rng.getstate())

    def test_runtime_pauses_and_explains_model_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=LearningRuntime(Path(folder))
            try:
                add_laya(runtime.population)
                with patch.object(runtime.population.brains['laya'],'_load_engine',side_effect=RuntimeError('test NPU offline')):
                    runtime.manual_pause=runtime.paused=False
                    runtime._tick()
                self.assertTrue(runtime.paused)
                self.assertTrue(runtime.manual_pause)
                self.assertEqual(runtime.world.tick,0)
                self.assertIn('test NPU offline',runtime.controller_error)
            finally:runtime.close()


if __name__=='__main__':unittest.main()
