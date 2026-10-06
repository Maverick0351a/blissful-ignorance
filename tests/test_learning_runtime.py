from pathlib import Path
import tempfile
import unittest

from agents.sequence_ppo import Population, Config
from agents.lifelong import Population as LegacyPopulation
from experiments.living_population import LearningRuntime,load_population


class LearningRuntimeTests(unittest.TestCase):
    def test_scarcity_stage_has_accessible_traveler_and_resumes_without_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'web').mkdir();(root/'runs').mkdir()
            runtime=LearningRuntime(root,seed=83,backend='recurrent-ppo',stage='scarcity')
            try:
                self.assertEqual(set(runtime.world.residents),{'r0','r1','player'})
                positions=[(a.x,a.y) for a in runtime.world.residents.values()]
                self.assertEqual(len(set(positions)),3)
                self.assertTrue(all(runtime.world.walkable(*p) for p in positions))
                runtime._tick();saved=runtime.world.to_dict()
            finally:runtime.close()
            restored=LearningRuntime(root,backend='recurrent-ppo',stage='near_food')
            try:self.assertEqual(restored.world.to_dict(),saved)
            finally:restored.close()

    def test_both_checkpoint_formats_and_sequence_inspector(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'web').mkdir();(root/'runs').mkdir()
            legacy=root/'runs'/'legacy.pt';LegacyPopulation(9).save(legacy)
            self.assertIsInstance(load_population(legacy),LegacyPopulation)
            p=Population(8,Config(rollout=8,epochs=1))
            for _ in range(17):p.step()
            p.save(root/'runs'/'population.pt')
            runtime=LearningRuntime(root,backend='recurrent-ppo')
            try:
                self.assertEqual(runtime.world.to_dict(),p.world.to_dict())
                state=json.loads(runtime.state_bytes())
                self.assertEqual(state['learning']['r0']['backend'],'recurrent-ppo')
                self.assertEqual(state['learning']['r0']['rollout_fill'],4)
                runtime._save('world.json');runtime._tick()
                self.assertTrue(runtime.control({'command':'load'})[0])
                self.assertEqual(runtime.world.to_dict(),p.world.to_dict())
            finally:runtime.close()


if __name__=='__main__':unittest.main()
