"""The playable population must learn without a scripted fallback or save reset."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from agents.sequence_ppo import Config, Population
from experiments.living_population import LearningRuntime, load_population
from server import create_runtime, WorldServer
from sim.world import World


class NeuralPopulationTests(unittest.TestCase):
    def test_saved_valley_migrates_without_changing_bodies_resources_or_memories(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=root/'runs';runs.mkdir()
            world=World(113);world.ecology_enabled=True
            world.save(runs/'world.json')
            for _ in range(16):
                world.step({rid:{'verb':'wait'} for rid in world.residents},scripted=False)
            world.residents['r5'].inventory['seed']=3
            world.save(runs/'autosave.json')
            original=(runs/'autosave.json').read_bytes();expected=copy.deepcopy(world.to_dict())
            runtime=create_runtime(root)
            try:
                self.assertEqual(runtime.world.to_dict(),expected)
                self.assertEqual(set(runtime.population.brains),set(world.residents)-{'player'})
                self.assertFalse(runtime.population.neighbors_scripted)
                self.assertTrue(runtime.manual_pause)
                report=json.loads((runs/'controller-transition.json').read_text())
                self.assertEqual((Path(report['legacy_backup'])/'autosave.json').read_bytes(),original)
                self.assertEqual((runs/'autosave.json').read_bytes(),original)
                with patch.object(World,'baseline_action',side_effect=AssertionError('Scripted choice used')):
                    for _ in range(5):runtime._tick()
                self.assertTrue(all(b.decisions==2 for b in runtime.population.brains.values()))
                runtime._save('world.json');saved=copy.deepcopy(runtime.world.to_dict())
                runtime._tick();self.assertTrue(runtime.control({'command':'load'})[0])
                self.assertEqual(runtime.world.to_dict(),saved)
                self.assertEqual(len(runtime.population.brains),8)
                state=json.loads(runtime.state_bytes())
                self.assertEqual(state['scriptedResidents'],0)
                self.assertEqual(set(state['learning']),set(world.residents)-{'player'})
            finally:runtime.close()
            resumed=create_runtime(root)
            try:self.assertEqual(resumed.world.to_dict(),saved)
            finally:resumed.close()

    def test_all_eight_learn_independently_and_resume_exactly(self):
        world=World(31)
        p=Population.from_world(world,Config(rollout=4,epochs=1))
        before={rid:copy.deepcopy(b.model.state_dict()) for rid,b in p.brains.items()}
        self.assertEqual(len({b.model.actor.weight.data_ptr() for b in p.brains.values()}),8)
        self.assertEqual(len({id(b.optimizer) for b in p.brains.values()}),8)
        with patch.object(World,'baseline_action',side_effect=AssertionError('Scripted choice used')):
            for _ in range(21):p.step()
        for rid,b in p.brains.items():
            self.assertEqual(b.updates,1)
            self.assertTrue(any(not torch.equal(v,b.model.state_dict()[k]) for k,v in before[rid].items()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'p.pt';p.save(path);q=Population.load(path)
            for _ in range(9):self.assertEqual(p.step(),q.step())
            self.assertEqual(p.world.to_dict(),q.world.to_dict())
            self.assertEqual(p.metrics,q.metrics)
            for rid,b in p.brains.items():
                self.assertTrue(all(torch.equal(v,q.brains[rid].model.state_dict()[k])
                                    for k,v in b.model.state_dict().items()))

    def test_promoting_existing_pair_preserves_both_learned_brains(self):
        from agents.lifelong import Population as LegacyPopulation
        for cls in (Population,LegacyPopulation):
            with self.subTest(cls=cls.__module__),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);runs=root/'runs';runs.mkdir()
                p=cls(29)
                for _ in range(9):p.step()
                expected=copy.deepcopy(p.world.to_dict())
                weights={rid:copy.deepcopy(b.model.state_dict()) for rid,b in p.brains.items()}
                updates={rid:b.updates for rid,b in p.brains.items()}
                p.save(runs/'population.pt')
                original=(runs/'population.pt').read_bytes()
                runtime=LearningRuntime(root)
                try:
                    self.assertEqual(runtime.world.to_dict(),expected)
                    self.assertEqual(len(runtime.population.brains),8)
                    self.assertEqual((runs/'before-all-residents-population.pt').read_bytes(),original)
                    for rid,old in weights.items():
                        brain=runtime.population.brains[rid]
                        self.assertEqual(brain.updates,updates[rid])
                        self.assertTrue(all(torch.equal(v,brain.model.state_dict()[k]) for k,v in old.items()))
                    with patch.object(World,'baseline_action',side_effect=AssertionError('Scripted choice used')):
                        runtime._tick()
                finally:runtime.close()
                self.assertEqual(len(load_population(runs/'population.pt').brains),8)

    def test_missing_brain_and_bad_legacy_save_fail_without_fallback(self):
        world=World(8)
        with patch.object(World,'baseline_action',side_effect=AssertionError('Scripted choice used')):
            with self.assertRaises(ValueError):world.step({'player':{'verb':'rest'}},scripted=False)
        self.assertEqual(world.tick,0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=root/'runs';runs.mkdir()
            source=runs/'autosave.json';source.write_text('unreadable save')
            with self.assertRaises(ValueError):create_runtime(root)
            self.assertEqual(source.read_text(),'unreadable save')
            self.assertFalse((runs/'population.pt').exists())
            source.unlink();runtime=create_runtime(root)
            removed=runtime.population.brains.pop('r7')
            try:
                self.assertFalse(runtime._tick())
                self.assertTrue(runtime.manual_pause)
                self.assertTrue(runtime.paused)
                self.assertIn('Every autonomous resident',runtime.controller_error)
                self.assertEqual(runtime.world.tick,0)
            finally:
                runtime.population.brains['r7']=removed;runtime.close()

    def test_learned_host_reports_all_brains_and_public_load_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=create_runtime(Path(folder),public_demo=True)
            runtime._save('world.json')
            server=WorldServer(('127.0.0.1',0),runtime)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                client.request('GET','/health');reply=client.getresponse()
                health=json.loads(reply.read());client.close()
                self.assertEqual(health['controllerMode'],'independent-learning')
                self.assertEqual(health['learners'],8)
                before=runtime.world.to_dict()
                self.assertFalse(runtime.control({'command':'load'})[0])
                self.assertEqual(runtime.world.to_dict(),before)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2);runtime.close()


if __name__=='__main__':unittest.main()
