import tempfile
from pathlib import Path
import unittest
import torch
from agents.lifelong import Population


class LifelongTests(unittest.TestCase):
    def test_independence_and_exact_checkpoint_continuation(self):
        p=Population(21)
        before={k:v.clone() for k,v in p.brains['r0'].model.state_dict().items()}
        self.assertNotEqual(p.brains['r0'].model.actor.weight.data_ptr(),p.brains['r1'].model.actor.weight.data_ptr())
        for _ in range(11):p.step()
        self.assertTrue(any(not torch.equal(v,p.brains['r0'].model.state_dict()[k]) for k,v in before.items()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'population.pt';p.save(path);q=Population.load(path)
            for _ in range(13):
                self.assertEqual(p.step(),q.step())
            self.assertEqual(p.world.to_dict(),q.world.to_dict())
            self.assertEqual(p.metrics,q.metrics)
            for rid in p.brains:
                self.assertEqual(p.brains[rid].updates,q.brains[rid].updates)
                self.assertTrue(all(torch.equal(v,q.brains[rid].model.state_dict()[k]) for k,v in p.brains[rid].model.state_dict().items()))


if __name__=='__main__':unittest.main()
