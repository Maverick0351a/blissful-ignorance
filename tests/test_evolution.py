import tempfile
from pathlib import Path
import unittest
import torch
from agents.replay_policy import Learner,encode
from agents.evolution import fork_candidate,load_parent,promotion_gate

STATE=(2,2,0,-1,0,0,1,0,False,5,0,0,False)

class EvolutionTests(unittest.TestCase):
    def test_growth_preserves_output_and_independence(self):
        parent=Learner(42,hidden_size=8)
        parent.curiosity.counts['scene']=3
        before={k:v.clone() for k,v in parent.network.state_dict().items()}
        child,lineage=fork_candidate(parent,43,hidden_size=12)
        self.assertTrue(torch.allclose(parent.network(encode(STATE)),child.network(encode(STATE)),atol=1e-6))
        self.assertTrue(torch.allclose(parent.target(encode(STATE)),child.target(encode(STATE)),atol=1e-6))
        self.assertGreater(sum(p.numel() for p in child.network.parameters()),sum(p.numel() for p in parent.network.parameters()))
        child.curiosity.counts['scene']=4
        self.assertEqual(parent.curiosity.counts['scene'],3)
        loss=child.network(encode(STATE)).sum();loss.backward();child.optimizer.step()
        self.assertTrue(all(torch.equal(before[k],v) for k,v in parent.network.state_dict().items()))
        self.assertIsNot(child.replay,parent.replay)
        self.assertEqual(len(child.replay),0)

    def test_adjustable_budget_and_checkpoint(self):
        parent=Learner(42,hidden_size=8)
        with self.assertRaises(ValueError):fork_candidate(parent,43,12,max_parameters=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'parent.pt';parent.save(path);loaded=load_parent(path)
            self.assertTrue(torch.equal(parent.network(encode(STATE)),loaded.network(encode(STATE))))

    def test_promotion_rejects_regression_and_unchanged_children(self):
        parent=[dict(seed=i,nutrition=100,zero_food_ticks=0,ticks=1000,harvested=4) for i in range(3)]
        self.assertFalse(promotion_gate(parent,parent)['passed'])
        child=[dict(r,nutrition=110) for r in parent]
        self.assertTrue(promotion_gate(parent,child)['passed'])
        child[0]['zero_food_ticks']=1
        self.assertFalse(promotion_gate(parent,child)['passed'])

if __name__=='__main__':unittest.main()
