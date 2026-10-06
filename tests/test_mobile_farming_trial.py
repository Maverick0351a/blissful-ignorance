import unittest
from experiments.mobile_farming_trial import arena, sense, Learner, episode
class MobileTrialTests(unittest.TestCase):
    def test_initial_conditions_and_masks(self):
        w=arena(4)
        self.assertEqual(len(w.residents),2)
        self.assertEqual(sum(t==0 for row in w.terrain for t in row),6)
        for rid,a in w.residents.items():
            self.assertEqual((a.food,a.inventory['food']),(40,2))
            state,mask,*_=sense(w,rid)
            self.assertEqual(len(mask),9)
            self.assertFalse(mask[7]); self.assertFalse(mask[8])
    def test_independent_learning(self):
        a,b=Learner(1),Learner(2)
        a.q[(1,)][0]=8
        self.assertEqual(b.q[(1,)][0],0)
        self.assertIsNot(a.trace,b.trace)
    def test_authoritative_ticks(self):
        row,trace=episode('scripted',123,{r:Learner(i) for i,r in enumerate(('player','r0'))},decisions=8)
        self.assertEqual(row['ticks'],960)
        self.assertEqual(len(trace),8)
        self.assertGreater(sum(a['planted'] for a in row['agents'].values()),0)
if __name__=='__main__': unittest.main()
