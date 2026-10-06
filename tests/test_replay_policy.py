import tempfile
import unittest
from pathlib import Path
import torch
from agents.replay_policy import Learner,encode
from experiments.mobile_farming_trial import episode
from experiments.neural_farming_trial import frozen_digest,SEEDS,signature,schedule

STATE=(2,2,0,-1,0,2,1,0,False,4,0,0,False)
MASK=(True,)*9
class ReplayTests(unittest.TestCase):
    def test_independent_and_matched_initialization(self):
        a,b=Learner(1),Learner(1)
        for x,y in zip(a.network.parameters(),b.network.parameters()):
            self.assertTrue(torch.equal(x,y));self.assertNotEqual(x.data_ptr(),y.data_ptr())
        a.update(STATE,0,1,STATE,MASK,True)
        self.assertEqual(len(b.replay),0)
        a.curiosity.counts['mine']=1
        self.assertNotIn('mine',b.curiosity.counts)
    def test_mask_and_terminal_targets(self):
        a=Learner(1)
        with torch.no_grad():
            for module in (a.network,a.target):
                for p in module.parameters(): p.zero_()
            a.network[-1].bias[0]=100
            a.network[-1].bias[1]=10
            a.target[-1].bias[1]=7
        mask=[False,True]+[False]*7
        self.assertEqual(a.choose(STATE,mask,False),1)
        targets=a.targets(torch.tensor([2.,2.]),torch.stack([encode(STATE)]*2),torch.tensor([mask]*2),torch.tensor([True,False]),torch.tensor([3,3]))
        self.assertTrue(torch.isfinite(targets).all())
        self.assertEqual(float(targets[0]),2.)
        self.assertAlmostEqual(float(targets[1]),2+.995**3*7,places=5)
    def test_nstep_flush_checkpoint(self):
        a=Learner(1)
        a.update(STATE,0,1,STATE,MASK,False)
        a.update(STATE,1,2,STATE,MASK,True)
        self.assertEqual(len(a.trace),0);self.assertEqual(len(a.replay),2)
        self.assertAlmostEqual(a.replay[0][2],1+.995*2)
        self.assertTrue(all(t[5] for t in a.replay))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'agent.pt';a.save(path)
            state=torch.load(path,weights_only=False)
            self.assertEqual(state['config']['dimensions'],[23,64,64,9])
            self.assertEqual(len(state['replay']),2)
    def test_actual_episode_and_frozen_evaluation(self):
        agents={rid:Learner(i) for i,rid in enumerate(('player','r0'))}
        row,_=episode('test',991,agents,True,decisions=4)
        self.assertEqual(row['ticks'],480)
        self.assertEqual(len(agents['player'].replay),4)
        for a in agents.values(): a.freeze_curiosity();a.curiosity.last=None
        before={rid:frozen_digest(a) for rid,a in agents.items()}
        episode('test',992,agents,False,decisions=4)
        self.assertEqual(before,{rid:frozen_digest(a) for rid,a in agents.items()})
    def test_frozen_populated_optimizer(self):
        a=Learner(1)
        for _ in range(64): a.update(STATE,0,1,STATE,MASK,True)
        self.assertGreater(a.updates,0)
        a.freeze_curiosity();a.curiosity.last=None
        before=frozen_digest(a)
        self.assertEqual(before,frozen_digest(a))
        b=Learner(2);b.freeze_curiosity();b.curiosity.last=None
        episode('test',992,{'player':a,'r0':b},False,decisions=4)
        self.assertEqual(before,frozen_digest(a))
    def test_global_unseen_split(self):
        training={signature(s+i*10000) for s in SEEDS for i in range(24)}
        used=set(training);evaluation=[]
        for s in SEEDS: evaluation.extend(signature(k) for k in schedule(s,used=used)[1])
        self.assertFalse(training.intersection(evaluation))
        self.assertEqual(len(set(evaluation)),9)
if __name__=='__main__':unittest.main()
