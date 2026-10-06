import copy
from pathlib import Path
import tempfile
import unittest

import torch
from agents.sequence_ppo import Brain, Config, Population, advantages
from agents.network import ACTIONS, encode_observation
from experiments.development import configure


class SequenceTests(unittest.TestCase):
    def test_delayed_credit_and_terminal_cut(self):
        values=torch.zeros(3)
        adv,returns=advantages([0,0,1],values,0,[False]*3,1,1)
        self.assertEqual(returns.tolist(),[1,1,1])
        adv,_=advantages([0,0,1],values,2,[False,True,False],1,1)
        self.assertEqual(adv.tolist(),[0,0,3])
        _,returns=advantages([0],torch.zeros(1),2,[False],.5,1)
        self.assertEqual(returns.tolist(),[1])

    def test_on_policy_sequence_and_frozen_evaluation(self):
        p=configure(Population(61,Config(rollout=8,epochs=2)),61)
        before={k:v.clone() for k,v in p.brains['r0'].model.state_dict().items()}
        for _ in range(32):p.step()
        self.assertEqual(p.brains['r0'].updates,0)
        self.assertTrue(all(torch.equal(v,p.brains['r0'].model.state_dict()[k]) for k,v in before.items()))
        p.step();brain=p.brains['r0']
        self.assertEqual(brain.updates,1);self.assertEqual(brain.diagnostics['sequence_length'],8)
        self.assertTrue(any(not torch.equal(v,brain.model.state_dict()[k]) for k,v in before.items()))
        self.assertGreater(len(brain.journal),0)
        brain.freeze();fixed=copy.deepcopy(brain.state())
        for _ in range(9):brain.decide(p.world.observe('r0'))
        for module in ('model','predictor'):
            self.assertTrue(all(torch.equal(v,brain.state()[module][k]) for k,v in fixed[module].items()))
        self.assertEqual(brain.updates,fixed['updates']);self.assertEqual(brain.journal,fixed['journal'])
        self.assertEqual(brain.buffer,[])

    def test_gradient_reaches_earlier_observations(self):
        p=Population(51);brain=p.brains['r0'];patch,features=encode_observation(p.world.observe('r0'))
        patches=patch.repeat(3,1,1,1).requires_grad_();feature=features.repeat(3,1)
        logits,_=brain._sequence(patches,feature,brain.model.initial_state())
        logits[-1,ACTIONS.index({'verb':'eat'})].backward()
        self.assertGreater(float(patches.grad[0].abs().sum()),0)

    def test_exact_resume_mid_rollout_and_independence(self):
        p=configure(Population(33,Config(rollout=8,epochs=2)),33,'hazards')
        self.assertNotEqual(p.brains['r0'].model.actor.weight.data_ptr(),p.brains['r1'].model.actor.weight.data_ptr())
        self.assertNotEqual(p.brains['r0'].predictor[0].weight.data_ptr(),p.brains['r1'].predictor[0].weight.data_ptr())
        for _ in range(21):p.step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'p.pt';p.save(path);q=Population.load(path)
            for _ in range(25):self.assertEqual(p.step(),q.step())
            self.assertEqual(p.world.to_dict(),q.world.to_dict());self.assertEqual(p.metrics,q.metrics)
            for rid in p.brains:
                self.assertEqual(p.brains[rid].journal,q.brains[rid].journal)
                self.assertEqual(p.brains[rid].diagnostics,q.brains[rid].diagnostics)
                for module in ('model','predictor'):
                    self.assertTrue(all(torch.equal(v,q.brains[rid].state()[module][k])
                                        for k,v in p.brains[rid].state()[module].items()))

    def test_development_keeps_local_interface_and_hazards(self):
        p=configure(Population(6),6,'hazards');o=p.world.observe('r0')
        harmful=ACTIONS.index({'verb':'eat','item':'amber_fruit'})
        from agents.affordances import action_mask
        self.assertTrue(action_mask(o,ACTIONS)[harmful]);self.assertFalse(p.neighbors_scripted)
        altered=copy.deepcopy(o);altered.update(player=True,secret_resources={'hidden':100})
        a,b=encode_observation(o);c,d=encode_observation(altered)
        self.assertTrue(torch.equal(a,c));self.assertTrue(torch.equal(b,d))

    def test_recovery_does_not_earn_reward_and_noisy_error_has_no_bonus(self):
        p=configure(Population(7),7);a=p.world.residents['r0'];a.health=30
        p.step()
        self.assertGreater(a.health,30)
        self.assertEqual(p.brains['r0'].reward_parts['injury'],0)
        self.assertEqual(p.brains['r0'].config.curiosity_coefficient,0)
        self.assertNotIn('curiosity',p.brains['r0'].reward_parts)
        p.brains['r0'].buffer=[{'version':-1}]
        with self.assertRaises(RuntimeError):p.brains['r0']._learn(0)


if __name__=='__main__':unittest.main()
