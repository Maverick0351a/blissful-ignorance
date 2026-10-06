import copy
import unittest
import torch
from agents.sequence_ppo import Config
from experiments.category_trial import population_for
from experiments.near_food_trial import make_world
from experiments.feeding_credit import (CreditBrain,canonical_inputs,credit_terms,fingerprint,
    gradient_components,probe,summarize)
from experiments.audit_feeding_credit import reference_gae


class FeedingCreditTests(unittest.TestCase):
    def test_food_credit_spreads_back_and_terminal_cuts_it(self):
        cfg=Config(gamma=.9,gae_lambda=.8)
        rows=[dict(reward=v,value=.2,terminal=i==2) for i,v in enumerate((-.1,-.1,.9))]
        journal=[dict(reward=dict(food=1. if i==2 else 0.,hunger=-.1,injury=0.)) for i in range(3)]
        adv,norm,returns,parts=credit_terms(rows,journal,5.,cfg)
        self.assertTrue(torch.allclose(parts['food'],torch.tensor([.72**2,.72,1.])))
        self.assertTrue(torch.allclose(sum(parts.values()),adv))
        expected=reference_gae([r['reward'] for r in rows],[.2]*3,5.,[False,False,True],.9,.8)
        self.assertTrue(torch.allclose(adv,torch.tensor(expected)))
        self.assertAlmostEqual(float(norm.mean()),0.,places=6)
        cut=reference_gae([0,0,1],[0,0,0],9,[False,True,True],1,1)
        self.assertEqual(cut,[0,0,1])

    def test_positive_meal_reward_can_have_negative_relative_advantage(self):
        rows=[dict(reward=v,value=0.,terminal=True) for v in (1.,3.,4.)]
        journal=[dict(reward=dict(food=r['reward'])) for r in rows]
        adv,norm,_,_=credit_terms(rows,journal,0.,Config())
        self.assertGreater(float(adv[0]),0);self.assertLess(float(norm[0]),0)

    def test_observer_reproduces_native_learning_and_preserves_rng(self):
        cfg=Config(rollout=8,epochs=2)
        native=population_for(93421,'category',make_world(92400,0,'carried')[0],cfg)
        observed=population_for(93421,'category',make_world(92400,0,'carried')[0],cfg)
        for rid,b in list(observed.brains.items()):
            instrument=CreditBrain(b.seed,cfg);instrument.restore(copy.deepcopy(b.state()))
            instrument.probes=canonical_inputs(instrument,rid);observed.brains[rid]=instrument
        for case in range(3):
            for p in (native,observed):
                p.world=make_world(92400+case,case,'carried')[0]
                for rid,b in p.brains.items():
                    if isinstance(b,CreditBrain):b.context=dict(candidate=rid,case=case,episode=str(case),training_seed=93421)
            for _ in range(32):
                self.assertEqual(native.step(),observed.step())
            for rid in native.brains:
                native.brains[rid].finish(native.world.observe(rid),terminal=True)
                observed.brains[rid].finish(observed.world.observe(rid),terminal=True)
                self.assertEqual(fingerprint(native.brains[rid].state()),fingerprint(observed.brains[rid].state()))
            self.assertEqual(native.world.to_dict(),observed.world.to_dict())
        for b in observed.brains.values():
            self.assertEqual(len(b.records),3)
            for record in b.records:
                self.assertGreater(record['gradient']['combined_norm'],0)
                self.assertLessEqual(record['gradient']['clipping_scale'],1)

    def test_probe_and_state_fingerprint_cover_private_history(self):
        b=CreditBrain(92701,Config());inputs=canonical_inputs(b,'r0')
        before=copy.deepcopy(b.state());digest=fingerprint(before);rng=torch.random.get_rng_state().clone()
        for v in inputs.values():probe(b,v);probe(b,v,False)
        self.assertEqual(digest,fingerprint(b.state()));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        b.decisions+=1;self.assertNotEqual(digest,fingerprint(b.state()))
        self.assertEqual(fingerprint(dict(a=1,b=2)),fingerprint(dict(b=2,a=1)))

    def test_short_or_negative_anchor_gain_is_not_retention_evidence(self):
        def record(update,first,before,after):
            view=lambda x:{'zero_context':{'eat':x},'fixed_context':{'eat':x}}
            return dict(candidate='c0',update=update,transitions=[],anchor_first=first,
                anchor_before=view(before),anchor_after=view(after),canonical_before={'carried-0':{'eat':.2},'adjacent-0':{'gather':.2}},
                canonical_after={'carried-0':{'eat':.3},'adjacent-0':{'gather':.3}},gradient={'norms':dict(actor=1.,value=1.,entropy=1.)})
        short=summarize([record(1,True,.2,.3),record(4,False,.3,.4)])
        self.assertFalse(short['candidates']['c0']['retention']['zero_context']['eligible'])
        loss=summarize([record(1,True,.3,.2),record(6,False,.2,.4)])
        self.assertIsNone(loss['candidates']['c0']['retention']['zero_context']['retains_half_gain'])
        lost=summarize([record(1,True,.2,.3),record(6,False,.3,.21)])
        self.assertFalse(lost['candidates']['c0']['retention']['zero_context']['retains_half_gain'])
        self.assertFalse(lost['milestone_confirmed'])


if __name__=='__main__':unittest.main()
