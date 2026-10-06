import copy
import unittest
import torch
from agents.network import ACTIONS,encode_observation
from agents.sequence_ppo import Config
from experiments.category_trial import population_for
from experiments.near_food_trial import make_world
from experiments.feeding_credit import fingerprint
from experiments.value_interference import detached_value_input,copy_brain,summarize
from experiments.audit_value_interference import ExplicitDetached


def rollout_with_optimizer():
    cfg=Config(rollout=8,epochs=2)
    p=population_for(95631,'category',make_world(95811,0,'carried')[0],cfg)
    for _ in range(32):p.step()
    for rid,b in p.brains.items():b.finish(p.world.observe(rid),terminal=True)
    p.world=make_world(95812,1,'carried')[0]
    for _ in range(32):p.step()
    b=p.brains['r0'];b._complete(p.world.observe('r0'),terminal=True)
    return b


class ValueInterferenceTests(unittest.TestCase):
    def test_detach_changes_only_value_gradient_path(self):
        b=rollout_with_optimizer();rows=b.buffer
        patch=torch.cat([r['patch'] for r in rows]);features=torch.cat([r['features'] for r in rows])
        params=dict(b.model.named_parameters())
        logits,values=b._sequence(patch,features,rows[0]['hidden'])
        original_logits=logits.detach().clone();original_values=values.detach().clone()
        native_value=torch.autograd.grad(values.sum(),tuple(params.values()),allow_unused=True)
        with detached_value_input(b):
            logits,values=b._sequence(patch,features,rows[0]['hidden'])
            self.assertTrue(torch.equal(logits,original_logits));self.assertTrue(torch.equal(values,original_values))
            detached=torch.autograd.grad(values.sum(),tuple(params.values()),retain_graph=True,allow_unused=True)
            actor=torch.autograd.grad(logits.sum(),tuple(params.values()),allow_unused=True)
        native_logits,_=b._sequence(patch,features,rows[0]['hidden'])
        native_actor=torch.autograd.grad(native_logits.sum(),tuple(params.values()),allow_unused=True)
        shared_nonzero=False
        for (name,param),v,d,a,n in zip(params.items(),native_value,detached,actor,native_actor):
            if name.startswith('value.'):
                self.assertIsNotNone(d);self.assertTrue(torch.equal(v,d))
            else:
                self.assertIsNone(d)
                shared_nonzero|=v is not None and float(v.abs().sum())>0
            self.assertEqual(a is None,n is None)
            if a is not None:self.assertTrue(torch.equal(a,n))
        self.assertTrue(shared_nonzero)

    def test_independent_graph_matches_hook_and_original_is_unchanged(self):
        original=rollout_with_optimizer();snapshot=copy.deepcopy(original.state());digest=fingerprint(snapshot)
        self.assertTrue(snapshot['optimizer']['state'])
        hooked=copy_brain(snapshot);explicit=ExplicitDetached(original.seed,original.config);explicit.restore(copy.deepcopy(snapshot))
        with detached_value_input(hooked):hooked._learn(0.)
        explicit._learn(0.)
        self.assertEqual(fingerprint(hooked.state()),fingerprint(explicit.state()))
        self.assertEqual(fingerprint(original.state()),digest);self.assertEqual(fingerprint(snapshot),digest)
        self.assertFalse(hooked.model.value._forward_pre_hooks)
        self.assertEqual(hooked.updates,original.updates+1)
        self.assertNotEqual(fingerprint(hooked.model.value.state_dict()),fingerprint(original.model.value.state_dict()))

    def test_hook_is_removed_on_exception(self):
        b=rollout_with_optimizer()
        with self.assertRaisesRegex(RuntimeError,'deliberate test'):
            with detached_value_input(b):
                self.assertTrue(b.model.value._forward_pre_hooks)
                raise RuntimeError('deliberate test')
        self.assertFalse(b.model.value._forward_pre_hooks)

    def test_identity_hook_is_exact_native_control(self):
        b=rollout_with_optimizer();control=copy_brain(b.state());identity=copy_brain(b.state())
        handle=identity.model.value.register_forward_pre_hook(lambda module,args:args)
        try:identity._learn(0.)
        finally:handle.remove()
        control._learn(0.)
        self.assertEqual(fingerprint(identity.state()),fingerprint(control.state()))

    def test_screen_rejects_tiny_or_narrow_improvements_and_food_regression(self):
        def rows(delta=.002,gather_delta=0.,positive=6,reinforced=True):
            result=[]
            for i in range(6):
                before={'carried-0':{'eat':.2},'adjacent-0':{'gather':.25}}
                standard={'carried-0':{'eat':.21},'adjacent-0':{'gather':.26}}
                detached={'carried-0':{'eat':.21+(delta if i<positive else -delta)},
                          'adjacent-0':{'gather':.26+gather_delta}}
                result.append(dict(candidate=f'c{i}',probes_before=before,probes_standard=standard,probes_detached=detached,
                    transitions=[dict(ordinary_meal=True,before=.2,standard=.21,detached=.22 if reinforced else .19)],
                    value_mse=dict(before=1.,standard=.9,detached=.95)))
            return result
        self.assertTrue(summarize(rows())['diagnostic_promising'])
        for data in (rows(delta=.00001),rows(positive=3),rows(gather_delta=-.002),rows(reinforced=False)):
            self.assertFalse(summarize(data)['diagnostic_promising'])
        self.assertFalse(summarize(rows())['milestone_confirmed'])


if __name__=='__main__':unittest.main()
