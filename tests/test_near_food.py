import copy
import unittest

import torch
from agents.affordances import action_mask
from agents.network import ACTIONS
from agents.sequence_ppo import Config
from experiments.near_food_trial import make_world, choice_probe, quick_success, summarize, ARMS, FIXTURES
from experiments.category_trial import population_for


class NearFoodTests(unittest.TestCase):
    def test_gather_then_eat_immediately_possible_without_moving(self):
        directions = {rid:set() for rid in ('r0','r1')}
        for case in range(8):
            w, meta = make_world(91000+case,case)
            starts = {rid:(a.x,a.y) for rid,a in w.residents.items()}
            for rid, a in w.residents.items():
                x,y = meta['locations'][rid]['food_at']; directions[rid].add((x-a.x,y-a.y))
                self.assertEqual(abs(x-a.x)+abs(y-a.y),1)
                self.assertFalse(any(a.inventory.values()))
                mask = action_mask(w.observe(rid),ACTIONS)
                self.assertTrue(mask[ACTIONS.index({'verb':'gather'})])
                self.assertFalse(mask[ACTIONS.index({'verb':'eat'})])
                self.assertFalse({'fixture','case','food_at','world_seed'} & set(w.observe(rid)))
            # Physics feasibility only. These commands are never given to learners.
            results = w.step({rid:{'verb':'gather'} for rid in w.residents},scripted=False)
            self.assertTrue(all(result[0] for result in results.values()))
            for _ in range(3):w.step({rid:{'verb':'wait'} for rid in w.residents},scripted=False)
            results = w.step({rid:{'verb':'eat'} for rid in w.residents},scripted=False)
            self.assertTrue(all(result[0] for result in results.values()))
            self.assertEqual(starts,{rid:(a.x,a.y) for rid,a in w.residents.items()})
            self.assertTrue(all(a.food>28 for a in w.residents.values()))
        self.assertTrue(all(len(d)==4 for d in directions.values()))

    def test_carried_probe_is_consumption_only_and_tones_remain(self):
        w,_ = make_world(91822,3,'carried')
        self.assertFalse(w.resources)
        for rid,a in w.residents.items():
            self.assertEqual(a.inventory['food'],1);self.assertEqual(sum(a.inventory.values()),1)
            mask = action_mask(w.observe(rid),ACTIONS)
            self.assertTrue(mask[ACTIONS.index({'verb':'eat'})])
            self.assertFalse(mask[ACTIONS.index({'verb':'gather'})])
            self.assertEqual(sum(ok for a,ok in zip(ACTIONS,mask) if a['verb']=='tone'),10)
            self.assertTrue(all(ok for a,ok in zip(ACTIONS,mask) if a['verb']=='move'))

    def test_probability_probe_has_no_side_effect_or_rng_draw(self):
        w,_ = make_world(91991,1)
        for kind in ('flat','category'):
            p = population_for(71,kind,w,Config()); b = p.brains['r0']; b.freeze()
            before = copy.deepcopy(b.state()); world = copy.deepcopy(w.to_dict())
            first = choice_probe(b,w.observe('r0')); second = choice_probe(b,w.observe('r0'))
            self.assertEqual(first,second);self.assertAlmostEqual(sum(first),1.,places=6)
            self.assertEqual(before['rng'],b.state()['rng'])
            self.assertEqual(before['decisions'],b.decisions)
            for initial,actual in zip(before['hidden'],b.hidden):self.assertTrue(torch.equal(initial,actual))
            for name,tensor in before['model'].items():self.assertTrue(torch.equal(tensor,b.model.state_dict()[name]))
            self.assertEqual(world,w.to_dict())

    def test_deadlines_and_empty_pack_acquisition_are_distinct(self):
        m=dict(first_meal_tick=60,first_gather_food_tick=0,first_zero_tick=None)
        self.assertTrue(quick_success(m,'adjacent'))
        self.assertFalse(quick_success(dict(m,first_meal_tick=64),'adjacent'))
        self.assertFalse(quick_success(dict(m,first_zero_tick=20),'adjacent'))
        self.assertFalse(quick_success(dict(m,first_gather_food_tick=None),'adjacent'))
        self.assertTrue(quick_success(dict(m,first_meal_tick=0,first_gather_food_tick=None),'carried'))
        self.assertFalse(quick_success(dict(m,first_meal_tick=16),'carried'))

    def test_equal_random_success_cannot_pass_a_learning_gate(self):
        rows=[]
        for seed in (1,2,3):
            for fixture in FIXTURES:
                for arm in ARMS:
                    m=dict(first_meal_tick=4,first_gather_food_tick=0 if fixture=='adjacent' else None,
                        first_zero_tick=None,timely_acquisition=fixture=='adjacent',safe_eaten=1,
                        amber_eaten=0,zero_food_ticks=0,unconscious_ticks=0)
                    rows.append(dict(metadata=dict(training_seed=seed,fixture=fixture,arm=arm,
                        choice_probes={rid:[1/len(ACTIONS)]*len(ACTIONS) for rid in ('r0','r1')}),
                        metrics={rid:dict(m) for rid in ('r0','r1')}))
        result=summarize(rows,(1,2,3))
        self.assertFalse(any(result['adjacent_learning_gate'].values()))
        self.assertFalse(any(result['carried_consumption_gate'].values()))
        self.assertFalse(result['milestone_confirmed'])


if __name__=='__main__':unittest.main()
