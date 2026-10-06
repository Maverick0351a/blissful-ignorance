import copy
import unittest

import torch
from agents.sequence_ppo import Config
from experiments.category_trial import MeasuredCategory
from experiments.culling_trial import (ARMS,branch_states,sample_seed,select_retirements,selection_scores,summarize)


def metric(success=True,meals=1):
    return dict(first_meal_tick=4 if success else None,first_gather_food_tick=0,
        first_zero_tick=None,timely_acquisition=success,safe_eaten=meals if success else 0,
        amber_eaten=0,zero_food_ticks=0,unconscious_ticks=0)


class CullingTests(unittest.TestCase):
    def test_two_screens_required_and_no_forced_retirement(self):
        rows=[]
        for case in range(16):
            rows.append(dict(metadata=dict(case=case,candidates={'r0':'c0','r1':'c1'}),
                metrics={'r0':metric(case>=8),'r1':metric(True)}))
        scores=selection_scores(rows,('c0','c1'),8)
        self.assertFalse(scores['c0']['eligible']);self.assertEqual(select_retirements(scores),[])
        with self.assertRaises(ValueError):selection_scores(rows[:-1],('c0','c1'),8)

    def test_retire_at_most_two_eligible_candidates_with_stable_ties(self):
        scores={f'c{i}':dict(eligible=i!=0,round_successes=[1,1],full_life=5,ordinary_meals=5) for i in range(6)}
        scores['c4']['round_successes']=[0,0]
        scores['c0']['round_successes']=[0,0]
        self.assertEqual(select_retirements(scores),['c4','c1'])

    def test_archived_parent_and_other_branch_remain_independent(self):
        config=Config();source={f'c{i}':copy.deepcopy(MeasuredCategory(911+i,config).state()) for i in range(2)}
        source['c0']['decisions']=4096;before=copy.deepcopy(source)
        kept,ids=branch_states(source,[],'continue',config)
        reset,new_ids=branch_states(source,['c1'],'selected-restart',config)
        self.assertEqual(ids['c1'],'c1');self.assertEqual(new_ids['c1'],'selected-restart-new-0')
        self.assertEqual(reset['c1']['decisions'],0);self.assertFalse(reset['c1']['optimizer']['state'])
        self.assertEqual(reset['c0']['decisions'],4096)
        tensor=next(iter(reset['c0']['model']));reset['c0']['model'][tensor].add_(1)
        self.assertTrue(torch.equal(source['c0']['model'][tensor],before['c0']['model'][tensor]))
        self.assertTrue(torch.equal(kept['c0']['model'][tensor],before['c0']['model'][tensor]))
        self.assertNotEqual(reset['c1']['seed'],source['c1']['seed'])

    def test_fresh_initializations_match_across_controls(self):
        config=Config();source={f'c{i}':MeasuredCategory(930+i,config).state() for i in range(3)}
        selected,_=branch_states(source,['c0','c2'],'selected-restart',config)
        random,_=branch_states(source,['c1','c2'],'random-restart',config)
        for a,b in ((selected['c0'],random['c1']),(selected['c2'],random['c2'])):
            for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]))

    def test_sampling_coordinates_have_no_cross_resident_case_collision(self):
        seeds=[sample_seed(stage,case,slot) for stage in ('selection','training','evaluation') for case in range(32) for slot in range(6)]
        self.assertEqual(len(seeds),len(set(seeds)))
        with self.assertRaises(ValueError):sample_seed('selection',0,6)

    def test_equal_performance_and_only_one_good_half_cannot_pass(self):
        rows=[]
        for arm in ARMS:
            for case in range(8):
                for pair in range(3):
                    rows.append(dict(metadata=dict(arm=arm,case=case),metrics={'r0':metric(),'r1':metric()}))
        self.assertFalse(summarize(rows,8)['pilot_promising'])
        for row in rows:
            if row['metadata']['arm']!='selected-restart' and row['metadata']['case']<4:
                row['metrics']={'r0':metric(False),'r1':metric(False)}
        self.assertFalse(summarize(rows,8)['pilot_promising'])
        for row in rows:
            if row['metadata']['arm']!='selected-restart':row['metrics']={'r0':metric(False),'r1':metric(False)}
        self.assertTrue(summarize(rows,8)['pilot_promising'])


if __name__=='__main__':unittest.main()
