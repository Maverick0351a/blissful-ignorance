import copy
import io
from pathlib import Path
import tempfile
import time
import unittest

from agents.sequence_ppo import Config
from experiments.category_trial import MeasuredCategory, run_episode
from experiments.feeding_credit import fingerprint
from experiments.near_food_trial import make_world
from experiments.practice_curve import (ARMS, coordinates, copy_population, advancement,
                                        summarize, validate_states)


class PracticeCurveTests(unittest.TestCase):
    def test_world_and_sampling_coordinates_do_not_collide(self):
        worlds=[];samplers=[]
        for stage in ('training','adjacent','carried'):
            for pair in range(3):
                for case in range(64 if stage=='training' else 16):
                    first=coordinates(stage,pair,case,0);second=coordinates(stage,pair,case,1)
                    self.assertEqual(first[0],second[0]);worlds.append(first[0]);samplers.extend((first[1],second[1]))
        self.assertEqual(len(worlds),len(set(worlds)));self.assertEqual(len(samplers),len(set(samplers)))
        for args in (('bad',0,0,0),('training',3,0,0),('adjacent',0,256,0),('carried',0,0,2)):
            with self.assertRaises(ValueError):coordinates(*args)

    def test_training_and_frozen_copies_preserve_full_source_and_checkpoint(self):
        config=Config(rollout=4)
        source={f'c{i}':copy.deepcopy(MeasuredCategory(12000+i,config).state()) for i in range(2)}
        source_hash=fingerprint(source)
        w,meta=make_world(12100,0);p=copy_population(source,0,w,config)
        with tempfile.TemporaryDirectory(prefix='godhood-practice-test-') as directory:
            out=Path(directory);trace=io.StringIO();deadline=time.perf_counter()+30
            run_episode(p,meta,'first',16,trace,out,deadline)
            prefix={f'c{i}':copy.deepcopy(p.brains[rid].state()) for i,rid in enumerate(('r0','r1'))}
            prefix_hash=fingerprint(prefix)
            self.assertTrue(prefix['c0']['optimizer']['state']);self.assertTrue(prefix['c0']['journal'])
            w2,meta2=make_world(12101,1)
            resumed=copy_population(prefix,0,copy.deepcopy(w2),config)
            p.world=w2;p.metrics={rid:p.new_metrics() for rid in p.brains}
            for candidate,label in ((p,'continued'),(resumed,'resumed')):
                run_episode(candidate,meta2,label,16,trace,out,deadline)
            for rid in ('r0','r1'):self.assertEqual(fingerprint(p.brains[rid].state()),fingerprint(resumed.brains[rid].state()))
            frozen=copy_population(prefix,0,make_world(12102,2)[0],config,frozen=True)
            frozen.step();self.assertFalse(frozen.brains['r0'].training)
            self.assertEqual(fingerprint(prefix),prefix_hash)
        self.assertEqual(fingerprint(source),source_hash)

    def test_wrong_source_history_or_missing_resident_is_rejected(self):
        config=Config();states={f'c{i}':MeasuredCategory(13000+i,config).state() for i in range(6)}
        with self.assertRaises(ValueError):validate_states(states,config)
        for state in states.values():state['decisions']=6144;state['updates']=48
        validate_states(states,config)
        states['c4']['training']=False
        with self.assertRaises(ValueError):validate_states(states,config)
        states['c4']['training']=True;states.pop('c5')
        with self.assertRaises(ValueError):validate_states(states,config)

    def test_missing_or_duplicated_matched_evaluation_fails_closed(self):
        rows=[dict(metadata=dict(fixture=f,arm=a,pair=p,case=0)) for f in ('adjacent','carried') for a in ARMS for p in range(3)]
        with self.assertRaises(ValueError):summarize(rows[:-1],cases=1)
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],cases=1)

    def test_screen_rejects_equal_rates_weak_absolute_and_regressions(self):
        def score(n):return dict(prompt_successes=n,lives=100,resident_ticks=51200,zero_food_ticks=0)
        good=dict(adjacent=dict(arms=dict(zip(ARMS,map(score,(50,60,80)))),
            seed_groups={str(p):dict(zip(ARMS,map(score,(50,60,80)))) for p in range(3)}),
            carried=dict(arms=dict(zip(ARMS,map(score,(75,80,80))))))
        self.assertTrue(all(advancement(good).values()))
        for change in ('equal','absolute','seeds','carried','deprivation'):
            trial=copy.deepcopy(good)
            if change=='equal':trial['adjacent']['arms']['after-16']['prompt_successes']=80
            if change=='absolute':trial['adjacent']['arms']['after-64']['prompt_successes']=79
            if change=='seeds':
                for p in ('0','1'):trial['adjacent']['seed_groups'][p]['after-64']['prompt_successes']=59
            if change=='carried':trial['carried']['arms']['after-64']['prompt_successes']=74
            if change=='deprivation':trial['adjacent']['arms']['after-64']['zero_food_ticks']=257
            self.assertFalse(all(advancement(trial).values()),change)


if __name__=='__main__':unittest.main()
