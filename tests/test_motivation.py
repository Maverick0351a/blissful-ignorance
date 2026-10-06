import unittest
from agents.rewards import Curiosity,hunger_reward,scene_key


def view(kind=None):
    return {'tiles':[{'dx':0,'dy':0,'terrain':0,'resource':{'kind':kind}}]}


class MotivationTests(unittest.TestCase):
    def test_hunger_pressure_and_relief(self):
        self.assertEqual(hunger_reward(0,100*120),-.25)
        self.assertEqual(hunger_reward(0,0),0)
        self.assertGreater(hunger_reward(25,20*120),0)
        with self.assertRaises(ValueError):hunger_reward(0,float('nan'))

    def test_novelty_private_fades_and_cannot_reward_idle_or_fainting(self):
        a,b=Curiosity(),Curiosity()
        self.assertEqual(a.reward(view()),.02)
        self.assertEqual(a.reward(view()),0)
        self.assertEqual(a.reward(view('berry'),eligible=False),0)
        self.assertEqual(a.reward(view('berry')),.02)
        self.assertLess(a.reward(view()),.02)
        self.assertEqual(b.reward(view()),.02)
        self.assertIsNot(a.counts,b.counts)

    def test_experiment_feedback_is_integrated_and_private(self):
        from experiments.mobile_farming_trial import Learner,episode
        pair={rid:Learner(i) for i,rid in enumerate(('player','r0'))}
        row,trace=episode('scripted',123,pair,decisions=8)
        for rid,a in row['agents'].items():
            self.assertGreater(a['hunger_cost'],0)
            self.assertAlmostEqual(a['total_reward'],a['nutrition']/25-a['hunger_cost']+a['curiosity_reward'])
            self.assertAlmostEqual(sum(t['agents'][rid]['reward'] for t in trace),a['total_reward'])
        self.assertIsNot(pair['player'].curiosity,pair['r0'].curiosity)

    def test_body_and_hidden_tiles_do_not_manufacture_novelty(self):
        a=view();b=view();b['body']={'health':1,'hunger_discomfort':100};b['tiles'].append({'dx':1,'dy':0,'terrain':-1,'resource':{'kind':'berry'}})
        self.assertEqual(scene_key(a),scene_key(b))


if __name__=='__main__':unittest.main()
