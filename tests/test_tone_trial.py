import unittest
import torch
from experiments.tone_trial import Policy,world_for,sender_input,receiver_input,episode,TRAIN_LAYOUTS,EVAL_LAYOUTS


class Fixed:
    def __init__(self, action):self.action=action
    def choose(self, features, training=False):return self.action


class ToneTrialTests(unittest.TestCase):
    def test_private_inputs_and_layout_split(self):
        self.assertFalse(set(TRAIN_LAYOUTS)&set(EVAL_LAYOUTS))
        for layout in TRAIN_LAYOUTS+EVAL_LAYOUTS:
            for lane in range(3):
                w,_=world_for(811,layout,lane)
                self.assertEqual(sender_input(w.observe('player')),[float(i==lane) for i in range(3)])
                self.assertEqual(sender_input(w.observe('r0')),[0.,0.,0.])
                self.assertEqual(receiver_input(w.observe('r0')),[0.,0.,0.,1.])

    def test_actual_food_outcome_and_channel_interventions(self):
        for lane in range(3):
            row=episode(Fixed(2),Fixed(lane),811,(4,3),lane)
            self.assertAlmostEqual(row['nutrition'],25)
            self.assertEqual(row['heard'],[0.,0.,1.,0.])
            wrong=episode(Fixed(2),Fixed((lane+1)%3),811,(4,3),lane)
            self.assertAlmostEqual(wrong['nutrition'],0)
        muted=episode(Fixed(2),Fixed(0),811,(4,3),0,'muted')
        shuffled=episode(Fixed(2),Fixed(0),811,(4,3),0,'shuffled',replacement=0)
        self.assertEqual(muted['heard'],[0.,0.,0.,1.])
        self.assertEqual(shuffled['heard'],[1.,0.,0.,0.])

    def test_independent_weights_and_frozen_choices(self):
        a,b=Policy(3,1),Policy(4,2)
        before=b.digest();a.learn([1.,0.,0.],0,1.)
        self.assertEqual(b.digest(),before)
        before=a.digest();a.choose([1.,0.,0.]);self.assertEqual(a.digest(),before)


if __name__=='__main__':unittest.main()
