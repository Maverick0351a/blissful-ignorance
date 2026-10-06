"""Accounting and learning-boundary checks for the disposable farming pilot."""
import unittest
from experiments.farming_trial import Learner, episode


class FarmingTrialTests(unittest.TestCase):
    def test_reference_farmer_and_food_accounting(self):
        for competition in (False, True):
            row, history = episode('scripted', 9991, competition, Learner(9991))
            self.assertGreater(row['meals'], 2)
            self.assertGreater(row['harvested'], 0)
            self.assertEqual(row['zero_food_ticks'], 0)
            self.assertEqual(row['final_inventory']['food'], 2 + row['harvested'] - row['meals'] - row['conversions'])
            self.assertEqual(row['final_inventory']['seed'], 2 * row['conversions'] - row['planted'])
            self.assertAlmostEqual(sum(h['reward'] * 25 for h in history), row['nutrition'])
            self.assertEqual(row['ticks'], 9600)

    def test_terminal_transition_has_no_bootstrap(self):
        learner = Learner(42)
        learner.q[('next',)] = [100.] * 5
        learner.update(('now',), 0, 1, ('next',), [True] * 5, True)
        self.assertAlmostEqual(learner.q[('now',)][0], .15)

    def test_frozen_evaluation_does_not_train(self):
        learner = Learner(42)
        episode('frozen', 9991, False, learner)
        self.assertTrue(all(v == 0 for values in learner.q.values() for v in values))


if __name__ == '__main__':
    unittest.main()
