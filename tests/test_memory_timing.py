import unittest

from experiments.review_memory_timing import before_outcome


class MemoryTimingTests(unittest.TestCase):
    def test_update_after_arrival_does_not_explain_the_return(self):
        event = dict(opened=120, closed=128)
        self.assertFalse(before_outcome(event, {128}))
        self.assertTrue(before_outcome(event, {120}))
        self.assertTrue(before_outcome(event, {124, 128}))
        self.assertFalse(before_outcome(event, {119, 129}))

    def test_last_observed_boundary_counts_for_end_censored_opportunity(self):
        event = dict(opened=120, closed=130)
        self.assertTrue(before_outcome(event, {128}))


if __name__ == '__main__':
    unittest.main()
