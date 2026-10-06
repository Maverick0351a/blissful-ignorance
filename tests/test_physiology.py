import unittest
from agents.physiology import food_relief


class NutritionTests(unittest.TestCase):
    def test_no_food_reward_at_floor_or_from_decay(self):
        self.assertEqual(food_relief(0,0),0)
        self.assertEqual(food_relief(.004,0),0)
        self.assertEqual(food_relief(20,19.992),0)
        self.assertAlmostEqual(food_relief(0,24.992),25)
        self.assertAlmostEqual(food_relief(95,99.992),5)

    def test_both_populations_record_no_food_at_zero(self):
        from agents.lifelong import Population as Legacy
        from agents.sequence_ppo import Population as Sequence
        for cls in (Legacy,Sequence):
            p=cls(10);p.world.resources.clear()
            for a in p.world.residents.values():
                for item in a.inventory:a.inventory[item]=0
            for rid in p.brains:p.world.residents[rid].food=0
            for _ in range(16):p.step()
            for m in p.metrics.values():self.assertEqual(m['nutrition'],0)


if __name__=='__main__':unittest.main()
