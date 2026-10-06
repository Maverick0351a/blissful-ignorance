import copy
import unittest

from sim.world import CROP_TICKS, World


class FarmingTests(unittest.TestCase):
    def setUp(self):
        self.w = World(713)
        self.w.resources.clear()
        self.w.structures.clear()
        self.w.residents = {k: self.w.residents[k] for k in ('player', 'r0')}
        self.a, self.b = self.w.residents.values()
        self.a.x, self.a.y = 32, 32
        self.b.x, self.b.y = 33, 32
        self.w.terrain[32][32] = self.w.terrain[32][33] = 0

    def act(self, verb, **fields):
        return self.w.apply_action('player', dict(verb=verb, **fields))[0]

    def wait(self, n):
        for _ in range(n):
            self.w.step({k: {'verb': 'wait'} for k in self.w.residents})

    def test_conversion_capacity_and_amber_rejection(self):
        self.a.inventory['amber_fruit'] = 1
        self.assertFalse(self.act('make_seeds'))
        self.a.inventory.update(food=1, wood=10)
        self.assertFalse(self.act('make_seeds'))
        self.a.inventory['wood'] = 9
        self.assertTrue(self.act('make_seeds'))
        self.assertEqual((self.a.inventory['food'], self.a.inventory['seed']), (0, 2))
        self.assertEqual(sum(self.a.inventory.values()), 12)

    def test_maturity_save_and_shared_last_fruit(self):
        self.a.inventory['seed'] = 1
        self.assertTrue(self.act('plant'))
        self.assertFalse(self.act('gather'))
        self.assertFalse(self.act('dismantle'))
        self.wait(CROP_TICKS - 1)
        saved = World.from_dict(copy.deepcopy(self.w.to_dict()))
        self.assertEqual(saved.to_dict(), self.w.to_dict())
        self.wait(1)
        saved.step({k: {'verb': 'wait'} for k in saved.residents})
        self.assertEqual(saved.to_dict(), self.w.to_dict())
        self.assertEqual(self.w.resources['32,32'], {'kind': 'berry', 'amount': 3})
        # The non-planter can harvest, and two claims never duplicate the last unit.
        for _ in range(2):
            self.assertTrue(self.w.apply_action('r0', {'verb': 'gather'})[0])
        result = self.w.step({k: {'verb': 'gather'} for k in self.w.residents})
        self.assertEqual(sum(ok for ok, _ in result.values()), 1)
        self.assertEqual(sum(a.inventory['food'] for a in self.w.residents.values()), 3)
        self.assertNotIn('32,32', self.w.resources)
        self.wait(1200)
        self.assertNotIn('32,32', self.w.resources)

    def test_placement_is_atomic_and_protects_land(self):
        self.a.inventory['seed'] = 3
        self.w.structures['32,32'] = {'kind': 'floor'}
        self.assertFalse(self.act('plant'))
        self.assertFalse(self.act('plant', x=35, y=32))
        self.w.resources['33,32'] = {'kind': 'seed', 'amount': 1}
        self.assertFalse(self.act('plant', direction='east'))
        self.assertEqual(self.a.inventory['seed'], 3)
        self.w.resources.clear()
        self.assertTrue(self.act('plant', direction='east'))
        self.assertFalse(self.act('plant', direction='east'))
        self.assertEqual(self.a.inventory['seed'], 2)

    def test_visible_stage_without_exact_clock_or_owner(self):
        self.a.inventory['seed'] = 1
        self.act('plant')
        resource = next(t['resource'] for t in self.w.observe('player')['tiles'] if (t['dx'], t['dy']) == (0, 0))
        self.assertEqual(resource, {'kind': 'crop', 'amount': 0, 'stage': 0})
        self.wait(160)
        resource = next(t['resource'] for t in self.w.observe('player')['tiles'] if (t['dx'], t['dy']) == (0, 0))
        self.assertEqual(resource['stage'], 1)

    def test_wild_stock_no_longer_refills(self):
        self.w.resources['32,32'] = {'kind': 'berry', 'amount': 1}
        self.w.resources['33,32'] = {'kind': 'amber_bush', 'amount': 0}
        self.wait(600)
        self.assertEqual(self.w.resources['32,32']['amount'], 1)
        self.assertEqual(self.w.resources['33,32']['amount'], 0)


if __name__ == '__main__':
    unittest.main()
