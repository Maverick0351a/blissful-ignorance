import unittest
from sim.world import World
from sim import ecology


class EcologyTests(unittest.TestCase):
    def setUp(self):
        self.w=World(19);self.w.ecology_enabled=True
        self.w.resources.clear();self.w.structures.clear();self.w.terrain=[[0]*64 for _ in range(64)]
        self.a=self.w.residents['player'];self.a.x=self.a.y=32
        self.a.inventory.update(food=3,wood=3,seed=2)

    def act(self,verb,**kwargs):return self.w.apply_action('player',{'verb':verb,**kwargs})[0]

    def test_dry_growth_stalls_water_and_fertility(self):
        w=self.w;w.soils['32,32']={'fertility':1.,'moisture':0.}
        self.assertTrue(self.act('plant'))
        for _ in range(20):w.tick+=1;ecology.tick(w)
        self.assertEqual(w.resources['32,32']['growth'],0)
        self.a.irrigation_water=1;self.assertTrue(self.act('water_crop'))
        for _ in range(481):w.tick+=1;ecology.tick(w)
        self.assertEqual(w.resources['32,32'],{'kind':'berry','amount':3})
        self.assertLess(w.soils['32,32']['fertility'],1.)
        fertility=w.soils['32,32']['fertility'];w.tick+=1;ecology.tick(w)
        self.assertGreater(w.soils['32,32']['fertility'],fertility)

    def test_finite_supply_and_rain(self):
        w=self.w;w.terrain[32][33]=1;w.irrigation_supply=1
        self.assertTrue(self.act('fill_water'));self.assertEqual(w.irrigation_supply,0)
        self.assertFalse(self.act('fill_water'));self.assertEqual(self.a.irrigation_water,1)
        w.tick=2400;ecology.tick(w);self.assertAlmostEqual(w.irrigation_supply,.1)

    def test_cache_conservation_carry_and_atomic_competition(self):
        w=self.w;self.assertTrue(self.act('make_cache'));self.assertEqual(self.a.inventory['wood'],0)
        self.assertTrue(self.act('stash'));self.assertEqual(self.a.inventory['food'],2)
        self.assertTrue(self.act('move_cache'));self.assertFalse(w.caches)
        self.assertEqual(w.capacity(self.a),3) # 2 food + 2 seed + 4 basket + 1 stored food
        self.assertTrue(self.act('move_cache'));self.assertIsNone(self.a.carried_cache)
        other=w.residents['r0'];other.x,other.y=33,32
        w.residents={'player':self.a,'r0':other}
        before=sum(a.inventory['food'] for a in w.residents.values())+1
        results=w.step({rid:{'verb':'take_food'} for rid in w.residents})
        self.assertEqual(sum(r[0] for r in results.values()),1)
        self.assertEqual(sum(a.inventory['food'] for a in w.residents.values())+w.caches['32,32']['food'],before)
        self.assertFalse(self.act('plant'));self.assertFalse(self.act('build',kind='floor'))

    def test_save_sensors_and_legacy_crop(self):
        w=self.w;self.assertTrue(self.act('make_cache'));self.assertTrue(self.act('stash'))
        clone=World.from_dict(w.to_dict());self.assertEqual(clone.caches,w.caches)
        observed=clone.observe('player');tile=next(t for t in observed['tiles'] if t['dx']==t['dy']==0)
        self.assertEqual(tile['cache']['food'],1);self.assertIn('fertility',tile['soil'])
        w.resources['30,30']={'kind':'crop','amount':0,'planted':0,'ready':1}
        w.step({rid:{'verb':'wait'} for rid in w.residents})
        self.assertEqual(w.resources['30,30']['kind'],'berry')


if __name__=='__main__':unittest.main()
