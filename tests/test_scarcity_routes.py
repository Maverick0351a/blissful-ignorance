from collections import deque
import copy
import time
import unittest

from agents.sequence_ppo import Config, Population
from experiments.scarcity_routes import Arena, Routes, configure, food_stock, safe_oracle
from experiments.scarcity_trial import gate, run
from sim.world import DIRECTIONS


class ScarcityTests(unittest.TestCase):
    def test_detour_is_longer_and_food_is_finite_and_private(self):
        p=Population(41);arenas=configure(p,81)
        self.assertEqual(set(p.world.residents),{'r0','r1'})
        for rid,arena in arenas.items():
            self.assertEqual(food_stock(p.world,arena),4)
            self.assertEqual(p.world.residents[rid].food,8)
            observation=p.world.observe(rid)
            self.assertEqual(observation['others'],[])
            self.assertNotIn('arenas',observation)
            self.assertNotIn('route',observation)
            queue=deque([(arena.point(1,0),0)]);seen=set();length=None
            while queue:
                point,n=queue.popleft()
                if point in seen:continue
                seen.add(point)
                if point==arena.point(3,0):length=n;break
                for dx,dy in DIRECTIONS.values():
                    q=(point[0]+dx,point[1]+dy)
                    if q in arena.cells() and q!=arena.hazard:queue.append((q,n+1))
            self.assertEqual(length,2+2*abs(arena.detour))
            self.assertGreater(length,2)
        self.assertTrue(arenas['r0'].cells().isdisjoint(arenas['r1'].cells()))

    def test_completed_routes_require_physical_crossings(self):
        arena=Arena((20,30),(1,0),-2);routes=Routes(arena)
        for _ in range(10):routes.observe(arena.origin)
        self.assertEqual(routes.summary()['crossings'],0)
        for point in ((1,0),(2,0),(3,0)):routes.observe(arena.point(*point))
        self.assertEqual(routes.summary()['unsafe_crossings'],1)
        for point in ((3,-1),(3,-2),(2,-2),(1,-2),(1,-1)):
            routes.observe(arena.point(*point))
        self.assertEqual(routes.summary()['safe_crossings'],1)
        routes.observe(arena.point(1,-2));routes.observe(arena.point(1,-1))
        self.assertEqual(routes.summary()['aborted_crossings'],1)
        with self.assertRaises(ValueError):routes.observe(arena.point(4,0))

    def test_safe_oracle_can_feed_without_injury(self):
        # Exercise the authoritative physics, not model outputs or route labels.
        p=Population(1);arenas=configure(p,731)
        for _ in range(1200):
            p.world.step({rid:safe_oracle(p.world,rid) if p.world.tick%4==0 else {'verb':'wait'}
                          for rid in p.brains})
        for rid,a in p.world.residents.items():
            self.assertGreater(a.food,8)
            self.assertEqual(a.thorn_contacts,0)
            self.assertAlmostEqual(a.food+25*food_stock(p.world,arenas[rid]),8+100-1200*.008,places=6)

    def test_accounted_rollout_and_no_hidden_hints(self):
        p=Population(9,Config(rollout=8,epochs=1));arenas=configure(p,9)
        observations={rid:p.world.observe(rid) for rid in p.brains}
        for rid in p.brains:
            self.assertEqual(observations[rid]['others'],[])
            p.brains[rid].freeze()
            p.brains[rid].decide=lambda o,rid=rid:safe_oracle(p.world,rid)
        stats=run(p,arenas,256,time.perf_counter()+30)
        for m in stats.values():
            self.assertGreater(m['safe_foodward_crossings'],0)
            self.assertEqual(m['unsafe_crossings'],0)
            self.assertEqual(m['remaining_food_stock'],4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions'])
            self.assertGreater(m['nutrition'],0)

    def test_gate_rejects_inaction_despite_good_survival(self):
        initial=dict(zero_food_ticks=1000,mean_fullness=1.,thorn_contacts=10,conscious_decisions=100,
                     foodward_crossings=0,safe_foodward_crossings=0,safe_crossings=0,crossings=0)
        trained={**initial,'zero_food_ticks':0,'mean_fullness':40.,'thorn_contacts':0}
        rows=[dict(seed=1,condition='initial',metrics={'r0':initial,'r1':copy.deepcopy(initial)}),
              dict(seed=1,condition='trained',metrics={'r0':trained,'r1':copy.deepcopy(trained)})]
        result=gate(rows,12000)
        self.assertEqual(result['reliable_lives'],2)
        self.assertFalse(result['passed'])

    def test_seed_conversion_and_real_crop_yield_are_accounted(self):
        p=Population(11);arenas=configure(p,13);arena=arenas['r0']
        p.world.residents['r0'].inventory['food']=1
        p.world.resources[p.world.key(*arena.point(4,0))]['amount']=3
        target=arena.point(0,1);dx=target[0]-arena.origin[0];dy=target[1]-arena.origin[1]
        direction=next(d for d,delta in DIRECTIONS.items() if delta==(dx,dy))
        p.world.soils[p.world.key(*target)]={'fertility':1.,'moisture':1.}
        actions=iter(({'verb':'make_seeds'},{'verb':'plant','direction':direction}))
        for rid,b in p.brains.items():
            b.freeze();b.decide=lambda o:{'verb':'rest'}
        p.brains['r0'].decide=lambda o:next(actions,{'verb':'rest'})
        stats=run(p,arenas,512,time.perf_counter()+30)['r0']
        self.assertEqual(stats['seed_conversions'],1)
        self.assertEqual(stats['planted'],1)
        self.assertEqual(stats['matured_food'],3)
        self.assertEqual(stats['remaining_food_stock'],6)


if __name__=='__main__':unittest.main()
