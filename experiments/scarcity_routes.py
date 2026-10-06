"""Finite-food fork arenas and physical route accounting. No policy hints."""
from collections import deque
from dataclasses import dataclass
from functools import cache
import random

from sim.world import DIRECTIONS, SIZE


@dataclass(frozen=True)
class Arena:
    origin: tuple
    forward: tuple
    detour: int

    def point(self, u, v):
        dx, dy = self.forward
        return self.origin[0] + u*dx - v*dy, self.origin[1] + u*dy + v*dx

    def local(self, point):
        x, y = point[0]-self.origin[0], point[1]-self.origin[1]
        dx, dy = self.forward
        return x*dx+y*dy, -x*dy+y*dx

    def room(self, point):
        u, v = self.local(point)
        if abs(v) <= 1:
            if -1 <= u <= 1:return 'home'
            if 3 <= u <= 5:return 'food'
        return None

    @property
    def hazard(self):return self.point(2, 0)

    @cache
    def cells(self):
        result = {self.point(u, v) for u in (-1, 0, 1, 3, 4, 5) for v in (-1, 0, 1)}
        result.add(self.hazard)
        direction = 1 if self.detour > 0 else -1
        for v in range(0, self.detour+direction, direction):
            result.update((self.point(1, v), self.point(3, v)))
        result.update(self.point(u, self.detour) for u in range(1, 4))
        return frozenset(result)


def configure(population, seed, stage='scarcity'):
    if stage not in ('feeding', 'routes', 'scarcity'):raise ValueError('Unknown scarcity stage')
    w = population.world
    for store in (w.resources, w.structures, w.caches, w.soils, w.visual_memories,
                  w.auditory_memories, w.attention_events, w.exploration):store.clear()
    w.sounds.clear();w.events.clear();w.terrain = [[4]*SIZE for _ in range(SIZE)]
    population.neighbors_scripted = False
    w.residents = {rid:w.residents[rid] for rid in population.brains}
    rng = random.Random(seed);arenas = {}
    for i, rid in enumerate(population.brains):
        arena = Arena((18+28*i+rng.randrange(-2, 3), 30+rng.randrange(-3, 4)),
                      rng.choice(tuple(DIRECTIONS.values())), rng.choice((-3, -2, 2, 3)))
        arenas[rid] = arena
        for x, y in arena.cells():w.terrain[y][x] = 0
        water = arena.point(-2, 0);w.terrain[water[1]][water[0]] = 1
        w.resources[w.key(*arena.hazard)] = {'kind':'thorns', 'amount':1}
        # Four units total, no carried food and no replenishment interventions.
        # Easier training stages move some of that same stock beside the home.
        nearby = 2 if stage == 'feeding' else 1 if stage == 'routes' else 0
        if nearby:w.resources[w.key(*arena.point(0, 1))] = {'kind':'berry', 'amount':nearby}
        w.resources[w.key(*arena.point(4, 0))] = {'kind':'berry', 'amount':4-nearby}
        a = w.residents[rid];a.x, a.y = arena.origin
        a.facing = next(d for d, delta in DIRECTIONS.items() if delta == arena.forward)
        a.food=8.;a.water=90.;a.energy=a.warmth=a.health=100.;a.pain=0.
        a.unconscious=False;a.starvation_ticks=a.hunger_faint_until=a.next_move_tick=0
        a.last_hurt=a.last_thorn_contact=a.last_tap_tick=a.last_tone_tick=-1000
        a.inventory = {item:0 for item in a.inventory}
        a.irrigation_water=0;a.carried_cache=None
    w.revision += 1
    return arenas


class Routes:
    """Count completed room-to-room journeys, including aborted attempts."""
    def __init__(self, arena):
        self.arena=arena;self.previous=arena.origin;self.last_room='home'
        self.active=False;self.hazardous=False;self.length=0
        self.completed=self.safe=self.unsafe=self.aborted=0
        self.foodward=self.safe_foodward=0;self.completed_steps=0

    def observe(self, point):
        point=tuple(point)
        distance=sum(abs(a-b) for a,b in zip(point,self.previous))
        if distance>1 or point not in self.arena.cells():raise ValueError('Nonphysical route transition')
        room=self.arena.room(point)
        if room is None:
            self.active=True
        if self.active:
            self.length+=distance;self.hazardous |= point==self.arena.hazard
            if room is not None:
                if room==self.last_room:self.aborted+=1
                else:
                    self.completed+=1;self.completed_steps+=self.length
                    self.safe+=not self.hazardous;self.unsafe+=self.hazardous
                    if room=='food':
                        self.foodward+=1;self.safe_foodward+=not self.hazardous
                self.active=False;self.hazardous=False;self.length=0;self.last_room=room
        self.previous=point

    def summary(self):
        return dict(crossings=self.completed, safe_crossings=self.safe, unsafe_crossings=self.unsafe,
                    aborted_crossings=self.aborted, foodward_crossings=self.foodward,
                    safe_foodward_crossings=self.safe_foodward, crossing_steps=self.completed_steps)


def food_stock(world, arena):
    """Unconsumed ordinary food, including planted yield after maturation."""
    cells=arena.cells();result=0
    for key, resource in world.resources.items():
        point=tuple(map(int,key.split(',')))
        if point in cells and resource['kind'] in ('berry','food'):result+=resource['amount']
    for key, cache in world.caches.items():
        if tuple(map(int,key.split(','))) in cells:result+=cache['food']
    for a in world.residents.values():
        if (a.x,a.y) in cells:
            result+=a.inventory['food']
            if a.carried_cache:result+=a.carried_cache['food']
    return result


def safe_oracle(world, rid):
    """Privileged feasibility reference. Never used as a teacher or learner input."""
    a=world.residents[rid]
    if a.unconscious:return {'verb':'rest'}
    if a.food<70 and a.inventory['food']:return {'verb':'eat'}
    if a.inventory['food']>=4:return {'verb':'rest'}
    nearby=next((world.resources.get(world.key(x,y)) for x,y in world.nearby(a)
                 if world.resources.get(world.key(x,y),{}).get('amount',0)>0),None)
    if nearby and nearby['kind'] in ('berry','food'):return {'verb':'gather'}
    goals={tuple(map(int,k.split(','))) for k,r in world.resources.items()
           if r['kind'] in ('berry','food') and r['amount']>0}
    start=(a.x,a.y);queue=deque([(start,None)]);visited={start}
    while queue:
        point, first = queue.popleft()
        if point in goals and first:return {'verb':'move','direction':first}
        for direction,(dx,dy) in DIRECTIONS.items():
            target=(point[0]+dx,point[1]+dy)
            if target in visited or not world.walkable(*target):continue
            if world.resources.get(world.key(*target),{}).get('kind')=='thorns':continue
            if any(b.id!=rid and (b.x,b.y)==target for b in world.residents.values()):continue
            visited.add(target);queue.append((target,first or direction))
    return {'verb':'rest'}
