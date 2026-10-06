"""Experimental progression; ordinary actions and own senses remain unchanged."""
import random
from sim.world import SIZE

STAGES = ('near_food', 'depletion', 'hazards', 'farming', 'cooperation')


def configure(population, seed, stage='near_food', moved=False):
    if stage not in STAGES:raise ValueError('Unknown development stage')
    w = population.world; w.resources.clear(); w.structures.clear(); w.caches.clear(); w.soils.clear()
    w.visual_memories.clear(); w.sounds.clear(); w.auditory_memories.clear(); w.attention_events.clear()
    w.terrain = [[4]*SIZE for _ in range(SIZE)]
    population.neighbors_scripted = False
    rng = random.Random(seed)
    # Separate starting rooms expose each resident to its own actions. The final
    # stage joins them so food, baskets and local communication can matter.
    centers = (22,36)
    for center in centers:
        for y in range(27,34):
            for x in range(center-3,center+4):w.terrain[y][x] = 0
        for y in range(27,34):w.terrain[y][center+4] = 1
    if stage == 'cooperation':
        for y in range(29,32):
            for x in range(25,34):w.terrain[y][x] = 0
    for i,rid in enumerate(population.brains):
        a = w.residents[rid]; center = centers[i]
        a.x=center; a.y=30; a.facing='south'; a.food=20; a.water=70; a.health=100; a.pain=0
        a.energy=100; a.warmth=100; a.unconscious=False; a.starvation_ticks=0
        a.hunger_faint_until=0; a.next_move_tick=0; a.irrigation_water=0; a.carried_cache=None
        a.last_hurt=a.last_thorn_contact=a.last_tap_tick=a.last_tone_tick=-1000
        for item in a.inventory:a.inventory[item]=0
        a.inventory.update(food=1,wood=0 if stage=='near_food' else 3)
        if stage=='near_food':
            spots=[(center,31),(center+1,30),(center-1,30),(center,29)]
            x,y=rng.choice(spots);w.resources[w.key(x,y)]={'kind':'berry','amount':16}
        elif stage=='farming':
            a.inventory.update(food=2,seed=2)
            for x,y in ((center-1,29),(center+1,31)):
                w.resources[w.key(x,y)]={'kind':'berry','amount':2}
            # Moisture and fertility are visible; rain is unchanged. Crops retain
            # the real game's growth and watering requirements.
        else:
            side = 1 if moved else -1
            spots=[(center+side*2,y) for y in range(28,33)]
            rng.shuffle(spots)
            for x,y in spots[:3]:w.resources[w.key(x,y)]={'kind':'berry','amount':8}
            if stage in ('hazards','cooperation'):
                for x,y in ((center,29),(center-side,31)):
                    w.resources[w.key(x,y)]={'kind':'thorns','amount':1}
                w.resources[w.key(center-side*2,30)]={'kind':'amber_bush','amount':8}
                a.inventory['amber_fruit']=1
    # Six neighbors and the traveler rest out of the learners' view. No scripted
    # food seeking, gifts or rescues can carry an individual test.
    for i,(rid,a) in enumerate(w.residents.items()):
        if rid not in population.brains:a.x=4+i;a.y=4
    if stage=='cooperation':
        # Asymmetry and shared storage create opportunities without assigning
        # meanings to tones, jobs, ownership or a reward for talking.
        w.caches[w.key(29,30)]={'food':0}
        w.residents['r0'].inventory['seed']=3
        w.residents['r1'].inventory['wood']=6
        for y in range(27,34):w.terrain[y][26]=0
        for key,r in list(w.resources.items()):
            x=int(key.split(',')[0])
            if x>29 and r['kind']=='berry':del w.resources[key]
    w.revision += 1
    return population
