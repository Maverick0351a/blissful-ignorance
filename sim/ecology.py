"""Optional resource tradeoffs. All changes use simulation time and conserved stores."""
VERBS = ('fill_water','water_crop','make_cache','stash','take_food','move_cache')


def soil(world,x,y):
    stored=world.soils.get(world.key(x,y))
    if stored is not None:return dict(stored)
    return {'fertility':.5+((x*17+y*31+world.seed)%51)/100, 'moisture':.7}


def action(world,rid,command):
    a=world.residents[rid];verb=command['verb']
    if not world.ecology_enabled:return False,'Resource ecology is off in this world.'
    if verb=='fill_water':
        if not any(world.terrain[y][x]==1 for x,y in world.nearby(a)):
            return False,'Stand beside water to fill your irrigation flask.'
        if a.irrigation_water>=3 or world.irrigation_supply<1:
            return False,'Flask full, or the irrigation supply is depleted.'
        a.irrigation_water+=1;world.irrigation_supply-=1
        return True,'Collected one irrigation charge.'
    if verb=='water_crop':
        tile=world.action_tile(a,command)
        if not any(k in command for k in ('x','y','direction')):
            tile=next(((x,y) for x,y in world.nearby(a) if world.resources.get(world.key(x,y),{}).get('ecological')),tile)
        if tile is None or world.terrain[tile[1]][tile[0]]!=0:return False,'Water reachable grass or a crop.'
        if not a.irrigation_water:return False,'Fill your irrigation flask first.'
        key=world.key(*tile);s=soil(world,*tile)
        if s['moisture']>=.99:return False,'This soil is already wet.'
        s['moisture']=min(1.,s['moisture']+.6);world.soils[key]=s;a.irrigation_water-=1
        return True,'Watered the soil.'
    if verb=='make_cache':
        key=world.key(a.x,a.y)
        if a.inventory['wood']<3:return False,'Need 3 wood to make a food basket.'
        if world.terrain[a.y][a.x]!=0 or key in world.resources or key in world.structures or key in world.caches:
            return False,'Make a basket on clear grass at your feet.'
        a.inventory['wood']-=3;world.caches[key]={'food':0}
        return True,'Made a shared food basket (capacity 12).'
    if verb=='move_cache' and a.carried_cache is not None:
        key=world.key(a.x,a.y)
        if world.terrain[a.y][a.x]!=0 or key in world.resources or key in world.structures or key in world.caches:
            return False,'Place the basket on clear grass at your feet.'
        world.caches[key]=a.carried_cache;a.carried_cache=None
        return True,'Placed the food basket.'
    key=next((world.key(x,y) for x,y in world.nearby(a)
              if world.key(x,y) in world.caches and world.visible(a,x,y)),None)
    if key is None:return False,'No food basket within reach.'
    cache=world.caches[key]
    if verb=='stash':
        if not a.inventory['food'] or cache['food']>=12:return False,'Need food and room in the basket.'
        a.inventory['food']-=1;cache['food']+=1
        return True,'Stored one food. Anyone can retrieve it.'
    if verb=='take_food':
        if not cache['food'] or world.capacity(a)<1:return False,'Basket empty, or pack full.'
        cache['food']-=1;a.inventory['food']+=1
        return True,'Retrieved one food.'
    if verb=='move_cache':
        if world.capacity(a)<4+cache['food']:return False,'Need room for basket (4 spaces) and its food.'
        a.carried_cache=world.caches.pop(key)
        return True,'Carrying the basket. Use carry/place again to set it down.'
    return False,'Unknown ecology action.'


def tick(world):
    if not world.ecology_enabled:return
    rain=world.weather=='Rain'
    if rain:world.irrigation_supply=min(500.,world.irrigation_supply+.1)
    for key,s in world.soils.items():
        crop=world.resources.get(key,{})
        growing=crop.get('ecological',False)
        s['moisture']=max(0.,min(1.,s['moisture']+(.002 if rain else -.001 if growing else -.0001)))
        if not growing:s['fertility']=min(1.,s['fertility']+.00002)
    for key,crop in list(world.resources.items()):
        if not crop.get('ecological'):continue
        s=world.soils[key]
        crop['growth']+=s['fertility'] if s['moisture']>.1 else 0
        crop['stage']=min(2,int(crop['growth']*3/480))
        if crop['growth']>=480:
            world.resources[key]={'kind':'berry','amount':2+int(s['fertility']>=.7)}
            s['fertility']=max(.2,s['fertility']-.15)
            world.revision+=1
