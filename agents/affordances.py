"""Physical feasibility from private observations, not goals or privileged state."""
from sim.world import DIRECTIONS, RECIPES
from sim.ecology import VERBS as ECOLOGY_VERBS


def action_mask(observation, actions):
    tiles={(t['dx'],t['dy']):t for t in observation['tiles'] if t['terrain']>=0}
    here=tiles.get((0,0),{})
    facing=DIRECTIONS.get(observation.get('facing'),(0,1))
    ordered=[(0,0),facing]+[v for v in DIRECTIONS.values() if v!=facing]
    nearby=[tiles[p] for p in ordered if p in tiles]
    neighbors=[o for o in observation['others'] if abs(o['dx'])+abs(o['dy'])<=1]
    occupied={(o['dx'],o['dy']) for o in observation['others']}|{(0,0)}
    inventory=observation['inventory'];eco=observation.get('ecology',{})
    carried=eco.get('carried_cache')
    capacity=12-sum(inventory.values())-(4+carried['food'] if carried is not None else 0)
    blocked={t['direction']:t['blocked'] for t in observation['touch']}
    cache=next((t['cache'] for t in nearby if t.get('cache')),None)
    def clear(tile):
        return tile.get('terrain')==0 and not any(tile.get(k) for k in ('resource','structure','cache'))
    def possible(command):
        verb=command['verb'];item=command.get('item','food')
        pos=DIRECTIONS.get(command.get('direction'),(0,0));tile=tiles.get(pos,{})
        if observation['body'].get('unconscious'):return verb=='rest'
        if verb=='move':return not blocked.get(command['direction'],True)
        if verb=='rest':return True
        if verb=='tone':return True  # Own cooldown fits the four-tick decision interval.
        if verb=='eat':return inventory.get(item,0)>0 and observation['needs'][0]<=95
        if verb=='drink':return any(t['terrain']==1 for t in nearby)
        if verb=='gather':return capacity>0 and any(t.get('resource',{}).get('amount',0)>0 for t in nearby)
        if verb=='give':return inventory.get(item,0)>0 and bool(neighbors) # Other packs are private.
        if verb=='drop':return inventory.get(item,0)>0 and (not here.get('resource') or here['resource']['kind']==item)
        if verb=='make_seeds':return inventory.get('food',0)>0 and capacity>=1
        if verb=='plant':return inventory.get('seed',0)>0 and clear(tile)
        if verb=='tap':return any(not o['unconscious'] for o in neighbors)
        if verb=='revive':return observation['needs'][2]>=10 and any(o['unconscious'] for o in neighbors)
        if verb=='build':
            kind=command.get('kind','shelter');structure=tile.get('structure',{})
            return (tile.get('terrain')==0 and not tile.get('resource') and not tile.get('cache')
                    and (not structure or structure.get('kind')=='floor' and kind!='floor')
                    and (kind not in ('wall','stone_wall','door') or pos not in occupied)
                    and all(inventory.get(k,0)>=v for k,v in RECIPES[kind]['cost'].items()))
        if verb=='toggle_door':return tile.get('structure',{}).get('kind')=='door' and not (tile['structure'].get('open') and pos in occupied)
        if verb=='dismantle':
            structure=tile.get('structure',{})
            if structure:return capacity>=sum(RECIPES[structure['kind']]['cost'].values())
            resource=tile.get('resource',{})
            return bool(resource) and resource.get('amount')==0 and resource['kind']!='crop'
        if verb in ECOLOGY_VERBS:
            if not eco.get('enabled'):return False
            if verb=='fill_water':return eco.get('water_carried',0)<3 and any(t['terrain']==1 for t in nearby) # Reserve is not visible.
            if verb=='water_crop':
                target=next((t for t in nearby if t.get('resource',{}).get('kind')=='crop'),here)
                return eco.get('water_carried',0)>0 and target.get('terrain')==0 and target.get('soil',{}).get('moisture',1)<.99
            if verb=='make_cache':return inventory.get('wood',0)>=3 and clear(here)
            if verb=='move_cache' and carried is not None:return clear(here)
            if cache is None:return False
            if verb=='stash':return inventory.get('food',0)>0 and cache['food']<12
            if verb=='take_food':return cache['food']>0 and capacity>=1
            if verb=='move_cache':return capacity>=4+cache['food']
        return True
    return [bool(possible(a)) for a in actions]
