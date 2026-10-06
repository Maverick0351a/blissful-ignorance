"""Local two-learner navigation and delayed farming trial; no external I/O."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.world import World, SIZE, DIRECTIONS
from agents.rewards import Curiosity, hunger_reward
from experiments.farming_trial import Learner as BaseLearner
SEEDS = (7413, 8524, 9635)
DECISIONS, STRIDE, TRAIN, EVAL = 160, 120, 8, 2
ACTIONS = tuple(DIRECTIONS) + ('rest', 'eat', 'make_seeds', 'plant', 'gather')

class Learner(BaseLearner):
    def __init__(self, seed):
        super().__init__(seed)
        self.q = defaultdict(lambda: [0.] * len(ACTIONS))
        self.curiosity = Curiosity()

def arena(seed):
    w = World(seed, memory_capacity=1)
    w.residents = {k: w.residents[k] for k in ('player', 'r0')}
    w.resources.clear(); w.structures.clear()
    w.terrain = [[4] * SIZE for _ in range(SIZE)]
    for y in range(28, 37):
        for x in range(28, 37): w.terrain[y][x] = 2
    rng = random.Random(seed)
    sites = [(x,y) for y in (29,32,35) for x in (29,32,35)]
    for x,y in rng.sample(sites, 6): w.terrain[y][x] = 0
    for rid, pos in zip(w.residents, ((30,30),(34,34))):
        a = w.residents[rid]; a.x,a.y = pos; a.food = 40
        a.inventory = {k: 0 for k in a.inventory}; a.inventory['food'] = 2
    return w

def nearest(tiles):
    return min(tiles, key=lambda t:(abs(t['dx'])+abs(t['dy']),t['dy'],t['dx'])) if tiles else None

def goalcode(tile):
    if tile is None: return (-1,0)
    dx,dy = tile['dx'],tile['dy']; dist=abs(dx)+abs(dy)
    direction = 4 if dist == 0 else (1 if dx>0 else 3) if abs(dx)>abs(dy) else (2 if dy>0 else 0)
    return direction, min(3,dist)

def sense(w,rid):
    o=w.observe(rid,include_memory=False); inv=o['inventory']; capacity=12-sum(inv.values())
    ripe=[t for t in o['tiles'] if t.get('resource',{}).get('kind')=='berry' and t['resource']['amount']>0]
    plots=[t for t in o['tiles'] if t['terrain']==0 and not t.get('resource') and not t.get('structure')]
    reach=lambda ts:[t for t in ts if abs(t['dx'])+abs(t['dy'])<=1]
    planting=nearest(reach(plots)); harvest=nearest(reach(ripe))
    crops=reach([t for t in o['tiles'] if t.get('resource',{}).get('kind')=='crop'])
    mask=[not t['blocked'] for t in o['touch']]+[True, inv['food']>0 and o['needs'][0]<=95,
          inv['food']>0 and capacity>=1, inv['seed']>0 and planting is not None, capacity>0 and harvest is not None]
    if o['body']['unconscious']: mask=[False]*9; mask[4]=True
    state=(int(o['needs'][0]//20),min(3,inv['food']),min(3,inv['seed']),*goalcode(nearest(ripe)),*goalcode(nearest(plots)),
           1+max((t['resource']['stage'] for t in crops),default=-1),bool(o['others']), int(o['needs'][2]//20),
           int(o['body'].get('hunger_discomfort',0)//20), int(o['body'].get('starvation_strain',0)//20), o['body']['unconscious'])
    brief={'state':state,'fullness':o['needs'][0],'inventory':inv,'ripe_goal':goalcode(nearest(ripe)), 'plot_goal':goalcode(nearest(plots))}
    return state,mask,brief,planting,harvest

def command(action,plot):
    verb=ACTIONS[action]
    if action<4: return {'verb':'move','direction':verb}
    if verb=='plant' and plot:
        delta=(plot['dx'],plot['dy'])
        d=next((d for d,v in DIRECTIONS.items() if v==delta),None)
        return {'verb':'plant',**({'direction':d} if d else {})}
    return {'verb':verb}

def scripted(mask,brief):
    food=brief['inventory']['food']; seeds=brief['inventory']['seed']
    if mask[5] and brief['fullness']<30: return 5
    if mask[8]: return 8
    if mask[7]: return 7
    if mask[6] and seeds==0 and (food>=2 or brief['fullness']>=30): return 6
    goal=brief['ripe_goal'] if brief['ripe_goal'][0]>=0 else brief['plot_goal'] if seeds else (-1,0)
    if goal[0] in range(4) and mask[goal[0]]: return goal[0]
    if mask[5] and brief['fullness']<70: return 5
    return 4

def episode(kind,seed,learners,training=False,decisions=DECISIONS):
    w=arena(seed); owners={}; history=[]
    stats={rid:dict(hunger_cost=0.,curiosity_reward=0.,total_reward=0.,unconscious_ticks=0,nutrition=0.,zero_food_ticks=0,fullness_sum=0.,planted=0,harvested=0,competitor_harvest=0,invalid=0) for rid in w.residents}
    positions={rid:{(a.x,a.y)} for rid,a in w.residents.items()}
    for i,(rid,l) in enumerate(learners.items()): l.trace.clear(); l.rng=random.Random(seed ^ (i+1)*0xACE)
    for rid,l in learners.items():
        l.curiosity.last=None
        l.curiosity.reward(w.observe(rid,include_memory=False))  # Initial scene is familiar, not a rewarded action.
    for decision in range(decisions):
        senses={rid:sense(w,rid) for rid in w.residents}; actions={}
        for rid,(state,mask,brief,plot,_) in senses.items(): actions[rid]=scripted(mask,brief) if kind=='scripted' else learners[rid].choose(state,mask,training)
        commands={rid:command(actions[rid],senses[rid][3]) for rid in w.residents}
        before={rid:a.food for rid,a in w.residents.items()}
        gathering={rid:next((w.key(x,y) for x,y in w.nearby(a) if w.resources.get(w.key(x,y),{}).get('amount',0)>0),None) for rid,a in w.residents.items()}
        awake_before={rid:not a.unconscious for rid,a in w.residents.items()}
        result=w.step(commands); rewards={}
        hunger_integral={rid:0. for rid in w.residents}
        trace={'decision':decision,'tick':w.tick,'agents':{}}
        for rid,a in w.residents.items():
            action=actions[rid]; s=stats[rid]; success=result[rid][0]
            nutrition=max(0,a.food-before[rid]+.008) if action==5 and success else 0
            rewards[rid]=nutrition/25; s['nutrition']+=nutrition; s['invalid']+=not success
            if success and action==7:
                s['planted']+=1; p=senses[rid][3]; owners[w.key(a.x+p['dx'],a.y+p['dy'])]=rid
            if success and action==8:
                s['harvested']+=1; s['competitor_harvest']+=owners.get(gathering[rid]) not in (None,rid)
            positions[rid].add((a.x,a.y))
            trace['agents'][rid]={'observation':senses[rid][2],'action':ACTIONS[action],'success':success,'nutrition':nutrition,'position':[a.x,a.y]}
        for tick in range(STRIDE):
            if tick: w.step({rid:{'verb':'wait'} for rid in w.residents})
            for rid,a in w.residents.items():
                stats[rid]['zero_food_ticks']+=a.food==0
                stats[rid]['fullness_sum']+=a.food
                stats[rid]['unconscious_ticks']+=a.unconscious
                hunger_integral[rid]+=w.hunger_state(a)['hunger_discomfort']
        for rid in w.residents:
            ns,nm,*_=sense(w,rid)
            novelty=learners[rid].curiosity.reward(w.observe(rid,include_memory=False),
                eligible=bool(result[rid][0] and awake_before[rid] and not w.residents[rid].unconscious))
            nutrition=trace['agents'][rid]['nutrition']
            rewards[rid]=hunger_reward(nutrition,hunger_integral[rid])+novelty
            stats[rid]['hunger_cost']+=hunger_integral[rid]/48000
            stats[rid]['curiosity_reward']+=novelty
            stats[rid]['total_reward']+=rewards[rid]
            trace['agents'][rid].update(hunger_cost=hunger_integral[rid]/48000,curiosity_reward=novelty,reward=rewards[rid])
            if training: learners[rid].update(senses[rid][0],actions[rid],rewards[rid],ns,nm,decision==decisions-1)
        history.append(trace)
    for rid,a in w.residents.items():
        s=stats[rid]; s['mean_fullness']=s.pop('fullness_sum')/w.tick; s['final_fullness']=a.food
        s['unique_positions']=len(positions[rid]); s['visited_states']=len(learners[rid].q); s['inventory']=dict(a.inventory)
    return {'kind':kind,'seed':seed,'training':training,'ticks':w.tick,'agents':stats},history

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--pilot',action='store_true'); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    prereg={'seeds':SEEDS,'train':TRAIN,'evaluation':EVAL,'decisions':DECISIONS,'stride':STRIDE,'initial_fullness':40,'initial_food':2,
      'hypothesis':'Trained pair has less mean zero-food ticks and greater mean useful nutrition than untrained pair in each of three seed groups.',
      'protocol':2,'reward':'Useful eating gain /25 minus integrated own hunger discomfort /48000 plus private fading scene novelty up to0.02 per decision','learning':{'alpha':.15,'gamma':.995,'lambda':.9,'train_epsilon':.2,'evaluation_epsilon':0,'initial_q':0,'independent_tables':True},
      'limits':'Nine-action interface and physical masks supplied. Fertile-site target chosen by nearest locally visible empty reachable grass. No mortality, no drink action, no live world integration; limited eight-episode training.',
      'hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('experiments/mobile_farming_trial.py','experiments/farming_trial.py','sim/world.py','agents/rewards.py')}}
    (out/'preregistration.json').write_text(json.dumps(prereg,indent=2))
    start=time.perf_counter(); rows=[]
    if args.pilot:
        row,history=episode('scripted',123456,{rid:Learner(i) for i,rid in enumerate(('player','r0'))})
        rows.append(row); (out/'pilot-trace.json').write_text(json.dumps(history))
    else:
        for seed in SEEDS:
            pair={rid:Learner(seed+i) for i,rid in enumerate(('player','r0'))}
            for ep in range(TRAIN):
                row,_=episode('trained',seed+ep*10000,pair,True); row.update(base_seed=seed,episode=ep); rows.append(row)
            for rid,l in pair.items(): (out/f'q-{seed}-{rid}.json').write_text(json.dumps([{'state':k,'q':v} for k,v in l.q.items()]))
            for rid,l in pair.items(): (out/f'curiosity-{seed}-{rid}.json').write_text(json.dumps(l.curiosity.snapshot()))
            for kind in ('trained','untrained','scripted'):
                policies=pair if kind=='trained' else {rid:Learner(seed+i) for i,rid in enumerate(('player','r0'))}
                for ep in range(EVAL):
                    row,history=episode(kind,seed+1000000+ep*10000,policies); row.update(base_seed=seed,episode=ep); rows.append(row)
                    (out/f'trace-{seed}-{kind}-{ep}.json').write_text(json.dumps(history))
            print(f'Completed seed {seed}; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2)); (out/'complete.json').write_text(json.dumps({'complete':True,'seconds':time.perf_counter()-start,'rows':len(rows)}))
if __name__=='__main__': main()
