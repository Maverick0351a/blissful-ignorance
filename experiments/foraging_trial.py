"""Disposable local foraging screen with persistent physiology and no live IO.

python experiments/foraging_trial.py --laya-adapter PATH/TO/laya_lite.py
Small learners use online one-step Q learning, NOT recurrent PPO.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim.world import World, SIZE, DIRECTIONS

SEEDS=(1701,2702,3703)
STEPS=384
NAMES=('north','east','south','west')
KINDS=('random','tabular','mlp','gru','fly','laya')


def arena(seed):
    w=World(seed); a=w.residents['player']; w.residents={'player':a}
    w.resources.clear(); w.structures.clear(); w.visual_memories.clear()
    w.terrain=[[4]*SIZE for _ in range(SIZE)]
    for y in range(27,38):
        for x in range(27,38):w.terrain[y][x]=0
    a.x=a.y=32; a.food=25; a.water=40
    rng=random.Random(seed)
    spots=[(x,y) for y in range(28,37) for x in range(28,37) if (x,y)!=(32,32)]
    rng.shuffle(spots)
    for kind,count in [('berry',8),('amber_bush',8),('thorns',10)]:
        for _ in range(count):
            x,y=spots.pop();w.resources[w.key(x,y)]={'kind':kind,'amount':5 if kind!='thorns' else 1}
    for _ in range(3):
        x,y=spots.pop();w.terrain[y][x]=1
    return w


def sense(w,seed):
    o=w.observe('player',include_memory=False)
    # These category names are arbitrary cues, not danger/safety labels.
    kinds=('berry','amber_bush') if seed%2 else ('amber_bush','berry')
    items=('food','amber_fruit') if seed%2 else ('amber_fruit','food')
    tiles={(t['dx'],t['dy']):t for t in o['tiles']}
    values=[v/100 for v in o['needs']]+[o['body']['health']/100,o['body']['pain']/100]
    values += [o['inventory'][k]/12 for k in items]
    values += [float(o['facing']==d) for d in NAMES]
    values += [float(t['blocked']) for t in o['touch']]
    for d in NAMES:
        t=tiles[DIRECTIONS[d]];k=t.get('resource',{}).get('kind')
        values += [float(k==kinds[0]),float(k==kinds[1]),float(k=='thorns'),float(t['terrain']==1)]
    for k in kinds:
        found=[t for t in o['tiles'] if t.get('resource',{}).get('kind')==k and t['resource']['amount']>0]
        t=min(found,key=lambda t:abs(t['dx'])+abs(t['dy'])) if found else None
        values += [float(t is not None),t['dx']/4 if t else 0,t['dy']/4 if t else 0]
    here=tiles[(0,0)].get('resource',{}).get('kind')
    values += [float(here==kinds[0]),float(here==kinds[1]),float(here=='thorns')]
    mask=[not t['blocked'] for t in o['touch']]+[True,True,
          bool(o['inventory'][items[0]]) and o['needs'][0]<=95,
          bool(o['inventory'][items[1]]) and o['needs'][0]<=95,
          any(t['terrain']==1 and abs(t['dx'])+abs(t['dy'])<=1 for t in o['tiles'])]
    if o['body']['unconscious']: mask=[False]*9;mask[4]=True
    commands=[{'verb':'move','direction':d} for d in NAMES]+[{'verb':'rest'},{'verb':'gather'},
              {'verb':'eat','item':items[0]},{'verb':'eat','item':items[1]},{'verb':'drink'}]
    brief={'fullness':round(o['needs'][0]),'water':round(o['needs'][1]),'health':round(o['body']['health']),
           'carry_A_B':[o['inventory'][k] for k in items],
           'adjacent_N_E_S_W':[['A' if t.get('resource',{}).get('kind')==kinds[0] else
                               'B' if t.get('resource',{}).get('kind')==kinds[1] else
                               'C' if t.get('resource',{}).get('kind')=='thorns' else '-',
                               'water' if t['terrain']==1 else 'ground'] for t in [tiles[DIRECTIONS[d]] for d in NAMES]],
           'nearest_A_B':values[32:38], 'on_A_B_C':values[-3:]}
    return values,mask,commands,brief


class Agent:
    def __init__(self,kind,seed,size,frozen=False,laya=None,adapter=None):
        self.kind=kind;self.frozen=frozen;self.rng=random.Random(seed);self.table={}
        self.laya=laya;self.adapter=adapter;self.size=size;self.parameters=0
        if kind in ('mlp','gru','fly'):
            import torch
            self.t=torch;torch.set_num_threads(1);torch.manual_seed(seed)
            if kind=='fly':
                self.projection=torch.randn(size,256)
                self.weights=torch.zeros(256,9);self.parameters=256*9
            elif kind=='mlp':
                self.model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(size*4,64),torch.nn.Tanh(),torch.nn.Linear(64,9))
            else:
                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__();self.rnn=torch.nn.GRU(size,32,batch_first=True);self.head=torch.nn.Linear(32,9)
                    def forward(self,x):
                        _,h=self.rnn(x);return self.head(h[-1])
                self.model=Model()
            if kind!='fly':
                self.opt=torch.optim.Adam(self.model.parameters(),lr=.001)
                self.parameters=sum(p.numel() for p in self.model.parameters())

    def key(self,history):return tuple(round(x*4) for x in history[-1])
    def code(self,history):
        code=self.t.zeros(256);code[(self.t.tensor(history[-1])@self.projection).topk(16).indices]=1
        return code
    def tensor(self,h):return self.t.tensor([[[0.]*self.size]*(4-len(h[-4:]))+h[-4:]])
    def values(self,h):
        if self.kind=='tabular':return self.table.setdefault(self.key(h),[0.]*9)
        with self.t.no_grad():
            if self.kind=='fly':return (self.code(h)@self.weights/16).tolist()
            return self.model(self.tensor(h))[0].tolist()
    def choose(self,h,mask,brief,recent):
        legal=[i for i,v in enumerate(mask) if v]
        if self.kind=='random':return self.rng.choice(legal)
        if self.kind=='laya':
            names=['N','E','S','W','rest','gather','eatA','eatB','drink']
            state={'senses':brief,'recent':recent[-3:]}
            q={'action':{'type':'choice','instructions':'Maintain food, water and health. Explore to find resources. Learn from recent reward. A, B and C are unfamiliar plants.',
                         'criteria':{names[i]:None for i in legal}}}
            seq,_=self.adapter.build_sequence(self.laya.tok,state,self.adapter.to_internal(q['action']),4096,self.laya.cfg.get('head_max_len',192))
            if len(seq)>self.adapter.L:raise ValueError('Laya observation truncation')
            answer=self.laya.system_one(state,q)['answers']['action']
            values=[answer['probabilities'].get(name,-1e9) for name in names]
        else:values=self.values(h)
        if self.rng.random()<.15:return self.rng.choice(legal)
        maximum=max(values[i] for i in legal)
        return self.rng.choice([i for i in legal if abs(values[i]-maximum)<1e-7])
    def update(self,h,action,reward,next_h,next_mask):
        if self.frozen or self.kind in ('random','laya'):return
        target=reward+.95*max(v for v,legal in zip(self.values(next_h),next_mask) if legal)
        if not math.isfinite(target):raise ValueError('nonfinite target')
        target=max(-10,min(10,target))
        if self.kind=='tabular':
            q=self.values(h);q[action]+=.2*(target-q[action])
        elif self.kind=='fly':
            code=self.code(h);pred=float(code@self.weights[:,action]/16)
            self.weights[:,action]+=.1*(target-pred)*code
        else:
            prediction=self.model(self.tensor(h))[0,action]
            loss=self.t.nn.functional.smooth_l1_loss(prediction,self.t.tensor(target))
            self.opt.zero_grad();loss.backward();self.t.nn.utils.clip_grad_norm_(self.model.parameters(),1);self.opt.step()


def worker(args):
    laya=adapter=None
    if args.kind=='laya':
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        spec=importlib.util.spec_from_file_location('local_laya',args.laya_adapter)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);laya=adapter.LayaLite('NPU')
    rows=[];start=time.perf_counter()
    for seed in SEEDS:
        w=arena(seed);v,mask,commands,brief=sense(w,seed)
        agent=Agent(args.kind,seed,len(v),args.frozen,laya,adapter)
        h=[v];recent=[];a=w.residents['player'];positions={(a.x,a.y)}
        m={'seed':seed,'safe_meals':0,'harmful_meals':0,'useful_nutrition':0.,'water_gained':0.,'invalid':0,
           'unconscious_ticks':0,'below20_food_ticks':0,'gathered':0,'half':[]}
        decision_ms=[]
        for tick in range(STEPS):
            t=time.perf_counter();action=agent.choose(h,mask,brief,recent);decision_ms.append((time.perf_counter()-t)*1000)
            food,water,damage,gathered=a.food,a.water,a.damage_taken,a.gathered
            m['unconscious_ticks']+=int(a.unconscious)
            result=w.step({'player':commands[action]})['player']
            nutrition=max(0,a.food-food+.008);drink=max(0,a.water-water+.012)
            reward=nutrition/25+drink/35-(a.damage_taken-damage)/10+.05*(a.gathered-gathered)-.002
            reward-=.01*int(not result[0])+.01*int(a.food<20)+.05*int(a.unconscious)
            next_v,next_mask,next_commands,next_brief=sense(w,seed);next_h=(h+[next_v])[-4:]
            agent.update(h,action,reward,next_h,next_mask)
            m['invalid']+=int(not result[0]);m['below20_food_ticks']+=int(a.food<20)
            m['useful_nutrition']+=nutrition;m['water_gained']+=drink
            if result[0] and commands[action]['verb']=='eat':
                m['harmful_meals' if commands[action]['item']=='amber_fruit' else 'safe_meals']+=1
            recent.append({'action':action,'reward':round(reward,2)})
            positions.add((a.x,a.y));h=next_h;v,mask,commands,brief=next_v,next_mask,next_commands,next_brief
            if tick+1 in (STEPS//2,STEPS):
                m['half'].append({'tick':tick+1,'nutrition':round(m['useful_nutrition'],3),'damage':a.damage_taken,'health':a.health})
        m.update(gathered=a.gathered,damage=round(a.damage_taken,3),health=round(a.health,3),fullness=round(a.food,3),
                 hydration=round(a.water,3),thorns=a.thorn_contacts,faints=a.fainted,unique_tiles=len(positions),
                 median_decision_ms=statistics.median(decision_ms),parameters=agent.parameters)
        rows.append(m);print(json.dumps(m),flush=True)
    result={'kind':args.kind,'frozen':args.frozen,'steps':STEPS,'rows':rows,'seconds':time.perf_counter()-start}
    name=args.kind+('-frozen' if args.frozen else '')
    (Path(args.output)/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=KINDS);p.add_argument('--frozen',action='store_true')
    p.add_argument('--laya-adapter');p.add_argument('--output',default=str(ROOT/'runs'/'foraging-trial'))
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.kind:worker(args);return
    if (out/'preregistration.json').exists():raise SystemExit('Choose a fresh output directory')
    plan={'seeds':SEEDS,'steps':STEPS,'hypotheses':['Online learners obtain more useful nutrition than paired frozen controls.',
          'Food-only ranking may not transfer to navigation.','Low damage without food acquisition is not success.'],
          'limits':'Exploratory 384-tick lives; no physiological resets, learning or inference state shared between residents. No claim of PPO or sustained survival.',
          'hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['sim/world.py','experiments/foraging_trial.py']}}
    (out/'preregistration.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    status={}
    for kind in KINDS:
        for frozen in ((False,True) if kind in ('tabular','mlp','gru','fly') else (False,)):
            name=kind+('-frozen' if frozen else '')
            if kind=='laya' and not args.laya_adapter:status[name]='blocked: local adapter required';continue
            cmd=[sys.executable,__file__,'--kind',kind,'--output',str(out)]
            if frozen:cmd+=['--frozen']
            if args.laya_adapter:cmd+=['--laya-adapter',args.laya_adapter]
            print('Starting '+name,flush=True)
            try:
                r=subprocess.run(cmd,text=True,capture_output=True,timeout=600)
                (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
                status[name]='complete' if r.returncode==0 else 'failed; see log'
            except subprocess.TimeoutExpired:status[name]='timed out after 600 seconds'
            print(name+': '+status[name],flush=True)
    (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


if __name__=='__main__':main()
