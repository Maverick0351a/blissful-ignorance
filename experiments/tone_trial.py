"""Sealed neural signaling diagnostic; route execution is explicitly scaffolded."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from sim.world import World
TONES = ("0", "1", "2")  # Fixed three-symbol protocol; current game has ten.

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (24051, 24062, 24073)
TRAIN_LAYOUTS = ((2,2), (2,3), (3,2), (3,3))
EVAL_LAYOUTS = ((4,2), (4,3), (2,4), (3,4))
torch.set_num_threads(1)


class Policy:
    def __init__(self, inputs, seed):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = nn.Sequential(nn.Linear(inputs,16), nn.Tanh(), nn.Linear(16,3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.rng = random.Random(seed)

    def choose(self, features, training=False):
        if training and self.rng.random() < .2:
            return self.rng.randrange(3)
        with torch.no_grad():
            return int(self.model(torch.tensor(features)).argmax())

    def learn(self, features, action, reward):
        prediction = self.model(torch.tensor(features))[action]
        loss = (prediction - reward)**2
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

    def digest(self):
        return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in self.model.state_dict().values())).hexdigest()


def world_for(seed, layout, lane):
    w = World(seed)
    w.resources.clear(); w.structures.clear(); w.sounds.clear()
    w.residents = {rid:w.residents[rid] for rid in ('player','r0')}
    w.terrain = [[0]*64 for _ in range(64)]
    rng = random.Random(seed)
    x,y = rng.randint(15,45),rng.randint(15,45)
    sender, receiver = w.residents.values()
    sender.x,sender.y=x,y
    receiver.x,receiver.y=x,y+3
    receiver.food=40
    dx,dy=layout
    routes=[(x-dx,y-dy),(x,y-dy),(x+dx,y-dy)]
    w.resources[w.key(*routes[lane])]={'kind':'berry','amount':1}
    return w,routes


def sender_input(observation):
    features=[0.,0.,0.]
    for tile in observation['tiles']:
        if tile['terrain'] >= 0 and tile.get('resource',{}).get('kind')=='berry':
            features[0 if tile['dx']<0 else 2 if tile['dx']>0 else 1]=1.
    return features


def receiver_input(observation):
    tones=[s['kind'] for s in observation['hearing'] if s['kind'].startswith('tone_')]
    heard=tones[-1] if tones else 'silence'
    return [float(heard=='tone_'+p) for p in TONES]+[float(heard=='silence')]


def travel_and_eat(w, destination):
    """Fixed route skill knows candidate locations, never which contains food."""
    a=w.residents['r0']; sender=w.residents['player']
    x,y=destination
    waypoints=[(x if x!=sender.x else x+1,a.y),(x if x!=sender.x else x+1,y),(x,y)]
    for tx,ty in waypoints:
        while (a.x,a.y)!=(tx,ty):
            direction=('east' if a.x<tx else 'west') if a.x!=tx else ('south' if a.y<ty else 'north')
            result=w.step({'player':{'verb':'wait'},'r0':{'verb':'move','direction':direction}})
            if not result['r0'][0]:raise AssertionError('Route scaffold blocked')
    gathered=w.step({'player':{'verb':'wait'},'r0':{'verb':'gather'}})['r0'][0]
    before=a.food
    w.step({'player':{'verb':'wait'},'r0':{'verb':'eat','item':'food'}})
    nutrition=max(0.,a.food-before+.008)
    assert abs(nutrition-(25 if gathered else 0))<1e-8
    return nutrition


def episode(sender,receiver,seed,layout,lane,condition='intact',training=False,replacement=None):
    w,routes=world_for(seed,layout,lane)
    assert not any(t.get('resource',{}).get('kind')=='berry' for t in w.observe('r0')['tiles'])
    s=sender_input(w.observe('player'))
    assert sum(s)==1
    sent=sender.choose(s,training)
    if condition!='muted':
        transmitted=sent if replacement is None else replacement
        assert w.apply_action('player',{'verb':'tone','tone':TONES[transmitted]})[0]
    r=receiver_input(w.observe('r0'))
    selected=receiver.choose(r,training)
    nutrition=travel_and_eat(w,routes[selected])
    if training:
        # Shared cooperative outcome; no direct supervision of pitch or route.
        sender.learn(s,sent,nutrition/25)
        receiver.learn(r,selected,nutrition/25)
    return dict(seed=seed,layout=layout,lane=lane,condition=condition,sent=sent,
                heard=r,route=selected,nutrition=nutrition,success=nutrition>24, ticks=w.tick)


def write(path,data):path.write_text(json.dumps(data,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    prereg=dict(seeds=SEEDS,training_episodes=1500,evaluation_episodes=240,
        train_layouts=TRAIN_LAYOUTS,eval_layouts=EVAL_LAYOUTS,
        gate='Each seed intact success >=80% and >=20 percentage points above both muted and shuffled; no live deployment',
        controls='Identical frozen policies and worlds; shuffled permutes the balanced intact message list, preserving message frequency; muted supplies silence',
        limits='Scaffolded routes, shared team reward, one-shot discrete signaling, three training seeds; no learned navigation or general language',
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('experiments/tone_trial.py','sim/world.py','sim/social.py')})
    write(out/'preregistration.json',prereg)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest())
    start=time.perf_counter();results=[];summaries=[]
    for seed in SEEDS:
        sender,receiver=Policy(3,seed),Policy(4,seed+1)
        rng=random.Random(seed)
        train=[]
        for i in range(1500):
            train.append(episode(sender,receiver,seed*10000+i,rng.choice(TRAIN_LAYOUTS),rng.randrange(3),training=True))
        write(out/f'training-{seed}.json',train)
        frozen=(sender.digest(),receiver.digest())
        schedule=[(seed*10000+5000+i,EVAL_LAYOUTS[(i//3)%4],i%3) for i in range(240)]
        messages=[sender.choose([float(j==lane) for j in range(3)]) for _,_,lane in schedule]
        shuffled=list(messages);random.Random(seed+777).shuffle(shuffled)
        scores={}
        for condition in ('intact','muted','shuffled'):
            rows=[episode(sender,receiver,k,layout,lane,condition,replacement=shuffled[i] if condition=='shuffled' else None)
                  for i,(k,layout,lane) in enumerate(schedule)]
            for row in rows:row['training_seed']=seed
            results.extend(rows);scores[condition]=sum(r['success'] for r in rows)/len(rows)
        assert frozen==(sender.digest(),receiver.digest()),'Evaluation changed weights'
        torch.save({'sender':sender.model.state_dict(),'receiver':receiver.model.state_dict()},out/f'policies-{seed}.pt')
        passed=scores['intact']>=.8 and all(scores['intact']-scores[c]>=.2 for c in ('muted','shuffled'))
        summaries.append(dict(seed=seed,success=scores,passed=passed,sender_code=messages[:3],frozen_hashes=frozen))
        write(out/'summary.partial.json',summaries)
        print(json.dumps(summaries[-1]),flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in prereg['hashes'].items())
    write(out/'results.json',results)
    write(out/'complete.json',dict(seconds=time.perf_counter()-start,summaries=summaries,passed=all(s['passed'] for s in summaries),source_hashes_verified=True))


if __name__=='__main__':main()
