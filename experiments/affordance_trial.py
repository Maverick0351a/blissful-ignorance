"""Preregistered local-only feasibility and long-life feeding comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.lifelong import Population

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(31041,31052,31063)


def write(path,value):path.write_text(json.dumps(value,indent=2))


def digest(brain):
    return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in brain.model.state_dict().values())).hexdigest()


def configure(population):
    # Fixed low reserves make evaluation exceed initial food endurance.
    for rid in population.brains:
        a=population.world.residents[rid];a.food=20
        a.inventory.update(food=1,wood=3,stone=0,seed=0,amber_fruit=0)


def run(population,ticks):
    trace=[]
    for i in range(ticks):
        population.step()
        if (i+1)%400==0:
            trace.append({'tick':population.world.tick,'agents':{rid:{'food':population.world.residents[rid].food,**dict(population.metrics[rid])} for rid in population.brains}})
    return trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=('experiments/affordance_trial.py','agents/lifelong.py','agents/preservation.py','agents/affordances.py','agents/network.py','sim/world.py','sim/ecology.py','sim/social.py')
    prereg={'seeds':SEEDS,'training_ticks':1200,'evaluation_ticks':8000,'conditions':['trained_masked','trained_unmasked','initial_masked'],
        'evaluation_world_seeds':[s+100000 for s in SEEDS],'starting_food':20,'starting_inventory_food':1,
        'gate':'In all three seeds, trained masked must increase nutrition and reduce zero-food ticks against BOTH controls; >=5/6 masked lives must spend <1% ticks at zero food.',
        'secondary':'Conscious invalid actions / conscious decisions; not a survival claim.',
        'limits':'One eval world per training seed; two learners compete with six scripted residents; short training; primitive actions; no curriculum or architecture superiority claim.',
        'hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    write(out/'preregistration.json',prereg);(out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest())
    start=time.perf_counter();rows=[]
    for seed in SEEDS:
        trained={}
        for masked in (True,False):
            p=Population(seed,masked=masked);configure(p)
            trace=run(p,1200);name='trained_masked' if masked else 'trained_unmasked'
            p.save(out/f'{seed}-{name}.pt');write(out/f'{seed}-{name}-training.json',trace)
            trained[name]=p
        for condition in prereg['conditions']:
            p=Population(seed+100000,masked=condition!='trained_unmasked');configure(p)
            # Initial baseline uses matching training initialization, not eval-world initialization.
            source=trained.get(condition) or Population(seed,masked=True)
            for i,(rid,brain) in enumerate(p.brains.items()):
                brain.model.load_state_dict(source.brains[rid].model.state_dict())
                brain.hidden=brain.model.initial_state();brain.pending=None;brain.training=False
                brain.rng=random.Random(seed+900000+i)
            before={rid:digest(b) for rid,b in p.brains.items()}
            trace=run(p,8000)
            assert before=={rid:digest(b) for rid,b in p.brains.items()}
            row={'seed':seed,'condition':condition,'metrics':p.metrics,'frozen_weights':before}
            rows.append(row);write(out/f'{seed}-{condition}-evaluation.json',trace)
            write(out/'results.partial.json',rows)
            print(json.dumps({'seed':seed,'condition':condition,'seconds':round(time.perf_counter()-start,1),'metrics':p.metrics}),flush=True)
    means=lambda row,k:sum(m[k] for m in row['metrics'].values())/2
    checks=[];reliable=0
    for seed in SEEDS:
        group={r['condition']:r for r in rows if r['seed']==seed};m=group['trained_masked']
        checks.append(all(means(m,'nutrition')>means(group[c],'nutrition') and means(m,'zero_food_ticks')<means(group[c],'zero_food_ticks') for c in ('trained_unmasked','initial_masked')))
        reliable+=sum(v['zero_food_ticks']/8000<.01 for v in m['metrics'].values())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in prereg['hashes'].items())
    write(out/'results.json',rows)
    write(out/'complete.json',{'seconds':time.perf_counter()-start,'seed_improvements':checks,'reliable_lives':reliable,'lives':6,'passed':all(checks) and reliable>=5,'source_hashes_valid':True})


if __name__=='__main__':main()
