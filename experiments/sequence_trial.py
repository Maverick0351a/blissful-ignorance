"""Bounded, preregistered development and resource-relocation comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from agents.sequence_ppo import Population, Config
from experiments.development import configure

SEEDS=(42071,42082,42093)
FILES=('agents/sequence_ppo.py','agents/preservation.py','agents/network.py','agents/affordances.py','agents/physiology.py','sim/world.py',
       'sim/ecology.py','sim/social.py','sim/visual_memory.py','experiments/development.py',
       'experiments/sequence_trial.py')


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def digest(brain):
    return {name:hashlib.sha256(b''.join(v.detach().numpy().tobytes()
                    for v in getattr(brain,name).state_dict().values())).hexdigest()
            for name in ('model','predictor')}


def offline(population):
    # Only the two independent residents remain. No scripted food transfers or
    # revivals and no player interventions are possible in these tests.
    population.world.residents={rid:population.world.residents[rid] for rid in population.brains}
    return population


def relocate(population, seed, moved):
    w=population.world;rng=random.Random(seed)
    # Resource movement changes what is seen; no goal or stage enters the brain.
    for key,r in list(w.resources.items()):
        if r['kind'] in ('berry','amber_bush'):del w.resources[key]
    for center in (22,36):
        side=1 if moved else -1
        ys=list(range(28,33));rng.shuffle(ys)
        for y in ys[:3]:w.resources[w.key(center+2*side,y)]={'kind':'berry','amount':8}
        w.resources[w.key(center-2*side,30)]={'kind':'amber_bush','amount':8}
    w.revision+=1


def run(population,ticks,deadline,shift=False):
    trace=[];exposures={rid:0 for rid in population.brains}
    for i in range(ticks):
        if time.perf_counter()>deadline:raise TimeoutError('Declared wall-time budget reached')
        if shift and i in (ticks//3,2*ticks//3):
            relocate(population,population.world.seed+i,i==ticks//3)
        if population.world.tick%4==0:
            for rid in population.brains:
                o=population.world.observe(rid)
                exposures[rid]+=int(not o['body']['unconscious'] and any(
                    t.get('resource',{}).get('kind')=='thorns' and abs(t['dx'])+abs(t['dy'])<=1
                    for t in o['tiles']))
        population.step()
        if (i+1)%512==0 or i+1==ticks:
            trace.append({'ticks':i+1,'world_tick':population.world.tick,
                          'agents':{rid:{**dict(population.metrics[rid]),'thorn_opportunities':exposures[rid],
                                        'fullness':population.world.residents[rid].food,
                                        'updates':b.updates} for rid,b in population.brains.items()}})
    return trace


def gate(rows,ticks):
    groups={};reliable=0;avoidant=0;lives=0
    for row in rows:
        if row['condition']!='trained':continue
        peers={r['condition']:r for r in rows if r['seed']==row['seed'] and r['evaluation_seed']==row['evaluation_seed']}
        initial=peers['initial'];scripted=peers['scripted']
        mean=lambda r,key:sum(m[key] for m in r['metrics'].values())/len(r['metrics'])
        improvement=mean(row,'nutrition')>mean(initial,'nutrition')
        sufficient=mean(row,'nutrition')>=.8*mean(scripted,'nutrition')
        groups[str(row['evaluation_seed'])]=dict(improved=improvement,sufficient_nutrition=sufficient)
        for rid,m in row['metrics'].items():
            lives+=1;reliable+=m['zero_food_ticks']/ticks<.01
            opportunities=m['thorn_opportunities'];control=initial['metrics'][rid]
            # No encounters cannot pass as learned avoidance. Injury frequency
            # is reported separately from physiological damage totals.
            avoidance=opportunities>0 and control['thorn_opportunities']>0 and (
                m['thorn_contacts']/opportunities <= .5*control['thorn_contacts']/control['thorn_opportunities'])
            avoidant+=avoidance
    return dict(groups=groups,reliable_lives=reliable,lives=lives,avoidant_lives=avoidant,
                passed=bool(groups) and all(v['improved'] and v['sufficient_nutrition'] for v in groups.values())
                       and reliable/lives>=.9 and avoidant/lives>=.8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--wall-seconds',type=int,default=1800)
    args=parser.parse_args();out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs'):parser.error('Output must be a new project runs directory')
    out.mkdir(parents=True,exist_ok=False)
    config=Config();episodes=1 if args.pilot else 4;decisions=128 if args.pilot else 512
    eval_ticks=512 if args.pilot else 12000;seeds=SEEDS[:1] if args.pilot else SEEDS
    evaluation_seeds={s:[s+100000] if args.pilot else [s+100000,s+200000] for s in seeds}
    stages=('near_food','depletion','hazards')
    manifest=dict(config=asdict(config),training_seeds=seeds,evaluation_seeds=evaluation_seeds,
        stages=stages,episodes_per_stage=episodes,decisions_per_episode=decisions,
        decisions_per_agent=len(stages)*episodes*decisions,evaluation_ticks=eval_ticks,
        conditions=('trained','initial','scripted'),wall_seconds=args.wall_seconds,pilot=args.pilot,
        reward='food relief/25; hunger distress -0.005/tick; observed injury -1/20; no curiosity or auxiliary policy loss',
        gate='Every held-out map improves nutrition over initial weights and obtains >=80% scripted nutrition; >=90% trained lives <1% zero-food; >=80% lives halve thorn contacts per adjacent opportunity vs initial, with nonzero exposure.',
        limits='Small separated rooms, fixed physiology/cues, spatial A-B-A resource shift with frozen weights; no full-valley, farming, social, online adaptation or language competence claim.',
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out/'preregistration.json',manifest)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest())
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in FILES:archive.write(ROOT/f,f)
    start=time.perf_counter();deadline=start+args.wall_seconds;rows=[]
    try:
        for seed in seeds:
            population=Population(seed,config);curves=[]
            for stage_i,stage in enumerate(stages):
                for episode in range(episodes):
                    # New episodes use new worlds, while each resident's learned
                    # weights and optimizer continue only from its own history.
                    fresh=Population(seed+stage_i*1000+episode,config)
                    fresh.brains=population.brains;population=offline(configure(fresh,fresh.world.seed,stage))
                    trace=run(population,decisions*4,deadline)
                    for rid,b in population.brains.items():b.finish(population.world.observe(rid),terminal=True)
                    curves.append({'stage':stage,'episode':episode,'metrics':population.metrics,
                                   'inspector':{rid:b.inspector() for rid,b in population.brains.items()}})
                    write(out/f'{seed}-learning-curve.json',curves)
                    print(json.dumps({'seed':seed,'stage':stage,'episode':episode,
                                      'seconds':round(time.perf_counter()-start,1),
                                      'metrics':population.metrics}),flush=True)
            population.save(out/f'{seed}-trained.pt')
            for evaluation_seed in evaluation_seeds[seed]:
                for condition in manifest['conditions']:
                    p=offline(configure(Population(evaluation_seed,config),evaluation_seed,'hazards'))
                    initial=Population(seed,config)
                    for i,(rid,b) in enumerate(p.brains.items()):
                        source=population.brains[rid] if condition=='trained' else initial.brains[rid]
                        b.model.load_state_dict(source.model.state_dict());b.predictor.load_state_dict(source.predictor.state_dict())
                        b.freeze();b.rng=random.Random(evaluation_seed+800000+i)
                        if condition=='scripted':b.decide=lambda observation,rid=rid:p.world.baseline_action(rid)
                    before={rid:digest(b) for rid,b in p.brains.items()}
                    trace=run(p,eval_ticks,deadline,shift=True)
                    after={rid:digest(b) for rid,b in p.brains.items()}
                    assert before==after
                    for rid,m in p.metrics.items():
                        m['thorn_contacts']=p.world.residents[rid].thorn_contacts
                        m['thorn_opportunities']=trace[-1]['agents'][rid]['thorn_opportunities']
                    row=dict(seed=seed,evaluation_seed=evaluation_seed,condition=condition,
                             metrics=p.metrics,frozen_weights=before,frozen_end=after)
                    rows.append(row);write(out/f'{evaluation_seed}-{condition}-trace.json',trace)
                    write(out/'results.partial.json',rows)
                    print(json.dumps({'evaluation_seed':evaluation_seed,'condition':condition,
                                      'seconds':round(time.perf_counter()-start,1),'metrics':p.metrics}),flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in manifest['hashes'].items())
        write(out/'results.json',rows);write(out/'complete.json',dict(seconds=time.perf_counter()-start,
              source_hashes_valid=True,gate=gate(rows,eval_ticks)))
    except TimeoutError as error:
        write(out/'budget-stop.json',dict(seconds=time.perf_counter()-start,reason=str(error),completed_evaluations=len(rows)))
        raise


if __name__=='__main__':main()
