"""Preregistered finite-food safe-route experiment; independent recurrent PPO."""
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
from agents.sequence_ppo import Config, Population
from experiments.scarcity_routes import Routes, configure, food_stock, safe_oracle
from experiments.sequence_trial import digest

SEEDS=(53081,53092,53103)
SCHEDULE=(('feeding',4,512),('routes',4,1024),('scarcity',4,1536))
FILES=('agents/sequence_ppo.py','agents/preservation.py','agents/network.py','agents/affordances.py','agents/physiology.py',
       'sim/world.py','sim/ecology.py','sim/social.py','sim/visual_memory.py',
       'experiments/scarcity_routes.py','experiments/scarcity_trial.py','experiments/sequence_trial.py',
       'experiments/development.py','experiments/audit_scarcity.py')


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def arena_record(arenas):return {rid:asdict(a) for rid,a in arenas.items()}


def run(population,arenas,ticks,deadline,trace_path=None):
    if ticks<=0 or ticks%4:raise ValueError('Runs require positive complete decision blocks')
    trackers={rid:Routes(a) for rid,a in arenas.items()};commands={}
    original={rid:b.decide for rid,b in population.brains.items()}
    def recorder(rid):
        def decide(observation):
            command=original[rid](observation);commands[rid]=dict(command);return command
        return decide
    for rid,b in population.brains.items():b.decide=recorder(rid)
    extra={rid:dict(thorn_opportunities=0,ordinary_eaten=0,seed_conversions=0,
                   planted=0,matured_food=0,mean_fullness_sum=0.) for rid in arenas}
    first_stock={rid:food_stock(population.world,a) for rid,a in arenas.items()}
    handle=trace_path.open('w',encoding='utf-8') if trace_path else None
    start_metrics={rid:dict(m) for rid,m in population.metrics.items()}
    # Predictor counters belong to a lifelong brain, rather than its new world.
    for rid,b in population.brains.items():
        start_metrics[rid]['prediction_error_sum']=b.prediction_error_total
        start_metrics[rid]['prediction_events']=b.observed_transitions
    try:
        for offset in range(ticks):
            if offset%128==0 and time.perf_counter()>deadline:raise TimeoutError('Declared wall-time budget reached')
            w=population.world
            if offset%4==0:
                before={rid:dict(position=[a.x,a.y],food=a.food,conscious=not a.unconscious,
                                metrics=dict(population.metrics[rid])) for rid,a in w.residents.items()}
                for rid,a in w.residents.items():
                    adjacent=sum(abs(v-u) for v,u in zip((a.x,a.y),arenas[rid].hazard))<=1
                    extra[rid]['thorn_opportunities']+=not a.unconscious and adjacent
            crops={key for key,r in w.resources.items() if r['kind']=='crop'}
            results=population.step()
            if offset%4==0:
                action_results={rid:results[rid][0] for rid in arenas}
                for rid,command in commands.items():
                    if not action_results[rid]:continue
                    verb=command['verb']
                    extra[rid]['ordinary_eaten']+=verb=='eat' and command.get('item','food')=='food'
                    extra[rid]['seed_conversions']+=verb=='make_seeds'
                    extra[rid]['planted']+=verb=='plant'
            for key in crops:
                r=w.resources.get(key,{})
                if r.get('kind')=='berry':
                    point=tuple(map(int,key.split(',')))
                    for rid,arena in arenas.items():
                        if point in arena.cells():extra[rid]['matured_food']+=r['amount']
            for rid,a in w.residents.items():
                trackers[rid].observe((a.x,a.y));extra[rid]['mean_fullness_sum']+=a.food
                actual=food_stock(w,arenas[rid])
                expected=first_stock[rid]+extra[rid]['matured_food']-extra[rid]['ordinary_eaten']-extra[rid]['seed_conversions']
                if actual!=expected:raise RuntimeError(f'Food conservation failure for {rid}: {actual} != {expected}')
            if handle and offset%4==3:
                row=dict(tick=offset+1,agents={})
                for rid,a in w.residents.items():
                    m=population.metrics[rid];old=before[rid]['metrics']
                    row['agents'][rid]=dict(position=[a.x,a.y],food=a.food,health=a.health,water=a.water,
                        conscious=before[rid]['conscious'],unconscious=a.unconscious,action=commands[rid],
                        success=action_results[rid],nutrition=m['nutrition']-old['nutrition'],
                        damage=m['damage']-old['damage'],zero_food_ticks=m['zero_food_ticks']-old['zero_food_ticks'],
                        unconscious_ticks=m['unconscious_ticks']-old['unconscious_ticks'],
                        thorn_contacts=m['thorn_contacts']-old['thorn_contacts'],
                        food_stock=food_stock(w,arenas[rid]),mean_fullness_sum=extra[rid]['mean_fullness_sum'],
                        matured_food=extra[rid]['matured_food'])
                handle.write(json.dumps(row,separators=(',',':'))+'\n')
    finally:
        if handle:handle.close()
        for rid,b in population.brains.items():b.decide=original[rid]
    metrics={}
    for rid,m in population.metrics.items():
        values={key:value-start_metrics[rid][key] for key,value in m.items()}
        metrics[rid]={**values,**extra[rid],**trackers[rid].summary(),
                      'mean_fullness':extra[rid]['mean_fullness_sum']/ticks,
                      'initial_food_stock':first_stock[rid], 'remaining_food_stock':food_stock(w,arenas[rid])}
    return metrics


def gate(rows,ticks):
    checks={};reliable=safe=lives=0
    for seed in sorted({row['seed'] for row in rows}):
        group={c:[m for row in rows if row['seed']==seed and row['condition']==c for m in row['metrics'].values()]
               for c in ('trained','initial')}
        mean=lambda c,k:sum(m[k] for m in group[c])/len(group[c])
        rate=lambda c:sum(m['thorn_contacts'] for m in group[c])/max(1,sum(m['conscious_decisions'] for m in group[c]))
        checks[str(seed)]=dict(less_starvation=mean('trained','zero_food_ticks')<mean('initial','zero_food_ticks'),
                              more_fullness=mean('trained','mean_fullness')>mean('initial','mean_fullness'),
                              lower_contact_rate=rate('trained')<=.5*rate('initial'))
        for m in group['trained']:
            lives+=1;reliable+=m['zero_food_ticks']<.01*ticks
            safe+=m['foodward_crossings']>0 and m['safe_foodward_crossings']>0 and m['safe_crossings']/m['crossings']>=.8
    passed=bool(checks) and all(all(v.values()) for v in checks.values()) and reliable/lives>=.8 and safe/lives>=.8
    return dict(seed_checks=checks,reliable_lives=reliable,safe_active_lives=safe,lives=lives,passed=passed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--wall-seconds',type=int,default=1800)
    args=parser.parse_args();out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs'):parser.error('Use a new project runs directory')
    if args.wall_seconds<=0:parser.error('Wall budget must be positive')
    out.mkdir(parents=True,exist_ok=False)
    config=Config();seeds=SEEDS[:1] if args.pilot else SEEDS
    schedule=(('feeding',1,128),('routes',1,128),('scarcity',1,128)) if args.pilot else SCHEDULE
    ticks=512 if args.pilot else 12000
    evaluation_seeds={s:[s+100000] if args.pilot else [s+100000,s+200000] for s in seeds}
    manifest=dict(config=asdict(config),training_seeds=seeds,evaluation_seeds=evaluation_seeds,
        training_schedule=schedule,decisions_per_agent=sum(n*d for _,n,d in schedule),evaluation_ticks=ticks,
        conditions=('trained','initial','scripted','safe_oracle'),pilot=args.pilot,wall_seconds=args.wall_seconds,
        starting_fullness=8,starting_inventory_food=0,food_units_per_agent=4,
        interventions='None. Initial stocks are finite. Ordinary player farming actions can generate real crop yields, separately accounted.',
        reward='Unchanged food relief/25, hunger distress -0.005/tick, observed injury -1/20. Curiosity zero; no route, movement, planting or signaling bonus.',
        gate='Every training-seed group reduces mean zero-food ticks, increases mean fullness, and halves thorn contacts per conscious decision versus matched initial weights. >=80% trained lives spend <1% time at zero food. >=80% have a foodward crossing, a safe foodward crossing and >=80% safe completed crossings. Idle agents cannot pass.',
        limits='Rotated/transposed small fork arenas with shared physiology and cue semantics. Frozen evaluation, two isolated independent learners, no competition, learned society, planning or farming competence claim. Safe oracle sees global resources and is feasibility only; it is never a teacher.',
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out/'preregistration.json',manifest)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest(),encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in FILES:archive.write(ROOT/f,f)
    start=time.perf_counter();deadline=start+args.wall_seconds;rows=[]
    try:
        for seed in seeds:
            trained=Population(seed,config);curves=[]
            for stage_i,(stage,episodes,decisions) in enumerate(schedule):
                for episode in range(episodes):
                    world_seed=seed+1000*stage_i+episode
                    fresh=Population(world_seed,config);fresh.brains=trained.brains;trained=fresh
                    arenas=configure(trained,world_seed,stage)
                    metrics=run(trained,arenas,decisions*4,deadline)
                    for rid,b in trained.brains.items():b.finish(trained.world.observe(rid),terminal=True)
                    curves.append(dict(stage=stage,episode=episode,world_seed=world_seed,arenas=arena_record(arenas),
                                       metrics=metrics,inspector={rid:b.inspector() for rid,b in trained.brains.items()}))
                    write(out/f'{seed}-learning-curve.json',curves)
                    print(json.dumps(dict(seed=seed,stage=stage,episode=episode,seconds=round(time.perf_counter()-start,1),metrics=metrics)),flush=True)
            trained.save(out/f'{seed}-trained.pt')
            initial=Population(seed,config)
            for evaluation_seed in evaluation_seeds[seed]:
                for condition in manifest['conditions']:
                    p=Population(evaluation_seed,config);arenas=configure(p,evaluation_seed)
                    for i,(rid,b) in enumerate(p.brains.items()):
                        source=trained.brains[rid] if condition=='trained' else initial.brains[rid]
                        b.model.load_state_dict(source.model.state_dict());b.predictor.load_state_dict(source.predictor.state_dict())
                        b.freeze();b.rng=random.Random(evaluation_seed+900000+i)
                        if condition=='scripted':b.decide=lambda observation,rid=rid:p.world.baseline_action(rid)
                        if condition=='safe_oracle':b.decide=lambda observation,rid=rid:safe_oracle(p.world,rid)
                    before={rid:digest(b) for rid,b in p.brains.items()}
                    metrics=run(p,arenas,ticks,deadline,out/f'{evaluation_seed}-{condition}-trace.jsonl')
                    after={rid:digest(b) for rid,b in p.brains.items()};assert before==after
                    row=dict(seed=seed,evaluation_seed=evaluation_seed,condition=condition,arenas=arena_record(arenas),
                             metrics=metrics,frozen_weights=before,frozen_end=after)
                    rows.append(row);write(out/'results.partial.json',rows)
                    print(json.dumps(dict(evaluation_seed=evaluation_seed,condition=condition,seconds=round(time.perf_counter()-start,1),metrics=metrics)),flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in manifest['hashes'].items())
        write(out/'results.json',rows)
        write(out/'complete.json',dict(seconds=time.perf_counter()-start,source_hashes_valid=True,
              total_ticks=sum(n*d*4 for _,n,d in schedule)*len(seeds)+ticks*len(rows),gate=gate(rows,ticks)))
    except TimeoutError as error:
        write(out/'budget-stop.json',dict(seconds=time.perf_counter()-start,reason=str(error),completed_evaluations=len(rows)))
        raise


if __name__=='__main__':main()
