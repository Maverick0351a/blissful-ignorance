"""Preregistered neural replay ablation using real farming physiology."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import pickle
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.replay_policy import Learner, RandomLearner
from experiments.mobile_farming_trial import arena,episode

SEEDS=(11031,12042,13053)
def signature(seed):
    world=arena(seed)
    return tuple((x,y) for y,row in enumerate(world.terrain) for x,v in enumerate(row) if v==0)
def schedule(seed,train=24,used=None):
    training=[seed+i*10000 for i in range(train)]
    if used is None: used={signature(s) for s in training}
    evaluation=[]; candidate=seed+1000000
    while len(evaluation)<3:
        sig=signature(candidate)
        if sig not in used:
            used.add(sig); evaluation.append(candidate)
        candidate+=1
    return training,evaluation
def pair(seed,coefficient=.02,random=False):
    cls=RandomLearner if random else Learner
    return {rid:cls(seed+i,coefficient) for i,rid in enumerate(('player','r0'))}
def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def frozen_digest(policy):
    """Exclude episode action RNG, empty trace and observational diagnostics."""
    weights=[tensor.detach().numpy().tobytes() for module in (policy.network,policy.target) for tensor in module.parameters()]
    return hashlib.sha256(pickle.dumps((weights,policy.optimizer.state_dict(),list(policy.replay),policy.updates,policy.curiosity.snapshot(),policy.replay_rng.getstate()))).hexdigest()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
    out=Path(args.output)
    if not args.pilot: out.mkdir(parents=True,exist_ok=False)
    else: out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if args.pilot:
        policies=pair(99817)
        row,_=episode('pilot',99817,policies,True,decisions=40)
        write(out/'pilot.json',dict(seconds=time.perf_counter()-start,result=row))
        print(json.dumps(dict(seconds=time.perf_counter()-start,result=row)),flush=True)
        return
    used={signature(s+i*10000) for s in SEEDS for i in range(24)}
    schedules={str(s):schedule(s,used=used) for s in SEEDS}
    prereg=dict(seeds=SEEDS,train_episodes=24,eval_episodes=3,decisions=160,stride=120,
        schedules=schedules,layout_signatures={str(s):[signature(k) for k in sum(schedules[str(s)],[])] for s in SEEDS},
        model='23 inputs from existing local sensor;64x64 ReLU;9 actions;independent DoubleDQN',
        learning=dict(replay=12000,batch=64,nstep=3,gamma=.995,lr=.001,epsilon=.25,target_period=100,gradient_clip=5),
        conditions={'curiosity_on':.02,'curiosity_off':0},
        evaluation='greedy frozen weights/replay;curiosity reward and counts disabled;unseen layouts;random and initial-weight baselines;scripted positive control',
        gate='Both conditions separately: each seed improves mean hunger cost and nutrition versus initial weights; >=90% of all evaluation agent lives have zero-food fraction<.01 and harvest>0',
        limitations='Partial sensor bins alias crop timing and capacity; replay retains historical intrinsic rewards; no mortality; no deployment unless gate passes',
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('agents/replay_policy.py','experiments/neural_farming_trial.py','experiments/mobile_farming_trial.py','agents/rewards.py','sim/world.py')})
    write(out/'preregistration.json',prereg)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest())
    rows=[]
    for seed in SEEDS:
        training,evaluation=schedules[str(seed)]
        for condition,coefficient in (('curiosity_on',.02),('curiosity_off',0)):
            policies=pair(seed,coefficient)
            for index,world_seed in enumerate(training):
                row,_=episode(condition,world_seed,policies,True)
                row.update(base_seed=seed,episode=index); rows.append(row)
            for rid,policy in policies.items(): policy.save(out/f'checkpoint-{seed}-{condition}-{rid}.pt')
            for policy in policies.values():
                policy.freeze_curiosity(); policy.curiosity.last=None
            for index,world_seed in enumerate(evaluation):
                before={rid:frozen_digest(policy) for rid,policy in policies.items()}
                row,trace=episode(condition,world_seed,policies)
                after={rid:frozen_digest(policy) for rid,policy in policies.items()}
                assert before==after,'Evaluation changed learned state'
                row['frozen_state_hashes']=after
                row.update(base_seed=seed,episode=index);rows.append(row)
                write(out/f'trace-{seed}-{condition}-{index}.json',trace)
            write(out/'results.partial.json',rows)
            print(f'{seed} {condition} complete in {time.perf_counter()-start:.1f}s',flush=True)
        for kind in ('untrained','random','scripted'):
            policies=pair(seed,0,kind=='random')
            for policy in policies.values(): policy.freeze_curiosity()
            for index,world_seed in enumerate(evaluation):
                row,trace=episode(kind,world_seed,policies)
                row.update(base_seed=seed,episode=index); rows.append(row)
                write(out/f'trace-{seed}-{kind}-{index}.json',trace)
    write(out/'results.json',rows)
    gates={}
    for condition in ('curiosity_on','curiosity_off'):
        seed_gates=[]; lives=[]
        for seed in SEEDS:
            selected=[a for row in rows if row['kind']==condition and not row['training'] and row['base_seed']==seed for a in row['agents'].values()]
            baseline=[a for row in rows if row['kind']=='untrained' and row['base_seed']==seed for a in row['agents'].values()]
            mean=lambda group,key:sum(a[key] for a in group)/len(group)
            seed_gates.append(mean(selected,'hunger_cost')<mean(baseline,'hunger_cost') and mean(selected,'nutrition')>mean(baseline,'nutrition'))
            lives.extend(a['zero_food_ticks']/19200<.01 and a['harvested']>0 for a in selected)
        gates[condition]=dict(seed_improvement=seed_gates,reliable_lives=sum(lives),lives=len(lives),passed=all(seed_gates) and sum(lives)/len(lives)>=.9)
    write(out/'complete.json',dict(complete=True,seconds=time.perf_counter()-start,rows=len(rows),gates=gates))
    print(json.dumps(gates),flush=True)
if __name__=='__main__': main()
