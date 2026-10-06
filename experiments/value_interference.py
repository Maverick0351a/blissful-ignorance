"""Matched, one-update value-gradient ablation on archived experimental brains.

Every shadow starts at an original on-policy update boundary. Shadows never
produce subsequent training data or control the continuing world.
"""
import argparse
from contextlib import contextmanager
import copy
import gzip
import json
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from agents.sequence_ppo import Config
from experiments.category_trial import (MeasuredCategory,population_for,check_budget,
    memory_usage,weight_hashes,bodies,sha,write)
from experiments.feeding_credit import (EAT,canonical_inputs,fingerprint,probe,sequence_probabilities)
from sim.world import World


@contextmanager
def detached_value_input(brain):
    """Preserve forward values and value-head learning; block its trunk gradient."""
    handle=brain.model.value.register_forward_pre_hook(lambda module,args:(args[0].detach(),))
    try:yield
    finally:handle.remove()


def copy_brain(state):
    brain=MeasuredCategory(state['seed'],Config(**state['config']))
    brain.restore(copy.deepcopy(state))
    return brain


def same(left,right):
    if json.dumps(left,sort_keys=True)!=json.dumps(right,sort_keys=True):
        raise AssertionError('Recorded data mismatch')


def value_mse(brain,rows,returns):
    with torch.no_grad():
        _,values=brain._sequence(torch.cat([r['patch'] for r in rows]),
                                torch.cat([r['features'] for r in rows]),rows[0]['hidden'])
        return float((values-torch.tensor(returns)).square().mean())


def paired_update(brain,reference,inputs):
    """Apply one intervention to a private clone; advance only native baseline."""
    snapshot=copy.deepcopy(brain.state());before_hash=fingerprint(snapshot)
    rows=brain.buffer;shadow=copy_brain(snapshot)
    if fingerprint(shadow.state())!=before_hash:raise AssertionError('Unmatched shadow start')
    before=sequence_probabilities(brain,rows)
    probes_before={key:probe(brain,value) for key,value in inputs.items()}
    same(probes_before,reference['canonical_before'])
    returns=[t['target_return'] for t in reference['transitions']]
    before_mse=value_mse(brain,rows,returns)
    # Only this backward path differs; native clipping, Adam momentum and PPO
    # epochs are preserved. Their coupled consequences belong to this treatment.
    with detached_value_input(shadow):shadow._learn(reference['bootstrap'])
    if fingerprint(brain.state())!=before_hash or fingerprint(snapshot)!=before_hash:
        raise AssertionError('Shadow changed the baseline or snapshot')
    brain._learn(reference['bootstrap'])
    if fingerprint(brain.state())!=reference['state_after']:
        raise AssertionError('Native baseline did not reproduce the recorded update')
    standard=sequence_probabilities(brain,rows);detached=sequence_probabilities(shadow,rows)
    transitions=[]
    for i,(row,original) in enumerate(zip(rows,reference['transitions'])):
        if float(before[i,EAT])!=original['eat_probability_before'] or float(standard[i,EAT])!=original['eat_probability_after']:
            raise AssertionError('Changed native meal probability')
        transitions.append(dict(index=i,decision=original['decision'],ordinary_meal=original['ordinary_meal'],
            action=original['action'],normalized_advantage=original['normalized_advantage'],
            before=float(before[i,EAT]),standard=float(standard[i,EAT]),detached=float(detached[i,EAT])))
    same(fingerprint(brain.state()['predictor']),fingerprint(shadow.state()['predictor']))
    record={k:reference[k] for k in ('candidate','case','episode','training_seed','update','bootstrap')}
    record.update(state_before=before_hash,state_standard=fingerprint(brain.state()),
        state_detached=fingerprint(shadow.state()),probes_before=probes_before,
        probes_standard={k:probe(brain,v) for k,v in inputs.items()},
        probes_detached={k:probe(shadow,v) for k,v in inputs.items()},transitions=transitions,
        diagnostics_standard=copy.deepcopy(brain.diagnostics),diagnostics_detached=copy.deepcopy(shadow.diagnostics),
        value_mse=dict(before=before_mse,standard=value_mse(brain,rows,returns),detached=value_mse(shadow,rows,returns)))
    return record,copy.deepcopy(shadow.state())


def mean(values):return sum(values)/len(values) if values else None


def probe_mean(record,arm,fixture,verb):
    return mean([value[verb] for key,value in record['probes_'+arm].items() if key.startswith(fixture)])


def summarize(records):
    candidates={}
    for cid in sorted({r['candidate'] for r in records}):
        rows=[r for r in records if r['candidate']==cid]
        meals=[t for r in rows for t in r['transitions'] if t['ordinary_meal']]
        fixtures={}
        for fixture,verb in (('carried','eat'),('adjacent','gather')):
            standard=[probe_mean(r,'standard',fixture,verb)-probe_mean(r,'before',fixture,verb) for r in rows]
            detached=[probe_mean(r,'detached',fixture,verb)-probe_mean(r,'before',fixture,verb) for r in rows]
            fixtures[fixture]=dict(mean_standard_gain=mean(standard),mean_detached_gain=mean(detached),
                mean_paired_difference=mean([b-a for a,b in zip(standard,detached)]),
                beneficial_updates=sum(b-a>1e-6 for a,b in zip(standard,detached)))
        candidates[cid]=dict(updates=len(rows),meals=len(meals),fixtures=fixtures,
            reinforced={arm:sum(t[arm]-t['before']>1e-6 for t in meals) for arm in ('standard','detached')},
            mean_meal_gain={arm:mean([t[arm]-t['before'] for t in meals]) for arm in ('standard','detached')},
            mean_value_mse={arm:mean([r['value_mse'][arm] for r in rows]) for arm in ('before','standard','detached')})
    n=len(candidates)
    carried=mean([c['fixtures']['carried']['mean_paired_difference'] for c in candidates.values()])
    adjacent=mean([c['fixtures']['adjacent']['mean_paired_difference'] for c in candidates.values()])
    positive=sum(c['fixtures']['carried']['mean_paired_difference']>1e-6 for c in candidates.values())
    reinforced={arm:sum(c['reinforced'][arm] for c in candidates.values()) for arm in ('standard','detached')}
    checks=dict(at_least_point_one_pp=carried>=.001-1e-12,positive_in_at_least_four_brains=positive>=4,
                no_fewer_reinforced_meals=reinforced['detached']>=reinforced['standard'],
                gathering_not_worse_by_point_one_pp=adjacent>=-.001-1e-12)
    return dict(candidates=candidates,brains=n,updates=len(records),meals=sum(c['meals'] for c in candidates.values()),
        mean_paired_carried_difference=carried,mean_paired_adjacent_difference=adjacent,
        positive_brains=positive,reinforced=reinforced,checks=checks,
        diagnostic_promising=all(checks.values()),milestone_confirmed=False,
        limitation='Each shadow is reset before each matched update. No chained detached policy, new life, retention or survival result.')


def replay(folder,callback,deadline):
    """Reconstruct every original boundary using native, sampled actions."""
    protocol=json.loads((folder/'parent-preregistration.json').read_text(encoding='utf-8'))
    original=json.loads((folder/'source-preregistration.json').read_text(encoding='utf-8'))
    references=json.loads((folder/'parent-updates.json').read_text(encoding='utf-8'))
    reference_index={(r['candidate'],r['case']):r for r in references}
    assert len(reference_index)==96
    initial=torch.load(folder/'source-continue-initial.pt',weights_only=True,map_location='cpu')
    expected=torch.load(folder/'source-continue-trained.pt',weights_only=True,map_location='cpu')
    config=Config(**protocol['config']);populations={};inputs={};seen=set();episodes=steps=decisions=0
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            if steps%64==0:check_budget(folder,deadline)
            row=json.loads(line)
            if row['type']=='start':
                meta=row['metadata'];pair=meta['pair'];w=World.from_dict(row['world'])
                assert meta['stage']=='training' and meta['arm']=='continue'
                if pair not in populations:
                    p=population_for(meta['training_seed'],'category',w,config)
                    for i,(rid,b) in enumerate(p.brains.items()):
                        cid=f'c{pair*2+i}';b.restore(copy.deepcopy(initial[cid]));inputs[cid]=canonical_inputs(b,rid)
                    populations[pair]=p
                p=populations[pair];p.world=w;p.metrics={rid:p.new_metrics() for rid in p.brains}
                assert meta['case']==sum(cid==f'c{pair*2}' for cid,case in seen)
                for i,b in enumerate(p.brains.values()):
                    b.rng=random.Random(original['bases']['training']*10+meta['case']*6+pair*2+i);b.reward_totals={}
                same({rid:weight_hashes(b) for rid,b in p.brains.items()},row['weights'])
            elif row['type']=='step':
                assert p.world.tick==row['tick'];deciding=p.world.tick%4==0
                result=p.step();same(result,row['results']);same(bodies(p.world),row['after'])
                for rid,b in p.brains.items():
                    same(b.last_action if deciding else {'verb':'wait'},row['commands'][rid])
                    same(b.last_reward,row['rewards'][rid])
                    if deciding:same(b.last_observation,row['observations'][rid])
                steps+=1;decisions+=len(p.brains) if deciding else 0
            elif row['type']=='end':
                for rid,b in p.brains.items():
                    cid=meta['candidates'][rid];key=(cid,meta['case']);reference=reference_index[key]
                    b._complete(p.world.observe(rid),terminal=True)
                    assert fingerprint(b.state())==reference['state_before'] and len(b.buffer)==128
                    callback(b,reference,inputs[cid])
                    assert fingerprint(b.state())==reference['state_after']
                    b.pending=None;b.hidden=b.model.initial_state();seen.add(key)
                same(p.world.to_dict(),row['world']);same({rid:weight_hashes(b) for rid,b in p.brains.items()},row['weights_after'])
                episodes+=1
                if episodes%8==0:print(json.dumps(dict(episodes=episodes,updates=len(seen))),flush=True)
            else:raise AssertionError('Unknown trace row')
    final={}
    for pair,p in populations.items():
        for i,b in enumerate(p.brains.values()):
            cid=f'c{pair*2+i}';assert fingerprint(b.state())==fingerprint(expected[cid]);final[cid]=copy.deepcopy(b.state())
    assert (episodes,steps,decisions,len(seen))==(48,24576,12288,96)
    return final,dict(episodes=episodes,replayed_ticks=steps,replayed_training_decisions=decisions,updates=len(seen))


def prepare_source(source,out):
    parent=json.loads((source/'preregistration.json').read_text(encoding='utf-8'))
    receipt=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    if parent['diagnostic']!='recorded-feeding-update' or not audit['passed']:raise ValueError('Requires audited feeding-credit run')
    if sha(source/'preregistration.json')!=(source/'preregistration.sha256').read_text().strip():raise ValueError('Parent protocol changed')
    for name,value in receipt['evidence_hashes'].items():
        if sha(source/name)!=value:raise ValueError('Parent evidence changed: '+name)
    for name,value in parent['hashes'].items():
        if sha(ROOT/name)!=value:raise ValueError('Parent code changed: '+name)
    out.mkdir(parents=True,exist_ok=False)
    mapping={name:name for name in ('trace.jsonl.gz','source-continue-initial.pt','source-continue-trained.pt',
                                  'source-preregistration.json','source-complete.json','source-audit.json')}
    mapping.update({name:'parent-'+name for name in ('preregistration.json','complete.json','audit.json','updates.json')})
    for name,destination in mapping.items():shutil.copy2(source/name,out/destination)
    return parent,mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/feeding-credit-20261006')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=600)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    if any(not p.is_relative_to(ROOT/'runs') or p==ROOT/'runs' for p in (source,out)):parser.error('Use local experimental run directories')
    if not 1<=args.wall_seconds<=600:parser.error('Wall cap must be 1..600 seconds')
    parent,mapping=prepare_source(source,out)
    files=set(parent['hashes']) | {'experiments/value_interference.py','experiments/audit_value_interference.py','tests/test_value_interference.py'}
    protocol=dict(protocol=1,milestone='GT-01',diagnostic='matched-value-gradient-interference',
        parent=str(source.relative_to(ROOT)),parent_files={destination:sha(out/destination) for destination in mapping.values()},
        config=parent['config'],replay_ticks=24576,paired_updates=96,shadow_optimizer_calls=96,
        wall_seconds=args.wall_seconds,audit_wall_seconds=600,evidence_cap_bytes=512*1024*1024,
        primary='Mean paired change in ordinary eating probability on four fixed carried-food probes, equally weighted across six brains and sixteen original updates per brain.',
        treatment='Native PPO update with a temporary value-head input detach. Only new value-loss gradients into the shared CNN/features/LSTM are blocked. The value head still trains. Native Adam states, value targets, rewards, entropy, clipping, all actions, epochs and KL stopping are unchanged. Historical optimizer momentum remains.',
        controls='Before-update probability is the no-update control. Standard native PPO must reproduce all archived states. Each detached shadow starts from the exact original pre-update model, predictor, optimizer, RNG and history, receives that one original on-policy rollout, and is then retired. NEVER chain shadow updates or feed baseline data to a diverged shadow.',
        gate='Diagnostic promising only if mean paired carried-food improvement >=0.001 (0.1 percentage point), positive (>0.000001) in at least four of six brains, at least as many recorded meals reinforced as native PPO, and mean adjacent-gathering change >=-0.001. No significance, memory, survival, milestone or deployment claim.',
        secondary='Report all brains, c3/c5 individually, every update, meal-probability changes, value-target MSE, update epochs/KL, and any gathering loss. Fixed probe scenes were already evaluated in the prior diagnostic, never used to train. These are not an experimenter-blind confirmation set.',
        limitations='Immediate matched effects include changes to shared global gradient clipping and current Adam steps. No new lives, accumulated detached learner, long-term retention, Laya comparison or live change. A favorable result requires a separate fresh behavioral trial.',
        hashes={name:sha(ROOT/name) for name in sorted(files)})
    write(out/'preregistration.json',protocol);(out/'preregistration.sha256').write_text(sha(out/'preregistration.json'),encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):archive.write(ROOT/name,name)
    start=time.perf_counter();deadline=start+args.wall_seconds;records=[];last_shadows={}
    try:
        def callback(brain,reference,inputs):
            record,shadow=paired_update(brain,reference,inputs);records.append(record)
            if reference['case']==15:last_shadows[reference['candidate']]=shadow
            write(out/'pairs.partial.json',records)
        final,counts=replay(out,callback,deadline)
        torch.save(final,out/'native-final.pt');torch.save(last_shadows,out/'last-independent-shadows.pt')
        write(out/'pairs.json',records);write(out/'analysis.json',summarize(records));check_budget(out,deadline)
        assert all(sha(ROOT/name)==value for name,value in protocol['hashes'].items())
        assert all(sha(source/name)==protocol['parent_files'][destination] for name,destination in mapping.items())
        write(out/'complete.json',dict(seconds=time.perf_counter()-start,**counts,shadow_optimizer_calls=len(records),
            native_final_states_match=True,live_population_changed=False,memory=memory_usage(),
            evidence_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(complete=True,seconds=time.perf_counter()-start,summary=summarize(records))),flush=True)
    except Exception as exc:
        write(out/'incomplete.json',dict(error=type(exc).__name__,reason=str(exc),paired_updates=len(records),
                                       seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
