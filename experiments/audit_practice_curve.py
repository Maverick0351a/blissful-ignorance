"""Separate native-policy, physics, full-checkpoint and outcome audit."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.affordances import action_mask
from agents.category_ppo import Brain
from agents.network import ACTIONS, encode_observation
from agents.sequence_ppo import Config
from experiments.feeding_credit import fingerprint
from sim.world import World


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value, sort_keys=True, separators=(',', ':'))


def weight_hashes(brain):
    return {name:hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in
            getattr(brain, name).state_dict().values())).hexdigest() for name in ('model','predictor')}


def prompt(m, fixture):
    meal, gather, zero = (m[k] for k in ('first_meal_tick','first_gather_food_tick','first_zero_tick'))
    return (meal is not None and meal < (64 if fixture=='adjacent' else 16)
            and (zero is None or meal < zero) and (fixture=='carried' or (gather is not None and gather < meal)))


def audit(folder):
    started = time.perf_counter()
    manifest = json.loads((folder/'preregistration.json').read_text())
    receipt = json.loads((folder/'complete.json').read_text())
    deadline = started+manifest['audit_wall_seconds']
    assert manifest['experiment']=='retained-practice-amount' and manifest['audit_wall_seconds']==600
    assert digest(folder/'preregistration.json')==(folder/'preregistration.sha256').read_text().strip()
    for name,h in receipt['evidence_hashes'].items():assert digest(folder/name)==h,name
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name,h in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==h and digest(ROOT/name)==h,name
    source=(ROOT/manifest['source']).resolve();assert source.is_relative_to(ROOT/'runs')
    for name,h in manifest['source_hashes'].items():assert digest(source/name)==digest(folder/('source-'+name))==h
    original=torch.load(folder/'source-continue-trained.pt',weights_only=True,map_location='cpu')
    assert set(original)=={f'c{i}' for i in range(6)}
    config=Config(**manifest['config']);checkpoints={}
    for cid,state in original.items():
        assert fingerprint(state)==manifest['original_state_hashes'][cid]
        assert state['decisions']==6144 and state['updates']==48 and state['training']
        assert state['pending'] is None and not state['buffer']
    for pair in range(3):
        for count in (16,32,48,64):
            states=torch.load(folder/f'pair-{pair}-after-{count}.pt',weights_only=True,map_location='cpu')
            assert set(states)=={f'c{pair*2}',f'c{pair*2+1}'}
            for cid,state in states.items():
                assert state['decisions']==6144+count*128 and state['updates']==48+count
                assert state['seed']==original[cid]['seed'] and state['config']==manifest['config']
                assert not state['buffer'] and state['pending'] is None and state['training']
            checkpoints[pair,count]=states
    saved_rows={r['episode']:r for name in ('training.json','results.json') for r in json.loads((folder/name).read_text())}
    assert len(saved_rows)==480
    ids=set();coordinates=set();starts={};training_brains={};training_counts={};rows=[]
    steps=training_decisions=frozen_decisions=updates=checkpoint_checks=0
    w=None
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            if steps%64==0:
                if time.perf_counter()>deadline:raise TimeoutError('Audit wall cap reached; evidence preserved')
                if (folder/'STOP').exists():raise InterruptedError('Audit STOP requested; evidence preserved')
            row=json.loads(line)
            if row['type']=='start':
                assert w is None and row['episode'] not in ids;ids.add(row['episode'])
                header=row;episode=row['episode'];meta=row['metadata']
                stage,arm,pair,case,fixture=(meta[k] for k in ('stage','arm','pair','case','fixture'))
                assert stage in ('training','evaluation') and pair in range(3)
                assert meta['training_seed']==manifest['seeds'][pair]
                assert meta['candidates']=={'r0':f'c{pair*2}','r1':f'c{pair*2+1}'}
                coordinate=(stage,arm,pair,case,fixture);assert coordinate not in coordinates;coordinates.add(coordinate)
                if stage=='training':assert arm=='practice' and fixture=='adjacent' and case in range(64)
                else:
                    assert arm in manifest['arms'] and fixture in manifest['fixtures'] and case in range(16)
                    assert sum(training_counts.values())==192,'Evaluation preceded completed fixed training'
                w=World.from_dict(copy.deepcopy(row['world']))
                expected_seed=manifest['bases']['training' if stage=='training' else fixture]+pair*1000+case
                assert w.tick==0 and w.seed==expected_seed and set(w.residents)=={'r0','r1'} and not w.ecology_enabled
                assert starts.setdefault((stage,pair,case,fixture),canonical(row['world']))==canonical(row['world'])
                if stage=='training':
                    assert case==training_counts.get(pair,0)
                    if pair not in training_brains:
                        training_brains[pair]={}
                        for i,rid in enumerate(('r0','r1')):
                            state=original[f'c{pair*2+i}'];b=Brain(state['seed'],config)
                            b.restore(copy.deepcopy(state));training_brains[pair][rid]=b
                    brains=training_brains[pair];training_counts[pair]=case+1
                else:
                    states=original if arm=='starting' else checkpoints[pair,int(arm.split('-')[1])]
                    brains={}
                    for i,rid in enumerate(('r0','r1')):
                        state=states[f'c{pair*2+i}'];b=Brain(state['seed'],config)
                        b.restore(copy.deepcopy(state));b.freeze();brains[rid]=b
                m={};positions={};counters={};frozen_state={}
                for i,(rid,b) in enumerate(brains.items()):
                    a=w.residents[rid]
                    assert a.food==4 and a.water==a.health==a.energy==a.warmth==100 and not a.unconscious
                    assert a.inventory['food']==(1 if fixture=='carried' else 0)
                    assert sum(a.inventory.values())==a.inventory['food']
                    if fixture=='adjacent':
                        loc=meta['locations'][rid]['food_at'];assert abs(a.x-loc[0])+abs(a.y-loc[1])==1
                        assert w.resources[w.key(*loc)]==dict(kind='berry',amount=4)
                    else:assert not w.resources
                    b.rng=random.Random(manifest['bases']['training' if stage=='training' else fixture]*10+pair*10000+case*2+i)
                    assert weight_hashes(b)==row['weights'][rid]
                    counters[rid]=(b.decisions,b.updates)
                    if stage=='evaluation':
                        frozen_state[rid]={k:fingerprint(b.state()[k]) for k in ('model','predictor','optimizer','predictor_optimizer')}
                        observation=w.observe(rid);patch,features=encode_observation(observation)
                        with torch.no_grad():
                            logits,_,_=b.model(patch,features,b.hidden)
                            probabilities=b.distribution(logits,torch.tensor([action_mask(observation,ACTIONS)])).probs[0].tolist()
                        assert probabilities==meta['choice_probes'][rid]
                    m[rid]=dict(first_meal_tick=None,first_gather_food_tick=None,first_zero_tick=None,
                        first_food_seen_tick=None,safe_eaten=0,amber_eaten=0,zero_food_ticks=0,unconscious_ticks=0,
                        nutrition=0.,damage=0.,invalid=0,conscious_invalid=0,conscious_decisions=0,
                        reward_totals={},actions={},successful_actions={},conscious_tones=0)
                    positions[rid]={(a.x,a.y)}
            elif row['type']=='step':
                assert row['episode']==episode and row['tick']==w.tick
                deciding=w.tick%4==0
                before={rid:(a.food,a.health,a.pain,a.unconscious,a.inventory['food']) for rid,a in w.residents.items()}
                if deciding:
                    for rid,b in brains.items():
                        observation=w.observe(rid);cmd=row['commands'][rid];s=m[rid]
                        assert canonical(observation)==canonical(row['observations'][rid])
                        assert action_mask(observation,ACTIONS)[ACTIONS.index(cmd)]
                        assert b.decide(observation)==cmd,(episode,w.tick,rid,'native policy diverged')
                        if stage=='training':training_decisions+=1
                        else:frozen_decisions+=1
                        visible=any(t['terrain']>=0 and t.get('resource',{}).get('kind') in ('berry','food')
                            and t['resource'].get('amount',0)>0 for t in observation['tiles'])
                        if visible and s['first_food_seen_tick'] is None:s['first_food_seen_tick']=w.tick
                        verb=cmd['verb'];s['actions'][verb]=s['actions'].get(verb,0)+1
                        s['conscious_tones']+=verb=='tone' and not observation['body']['unconscious']
                else:assert all(cmd=={'verb':'wait'} for cmd in row['commands'].values())
                actual=w.step(row['commands'],scripted=False)
                assert canonical(actual)==canonical(row['results'])
                for rid,a in w.residents.items():
                    s=m[rid];food,health,pain,unconscious,inventory=before[rid]
                    assert {k:getattr(a,k) for k in row['after'][rid]}==row['after'][rid]
                    decay=.008 if a.food>0 else min(.008,max(0.,food))
                    gain=max(0.,a.food-food+decay)
                    if gain<1e-7:gain=0.
                    injury=max(0.,health-a.health,a.pain-pain)
                    reward=dict(food=gain/25,hunger=-min(1.,max(0.,(60-a.food)/60))*.005,injury=-injury/20)
                    assert reward==row['rewards'][rid],(episode,row['tick'],rid,'reward mismatch')
                    brains[rid].add_reward(reward)
                    for key,value in reward.items():s['reward_totals'][key]=s['reward_totals'].get(key,0.)+value
                    s['nutrition']+=gain;s['damage']+=injury;s['zero_food_ticks']+=a.food<=0;s['unconscious_ticks']+=a.unconscious
                    if a.food<=0 and s['first_zero_tick'] is None:s['first_zero_tick']=row['tick']
                    positions[rid].add((a.x,a.y))
                    if deciding:
                        cmd=row['commands'][rid];verb=cmd['verb'];success=actual[rid][0];s['invalid']+=not success
                        if not unconscious:s['conscious_decisions']+=1;s['conscious_invalid']+=not success
                        if success:
                            s['successful_actions'][verb]=s['successful_actions'].get(verb,0)+1
                            if verb=='gather' and a.inventory['food']>inventory and s['first_gather_food_tick'] is None:s['first_gather_food_tick']=row['tick']
                            if verb=='eat':
                                s['amber_eaten' if cmd.get('item')=='amber_fruit' else 'safe_eaten']+=1
                                if cmd.get('item','food')=='food' and s['first_meal_tick'] is None:s['first_meal_tick']=row['tick']
                steps+=1
            else:
                assert row['type']=='end' and row['episode']==episode
                assert w.tick==row['ticks']==manifest['horizons'][fixture]
                assert canonical(w.to_dict())==canonical(row['world'])
                saved=dict(row);saved.pop('type');saved.pop('world');assert saved==saved_rows[episode]
                for rid,b in brains.items():
                    b.finish(w.observe(rid),terminal=True)
                    assert weight_hashes(b)==row['weights_after'][rid]
                    old_decisions,old_updates=counters[rid]
                    assert b.decisions-old_decisions==row['timings'][rid]['decisions']==w.tick//4
                    assert b.updates-old_updates==row['timings'][rid]['updates']==(1 if stage=='training' else 0)
                    updates+=b.updates-old_updates
                    if stage=='training' and case+1 in (16,32,48,64):
                        expected=checkpoints[pair,case+1][meta['candidates'][rid]]
                        assert fingerprint(b.state())==fingerprint(expected),(episode,rid,'full state mismatch')
                        checkpoint_checks+=1
                    if stage=='evaluation':
                        assert frozen_state[rid]=={k:fingerprint(b.state()[k]) for k in frozen_state[rid]}
                        assert row['weights_after'][rid]==header['weights'][rid]
                    s=m[rid];meal,gather,zero=(s[k] for k in ('first_meal_tick','first_gather_food_tick','first_zero_tick'))
                    s['timely_acquisition']=meal is not None and gather is not None and gather<meal and (zero is None or meal<zero)
                    s['unique_tiles']=len(positions[rid])
                    for key,value in s.items():
                        measured=row['metrics'][rid][key]
                        if isinstance(value,float):assert abs(value-measured)<1e-6,(episode,rid,key)
                        elif key=='reward_totals':assert all(abs(v-measured[k])<1e-6 for k,v in value.items())
                        else:assert value==measured,(episode,rid,key)
                rows.append(saved);w=None
                if len(ids)%16==0:print(json.dumps(dict(audited_episodes=len(ids),ticks=steps,
                    seconds=round(time.perf_counter()-started,2))),flush=True)
    assert w is None and ids==set(saved_rows)
    assert (steps,training_decisions,frozen_decisions,updates,checkpoint_checks)==(181248,49152,41472,384,24)
    assert training_counts=={0:64,1:64,2:64}
    expected_coordinates={('training','practice',p,c,'adjacent') for p in range(3) for c in range(64)} | {
        ('evaluation',a,p,c,f) for a in manifest['arms'] for p in range(3) for c in range(16) for f in manifest['fixtures']}
    assert coordinates==expected_coordinates
    # Recount reported groups and the advancement rule without runner helpers.
    for fixture in manifest['fixtures']:
        section=receipt['result']['fixtures'][fixture]
        groups=[(section['arms'],lambda meta,rid:True)]
        groups += [(value,lambda meta,rid,p=int(pair):meta['pair']==p) for pair,value in section['seed_groups'].items()]
        groups += [(value,lambda meta,rid,c=cid:meta['candidates'][rid]==c) for cid,value in section['candidates'].items()]
        for reported,predicate in groups:
            for arm,score in reported.items():
                items=[(r,rid) for r in rows if r['metadata']['stage']=='evaluation'
                    and r['metadata']['fixture']==fixture and r['metadata']['arm']==arm
                    for rid in ('r0','r1') if predicate(r['metadata'],rid)]
                assert len(items)==score['lives']
                assert sum(prompt(r['metrics'][rid],fixture) for r,rid in items)==score['prompt_successes']
                for target,key in (('ordinary_meals','safe_eaten'),('amber_meals','amber_eaten'),
                    ('zero_food_ticks','zero_food_ticks'),('unconscious_ticks','unconscious_ticks')):
                    assert sum(r['metrics'][rid][key] for r,rid in items)==score[target]
                assert sum(r['ticks'] for r,rid in items)==score['resident_ticks']
                assert sum(r['metrics'][rid]['unique_tiles'] for r,rid in items)/len(items)==score['mean_unique_tiles']
                for target,key in (('actions','actions'),('successful_actions','successful_actions'),('reward_totals','reward_totals')):
                    counts={}
                    for r,rid in items:
                        for k,v in r['metrics'][rid][key].items():counts[k]=counts.get(k,0)+v
                    assert counts==score[target]
    a=receipt['result']['fixtures']['adjacent'];c=receipt['result']['fixtures']['carried']
    rate=lambda score:score['prompt_successes']/score['lives']
    deprivation=lambda score:score['zero_food_ticks']/score['resident_ticks']
    long=a['arms']['after-64']
    checks=dict(long_prompt_at_least_80_percent=rate(long)>=.8,
        long_exceeds_16_by_10_points=rate(long)-rate(a['arms']['after-16'])>=.1-1e-12,
        long_exceeds_start_by_10_points=rate(long)-rate(a['arms']['starting'])>=.1-1e-12,
        positive_long_minus_16_in_at_least_two_seed_groups=sum(rate(g['after-64'])>rate(g['after-16']) for g in a['seed_groups'].values())>=2,
        carried_regression_no_more_than_5_points=all(rate(c['arms']['after-64'])-rate(c['arms'][arm])>=-.05-1e-12 for arm in ('starting','after-16')),
        deprivation_increase_no_more_than_half_point=all(deprivation(long)-deprivation(a['arms'][arm])<=.005+1e-12 for arm in ('starting','after-16')))
    assert checks==receipt['result']['checks'] and all(checks.values())==receipt['result']['pilot_promising']
    assert receipt['result']['milestone_confirmed'] is False
    for name,h in manifest['source_hashes'].items():assert digest(source/name)==h
    for name,h in manifest['hashes'].items():assert digest(ROOT/name)==h
    return dict(passed=True,seconds=time.perf_counter()-started,episodes=len(ids),replayed_ticks=steps,
        replayed_training_decisions=training_decisions,replayed_frozen_decisions=frozen_decisions,
        reproduced_updates=updates,full_checkpoint_states=checkpoint_checks,checks=checks,
        shared_code='Native simulator and brain, observation encoder/mask and full-state fingerprint; rollout, reward reconstruction and outcome recount are separate implementations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs') or folder==ROOT/'runs':parser.error('Use a local experiment directory')
    try:
        result=audit(folder);(folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
    except Exception as exc:
        (folder/'audit-failed.json').write_text(json.dumps(dict(error=type(exc).__name__,reason=str(exc))),encoding='utf-8')
        raise
