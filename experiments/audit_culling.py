"""Separate replay and retirement audit for the archived restart comparison."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from agents.affordances import action_mask
from agents.network import ACTIONS
from agents.sequence_ppo import Config
from experiments.category_trial import MeasuredCategory
from sim.world import World


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def weights(state):
    return {name:hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in state[name].values())).hexdigest()
            for name in ('model','predictor')}


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def prompt(m):
    meal,gather,zero=(m[k] for k in ('first_meal_tick','first_gather_food_tick','first_zero_tick'))
    return meal is not None and meal<64 and gather is not None and gather<meal and (zero is None or meal<zero)


def audit(folder,wall_seconds):
    started=time.perf_counter();deadline=started+wall_seconds
    manifest=json.loads((folder/'preregistration.json').read_text())
    receipt=json.loads((folder/'complete.json').read_text())
    assert digest(folder/'preregistration.json')==(folder/'preregistration.sha256').read_text().strip()
    for name,h in receipt['evidence_hashes'].items():assert digest(folder/name)==h,name
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name,h in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==h and digest(ROOT/name)==h,name
    source=ROOT/manifest['source_run']
    assert source.resolve().is_relative_to(ROOT/'runs')
    assert digest(source/'complete.json')==manifest['source_complete_sha256']
    assert digest(source/'audit.json')==manifest['source_audit_sha256']
    original={};checkpoints={};config=Config(**manifest['config'])
    for pair,seed in enumerate(manifest['seeds']):
        name=f'archive-{seed}.pt'
        assert digest(folder/name)==manifest['archive_hashes'][name]==digest(source/f'{seed}-category-trained.pt')
        states=torch.load(folder/name,weights_only=True,map_location='cpu')
        for i,rid in enumerate(('r0','r1')):
            s=states[rid];assert s['decisions']==4096 and s['updates']==32
            assert s['backend']=='experimental-category-ppo' and s['config']==manifest['config']
            original[f'c{pair*2+i}']=s
    decision=json.loads((folder/'retirement.json').read_text())
    assert decision['live_population_changed'] is False
    for arm in manifest['arms']:
        for variant in ('initial','trained'):
            checkpoints[arm,variant]=torch.load(folder/f'{arm}-{variant}.pt',weights_only=True,map_location='cpu')
            assert set(checkpoints[arm,variant])==set(original)
        retired=sorted(decision['retired'][arm])
        for cid,state in checkpoints[arm,'initial'].items():
            if cid not in retired:assert equal(state,original[cid]),'Survivor history changed'
            else:
                b=MeasuredCategory(manifest['fresh_seeds'][retired.index(cid)],config)
                assert equal(state,b.state()),'Replacement is not the declared fresh private brain'
            final=checkpoints[arm,'trained'][cid]
            assert final['decisions']-state['decisions']==manifest['training_lives']*128
            assert final['updates']-state['updates']==manifest['training_lives']
            assert final['seed']==state['seed'] and not final['buffer'] and final['pending'] is None
    saved_rows={r['episode']:r for name in ('selection.json','training.json','results.json')
                for r in json.loads((folder/name).read_text())}
    ids=set();starts={};last_weights={};training_counts={};rows=[];w=None
    steps=decisions=0
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            if steps%64==0 and time.perf_counter()>deadline:raise TimeoutError('Audit wall cap reached; evidence preserved')
            row=json.loads(line)
            if row['type']=='start':
                assert w is None and row['episode'] not in ids;ids.add(row['episode'])
                header=row;episode=row['episode'];meta=row['metadata'];stage=meta['stage'];arm=meta['arm']
                assert stage in ('selection','training','evaluation') and meta['fixture']=='adjacent'
                assert arm==('source' if stage=='selection' else arm) and (stage=='selection' or arm in manifest['arms'])
                assert meta['training_seed']==manifest['seeds'][meta['pair']]
                assert meta['candidates']=={rid:f'c{meta["pair"]*2+i}' for i,rid in enumerate(('r0','r1'))}
                w=World.from_dict(copy.deepcopy(row['world']));assert w.tick==0 and set(w.residents)=={'r0','r1'}
                assert w.seed==manifest['bases'][stage]+meta['case']
                assert starts.setdefault((stage,meta['case']),canonical(row['world']))==canonical(row['world'])
                m={};positions={};brains={}
                for rid,a in w.residents.items():
                    assert a.food==4 and not any(a.inventory.values()) and a.health==a.energy==a.water==a.warmth==100
                    assert action_mask(w.observe(rid),ACTIONS)[ACTIONS.index({'verb':'gather'})]
                    m[rid]=dict(first_meal_tick=None,first_gather_food_tick=None,first_zero_tick=None,
                        safe_eaten=0,amber_eaten=0,zero_food_ticks=0,unconscious_ticks=0,invalid=0,
                        nutrition=0.,damage=0.,reward_totals={},actions={},successful_actions={},conscious_tones=0)
                    positions[rid]={(a.x,a.y)}
                    cid=meta['candidates'][rid]
                    expected_identity=cid
                    if stage!='selection' and cid in decision['retired'][arm]:
                        expected_identity=f'{arm}-new-{sorted(decision["retired"][arm]).index(cid)}'
                    assert meta['identities'][rid]==expected_identity
                    state=original[cid] if stage=='selection' else checkpoints[arm,'trained' if stage=='evaluation' else 'initial'][cid]
                    if stage!='training':
                        b=MeasuredCategory(state['seed'],config);b.restore(copy.deepcopy(state));b.freeze()
                        b.rng=random.Random(manifest['bases'][stage]*10+meta['case']*6+int(cid[1:]))
                        brains[rid]=b;assert weights(state)==row['weights'][rid]
                    else:
                        key=(arm,cid);assert meta['case']==training_counts.get(key,0)
                        assert row['weights'][rid]==last_weights.get(key,weights(state))
                        training_counts[key]=training_counts.get(key,0)+1
            elif row['type']=='step':
                assert row['episode']==episode and row['tick']==w.tick
                deciding=w.tick%4==0
                before={rid:(a.food,a.health,a.pain,a.inventory['food']) for rid,a in w.residents.items()}
                if deciding:
                    for rid in w.residents:
                        obs=w.observe(rid);cmd=row['commands'][rid]
                        assert canonical(obs)==canonical(row['observations'][rid])
                        assert action_mask(obs,ACTIONS)[ACTIONS.index(cmd)]
                        if brains:assert brains[rid].decide(obs)==cmd;decisions+=1
                        verb=cmd['verb'];s=m[rid];s['actions'][verb]=s['actions'].get(verb,0)+1
                        s['conscious_tones']+=verb=='tone' and not obs['body']['unconscious']
                else:assert all(cmd=={'verb':'wait'} for cmd in row['commands'].values())
                result=w.step(row['commands'],scripted=False);assert canonical(result)==canonical(row['results'])
                for rid,a in w.residents.items():
                    s=m[rid];food,health,pain,inventory=before[rid]
                    assert {k:getattr(a,k) for k in row['after'][rid]}==row['after'][rid]
                    gain=max(0.,a.food+.008-food) if a.food>0 else 0.
                    if gain<1e-7:gain=0.
                    injury=max(0.,health-a.health,a.pain-pain)
                    reward=dict(food=gain/25,hunger=-min(1.,max(0.,(60-a.food)/60))*.005,injury=-injury/20)
                    assert set(row['rewards'][rid])==set(reward)
                    for key,value in reward.items():
                        assert abs(value-row['rewards'][rid][key])<1e-9
                        s['reward_totals'][key]=s['reward_totals'].get(key,0.)+value
                    s['nutrition']+=gain;s['damage']+=injury;s['zero_food_ticks']+=a.food<=0;s['unconscious_ticks']+=a.unconscious
                    if a.food<=0 and s['first_zero_tick'] is None:s['first_zero_tick']=row['tick']
                    positions[rid].add((a.x,a.y))
                    if deciding:
                        cmd=row['commands'][rid];verb=cmd['verb'];success=result[rid][0];s['invalid']+=not success
                        if success:
                            s['successful_actions'][verb]=s['successful_actions'].get(verb,0)+1
                            if verb=='gather' and a.inventory['food']>inventory and s['first_gather_food_tick'] is None:
                                s['first_gather_food_tick']=row['tick']
                            if verb=='eat':
                                s['amber_eaten' if cmd.get('item')=='amber_fruit' else 'safe_eaten']+=1
                                if cmd.get('item','food')=='food' and s['first_meal_tick'] is None:s['first_meal_tick']=row['tick']
                steps+=1
            else:
                assert row['type']=='end' and row['episode']==episode and w.tick==512
                assert canonical(w.to_dict())==canonical(row['world'])
                saved=dict(row);saved.pop('type');saved.pop('world');assert saved==saved_rows[episode]
                for rid,s in m.items():
                    meal,gather,zero=(s[k] for k in ('first_meal_tick','first_gather_food_tick','first_zero_tick'))
                    s['timely_acquisition']=meal is not None and gather is not None and gather<meal and (zero is None or meal<zero)
                    s['unique_tiles']=len(positions[rid])
                    for key,value in s.items():
                        actual=row['metrics'][rid][key]
                        if isinstance(value,float):assert abs(actual-value)<1e-6,(episode,rid,key)
                        elif key=='reward_totals':assert all(abs(actual[k]-v)<1e-6 for k,v in value.items())
                        else:assert actual==value,(episode,rid,key)
                    if stage=='training':last_weights[arm,meta['candidates'][rid]]=row['weights_after'][rid]
                    else:assert row['weights_before'][rid]==row['weights_after'][rid]==header['weights'][rid]==weights(brains[rid].state())
                rows.append(saved);w=None
    assert w is None and len(ids)==len(saved_rows) and steps==manifest['total_ticks']==receipt['total_ticks']
    for arm in manifest['arms']:
        for cid in original:
            assert training_counts[arm,cid]==manifest['training_lives']
            assert last_weights[arm,cid]==weights(checkpoints[arm,'trained'][cid])
    # Recount selection and the gate without calling runner selection helpers.
    screens={cid:dict(round_successes=[0,0],round_lives=[0,0],full_life=0,ordinary_meals=0) for cid in original}
    for row in rows:
        meta=row['metadata']
        if meta['stage']!='selection':continue
        index=meta['case']//manifest['cases_per_round']
        for rid,m in row['metrics'].items():
            s=screens[meta['candidates'][rid]];s['round_successes'][index]+=prompt(m);s['round_lives'][index]+=1
            s['full_life']+=m['timely_acquisition'];s['ordinary_meals']+=m['safe_eaten']
    for s in screens.values():
        assert s['round_lives']==[manifest['cases_per_round']]*2
        s['eligible']=all(n/manifest['cases_per_round']<.8 for n in s['round_successes'])
    assert screens==decision['scores']
    chosen=sorted((cid for cid,s in screens.items() if s['eligible']),key=lambda cid:(
        sum(screens[cid]['round_successes']),screens[cid]['full_life'],screens[cid]['ordinary_meals'],cid))[:2]
    assert chosen==decision['retired']['selected-restart'] and decision['retired']['continue']==[]
    assert sorted(random.Random(manifest['random_selection_seed']).sample(sorted(original),len(chosen)))==decision['retired']['random-restart']
    summaries={}
    for arm in manifest['arms']:
        chosen_rows=[r for r in rows if r['metadata']['stage']=='evaluation' and r['metadata']['arm']==arm]
        values=[m for r in chosen_rows for m in r['metrics'].values()]
        n=len(values);assert n==6*manifest['evaluation_cases']
        summaries[arm]=dict(lives=n,prompt_successes=sum(prompt(m) for m in values),
            full_life_successes=sum(m['timely_acquisition'] for m in values),
            ordinary_meals=sum(m['safe_eaten'] for m in values),amber_meals=sum(m['amber_eaten'] for m in values),
            zero_food_ticks=sum(m['zero_food_ticks'] for m in values),unconscious_ticks=sum(m['unconscious_ticks'] for m in values),
            halves=[sum(prompt(m) for r in chosen_rows if (r['metadata']['case']<manifest['evaluation_cases']//2)==(h==0)
                     for m in r['metrics'].values()) for h in (0,1)])
    assert summaries==receipt['result']['arms']
    selected=summaries['selected-restart'];n=selected['lives']
    passed=selected['prompt_successes']/n>=.8 and all(
        (selected['prompt_successes']-summaries[arm]['prompt_successes'])/n>=.1-1e-12 and all(
        (selected['halves'][h]-summaries[arm]['halves'][h])/(n/2)>=.1-1e-12 for h in (0,1))
        for arm in ('continue','random-restart'))
    assert receipt['result']['pilot_promising']==passed and receipt['result']['milestone_confirmed'] is False
    return dict(passed=True,seconds=time.perf_counter()-started,episodes=len(ids),replayed_ticks=steps,
        replayed_frozen_decisions=decisions,retirements=decision['retired'],
        checks=['source/archive integrity','private retained and fresh states','new identities',
                'matched worlds and unique sampling coordinates','physics, private observations, masks and rewards',
                'all frozen selection/evaluation decisions','equal continuation budget and training weight continuity',
                'two-round selection, random control and independent outcome gate recount'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--wall-seconds',type=int,default=600);args=parser.parse_args();folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local run directory')
    try:
        result=audit(folder,args.wall_seconds)
        (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    except Exception as exc:
        (folder/'audit-failed.json').write_text(json.dumps(dict(error=type(exc).__name__,reason=str(exc))),encoding='utf-8')
        raise
