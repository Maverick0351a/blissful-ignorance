"""Separate native replay and numerical audit of the feeding-update diagnostic."""
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
from agents.network import ACTIONS,encode_observation
from agents.sequence_ppo import Config
from experiments.category_trial import population_for,sha,write,bodies,weight_hashes
from experiments.near_food_trial import make_world
from experiments.feeding_credit import fingerprint,gradient_components,summarize
from sim.world import World


def same(left,right):
    assert json.dumps(left,sort_keys=True)==json.dumps(right,sort_keys=True),'Recorded data mismatch'


def near(actual,expected,tolerance=3e-5):
    if isinstance(actual,dict):
        assert set(actual)==set(expected)
        for key in actual:near(actual[key],expected[key],tolerance)
    elif isinstance(actual,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):near(a,b,tolerance)
    elif isinstance(actual,(float,int)) and not isinstance(actual,bool):
        assert abs(actual-expected)<=tolerance*max(1,abs(expected)),(actual,expected)
    else:assert actual==expected,(actual,expected)


def reference_gae(rewards,values,bootstrap,terminals,gamma,lam):
    """Independent scalar recurrence, including terminal and bootstrap handling."""
    out=[0.]*len(values); following=bootstrap; accumulated=0.
    for i in range(len(values)-1,-1,-1):
        alive=0. if terminals[i] else 1.
        delta=rewards[i]+gamma*following*alive-values[i]
        accumulated=delta+gamma*lam*alive*accumulated
        out[i]=accumulated;following=values[i]
    return out


def sequence(brain,rows):
    with torch.no_grad():
        logits,_=brain._sequence(torch.cat([r['patch'] for r in rows]),torch.cat([r['features'] for r in rows]),rows[0]['hidden'])
        return brain.distribution(logits,torch.cat([r['mask'] for r in rows])).probs


def evaluate(brain,entry,zero=True):
    hidden=brain.model.initial_state() if zero else entry['hidden']
    with torch.no_grad():
        logits,_,_=brain.model(entry['patch'],entry['features'],hidden)
        probs=brain.distribution(logits,entry['mask']).probs[0]
    return {name:sum(float(probs[i]) for i,a in enumerate(ACTIONS) if
                    (a=={'verb':'eat'} if name=='eat' else a['verb']==name))
            for name in ('eat','gather','tone')}


def audit(folder,wall_seconds=600):
    start=time.perf_counter();deadline=start+wall_seconds
    receipt=json.loads((folder/'complete.json').read_text());manifest=json.loads((folder/'preregistration.json').read_text())
    assert manifest['diagnostic']=='recorded-feeding-update'
    assert sha(folder/'preregistration.json')==(folder/'preregistration.sha256').read_text().strip()
    for name,digest in receipt['evidence_hashes'].items():assert sha(folder/name)==digest,name
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name,digest in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
            assert sha(ROOT/name)==digest,'Source changed: '+name
    original_manifest=json.loads((folder/'source-preregistration.json').read_text())
    original_receipt=json.loads((folder/'source-complete.json').read_text())
    assert sha(folder/'source-complete.json')==manifest['source_complete_sha256']
    assert sha(folder/'source-audit.json')==manifest['source_audit_sha256']
    assert json.loads((folder/'source-audit.json').read_text())['passed']
    for name in ('continue-initial.pt','continue-trained.pt','preregistration.json'):
        assert sha(folder/('source-'+name))==original_receipt['evidence_hashes'][name]
    origin=(ROOT/manifest['source']).resolve()
    assert origin.is_relative_to(ROOT/'runs') and sha(origin/'trace.jsonl.gz')==manifest['source_trace_sha256']
    # Prove the retained trace is exactly the original continuation subset.
    original_digest=hashlib.sha256();subset_digest=hashlib.sha256();active=False
    with gzip.open(origin/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['type']=='start':active=row['metadata']['stage']=='training' and row['metadata']['arm']=='continue'
            if active:original_digest.update(line.encode())
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:subset_digest.update(line.encode())
    assert original_digest.digest()==subset_digest.digest(),'Changed or omitted trace rows'
    initial=torch.load(folder/'source-continue-initial.pt',weights_only=True,map_location='cpu')
    final=torch.load(folder/'source-continue-trained.pt',weights_only=True,map_location='cpu')
    reproduced=torch.load(folder/'reproduced-final.pt',weights_only=True,map_location='cpu')
    records=json.loads((folder/'updates.json').read_text());index={(r['candidate'],r['case']):r for r in records}
    assert len(index)==len(records)==96
    config=Config(**manifest['config']);populations={};anchors={};probes={};seen=set()
    steps=decisions=episodes=meals=positive=reinforced=0
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            if steps%64==0:
                if time.perf_counter()>deadline:raise TimeoutError('Audit wall cap reached')
                if (folder/'STOP').exists():raise InterruptedError('STOP requested')
            row=json.loads(line)
            if row['type']=='start':
                meta=row['metadata'];pair=meta['pair'];w=World.from_dict(row['world'])
                if pair not in populations:
                    p=population_for(meta['training_seed'],'category',w,config)
                    for i,(rid,b) in enumerate(p.brains.items()):
                        cid=f'c{pair*2+i}';b.restore(copy.deepcopy(initial[cid]));inputs={}
                        for fixture in ('adjacent','carried'):
                            for case in range(4):
                                obs=make_world(manifest['probes_base']+case,case,fixture)[0].observe(rid)
                                patch,features=encode_observation(obs)
                                inputs[f'{fixture}-{case}']=dict(patch=patch,features=features,
                                    mask=torch.tensor([action_mask(obs,ACTIONS)]))
                        probes[cid]=inputs
                    populations[pair]=p
                p=populations[pair];p.world=w;p.metrics={rid:p.new_metrics() for rid in p.brains}
                assert meta['case']==sum(1 for cid,case in seen if cid==f'c{pair*2}')
                for i,(rid,b) in enumerate(p.brains.items()):
                    b.rng=random.Random(original_manifest['bases']['training']*10+meta['case']*6+pair*2+i)
                    b.reward_totals={}
                same({rid:weight_hashes(b) for rid,b in p.brains.items()},row['weights'])
            elif row['type']=='step':
                assert row['tick']==p.world.tick
                before=bodies(p.world);deciding=p.world.tick%4==0;result=p.step()
                same(result,row['results']);same(bodies(p.world),row['after'])
                for rid,b in p.brains.items():
                    command=b.last_action if deciding else {'verb':'wait'};same(command,row['commands'][rid])
                    if deciding:
                        same(b.last_observation,row['observations'][rid])
                        assert action_mask(b.last_observation,ACTIONS)[ACTIONS.index(command)]
                    a=p.world.residents[rid];old=before[rid]
                    gain=max(0.,a.food+.008-old['food']) if a.food>0 else 0.
                    if gain<1e-7:gain=0.
                    reward=dict(food=gain/25,hunger=-min(1.,max(0.,(60-a.food)/60))*.005,
                        injury=-max(0.,old['health']-a.health,a.pain-old['pain'])/20)
                    near(reward,row['rewards'][rid],1e-9);near(reward,b.last_reward,1e-9)
                steps+=1;decisions+=len(p.brains) if deciding else 0
            elif row['type']=='end':
                for rid,b in p.brains.items():
                    cid=meta['candidates'][rid];r=index[cid,meta['case']];seen.add((cid,meta['case']))
                    b._complete(p.world.observe(rid),terminal=True)
                    assert fingerprint(b.state())==r['state_before']
                    rows=b.buffer;journal=b.journal[-len(rows):];n=len(rows);assert n==128
                    vals=[t['value'] for t in rows];terms=[t['terminal'] for t in rows]
                    adv=reference_gae([t['reward'] for t in rows],vals,r['bootstrap'],terms,config.gamma,config.gae_lambda)
                    average=sum(adv)/n;std=(sum((v-average)**2 for v in adv)/n)**.5
                    norm=[(v-average)/(std+1e-8) for v in adv];returns=[a+v for a,v in zip(adv,vals)]
                    components={key:reference_gae([j['reward'].get(key,0) for j in journal],[0.]*n,0.,terms,
                                config.gamma,config.gae_lambda) for key in ('food','hunger','injury')}
                    components['value_residual']=reference_gae([0.]*n,vals,r['bootstrap'],terms,config.gamma,config.gae_lambda)
                    probs_before=sequence(b,rows)
                    meal_rows=[i for i,t in enumerate(rows) if ACTIONS[t['chosen']]=={'verb':'eat'} and journal[i]['reward']['food']>0]
                    first=cid not in anchors and bool(meal_rows);assert first==r['anchor_first']
                    if first:
                        k=meal_rows[0];anchors[cid]={name:copy.deepcopy(rows[k][name]) for name in ('patch','features','mask','hidden')}
                        same(r['anchor_identity'],dict(decision=journal[k]['decision'],case=meta['case'],row=k,
                                                     fullness=float(rows[k]['body'][0])*25))
                    near({k:evaluate(b,v) for k,v in probes[cid].items()},r['canonical_before'],2e-6)
                    if cid in anchors:near({k:evaluate(b,anchors[cid],k=='zero_context') for k in ('zero_context','fixed_context')},r['anchor_before'],2e-6)
                    near(gradient_components(b,rows,torch.tensor(norm),torch.tensor(returns)),r['gradient'],2e-4)
                    b._learn(r['bootstrap'])
                    assert fingerprint(b.state())==r['state_after']
                    same(b.diagnostics,r['learner_diagnostics'])
                    probs_after=sequence(b,rows)
                    near({k:evaluate(b,v) for k,v in probes[cid].items()},r['canonical_after'],2e-6)
                    if cid in anchors:near({k:evaluate(b,anchors[cid],k=='zero_context') for k in ('zero_context','fixed_context')},r['anchor_after'],2e-6)
                    for i,t in enumerate(rows):
                        logged=r['transitions'][i];chosen=t['chosen'];eat=ACTIONS.index({'verb':'eat'})
                        same(logged['action'],ACTIONS[chosen]);same(logged['reward_parts'],journal[i]['reward'])
                        near([logged['advantage'],logged['normalized_advantage'],logged['target_return']],[adv[i],norm[i],returns[i]])
                        near(logged['components'],{k:v[i] for k,v in components.items()})
                        near([logged['chosen_probability_before'],logged['chosen_probability_after'],logged['eat_probability_before'],logged['eat_probability_after']],
                             [float(probs_before[i,chosen]),float(probs_after[i,chosen]),float(probs_before[i,eat]),float(probs_after[i,eat])],2e-6)
                        assert logged['ordinary_meal']==(i in meal_rows)
                        if i in meal_rows:
                            meals+=1;positive+=norm[i]>0
                            reinforced+=float(probs_after[i,eat])-float(probs_before[i,eat])>1e-6
                    b.pending=None;b.hidden=b.model.initial_state()
                same(p.world.to_dict(),row['world']);same({rid:weight_hashes(b) for rid,b in p.brains.items()},row['weights_after'])
                episodes+=1
                if episodes%8==0:print(json.dumps(dict(audited_episodes=episodes,seconds=time.perf_counter()-start)),flush=True)
            else:raise AssertionError('Unknown row type')
    for pair,p in populations.items():
        for i,b in enumerate(p.brains.values()):
            cid=f'c{pair*2+i}';assert fingerprint(b.state())==fingerprint(final[cid])==fingerprint(reproduced[cid])
    assert len(seen)==96 and (episodes,steps,decisions)==(48,24576,12288)
    analysis=json.loads((folder/'analysis.json').read_text());same(summarize(records),analysis)
    assert (analysis['ordinary_meals'],analysis['positive_meal_advantages'],analysis['reinforced_meals'])==(meals,positive,reinforced)
    return dict(passed=True,seconds=time.perf_counter()-start,episodes=episodes,replayed_ticks=steps,
        replayed_training_decisions=decisions,updates=len(seen),ordinary_meals=meals,
        positive_meal_advantages=positive,reinforced_meals=reinforced,exact_final_states=True,
        checks=['source and evidence hashes','exact original continuation trace subset','native unscripted training replay',
                'private observations, masks, physics and rewards','scalar GAE and additive credit decomposition',
                'matched pre/post sequence and fixed-observation probabilities','full optimizer, model, history and RNG state equality',
                'first-epoch gradient recomputation (shared measurement helper)','chronological anchor selection and summary recount'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--wall-seconds',type=int,default=600);args=parser.parse_args();folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs') or folder==ROOT/'runs':parser.error('Use a local run directory')
    if not 1<=args.wall_seconds<=600:parser.error('Wall cap must be 1..600 seconds')
    try:
        result=audit(folder,args.wall_seconds);write(folder/'audit.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        write(folder/'audit-failed.json',dict(error=type(exc).__name__,reason=str(exc)));raise
