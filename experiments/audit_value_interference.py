"""Audit the paired intervention using an explicit detached recurrent unroll.

This verifier does not use the runner's temporary value-head hook. Trace replay
and hashing infrastructure are shared; the intervention graph and effect recount
are implemented separately.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from agents.sequence_ppo import Config
from experiments.category_trial import MeasuredCategory,sha,write
from experiments.feeding_credit import fingerprint,EAT
from experiments.value_interference import replay,same


class ExplicitDetached(MeasuredCategory):
    """Only the value branch's input is detached; no hooks or copied PPO loop."""
    def _sequence(self,patch,features,hidden):
        encoded=torch.cat((self.model.patch_encoder(patch),self.model.feature_encoder(features)),-1)
        logits=[];values=[]
        for entry in encoded:
            hidden=self.model.recurrent(entry[None],hidden)
            logits.append(self.model.actor(hidden[0]))
            values.append(self.model.value(hidden[0].detach()).squeeze(-1))
        return torch.cat(logits),torch.cat(values)


def seq(brain,rows):
    with torch.no_grad():
        logits,values=brain._sequence(torch.cat([r['patch'] for r in rows]),torch.cat([r['features'] for r in rows]),rows[0]['hidden'])
        return brain.distribution(logits,torch.cat([r['mask'] for r in rows])).probs,values


def probes(brain,inputs):
    from agents.network import ACTIONS
    result={}
    with torch.no_grad():
        for key,entry in inputs.items():
            logits,_,_=brain.model(entry['patch'],entry['features'],brain.model.initial_state())
            probabilities=brain.distribution(logits,entry['mask']).probs[0]
            result[key]={verb:float(probabilities[[i for i,a in enumerate(ACTIONS)
                if (a=={'verb':'eat'} if verb=='eat' else a['verb']==verb)]].sum()) for verb in ('eat','gather','tone')}
    return result


def audit(folder,wall_seconds=600):
    start=time.perf_counter();deadline=start+wall_seconds
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    protocol=read('preregistration.json');receipt=read('complete.json')
    assert protocol['diagnostic']=='matched-value-gradient-interference'
    assert sha(folder/'preregistration.json')==(folder/'preregistration.sha256').read_text().strip()
    for name,value in receipt['evidence_hashes'].items():assert sha(folder/name)==value,name
    for name,value in protocol['parent_files'].items():assert sha(folder/name)==value,name
    assert read('parent-audit.json')['passed']
    parent_receipt=read('parent-complete.json')
    for name in ('trace.jsonl.gz','source-continue-initial.pt','source-continue-trained.pt','source-preregistration.json'):
        assert sha(folder/name)==parent_receipt['evidence_hashes'][name]
    assert sha(folder/'parent-updates.json')==parent_receipt['evidence_hashes']['updates.json']
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name,value in protocol['hashes'].items():
            assert sha(ROOT/name)==hashlib.sha256(archive.read(name)).hexdigest()==value,name
    saved=read('pairs.json');index={(r['candidate'],r['case']):r for r in saved}
    assert len(index)==len(saved)==96
    expected_shadows=torch.load(folder/'last-independent-shadows.pt',weights_only=True,map_location='cpu')
    observed={};counter=0
    def callback(brain,reference,inputs):
        nonlocal counter
        cid=reference['candidate'];record=index[cid,reference['case']]
        original=copy.deepcopy(brain.state());rows=brain.buffer
        assert fingerprint(original)==record['state_before']==reference['state_before']
        shadow=ExplicitDetached(original['seed'],Config(**original['config']));shadow.restore(copy.deepcopy(original))
        assert fingerprint(shadow.state())==fingerprint(original)
        before,before_values=seq(brain,rows);same(probes(brain,inputs),record['probes_before'])
        shadow._learn(reference['bootstrap'])
        assert fingerprint(brain.state())==fingerprint(original)
        brain._learn(reference['bootstrap'])
        assert fingerprint(brain.state())==record['state_standard']==reference['state_after']
        assert fingerprint(shadow.state())==record['state_detached']
        standard,standard_values=seq(brain,rows);detached,detached_values=seq(shadow,rows)
        same(probes(brain,inputs),record['probes_standard']);same(probes(shadow,inputs),record['probes_detached'])
        same(brain.diagnostics,record['diagnostics_standard']);same(shadow.diagnostics,record['diagnostics_detached'])
        targets=torch.tensor([t['target_return'] for t in reference['transitions']])
        for arm,values in (('before',before_values),('standard',standard_values),('detached',detached_values)):
            assert float((values-targets).square().mean())==record['value_mse'][arm]
        assert len(record['transitions'])==128
        for i,t in enumerate(record['transitions']):
            r=reference['transitions'][i]
            same({k:t[k] for k in ('index','decision','ordinary_meal','action','normalized_advantage')},
                 {k:r[k] for k in ('index','decision','ordinary_meal','action','normalized_advantage')})
            assert t['before']==float(before[i,EAT]) and t['standard']==float(standard[i,EAT]) and t['detached']==float(detached[i,EAT])
        for key in ('predictor','predictor_optimizer','rng','journal','decisions','updates','policy_version'):
            assert fingerprint(brain.state()[key])==fingerprint(shadow.state()[key]),key
        if reference['case']==15:assert fingerprint(shadow.state())==fingerprint(expected_shadows[cid])
        observed.setdefault(cid,[]).append(record);counter+=1
    final,counts=replay(folder,callback,deadline)
    actual_final=torch.load(folder/'native-final.pt',weights_only=True,map_location='cpu')
    assert fingerprint(final)==fingerprint(actual_final) and counter==96
    # Independent effect and screen calculation; never call the runner summary.
    mean=lambda xs:sum(xs)/len(xs)
    carried={};adjacent={};reinforced=dict(standard=0,detached=0);summary=read('analysis.json');meal_count=0
    for cid,records in observed.items():
        assert len(records)==16
        for fixture,verb,destination in (('carried','eat',carried),('adjacent','gather',adjacent)):
            averages={arm:[mean([v[verb] for k,v in r['probes_'+arm].items() if k.startswith(fixture)]) for r in records]
                      for arm in ('before','standard','detached')}
            delta=[d-s for d,s in zip(averages['detached'],averages['standard'])];destination[cid]=mean(delta)
            measured=summary['candidates'][cid]['fixtures'][fixture]
            assert abs(measured['mean_paired_difference']-mean(delta))<1e-12
            assert measured['beneficial_updates']==sum(v>1e-6 for v in delta)
            for arm in ('standard','detached'):
                gain=mean([a-b for a,b in zip(averages[arm],averages['before'])])
                assert abs(gain-measured['mean_'+arm+'_gain'])<1e-12
        meals=[t for r in records for t in r['transitions'] if t['ordinary_meal']];meal_count+=len(meals)
        for arm in reinforced:
            value=sum(t[arm]-t['before']>1e-6 for t in meals);reinforced[arm]+=value
            assert value==summary['candidates'][cid]['reinforced'][arm]
            gain=mean([t[arm]-t['before'] for t in meals])
            assert abs(gain-summary['candidates'][cid]['mean_meal_gain'][arm])<1e-12
        for arm in ('before','standard','detached'):
            assert abs(mean([r['value_mse'][arm] for r in records])-summary['candidates'][cid]['mean_value_mse'][arm])<1e-12
    primary=mean(list(carried.values()));gathering=mean(list(adjacent.values()));positive=sum(v>1e-6 for v in carried.values())
    checks=dict(at_least_point_one_pp=primary>=.001-1e-12,positive_in_at_least_four_brains=positive>=4,
        no_fewer_reinforced_meals=reinforced['detached']>=reinforced['standard'],
        gathering_not_worse_by_point_one_pp=gathering>=-.001-1e-12)
    same(checks,summary['checks']);same(reinforced,summary['reinforced'])
    assert summary['diagnostic_promising']==all(checks.values()) and summary['milestone_confirmed'] is False
    assert summary['positive_brains']==positive and abs(summary['mean_paired_carried_difference']-primary)<1e-12
    assert summary['brains']==6 and summary['updates']==96 and summary['meals']==meal_count==214
    return dict(passed=True,seconds=time.perf_counter()-start,**counts,shadow_updates=counter,
        native_final_states_match=True,independent_detach_graph_matches=True,diagnostic_promising=all(checks.values()),
        checks=['source, evidence and parent trace hashes','native sampled-action replay and exact update boundaries',
                'independent explicit detach versus temporary hook','matched complete optimizer/model/history states',
                'fixed probes, sequence probabilities, value MSE and predictor isolation','independent paired-effect and screen recount'],
        shared_infrastructure='Original trace replay and state hashing helpers; detached graph and effect recount are separate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--wall-seconds',type=int,default=600);args=parser.parse_args();folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs') or folder==ROOT/'runs':parser.error('Use a local run directory')
    if not 1<=args.wall_seconds<=600:parser.error('Wall cap must be 1..600 seconds')
    try:
        result=audit(folder,args.wall_seconds);write(folder/'audit.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        write(folder/'audit-failed.json',dict(error=type(exc).__name__,reason=str(exc)));raise
