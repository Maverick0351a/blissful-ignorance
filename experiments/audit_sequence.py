"""Separate recount of completed sequence trials; no simulator rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    parser.add_argument('--archive-only',action='store_true',help='Recount historical source after a documented implementation change')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.directory)
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    manifest=read('preregistration.json');rows=read('results.json');complete=read('complete.json')
    prereg_hash=hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    assert prereg_hash==(out/'preregistration.sha256').read_text()
    with zipfile.ZipFile(out/'source.zip') as archive:
        for name,digest in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
            if not args.archive_only:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    expected={(int(seed),world,condition) for seed,worlds in manifest['evaluation_seeds'].items()
              for world in worlds for condition in manifest['conditions']}
    assert expected=={(r['seed'],r['evaluation_seed'],r['condition']) for r in rows}
    assert len(rows)==len(expected)
    for seed in manifest['training_seeds']:
        curves=read(f'{seed}-learning-curve.json')
        assert len(curves)==len(manifest['stages'])*manifest['episodes_per_stage']
        assert {(r['stage'],r['episode']) for r in curves}=={
            (stage,i) for stage in manifest['stages'] for i in range(manifest['episodes_per_stage'])}
        for rid in ('r0','r1'):
            assert curves[-1]['inspector'][rid]['decisions']==manifest['decisions_per_agent']
        # Nutrition clipping fix added after the run has no effect when no
        # observed transition reaches the zero-fullness floor.
        assert all(m['zero_food_ticks']==0 for row in curves for m in row['metrics'].values())
    checks={};reliable=0;avoidant=0;lives=0
    for row in rows:
        assert row['frozen_weights']==row['frozen_end']
        trace=read(f"{row['evaluation_seed']}-{row['condition']}-trace.json")
        assert trace[-1]['ticks']==manifest['evaluation_ticks']
        for rid,metrics in row['metrics'].items():
            for key,value in metrics.items():assert trace[-1]['agents'][rid][key]==value
        if row['condition']=='trained':
            group={r['condition']:r for r in rows if r['evaluation_seed']==row['evaluation_seed']}
            avg=lambda r,k:statistics.mean(m[k] for m in r['metrics'].values())
            checks[str(row['evaluation_seed'])]=dict(improved=avg(row,'nutrition')>avg(group['initial'],'nutrition'),
                     sufficient_nutrition=avg(row,'nutrition')>=.8*avg(group['scripted'],'nutrition'))
            for rid,m in row['metrics'].items():
                lives+=1;reliable+=m['zero_food_ticks']<.01*manifest['evaluation_ticks']
                control=group['initial']['metrics'][rid]
                if m['thorn_opportunities'] and control['thorn_opportunities']:
                    avoidant+=m['thorn_contacts']/m['thorn_opportunities']<=.5*control['thorn_contacts']/control['thorn_opportunities']
    passed=all(c['improved'] and c['sufficient_nutrition'] for c in checks.values()) and reliable/lives>=.9 and avoidant/lives>=.8
    gate=dict(groups=checks,reliable_lives=reliable,lives=lives,avoidant_lives=avoidant,passed=passed)
    assert gate==complete['gate']
    summaries={}
    for condition in manifest['conditions']:
        metrics=[m for r in rows if r['condition']==condition for m in r['metrics'].values()]
        summaries[condition]={key:statistics.mean(m[key] for m in metrics) for key in
            ('nutrition','zero_food_ticks','unconscious_ticks','damage','safe_eaten','amber_eaten','thorn_contacts')}
        summaries[condition]['reliable_lives']=sum(m['zero_food_ticks']<.01*manifest['evaluation_ticks'] for m in metrics)
        summaries[condition]['conscious_invalid_percent']=100*sum(m['conscious_invalid'] for m in metrics)/sum(m['conscious_decisions'] for m in metrics)
        summaries[condition]['normalized_prediction_mae']=sum(m['prediction_error_sum'] for m in metrics)/max(1,sum(m['prediction_events'] for m in metrics)) if condition!='scripted' else None
    report=dict(verified_hashes=True,trace_totals_match=True,preregistration_sha256=prereg_hash,
                seconds=complete['seconds'],gate=gate,summary=summaries)
    if args.archive_only:
        report['current_sources_match']=all(hashlib.sha256((root/f).read_bytes()).hexdigest()==h for f,h in manifest['hashes'].items())
    (out/('audit-archive.json' if args.archive_only else 'audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
