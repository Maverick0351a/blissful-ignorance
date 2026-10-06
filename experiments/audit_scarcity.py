"""Recount safe-route trials from physical action traces, separately from runner."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def recount(path,arenas,ticks):
    totals={rid:dict(nutrition=0.,damage=0.,zero_food_ticks=0,unconscious_ticks=0,thorn_contacts=0,
                    conscious_decisions=0,conscious_invalid=0,invalid=0,ordinary_eaten=0,
                    seed_conversions=0,planted=0,thorn_opportunities=0,safe_crossings=0,
                    unsafe_crossings=0,crossings=0,foodward_crossings=0,safe_foodward_crossings=0,
                    aborted_crossings=0,crossing_steps=0) for rid in arenas}
    positions={rid:tuple(a['origin']) for rid,a in arenas.items()}
    states={rid:dict(room='home',active=False,hazard=False,length=0) for rid in arenas}
    def local(rid,point):
        a=arenas[rid];x=point[0]-a['origin'][0];y=point[1]-a['origin'][1];dx,dy=a['forward']
        return x*dx+y*dy,-x*dy+y*dx
    last={};blocks=0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line);blocks+=1;assert row['tick']==blocks*4
            assert set(row['agents'])==set(arenas)
            for rid,m in row['agents'].items():
                t=totals[rid];previous=positions[rid];u0,v0=local(rid,previous)
                t['thorn_opportunities']+=m['conscious'] and abs(u0-2)+abs(v0)<=1
                point=tuple(m['position']);distance=sum(abs(x-y) for x,y in zip(previous,point))
                assert distance<=1
                if distance:
                    assert m['success'] and m['action']['verb']=='move'
                elif m['success'] and m['action']['verb']=='move':raise AssertionError('Successful movement did not move')
                u,v=local(rid,point);room='home' if -1<=u<=1 and abs(v)<=1 else 'food' if 3<=u<=5 and abs(v)<=1 else None
                s=states[rid]
                if room is None:s['active']=True
                if s['active']:
                    s['length']+=distance;s['hazard'] |= (u,v)==(2,0)
                    if room:
                        if room==s['room']:t['aborted_crossings']+=1
                        else:
                            t['crossings']+=1;t['crossing_steps']+=s['length']
                            t['unsafe_crossings' if s['hazard'] else 'safe_crossings']+=1
                            if room=='food':
                                t['foodward_crossings']+=1;t['safe_foodward_crossings']+=not s['hazard']
                        s.update(room=room,active=False,hazard=False,length=0)
                positions[rid]=point
                for key in ('nutrition','damage','zero_food_ticks','unconscious_ticks','thorn_contacts'):t[key]+=m[key]
                t['conscious_decisions']+=m['conscious'];t['invalid']+=not m['success']
                t['conscious_invalid']+=m['conscious'] and not m['success']
                if m['success']:
                    t['ordinary_eaten']+=m['action']['verb']=='eat' and m['action'].get('item','food')=='food'
                    t['seed_conversions']+=m['action']['verb']=='make_seeds'
                    t['planted']+=m['action']['verb']=='plant'
                assert m['food_stock']==4+m['matured_food']-t['ordinary_eaten']-t['seed_conversions']
                last[rid]=m
    assert blocks*4==ticks
    for rid,t in totals.items():
        m=last[rid];t.update(mean_fullness_sum=m['mean_fullness_sum'],mean_fullness=m['mean_fullness_sum']/ticks,
                           matured_food=m['matured_food'],initial_food_stock=4,remaining_food_stock=m['food_stock'])
    return totals


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    parser.add_argument('--archive-only',action='store_true');args=parser.parse_args()
    out=Path(args.directory);root=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    manifest=read('preregistration.json');rows=read('results.json');complete=read('complete.json')
    prereg_hash=hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    assert prereg_hash==(out/'preregistration.sha256').read_text(encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip') as archive:
        for name,digest in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
            if not args.archive_only:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    expected={(int(s),world,c) for s,worlds in manifest['evaluation_seeds'].items() for world in worlds for c in manifest['conditions']}
    assert expected=={(r['seed'],r['evaluation_seed'],r['condition']) for r in rows};assert len(rows)==len(expected)
    for seed in manifest['training_seeds']:
        curve=read(f'{seed}-learning-curve.json');schedule=manifest['training_schedule']
        assert [(r['stage'],r['episode']) for r in curve]==[(stage,i) for stage,n,_ in schedule for i in range(n)]
        for rid in ('r0','r1'):assert curve[-1]['inspector'][rid]['decisions']==manifest['decisions_per_agent']
        for entry in curve:
            for m in entry['metrics'].values():
                assert m['remaining_food_stock']==4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions']
    for row in rows:
        assert row['frozen_weights']==row['frozen_end']
        matched=[r for r in rows if r['evaluation_seed']==row['evaluation_seed']]
        assert all(r['arenas']==row['arenas'] for r in matched)
        totals=recount(out/f"{row['evaluation_seed']}-{row['condition']}-trace.jsonl",row['arenas'],manifest['evaluation_ticks'])
        for rid,t in totals.items():
            for key,value in t.items():
                actual=row['metrics'][rid][key]
                assert abs(actual-value)<1e-7,(row['condition'],rid,key,actual,value)
    checks={};reliable=safe=lives=0
    for seed in manifest['training_seeds']:
        groups={c:[m for r in rows if r['seed']==seed and r['condition']==c for m in r['metrics'].values()]
                for c in ('trained','initial')}
        average=lambda c,k:statistics.mean(m[k] for m in groups[c])
        rate=lambda c:sum(m['thorn_contacts'] for m in groups[c])/max(1,sum(m['conscious_decisions'] for m in groups[c]))
        checks[str(seed)]=dict(less_starvation=average('trained','zero_food_ticks')<average('initial','zero_food_ticks'),
                              more_fullness=average('trained','mean_fullness')>average('initial','mean_fullness'),
                              lower_contact_rate=rate('trained')<=.5*rate('initial'))
        for m in groups['trained']:
            lives+=1;reliable+=m['zero_food_ticks']<.01*manifest['evaluation_ticks']
            safe+=m['foodward_crossings']>0 and m['safe_foodward_crossings']>0 and m['safe_crossings']/m['crossings']>=.8
    gate=dict(seed_checks=checks,reliable_lives=reliable,safe_active_lives=safe,lives=lives,
              passed=all(all(v.values()) for v in checks.values()) and reliable/lives>=.8 and safe/lives>=.8)
    assert gate==complete['gate']
    total_ticks=sum(n*d*4 for _,n,d in manifest['training_schedule'])*len(manifest['training_seeds'])+manifest['evaluation_ticks']*len(rows)
    assert total_ticks==complete['total_ticks']
    summary={}
    for condition in manifest['conditions']:
        metrics=[m for r in rows if r['condition']==condition for m in r['metrics'].values()]
        mean=lambda k:statistics.mean(m[k] for m in metrics)
        summary[condition]={k:mean(k) for k in ('nutrition','zero_food_ticks','unconscious_ticks','damage','thorn_contacts',
                              'mean_fullness','crossings','safe_crossings','unsafe_crossings','foodward_crossings',
                              'safe_foodward_crossings','ordinary_eaten','seed_conversions','planted','matured_food')}
        summary[condition]['safe_crossing_fraction']=sum(m['safe_crossings'] for m in metrics)/max(1,sum(m['crossings'] for m in metrics))
        summary[condition]['contacts_per_1000_conscious_decisions']=1000*sum(m['thorn_contacts'] for m in metrics)/max(1,sum(m['conscious_decisions'] for m in metrics))
        summary[condition]['reliable_lives']=sum(m['zero_food_ticks']<.01*manifest['evaluation_ticks'] for m in metrics)
    report=dict(verified_hashes=True,matched_arenas=True,trace_totals_match=True,food_conserved=True,
                seconds=complete['seconds'],total_ticks=total_ticks,preregistration_sha256=prereg_hash,gate=gate,summary=summary)
    if args.archive_only:report['current_sources_match']=all(hashlib.sha256((root/f).read_bytes()).hexdigest()==h for f,h in manifest['hashes'].items())
    (out/('audit-archive.json' if args.archive_only else 'audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
