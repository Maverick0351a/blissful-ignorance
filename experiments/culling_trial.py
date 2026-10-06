"""Archived, experimental restart comparison; never reads or changes live saves."""
import argparse
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.sequence_ppo import Config
from experiments.category_trial import (MeasuredCategory, population_for, run_episode,
    check_budget, memory_usage, write, sha, RESIDENTS, MAX_BYTES)
from experiments.near_food_trial import make_world, quick_success

ARMS = ('continue', 'selected-restart', 'random-restart')
FRESH_SEEDS = (79201, 79213)
RANDOM_SELECTION_SEED = 79107
BASES = dict(selection=79100000, training=79200000, evaluation=79300000)


def sample_seed(stage, case, slot):
    """Unique within each stage/case/slot; matched across comparison arms."""
    if stage not in BASES or not 0 <= case < 1000 or not 0 <= slot < 6:
        raise ValueError('Invalid sampling coordinate')
    return BASES[stage]*10 + case*6 + slot


def load_sources(folder):
    protocol = json.loads((folder/'preregistration.json').read_text())
    receipt = json.loads((folder/'complete.json').read_text())
    audit = json.loads((folder/'audit.json').read_text())
    if not audit['passed'] or receipt['smoke'] or protocol.get('diagnostic') != 'near-food-chain':
        raise ValueError('Requires the audited full nearby-food diagnostic')
    if sha(folder/'preregistration.json') != (folder/'preregistration.sha256').read_text().strip():
        raise ValueError('Source preregistration changed')
    if len(protocol['seeds']) != 3:raise ValueError('Expected six source candidates')
    states = {}; config = Config(**protocol['config'])
    for pair, seed in enumerate(protocol['seeds']):
        name = f'{seed}-category-trained.pt'
        if sha(folder/name) != receipt['evidence_hashes'][name]:raise ValueError('Source checkpoint changed')
        saved = torch.load(folder/name, weights_only=True, map_location='cpu')
        if set(saved) != set(RESIDENTS):raise ValueError('Source roster changed')
        for i, rid in enumerate(RESIDENTS):
            state = saved[rid]
            if (state['backend'] != 'experimental-category-ppo' or state['config'] != asdict(config)
                    or state['decisions'] != 4096 or state['updates'] != 32
                    or state['buffer'] or state['pending'] is not None):
                raise ValueError('Candidate does not have the declared equal training history')
            states[f'c{pair*2+i}'] = copy.deepcopy(state)
    return protocol['seeds'], config, states


def selection_scores(rows, candidates, per_round):
    scores = {cid:dict(round_successes=[0,0], round_lives=[0,0], full_life=0, ordinary_meals=0)
              for cid in candidates}
    for row in rows:
        meta = row['metadata']; round_index = meta['case']//per_round
        if round_index not in (0,1):raise ValueError('Invalid selection round')
        for rid, metric in row['metrics'].items():
            score = scores[meta['candidates'][rid]]
            score['round_lives'][round_index] += 1
            score['round_successes'][round_index] += quick_success(metric,'adjacent')
            score['full_life'] += metric['timely_acquisition']
            score['ordinary_meals'] += metric['safe_eaten']
    for score in scores.values():
        if score['round_lives'] != [per_round,per_round]:raise ValueError('Incomplete matched selection screens')
        score['eligible'] = all(n/per_round < .8 for n in score['round_successes'])
    return scores


def select_retirements(scores):
    eligible = [cid for cid,s in scores.items() if s['eligible']]
    return sorted(eligible, key=lambda cid:(sum(scores[cid]['round_successes']),
                  scores[cid]['full_life'], scores[cid]['ordinary_meals'], cid))[:2]


def branch_states(source_states, retirements, arm, config):
    states = copy.deepcopy(source_states)
    identities = {cid:cid for cid in states}
    for index, cid in enumerate(sorted(retirements)):
        states[cid] = copy.deepcopy(MeasuredCategory(FRESH_SEEDS[index], config).state())
        identities[cid] = f'{arm}-new-{index}'
    return states, identities


def summarize(rows, cases):
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r['metadata']['arm']==arm]
        metrics = [m for r in selected for m in r['metrics'].values()]
        if len(metrics) != 6*cases:raise ValueError('Incomplete final evaluation')
        result[arm] = dict(lives=len(metrics), prompt_successes=sum(quick_success(m,'adjacent') for m in metrics),
            full_life_successes=sum(m['timely_acquisition'] for m in metrics),
            ordinary_meals=sum(m['safe_eaten'] for m in metrics), amber_meals=sum(m['amber_eaten'] for m in metrics),
            zero_food_ticks=sum(m['zero_food_ticks'] for m in metrics),
            unconscious_ticks=sum(m['unconscious_ticks'] for m in metrics),
            halves=[sum(quick_success(m,'adjacent') for r in selected if
                     (r['metadata']['case'] < cases//2)==(half==0) for m in r['metrics'].values())
                    for half in (0,1)])
    rate = lambda arm: result[arm]['prompt_successes']/result[arm]['lives']
    promising = rate('selected-restart') >= .8 and all(
        rate('selected-restart')-rate(other) >= .1-1e-12 and all(
            (result['selected-restart']['halves'][half]-result[other]['halves'][half])/(3*cases) >= .1-1e-12
            for half in (0,1)) for other in ('continue','random-restart'))
    return dict(arms=result, pilot_promising=promising, milestone_confirmed=False,
                limitation='One selected six-candidate pool; descriptive development comparison, not independent population replications')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/near-food-pilot-20261005')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args(); source = args.source.resolve(); out = args.output.resolve()
    if any(not p.is_relative_to(ROOT/'runs') or p==ROOT/'runs' for p in (source,out)):
        parser.error('Use local experimental run directories')
    if not 1 <= args.wall_seconds <= 600:parser.error('Wall cap must be <=600 seconds')
    seeds, config, originals = load_sources(source)
    per_round, training_lives, cases = (2,2,2) if args.smoke else (8,16,8)
    out.mkdir(parents=True, exist_ok=False)
    for seed in seeds:
        shutil.copy2(source/f'{seed}-category-trained.pt', out/f'archive-{seed}.pt')
    sources = sorted({str(p.relative_to(ROOT)).replace('\\','/') for folder in ('agents','sim')
                      for p in (ROOT/folder).glob('*.py')} |
        {'experiments/culling_trial.py','experiments/audit_culling.py','experiments/culling_report.py',
         'experiments/category_trial.py','experiments/coordination_trial.py','experiments/near_food_trial.py',
         'experiments/category_replay.py','tests/test_culling_trial.py'})
    manifest = dict(protocol=1, milestone='GT-01', experiment='archived-selective-restart', smoke=args.smoke,
        seeds=seeds, arms=ARMS, config=asdict(config), rounds=2, cases_per_round=per_round,
        training_lives=training_lives, evaluation_cases=cases, ticks_per_life=512, quick_window=64,
        total_ticks=3*(2*per_round+3*training_lives+3*cases)*512,
        wall_seconds=args.wall_seconds, audit_wall_seconds=600, evidence_cap_bytes=MAX_BYTES,
        source_run=str(source.relative_to(ROOT)), source_complete_sha256=sha(source/'complete.json'),
        source_audit_sha256=sha(source/'audit.json'),
        archive_hashes={f'archive-{seed}.pt':sha(out/f'archive-{seed}.pt') for seed in seeds},
        bases=BASES, fresh_seeds=FRESH_SEEDS, random_selection_seed=RANDOM_SELECTION_SEED,
        selection='Six previously trained private category-PPO candidates, each 4096 decisions and 32 updates. Two fresh eight-case frozen screens. Eligible only below 80% prompt acquisition in BOTH screens. Retire at most two eligible candidates, ranked by total prompt successes, then full-life feeding, then ordinary meals, then stable candidate ID. Every source archived before selection. No forced minimum retirement.',
        controls='Same archived pool in continue, selected-restart and random-restart. Random restart retires the same number via fixed seed; fresh initializations match by replacement ordinal. Retained brains keep weights, optimizers and histories. Fresh brains have new identities, seeds and optimizers; no inheritance or mutation. Equal FUTURE interactions, not equal lifetime training for newborns.',
        schedule='World seed=stage base+case, identical across pairs and arms; balanced cardinal direction. Sampling seed=stage base*10+case*6+slot, unique within stage/case/slot and matched across arms. Existing private observations, rewards, actions, config and physical masks unchanged. Three separated pairs make ONE six-candidate pool, not three selection replications.',
        gate='Development promising only if selected-restart reaches >=80% prompt acquisition and exceeds BOTH continuation and random restart by >=10 percentage points overall and in each four-map evaluation half. All weights frozen during final evaluation. No GT-01 confirmation or live migration.',
        boundaries='Local copies only. No main population save, Laya, mortality, scripted survival, network, downloads or publication. No claim of genetic evolution, communication or sustainable farming. Hold-out evaluation not used for retirement. Caps preserve partial evidence without retry.',
        hashes={name:sha(ROOT/name) for name in sources})
    write(out/'preregistration.json',manifest)
    (out/'preregistration.sha256').write_text(sha(out/'preregistration.json'),encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    started=time.perf_counter(); deadline=started+args.wall_seconds
    selections=[]; training=[]; evaluations=[]
    try:
        with gzip.open(out/'trace.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stream:
            def episodes(states, identities, stage, arm, count):
                rows=[]; final=copy.deepcopy(states)
                for pair, seed in enumerate(seeds):
                    w,_=make_world(BASES[stage],0)
                    p=population_for(seed,'category',w,config)
                    for i,rid in enumerate(RESIDENTS):
                        cid=f'c{pair*2+i}'; saved=copy.deepcopy(states[cid])
                        p.brains[rid]=MeasuredCategory(saved['seed'],config);p.brains[rid].restore(saved)
                    for case in range(count):
                        if stage!='training':
                            for i,rid in enumerate(RESIDENTS):
                                p.brains[rid].restore(copy.deepcopy(states[f'c{pair*2+i}']));p.brains[rid].freeze()
                        p.world,meta=make_world(BASES[stage]+case,case)
                        p.metrics={rid:p.new_metrics() for rid in RESIDENTS}
                        for i,rid in enumerate(RESIDENTS):p.brains[rid].rng=random.Random(sample_seed(stage,case,pair*2+i))
                        meta.update(stage=stage,arm=arm,case=case,training_seed=seed,pair=pair,
                            candidates={rid:f'c{pair*2+i}' for i,rid in enumerate(RESIDENTS)},
                            identities={rid:identities[f'c{pair*2+i}'] for i,rid in enumerate(RESIDENTS)})
                        row=run_episode(p,meta,f'{stage}-{arm}-{pair}-{case}',512,stream,out,deadline)
                        rows.append(row)
                        destination={'selection':selections,'training':training,'evaluation':evaluations}[stage]
                        destination.append(row);write(out/(stage+'.partial.json'),destination)
                    if stage=='training':
                        for i,rid in enumerate(RESIDENTS):final[f'c{pair*2+i}']=copy.deepcopy(p.brains[rid].state())
                    print(json.dumps(dict(stage=stage,arm=arm,pair=pair,seconds=round(time.perf_counter()-started,2))),flush=True)
                return final
            episodes(originals,{cid:cid for cid in originals},'selection','source',2*per_round)
            scores=selection_scores(selections,originals,per_round); retired=select_retirements(scores)
            random_retired=sorted(random.Random(RANDOM_SELECTION_SEED).sample(sorted(originals),len(retired)))
            decision=dict(scores=scores,retired={'continue':[],'selected-restart':retired,'random-restart':random_retired},
                          archive_hashes=manifest['archive_hashes'],live_population_changed=False)
            write(out/'selection.json',selections);write(out/'retirement.json',decision)
            print(json.dumps(dict(retirements=decision['retired'])),flush=True)
            for arm in ARMS:
                initial,identities=branch_states(originals,decision['retired'][arm],arm,config)
                torch.save(initial,out/(arm+'-initial.pt'))
                final=episodes(initial,identities,'training',arm,training_lives)
                torch.save(final,out/(arm+'-trained.pt'))
                episodes(final,identities,'evaluation',arm,cases)
        check_budget(out,deadline)
        assert all(sha(ROOT/name)==digest for name,digest in manifest['hashes'].items())
        assert all(sha(out/name)==digest for name,digest in manifest['archive_hashes'].items())
        for seed in seeds:
            assert sha(source/f'{seed}-category-trained.pt')==manifest['archive_hashes'][f'archive-{seed}.pt']
        write(out/'training.json',training);write(out/'results.json',evaluations)
        write(out/'complete.json',dict(seconds=time.perf_counter()-started,total_ticks=manifest['total_ticks'],
            result=summarize(evaluations,cases),memory=memory_usage(),live_population_changed=False,
            evidence_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
        print(json.dumps(summarize(evaluations,cases)),flush=True)
    except Exception as exc:
        write(out/'incomplete.json',dict(error=type(exc).__name__,reason=str(exc),seconds=time.perf_counter()-started,
            completed_selection=len(selections),completed_training=len(training),completed_evaluation=len(evaluations)))
        raise


if __name__=='__main__':main()
