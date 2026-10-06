"""GT-01: retained experimental brains after 0, 16 and 64 more practice lives.

Preparation and execution are separate. Never reads a live save or endpoint.
"""
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
from experiments.category_trial import (MeasuredCategory, RESIDENTS, MAX_BYTES,
    population_for, run_episode, check_budget, memory_usage, write, sha)
from experiments.near_food_trial import make_world, quick_success, choice_probe
from experiments.feeding_credit import fingerprint, validate_source

ARMS = ('starting', 'after-16', 'after-64')
FIXTURES = ('adjacent', 'carried')
BASES = dict(training=81100000, adjacent=81200000, carried=81300000)
HORIZONS = dict(adjacent=512, carried=64)
CHECKPOINTS = (16, 32, 48, 64)
CASES = 16
WEAK = ('c0', 'c3', 'c5')


def coordinates(stage, pair, case, resident_index):
    if stage not in BASES or not 0 <= pair < 3 or not 0 <= case < 256 or resident_index not in (0, 1):
        raise ValueError('Invalid practice/evaluation coordinate')
    return BASES[stage]+pair*1000+case, BASES[stage]*10+pair*10000+case*2+resident_index


def copy_population(states, pair, world, config, frozen=False):
    p = population_for(states[f'c{pair*2}']['seed'], 'category', world, config)
    for i, rid in enumerate(RESIDENTS):
        saved = copy.deepcopy(states[f'c{pair*2+i}'])
        brain = MeasuredCategory(saved['seed'], config)
        brain.restore(saved)
        if frozen:brain.freeze()
        p.brains[rid] = brain
    return p


def metrics_summary(items, fixture):
    """Items are (episode, resident id); all rates have resident-life denominators."""
    values = [row['metrics'][rid] for row, rid in items]
    n = len(values)
    if not n:raise ValueError('Empty metric group')
    actions = {}; successes = {}; rewards = {}
    for m in values:
        for target, name in ((actions, 'actions'), (successes, 'successful_actions'), (rewards, 'reward_totals')):
            for key, value in m[name].items():target[key] = target.get(key, 0)+value
    return dict(lives=n, resident_ticks=sum(row['ticks'] for row, _ in items),
        prompt_successes=sum(quick_success(m, fixture) for m in values),
        full_life_successes=sum(m['timely_acquisition'] for m in values) if fixture == 'adjacent' else None,
        gathered_lives=sum(m['first_gather_food_tick'] is not None for m in values),
        ate_lives=sum(m['first_meal_tick'] is not None for m in values),
        ordinary_meals=sum(m['safe_eaten'] for m in values), amber_meals=sum(m['amber_eaten'] for m in values),
        zero_food_ticks=sum(m['zero_food_ticks'] for m in values),
        unconscious_ticks=sum(m['unconscious_ticks'] for m in values),
        damage=sum(m['damage'] for m in values), invalid=sum(m['invalid'] for m in values),
        mean_unique_tiles=sum(m['unique_tiles'] for m in values)/n,
        actions=actions, successful_actions=successes, reward_totals=rewards,
        decisions=sum(row['timings'][rid]['decisions'] for row, rid in items),
        updates=sum(row['timings'][rid]['updates'] for row, rid in items),
        decision_seconds=sum(row['timings'][rid]['decision_seconds'] for row, rid in items),
        learning_seconds=sum(row['timings'][rid]['learning_seconds'] for row, rid in items))


def advancement(fixtures):
    a = fixtures['adjacent']; c = fixtures['carried']
    rate = lambda score:score['prompt_successes']/score['lives']
    deprivation = lambda score:score['zero_food_ticks']/score['resident_ticks']
    long = a['arms']['after-64']
    checks = dict(
        long_prompt_at_least_80_percent=rate(long) >= .8,
        long_exceeds_16_by_10_points=rate(long)-rate(a['arms']['after-16']) >= .1-1e-12,
        long_exceeds_start_by_10_points=rate(long)-rate(a['arms']['starting']) >= .1-1e-12,
        positive_long_minus_16_in_at_least_two_seed_groups=sum(
            rate(g['after-64']) > rate(g['after-16']) for g in a['seed_groups'].values()) >= 2,
        carried_regression_no_more_than_5_points=all(
            rate(c['arms']['after-64'])-rate(c['arms'][arm]) >= -.05-1e-12 for arm in ARMS[:2]),
        deprivation_increase_no_more_than_half_point=all(
            deprivation(long)-deprivation(a['arms'][arm]) <= .005+1e-12 for arm in ARMS[:2]))
    return checks


def summarize(rows, cases=CASES):
    expected = {(f, arm, pair, case) for f in FIXTURES for arm in ARMS for pair in range(3) for case in range(cases)}
    keys = [(r['metadata']['fixture'], r['metadata']['arm'], r['metadata']['pair'], r['metadata']['case']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:raise ValueError('Missing or duplicated matched evaluation')
    result = {}
    for fixture in FIXTURES:
        relevant = [r for r in rows if r['metadata']['fixture'] == fixture]
        def scores(predicate):
            return {arm:metrics_summary([(r, rid) for r in relevant if r['metadata']['arm'] == arm
                        for rid in RESIDENTS if predicate(r, rid)], fixture) for arm in ARMS}
        result[fixture] = dict(arms=scores(lambda r, rid:True),
            seed_groups={str(pair):scores(lambda r, rid, p=pair:r['metadata']['pair'] == p) for pair in range(3)},
            candidates={f'c{slot}':scores(lambda r, rid, s=slot:r['metadata']['candidates'][rid] == f'c{s}') for slot in range(6)})
    checks = advancement(result)
    weak = result['adjacent']['candidates']
    return dict(fixtures=result, checks=checks, pilot_promising=all(checks.values()), milestone_confirmed=False,
        weak_candidates=list(WEAK), weak_improved_vs_16=[cid for cid in WEAK if
            weak[cid]['after-64']['prompt_successes'] > weak[cid]['after-16']['prompt_successes']],
        limitation='Retained six-brain development pool and unequal practice budgets; three historical seed groups, not fresh training replications or a GT-01 confirmation.')


def validate_states(states, config):
    if set(states) != {f'c{i}' for i in range(6)}:raise ValueError('Expected all six experimental candidates')
    for cid, state in states.items():
        if (state['backend'] != 'experimental-category-ppo' or state['config'] != asdict(config)
                or state['decisions'] != 6144 or state['updates'] != 48 or not state['training']
                or state['buffer'] or state['pending'] is not None):
            raise ValueError('Unexpected retained state: '+cid)


def prepare(source, out):
    parent = validate_source(source)
    config = Config(**parent['config'])
    original = torch.load(source/'continue-trained.pt', weights_only=True, map_location='cpu')
    validate_states(original, config)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('continue-trained.pt', 'preregistration.json', 'complete.json', 'audit.json'):
        shutil.copy2(source/name, out/('source-'+name))
    names = set(parent['hashes']) | {'experiments/feeding_credit.py', 'experiments/practice_curve.py',
        'experiments/audit_practice_curve.py', 'tests/test_practice_curve.py'}
    manifest = dict(protocol=1, milestone='GT-01', experiment='retained-practice-amount',
        seeds=parent['seeds'], config=asdict(config), source=str(source.relative_to(ROOT)),
        source_hashes={name:sha(source/name) for name in ('continue-trained.pt','preregistration.json','complete.json','audit.json')},
        original_state_hashes={cid:fingerprint(s) for cid,s in original.items()},
        arms=ARMS, fixtures=FIXTURES, checkpoints=CHECKPOINTS, training_lives=64, evaluation_cases=CASES,
        bases=BASES, horizons=HORIZONS, quick_windows=dict(adjacent=64, carried=16),
        total_ticks=181248, training_decisions=49152, frozen_decisions=41472, updates=384,
        wall_seconds=600, audit_wall_seconds=600, evidence_cap_bytes=MAX_BYTES,
        primary='Frozen ordinary-food gather-then-eat before tick 64 and before first zero fullness, starting with empty pack and four adjacent berries. Compare 64 versus 16 more practice lives and the frozen retained starting policy on the same 16 fresh maps per seed group (96 resident lives per condition).',
        training='Native category PPO, unchanged config/rewards/observations/actions and 512-tick life length. Start each of six retained brains at 6144 decisions/48 updates. One 64-life continuation per brain; first 16 lives are the EXACT shared prefix. Save full states every 16 lives. No culling, inherited replacement, scripted actions, weight import or detached-value change. Body/world reset between separate practice lives; brain/optimizer/private learner journal retained with native terminal recurrent reset.',
        evaluation='After all training, evaluate separate frozen copies of starting, after-16 and after-64 states. Adjacent horizon 512, carried-food horizon 64 with prompt consumption before tick 16. Carried probe supplies one fruit but no ground food and is NEVER used for training. Evaluation does not enter the learner state. All ten tones and all primitives remain under existing physical masks.',
        schedule='World seed=stage base+pair*1000+case; sampling seed=stage base*10+pair*10000+case*2+resident index. Stage is training/adjacent/carried. Each life reseeds only its action sampler, as in the source pilot. Unique coordinates, same evaluation randomness across arms. Worlds are private separated rooms, rotated/translated within the existing fixture, no hazards/refill/social contact.',
        gate='Development screen: after-64 adjacent prompt success >=80%; >=10 percentage points above after-16 AND starting; positive after-64-minus-16 in >=2/3 historical seed groups; carried prompt rate no more than 5 points below either shorter control; adjacent zero-food fraction no more than 0.5 points above either. All conditions required. These are pilot thresholds, not GT-01 confirmation.',
        weak_candidates=WEAK, weak_analysis='Previously weak policy-gain responders c0,c3,c5 fixed before run; report individually without excluding other candidates or using this subgroup for the primary gate.',
        limits='No alternative training seeds or random-policy arm in this dose comparison. Correlated repeated maps/actions within historical groups; descriptive rates, not tick-wise significance. Success in this easy adjacent-food fixture is not navigation, hazard avoidance, productive farming, learned communication, memory retention across tasks or general survival.',
        audit='Separate implementation replays all recorded native decisions, world steps, rewards and full training states at four checkpoints. Independently recounts outcomes and screen. Shares world/brain implementations, not the experiment rollout or summary functions. One 600-second audit cap.',
        budgets='One 600-second development run and one 600-second audit; 512 MiB evidence cap. The 181248-tick workload matches the earlier near-food diagnostic (about 500 seconds). Preserve partial results on failure, without automatic retry or extension.',
        live_population_changed=False, python=sys.version, torch=torch.__version__,
        hashes={name:sha(ROOT/name) for name in sorted(names)})
    write(out/'preregistration.json', manifest)
    digest = sha(out/'preregistration.json')
    (out/'preregistration.sha256').write_text(digest, encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(ROOT/name, name)
    print(json.dumps(dict(prepared=True, preregistration_sha256=digest, total_ticks=manifest['total_ticks'])), flush=True)


def run(out):
    if any((out/name).exists() for name in ('started.json','complete.json','incomplete.json','trace.jsonl.gz')):
        raise ValueError('Run already attempted; preserve existing evidence without retry')
    manifest = json.loads((out/'preregistration.json').read_text())
    assert sha(out/'preregistration.json') == (out/'preregistration.sha256').read_text().strip()
    assert manifest['experiment'] == 'retained-practice-amount'
    for name, digest in manifest['hashes'].items():assert sha(ROOT/name) == digest, name
    source = (ROOT/manifest['source']).resolve()
    assert source.is_relative_to(ROOT/'runs')
    for name, digest in manifest['source_hashes'].items():
        assert sha(source/name) == sha(out/('source-'+name)) == digest
    original = torch.load(out/'source-continue-trained.pt', weights_only=True, map_location='cpu')
    config = Config(**manifest['config']); validate_states(original, config)
    write(out/'started.json', dict(preregistration_sha256=sha(out/'preregistration.json'), attempt=1))
    started = time.perf_counter(); deadline = started+manifest['wall_seconds']
    training = []; evaluations = []
    try:
        with gzip.open(out/'trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as stream:
            for pair, seed in enumerate(manifest['seeds']):
                p = copy_population(original, pair, make_world(BASES['training']+pair*1000, 0)[0], config)
                for case in range(64):
                    world_seed, _ = coordinates('training', pair, case, 0)
                    p.world, meta = make_world(world_seed, case)
                    p.metrics = {rid:p.new_metrics() for rid in RESIDENTS}
                    for i, rid in enumerate(RESIDENTS):p.brains[rid].rng = random.Random(coordinates('training', pair, case, i)[1])
                    meta.update(stage='training', arm='practice', pair=pair, training_seed=seed, case=case,
                                candidates={rid:f'c{pair*2+i}' for i,rid in enumerate(RESIDENTS)})
                    training.append(run_episode(p, meta, f'train-{pair}-{case}', 512, stream, out, deadline))
                    write(out/'training.partial.json', training)
                    if case+1 in CHECKPOINTS:
                        saved = {f'c{pair*2+i}':copy.deepcopy(p.brains[rid].state()) for i,rid in enumerate(RESIDENTS)}
                        torch.save(saved, out/f'pair-{pair}-after-{case+1}.pt')
                        print(json.dumps(dict(stage='training', pair=pair, additional_lives=case+1,
                            seconds=round(time.perf_counter()-started,2))), flush=True)
            # No held-out result is inspected before completing the fixed training budget.
            for pair, seed in enumerate(manifest['seeds']):
                states = {'starting':original}
                for arm in ARMS[1:]:states[arm] = torch.load(out/f'pair-{pair}-{arm}.pt', weights_only=True, map_location='cpu')
                for case in range(CASES):
                    for fixture in FIXTURES:
                        for arm in ARMS:
                            world_seed, _ = coordinates(fixture, pair, case, 0)
                            w, meta = make_world(world_seed, case, fixture)
                            p = copy_population(states[arm], pair, w, config, frozen=True)
                            for i,rid in enumerate(RESIDENTS):p.brains[rid].rng = random.Random(coordinates(fixture,pair,case,i)[1])
                            meta.update(stage='evaluation', arm=arm, pair=pair, training_seed=seed, case=case,
                                candidates={rid:f'c{pair*2+i}' for i,rid in enumerate(RESIDENTS)},
                                choice_probes={rid:choice_probe(b,w.observe(rid)) for rid,b in p.brains.items()})
                            evaluations.append(run_episode(p,meta,f'eval-{pair}-{case}-{fixture}-{arm}',
                                                           HORIZONS[fixture],stream,out,deadline))
                            write(out/'results.partial.json',evaluations)
                    if (case+1)%4 == 0:print(json.dumps(dict(stage='evaluation',pair=pair,cases=case+1,
                        seconds=round(time.perf_counter()-started,2))),flush=True)
        check_budget(out,deadline)
        assert all(sha(ROOT/name)==digest for name,digest in manifest['hashes'].items())
        assert all(sha(source/name)==digest for name,digest in manifest['source_hashes'].items())
        assert {cid:fingerprint(s) for cid,s in original.items()} == manifest['original_state_hashes']
        assert sum(r['ticks'] for r in training+evaluations) == manifest['total_ticks']
        assert sum(v['decisions'] for r in training for v in r['timings'].values()) == manifest['training_decisions']
        assert sum(v['updates'] for r in training for v in r['timings'].values()) == manifest['updates']
        assert sum(v['decisions'] for r in evaluations for v in r['timings'].values()) == manifest['frozen_decisions']
        write(out/'training.json',training); write(out/'results.json',evaluations)
        result = summarize(evaluations); write(out/'analysis.json',result)
        write(out/'complete.json',dict(seconds=time.perf_counter()-started,total_ticks=manifest['total_ticks'],
            training_episodes=len(training),evaluation_episodes=len(evaluations),source_hashes_valid=True,
            source_states_preserved=True,live_population_changed=False,memory=memory_usage(),result=result,
            evidence_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(complete=True,seconds=time.perf_counter()-started,checks=result['checks'],
                             pilot_promising=result['pilot_promising'])),flush=True)
    except Exception as exc:
        write(out/'incomplete.json',dict(error=type(exc).__name__,reason=str(exc),seconds=time.perf_counter()-started,
            training_episodes=len(training),evaluation_episodes=len(evaluations)))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/culling-pilot-20261006')
    args=parser.parse_args();out=args.output.resolve();source=args.source.resolve()
    if any(not p.is_relative_to(ROOT/'runs') or p==ROOT/'runs' for p in (source,out)):
        parser.error('Use isolated local experimental directories under runs')
    if args.mode=='prepare':prepare(source,out)
    else:run(out)
