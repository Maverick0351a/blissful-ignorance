"""Independent trace/checkpoint accounting and factorial effect report."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.retained_replay import Brain, Config
from agents.sequence_ppo import Population
from experiments.audit_scarcity import recount
from experiments.replay_comparison import digest_tree


def digest(brain):
    state = brain.state()
    return digest_tree({k: state[k] for k in ('model', 'target', 'optimizer', 'replay',
        'priorities', 'replay_rng', 'retention_rng', 'experiment', 'updates')})


def audit(directory, archive_only=False):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    manifest, rows, complete = map(read, ('preregistration.json', 'results.json', 'complete.json'))
    prereg_hash = hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    assert prereg_hash == (out/'preregistration.sha256').read_text(encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip') as archive:
        for file, sha in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(file)).hexdigest() == sha
            if not archive_only:
                assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == sha
    expected = {(int(s), w, c) for s, worlds in manifest['evaluation_seeds'].items()
                for w in worlds for c in manifest['conditions']}
    assert {(r['seed'], r['evaluation_seed'], r['condition']) for r in rows} == expected
    assert len(rows) == len(expected)
    assert not set(sum(manifest['training_worlds'].values(), [])) & set(sum(manifest['evaluation_seeds'].values(), []))
    ticks = manifest['evaluation_ticks']
    summaries, checkpoints, behaviors = {}, {}, {}
    for row in rows:
        assert row['frozen_before'] == row['frozen_after']
        assert all(r['arenas'] == row['arenas'] for r in rows if r['evaluation_seed'] == row['evaluation_seed'])
        trace = out/f"{row['evaluation_seed']}-{row['condition']}-trace.jsonl"
        for rid, values in recount(trace, row['arenas'], ticks).items():
            for key, value in values.items():
                assert abs(row['metrics'][rid][key]-value) < 1e-7, (row['condition'], rid, key)
        choices, repeats, previous, verbs = 0, 0, {}, Counter()
        tiles = {rid: {tuple(a['origin'])} for rid, a in row['arenas'].items()}
        with trace.open(encoding='utf-8') as handle:
            for line in handle:
                for rid, event in json.loads(line)['agents'].items():
                    tiles[rid].add(tuple(event['position']))
                    if not event['conscious']:
                        previous.pop(rid, None)
                        continue
                    symbol = json.dumps(event['action'], sort_keys=True)
                    choices += 1
                    repeats += previous.get(rid) == symbol
                    previous[rid] = symbol
                    verbs[event['action']['verb']] += 1
        behaviors[(row['evaluation_seed'], row['condition'])] = dict(
            choices=choices, repeats=repeats, tones=verbs['tone'],
            verbs=dict(verbs), tiles=list(map(len, tiles.values())))
    for condition in manifest['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        lives = [m for r in selected for m in r['metrics'].values()]
        bs = [behaviors[(r['evaluation_seed'], condition)] for r in selected]
        mean = lambda k: statistics.mean(m[k] for m in lives)
        n = max(1, sum(b['choices'] for b in bs))
        summaries[condition] = dict(lives=len(lives), zero_food_fraction=mean('zero_food_ticks')/ticks,
            fullness=mean('mean_fullness'), safe_meals=mean('ordinary_eaten'),
            thorn_contacts=mean('thorn_contacts'), unconscious_fraction=mean('unconscious_ticks')/ticks,
            safe_foodward_crossings=mean('safe_foodward_crossings'), plantings=mean('planted'),
            reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in lives),
            tones_fraction=sum(b['tones'] for b in bs)/n,
            repeated_choice_fraction=sum(b['repeats'] for b in bs)/n,
            unique_tiles=statistics.mean(x for b in bs for x in b['tiles']))
    for seed in manifest['seeds']:
        initial = {rid: Brain(seed+i, Config()) for i, rid in enumerate(('r0', 'r1'))}
        for arm in manifest['arms']:
            name = f'{seed}-{arm}-trained.pt'
            assert hashlib.sha256((out/name).read_bytes()).hexdigest() == complete['checkpoints'][name]
            curve = read(f'{seed}-{arm}-learning-curve.json')
            assert [r['world_seed'] for r in curve] == manifest['training_worlds'][str(seed)]
            assert [r['stage'] for r in curve] == [s[0] for s in manifest['training_schedule']]
            for entry in curve:
                for m in entry['metrics'].values():
                    assert m['remaining_food_stock'] == 4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions']
            p = Population.load(out/name)
            seen_pointers, details = set(), {}
            for rid, b in p.brains.items():
                assert b.seed == seed+int(rid[1:])
                assert b.decisions == manifest['decisions_per_agent']
                assert b.updates == b.decisions//b.config.unroll-b.config.warmup+1
                assert b.inspector() == curve[-1]['inspector'][rid]
                for parameter in list(b.model.parameters())+list(b.target.parameters()):
                    assert bool(torch.isfinite(parameter).all())
                    assert parameter.data_ptr() not in seen_pointers
                    seen_pointers.add(parameter.data_ptr())
                assert digest_tree(b.model.state_dict()) != digest_tree(initial[rid].model.state_dict())
                cfg = manifest['configs'][arm]
                assert b.config == Config(**cfg)
                assert len(b.replay) <= cfg['capacity']
                assert sum(c['protected'] for c in b.replay) <= cfg['reserved_chunks']
                assert len({c['serial'] for c in b.replay}) == len(b.replay)
                for chunk in b.replay:
                    assert not any(r['terminal'] for r in chunk['rows'][:-1])
                    if chunk['protected']:
                        assert cfg['retain_consequences']
                        assert any(r.get('nutrition') or r.get('injury') for r in chunk['rows'][chunk['burn_in']:])
                b.freeze()
                for row in rows:
                    if row['seed'] == seed and row['condition'] == arm:
                        assert digest(b) == row['frozen_before'][rid]
                details[rid] = b.inspector()
            checkpoints[name] = details
        for b in initial.values():
            b.freeze()
        for row in rows:
            if row['seed'] == seed and row['condition'] in ('initial', 'random', 'oracle'):
                assert {rid: digest(b) for rid, b in initial.items()} == row['frozen_before']
    # Check equal starting weights, parameter count, updates and capacity per seed.
    for seed in manifest['seeds']:
        entries = [t for t in complete['timing'] if t['seed'] == seed]
        assert len(entries) == len(manifest['arms'])
        assert len({t['parameters'] for t in entries}) == 1
        assert all(t['initial_model_digests'] == entries[0]['initial_model_digests'] for t in entries)
        assert len({i['updates'] for t in entries for i in t['inspectors'].values()}) == 1
    assert len({c['capacity'] for c in manifest['configs'].values()}) == 1
    effects, gates = {}, {}
    for seed in manifest['seeds']:
        means = {c: {k: statistics.mean(m[k] for r in rows if r['seed'] == seed and r['condition'] == c
                                       for m in r['metrics'].values())
                     for k in ('zero_food_ticks', 'mean_fullness')} for c in manifest['arms']}
        def value(c, k):
            return means[c][k]/ticks if k == 'zero_food_ticks' else means[c][k]
        effects[str(seed)] = {}
        for k in ('zero_food_ticks', 'mean_fullness'):
            ff, rf, fb, rb = [value(c, k) for c in ('fifo_flat', 'retained_flat', 'fifo_balanced', 'retained_balanced')]
            effects[str(seed)][k] = dict(retention=((rf-ff)+(rb-fb))/2,
                exploration=((fb-ff)+(rb-rf))/2, interaction=(rb-rf)-(fb-ff))
    for arm in manifest['arms']:
        checks, reliable, safe, count = {}, 0, 0, 0
        for seed in manifest['seeds']:
            groups = {c: [m for r in rows if r['seed'] == seed and r['condition'] == c
                           for m in r['metrics'].values()] for c in (arm, 'initial')}
            avg = lambda c, k: statistics.mean(m[k] for m in groups[c])
            rate = lambda c: sum(m['thorn_contacts'] for m in groups[c])/max(1, sum(m['conscious_decisions'] for m in groups[c]))
            checks[str(seed)] = dict(less_starvation=avg(arm, 'zero_food_ticks') < avg('initial', 'zero_food_ticks'),
                more_fullness=avg(arm, 'mean_fullness') > avg('initial', 'mean_fullness'),
                lower_contact_rate=rate(arm) <= .5*rate('initial'))
            for m in groups[arm]:
                count += 1
                reliable += m['zero_food_ticks'] < .01*ticks
                safe += m['foodward_crossings'] > 0 and m['safe_foodward_crossings'] > 0 and m['safe_crossings']/m['crossings'] >= .8
        gates[arm] = dict(seed_checks=checks, reliable_lives=reliable, safe_active_lives=safe, lives=count,
            passed=all(all(v.values()) for v in checks.values()) and reliable/count >= .8 and safe/count >= .8)
    assert gates == complete['gates']
    total = 4*manifest['decisions_per_agent']*len(manifest['seeds'])*len(manifest['arms'])+ticks*len(rows)
    assert total == complete['total_ticks']
    support = {factor: all(e['zero_food_ticks'][factor] < 0 and e['mean_fullness'][factor] > 0
                          for e in effects.values()) for factor in ('retention', 'exploration')}
    result = dict(preregistration_sha256=prereg_hash, hashes_verified=True, traces_recounted=True,
        food_conserved=True, matched_arenas=True, equal_budgets=True, frozen_learning=True,
        exact_evaluation_checkpoints=True, private_finite_parameters=True,
        summary=summaries, effects_by_seed=effects, directional_support=support,
        gates=gates, checkpoints=checkpoints, total_ticks=total, seconds=complete['seconds'])
    (out/('audit-archive.json' if archive_only else 'audit.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    result = audit(args.directory, args.archive_only)
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}, indent=2))
