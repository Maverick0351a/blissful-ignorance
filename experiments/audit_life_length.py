"""Independent trace, schedule and checkpoint audit for training-life comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.environment import LearningEnv
from agents.network import ACTIONS
from agents.sequence_ppo import Brain, Config
from experiments.audit_scarcity import recount
from experiments.learning_curve import make_env
from experiments.replay_comparison import digest_tree, learning_digest


def recount_behavior(out, row):
    totals = {rid: dict(visible_food_decisions=0, adjacent_food_decisions=0,
        food_carried_decisions=0, hungry_food_carried_decisions=0, tones=0, moves=0,
        amber_eaten=0) for rid in row['arenas']}
    points = {rid: tuple(arena['origin']) for rid, arena in row['arenas'].items()}
    visited = {rid: {point} for rid, point in points.items()}
    with (out/row['trace']).open(encoding='utf-8') as physical, (out/row['senses']).open(encoding='utf-8') as senses:
        for decision in range(row['decisions']):
            p, s = json.loads(next(physical)), json.loads(next(senses))
            assert p['tick'] == (decision+1)*4 and s['decision'] == decision
            assert set(p['agents']) == set(s['agents']) == set(points)
            for rid, observed in s['agents'].items():
                body, m = p['agents'][rid], totals[rid]
                assert tuple(observed['position']) == points[rid]
                assert observed['conscious'] == body['conscious']
                assert ACTIONS[observed['action']] == body['action']
                if row.get('condition') != 'oracle':
                    assert observed['action'] in observed['legal']
                visible = observed['visible_food']
                assert all(len(t) == 3 and abs(t[0]) <= 4 and abs(t[1]) <= 4 and t[2] > 0 for t in visible)
                conscious = observed['conscious']
                m['visible_food_decisions'] += conscious and bool(visible)
                m['adjacent_food_decisions'] += conscious and any(abs(t[0])+abs(t[1]) <= 1 for t in visible)
                m['food_carried_decisions'] += conscious and observed['food_carried'] > 0
                m['hungry_food_carried_decisions'] += conscious and observed['food_carried'] > 0 and observed['fullness'] < 60
                m['tones'] += conscious and body['action']['verb'] == 'tone'
                m['moves'] += conscious and body['action']['verb'] == 'move'
                m['amber_eaten'] += body['success'] and body['action']['verb'] == 'eat' and body['action'].get('item') == 'amber_fruit'
                points[rid] = tuple(body['position'])
                visited[rid].add(points[rid])
        assert not physical.readline() and not senses.readline()
    for rid, m in totals.items():
        m['unique_tiles'] = len(visited[rid])
        for key, value in m.items():
            assert row['metrics'][rid][key] == value, (row['trace'], rid, key)
    return totals


def audit(directory, archive_only=False):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest, training, rows, complete = map(read, ('preregistration.json', 'training.json', 'results.json', 'complete.json'))
    prereg = sha(out/'preregistration.json')
    assert prereg == (out/'preregistration.sha256').read_text(encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip') as archive:
        for file, value in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(file)).hexdigest() == value
            if not archive_only:
                assert sha(ROOT/file) == value
    expected_training = {(int(seed), arm, life) for seed, arms in manifest['plans'].items()
                         for arm, plan in arms.items() for life in range(len(plan))}
    assert len(training) == len(expected_training)
    assert {(r['seed'], r['arm'], r['life']) for r in training} == expected_training
    expected_eval = {(int(seed), world, condition) for seed, maps in manifest['test_maps'].items()
                     for world in maps for condition in manifest['conditions']}
    assert len(rows) == len(expected_eval)
    assert {(r['seed'], r['world_seed'], r['condition']) for r in rows} == expected_eval
    train_seeds = {p['world_seed'] for arms in manifest['plans'].values() for plan in arms.values() for p in plan}
    test_seeds = {world for maps in manifest['test_maps'].values() for world in maps}
    assert not train_seeds & test_seeds
    evidence = {r[key] for r in training+rows for key in ('trace', 'senses')}
    assert set(complete['evidence_hashes']) == evidence
    for file, value in complete['evidence_hashes'].items():
        assert sha(out/file) == value
    starts = {}
    for row in training+rows:
        start_key = (row['world_seed'], row['stage'])
        if start_key not in starts:
            env, _ = make_env(row['world_seed'], row['stage'], row['decisions'], 'native')
            restored = LearningEnv(('r0', 'r1'), max_cycles=row['decisions'])
            restored.restore(env.state())
            starts[start_key] = dict(world_digest=digest_tree(restored.world.to_dict()),
                residents={rid: vars(a) for rid, a in restored.world.residents.items()})
        assert row['start'] == starts[start_key]
        for values in row['start']['residents'].values():
            assert values['food'] == 8 and values['health'] == 100 and not values['unconscious']
            assert not any(values['inventory'].values())
        counted = recount(out/row['trace'], row['arenas'], row['decisions']*4)
        for rid, metrics in counted.items():
            for key, value in metrics.items():
                assert abs(row['metrics'][rid][key]-value) < 1e-7, (row['trace'], rid, key)
        recount_behavior(out, row)
        if 'condition' in row:
            assert row['decisions']*4 == manifest['evaluation_ticks']
            assert row['stage'] == 'scarcity'
            assert row['frozen_before'] == row['frozen_after']
    budget, rollout = manifest['decisions_per_resident'], manifest['config']['rollout']
    checkpoint_stats = {}
    for seed in manifest['seeds']:
        for arm in ('short', 'long'):
            entries = sorted((r for r in training if r['seed'] == seed and r['arm'] == arm), key=lambda r: r['life'])
            plan = manifest['plans'][str(seed)][arm]
            previous = {rid: learning_digest(Brain(seed+i, Config(**manifest['config']))) for i, rid in enumerate(('r0', 'r1'))}
            decisions = 0
            for entry, intended in zip(entries, plan):
                assert all(entry[key] == value for key, value in intended.items())
                assert entry['learning_before'] == previous
                previous = entry['learning_after']
                decisions += intended['decisions']
                for inspector in entry['inspectors'].values():
                    assert inspector['decisions'] == decisions
                    assert inspector['updates'] == decisions//rollout
                    assert inspector['diagnostics']['sequence_length'] == rollout
                    assert inspector['diagnostics']['recurrent_boundary'] == 'reset_after_update'
                    assert inspector['curiosity_coefficient'] == 0
            assert decisions == budget
            expected_length = (128 if manifest['pilot'] else 512) if arm == 'short' else (256 if manifest['pilot'] else 4096)
            assert all(entry['decisions'] == expected_length for entry in entries)
            assert [sum(e['decisions'] for e in entries if e['block'] == i) for i in range(4)] == [budget//4]*4
            assert [next(e['stage'] for e in entries if e['block'] == i) for i in range(4)] == ['feeding', 'routes', 'scarcity', 'scarcity']
            name = f'{seed}-{arm}-checkpoint.pt'
            assert sha(out/name) == complete['checkpoints'][name]
            data = torch.load(out/name, map_location='cpu', weights_only=True)
            env = LearningEnv(('r0', 'r1'), max_cycles=expected_length)
            env.restore(data['environment'])
            assert not env.agents and env.cycles == expected_length
            loaded, pointers = [], set()
            checkpoint_stats[name] = {}
            for rid, state in data['brains'].items():
                brain = Brain(state['seed'], Config(**state['config']))
                brain.restore(state)
                loaded.append(brain)
                assert state['config'] == manifest['config']
                assert brain.decisions == budget and brain.updates == budget//rollout
                assert not brain.buffer and brain.pending is None
                assert learning_digest(brain) == previous[rid]
                initial = Brain(brain.seed, brain.config)
                assert digest_tree(brain.model.state_dict()) != digest_tree(initial.model.state_dict())
                for parameter in list(brain.model.parameters())+list(brain.predictor.parameters()):
                    assert bool(torch.isfinite(parameter).all()) and parameter.data_ptr() not in pointers
                    pointers.add(parameter.data_ptr())
                checkpoint_stats[name][rid] = dict(updates=brain.updates, decisions=brain.decisions,
                    optimizer_steps=int(next(iter(brain.optimizer.state.values()))['step']),
                    policy_parameters=sum(p.numel() for p in brain.model.parameters()))
                brain.freeze()
                for row in rows:
                    if row['seed'] == seed and row['condition'] == arm:
                        assert learning_digest(brain) == row['frozen_before'][rid]
        for i, rid in enumerate(('r0', 'r1')):
            initial = Brain(seed+i, Config(**manifest['config']))
            initial.freeze()
            for row in rows:
                if row['seed'] == seed and row['condition'] in ('initial', 'random', 'oracle'):
                    assert learning_digest(initial) == row['frozen_before'][rid]
    assert len(complete['checkpoints']) == 2*len(manifest['seeds'])
    ticks, summary, seed_checks, gates = manifest['evaluation_ticks'], {}, {}, {}
    def group(condition, seed=None):
        return [m for r in rows if r['condition'] == condition and (seed is None or r['seed'] == seed) for m in r['metrics'].values()]
    def mean(condition, key, seed=None):
        return statistics.mean(m[key] for m in group(condition, seed))
    for condition in manifest['conditions']:
        metrics = group(condition)
        summary[condition] = {key: mean(condition, key) for key in ('mean_fullness', 'ordinary_eaten', 'amber_eaten',
            'thorn_contacts', 'planted', 'seed_conversions', 'matured_food', 'remaining_food_stock',
            'unique_tiles', 'visible_food_decisions', 'adjacent_food_decisions', 'food_carried_decisions',
            'hungry_food_carried_decisions', 'foodward_crossings', 'safe_foodward_crossings')}
        summary[condition].update(lives=len(metrics), zero_food_fraction=mean(condition, 'zero_food_ticks')/ticks,
            unconscious_fraction=mean(condition, 'unconscious_ticks')/ticks,
            tone_share=sum(m['tones'] for m in metrics)/max(1, sum(m['conscious_decisions'] for m in metrics)),
            reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in metrics),
            safe_active_lives=sum(m['safe_foodward_crossings'] > 0 and m['safe_crossings']/max(1, m['crossings']) >= .8 for m in metrics))
    for seed in manifest['seeds']:
        seed_checks[str(seed)] = dict(zero_food={c: mean(c, 'zero_food_ticks', seed)/ticks for c in manifest['conditions']},
            long_improved=mean('long', 'zero_food_ticks', seed) < mean('short', 'zero_food_ticks', seed))
    primary = all(v['long_improved'] for v in seed_checks.values()) and mean('long', 'mean_fullness') > mean('short', 'mean_fullness')
    for arm in ('short', 'long'):
        checks = {}
        for seed in manifest['seeds']:
            rate = lambda c: sum(m['thorn_contacts'] for m in group(c, seed))/max(1, sum(m['conscious_decisions'] for m in group(c, seed)))
            checks[str(seed)] = dict(better_than_initial=mean(arm, 'zero_food_ticks', seed) < mean('initial', 'zero_food_ticks', seed) and mean(arm, 'mean_fullness', seed) > mean('initial', 'mean_fullness', seed),
                better_than_random=mean(arm, 'zero_food_ticks', seed) < mean('random', 'zero_food_ticks', seed) and mean(arm, 'mean_fullness', seed) > mean('random', 'mean_fullness', seed),
                no_higher_thorn_rate=rate(arm) <= rate('initial'))
        s = summary[arm]
        gates[arm] = dict(seed_checks=checks, passed=all(all(v.values()) for v in checks.values())
            and s['reliable_lives']/s['lives'] >= .8 and s['safe_active_lives']/s['lives'] >= .8)
    total = sum(r['decisions']*4 for r in training+rows)
    assert total == complete['total_ticks'] == 2*len(manifest['seeds'])*budget*4+len(rows)*ticks
    result = dict(preregistration_sha256=prereg, source_archive_verified=True,
        trace_hashes_verified=True, physical_traces_recounted=True, local_opportunities_recounted=True,
        fresh_starts_matched=True, equal_decision_update_budgets=True, private_finite_parameters=True,
        frozen_learning_verified=True, checkpoint_provenance_verified=True, food_conserved=True,
        disjoint_seeds=True, seconds=complete['seconds'], total_ticks=total,
        checkpoints=checkpoint_stats, summary=summary, seed_checks=seed_checks,
        primary_hypothesis_passed=primary, competence_gates=gates,
        interpretation='Three-seed diagnostic of reset frequency including body/resource/perceptual refresh. Observed crop output and routes do not establish planning, social behavior or a general architecture winner.')
    (out/('audit-archive.json' if archive_only else 'audit.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.archive_only), indent=2))
