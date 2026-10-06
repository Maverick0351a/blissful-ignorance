"""Independent trace, checkpoint and paired-result audit for memory diagnostic."""
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
from agents.memory_ppo import Brain
from agents.network import ACTIONS
from agents.sequence_ppo import Config
from experiments.audit_scarcity import recount
from experiments.replay_comparison import learning_digest


def recount_context(out, row, decisions, periodic, interval):
    positions = {rid: list(a['origin']) for rid, a in row['arenas'].items()}
    seen = {rid: set() for rid in positions}
    active, events = dict.fromkeys(positions), {rid: [] for rid in positions}
    visited = {rid: set() for rid in positions}
    with (out/row['senses']).open(encoding='utf-8') as senses, (out/row['trace']).open(encoding='utf-8') as trace:
        for decision in range(decisions):
            sense, physical = json.loads(next(senses)), json.loads(next(trace))
            assert sense['decision'] == decision and physical['tick'] == 4*(decision+1)
            assert set(sense['agents']) == set(positions) == set(physical['agents'])
            for rid, item in sense['agents'].items():
                body = physical['agents'][rid]
                assert item['position'] == positions[rid]
                assert item['conscious'] == body['conscious']
                assert ACTIONS[item['action']] == body['action']
                assert item['boundary'] == bool(periodic and decision and decision % interval == 0)
                positions[rid] = body['position']
                point = tuple(item['position'])
                visited[rid].add(point)
                visible = set(map(tuple, item['visible_food']))
                assert all(max(abs(a-b) for a, b in zip(p, point)) <= 4 for p in visible)
                remaining = set(map(tuple, item['previous_food_remaining']))
                permitted = seen[rid] | ({tuple(active[rid]['target'])} if active[rid] else set())
                assert remaining <= permitted
                pending = active[rid]
                if pending is not None:
                    if item['boundary']:
                        pending['crossed_boundary'] = True
                    target = tuple(pending['target'])
                    outcome = None
                    if target not in remaining:
                        outcome = 'depleted'
                    elif abs(target[0]-point[0])+abs(target[1]-point[1]) <= 1:
                        outcome = 'returned'
                    elif decision-pending['opened'] >= 128:
                        outcome = 'expired'
                    if outcome is not None:
                        pending.update(outcome=outcome, closed=decision)
                        events[rid].append(pending)
                        active[rid] = None
                if active[rid] is None and item['conscious']:
                    candidates = []
                    for target in sorted(seen[rid]):
                        if (target not in visible and target in remaining
                                and abs(target[0]-point[0])+abs(target[1]-point[1]) > 1):
                            candidates.append(target)
                    if candidates:
                        active[rid] = dict(target=list(candidates[0]), cue=decision-1,
                                           opened=decision, crossed_boundary=item['boundary'])
                seen[rid] = visible if item['conscious'] else set()
        assert next(senses, None) is None and next(trace, None) is None
    for rid, pending in active.items():
        if pending:
            pending.update(outcome='censored', closed=decisions)
            events[rid].append(pending)
    assert events == row['events']
    for rid, records in events.items():
        totals = dict(food_loss_opportunities=len(records),
            food_returns=sum(e['outcome'] == 'returned' for e in records),
            boundary_opportunities=sum(e['crossed_boundary'] for e in records),
            boundary_returns=sum(e['crossed_boundary'] and e['outcome'] == 'returned' for e in records),
            expired=sum(e['outcome'] == 'expired' for e in records),
            depleted=sum(e['outcome'] == 'depleted' for e in records),
            censored=sum(e['outcome'] == 'censored' for e in records), unique_tiles=len(visited[rid]))
        assert all(row['metrics'][rid][key] == value for key, value in totals.items())


def audit(directory):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    manifest, rows, training, complete = map(read, ('preregistration.json', 'results.json', 'training.json', 'complete.json'))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    prereg = sha(out/'preregistration.json')
    assert prereg == (out/'preregistration.sha256').read_text(encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip') as archive:
        for name, digest in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest == sha(ROOT/name)
    for name, digest in complete['evidence_hashes'].items():
        assert sha(out/name) == digest
    expected = {(seed, mode, ep) for seed in manifest['seeds']
                for mode in ('reset', 'rebuild') for ep in range(manifest['episodes'])}
    assert {(r['seed'], r['mode'], r['episode']) for r in training} == expected
    assert len(training) == len(expected)
    expected = {(int(seed), world, condition) for seed, maps in manifest['test_maps'].items()
                for world in maps for condition in manifest['conditions']}
    assert {(r['seed'], r['world_seed'], r['condition']) for r in rows} == expected
    assert len(rows) == len(expected)
    train_seeds = set(sum(manifest['training_maps'].values(), []))
    test_seeds = set(sum(manifest['test_maps'].values(), []))
    assert not train_seeds & test_seeds
    assert len(train_seeds) == manifest['episodes']*len(manifest['seeds'])
    assert len(test_seeds) == sum(map(len, manifest['test_maps'].values()))
    for row in training+rows:
        is_training = 'episode' in row
        decisions = manifest['episode_decisions'] if is_training else manifest['evaluation_ticks']//4
        periodic = is_training or row['condition'].endswith('-reset')
        expected_metrics = recount(out/row['trace'], row['arenas'], decisions*4)
        for rid, values in expected_metrics.items():
            for key, value in values.items():
                assert abs(row['metrics'][rid][key]-value) < 1e-7, (row['trace'], rid, key)
            assert row['metrics'][rid]['amber_eaten'] == 0
        recount_context(out, row, decisions, periodic, manifest['config']['rollout'])
        if is_training:
            assert row['world_seed'] == manifest['training_maps'][str(row['seed'])][row['episode']]
            assert row['stage'] == manifest['stages'][row['episode'] % len(manifest['stages'])]
            for rid, inspector in row['inspectors'].items():
                assert inspector['decisions'] == (row['episode']+1)*decisions
                assert inspector['updates'] == inspector['decisions']//manifest['config']['rollout']
                initial = Brain(row['seed']+int(rid[1:]), Config(**manifest['config']), row['mode'])
                assert learning_digest(initial) == row['initial_digests'][rid]
        else:
            assert row['frozen_before'] == row['frozen_after']
            assert all(r['arenas'] == row['arenas'] for r in rows if r['world_seed'] == row['world_seed'])
    for seed in manifest['seeds']:
        for mode in ('reset', 'rebuild'):
            name = f'{seed}-{mode}-checkpoint.pt'
            assert sha(out/name) == complete['checkpoints'][name]
            data = torch.load(out/name, map_location='cpu', weights_only=True)
            assert not data['environment']['agents']
            pointers, loaded = set(), []
            for rid, state in data['brains'].items():
                b = Brain(state['seed'], Config(**state['config']), mode, manifest['memory_capacity'])
                b.restore(state)
                loaded.append(b)
                assert b.decisions == manifest['decisions_per_resident']
                assert b.updates == b.decisions//b.config.rollout == b.boundaries
                assert b.rebuilds == (b.updates if mode == 'rebuild' else 0)
                assert b.context_forward_steps == (b.decisions if mode == 'rebuild' else 0)
                assert not b.history and not b.buffer and b.pending is None
                for parameter in list(b.model.parameters())+list(b.predictor.parameters()):
                    assert parameter.data_ptr() not in pointers and bool(torch.isfinite(parameter).all())
                    pointers.add(parameter.data_ptr())
                b.freeze()
                for row in rows:
                    if row['seed'] == seed and row['condition'].startswith(mode+'-'):
                        assert learning_digest(b) == row['frozen_before'][rid]
        for rid in ('r0', 'r1'):
            initial = Brain(seed+int(rid[1:]), Config(**manifest['config']))
            initial.freeze()
            for row in rows:
                if row['seed'] == seed and row['condition'].split('-')[0] in ('initial', 'random', 'oracle'):
                    assert learning_digest(initial) == row['frozen_before'][rid]
    ticks = manifest['evaluation_ticks']
    summary = {}
    for condition in manifest['conditions']:
        group = [m for r in rows if r['condition'] == condition for m in r['metrics'].values()]
        summary[condition] = dict(lives=len(group),
            zero_food_fraction=statistics.mean(m['zero_food_ticks'] for m in group)/ticks,
            fullness=statistics.mean(m['mean_fullness'] for m in group),
            safe_meals=statistics.mean(m['ordinary_eaten'] for m in group),
            thorn_contacts=statistics.mean(m['thorn_contacts'] for m in group),
            unconscious_fraction=statistics.mean(m['unconscious_ticks'] for m in group)/ticks,
            plantings=statistics.mean(m['planted'] for m in group),
            unique_tiles=statistics.mean(m['unique_tiles'] for m in group),
            reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in group),
            opportunities=sum(m['food_loss_opportunities'] for m in group),
            returns=sum(m['food_returns'] for m in group),
            boundary_opportunities=sum(m['boundary_opportunities'] for m in group),
            boundary_returns=sum(m['boundary_returns'] for m in group))
    seed_checks = {}
    for seed in manifest['seeds']:
        means = {}
        for condition in ('reset-continuous', 'rebuild-continuous'):
            group = [m for r in rows if r['seed'] == seed and r['condition'] == condition for m in r['metrics'].values()]
            means[condition] = statistics.mean(m['zero_food_ticks'] for m in group)/ticks
        seed_checks[str(seed)] = dict(zero_food=means,
            improved=means['rebuild-continuous'] < means['reset-continuous'])
    passed = all(r['improved'] for r in seed_checks.values()) and summary['rebuild-continuous']['fullness'] > summary['reset-continuous']['fullness']
    total = len(training)*manifest['episode_decisions']*4+len(rows)*ticks
    assert total == complete['total_ticks']
    result = dict(preregistration_sha256=prereg, source_archive_verified=True,
        trace_hashes_verified=True, physical_traces_recounted=True, context_events_recounted=True,
        equal_decision_update_budgets=True, frozen_learning_verified=True,
        checkpoint_provenance_verified=True, private_finite_parameters=True,
        disjoint_seeds=True, food_conserved=True, total_ticks=total, seconds=complete['seconds'],
        summary=summary, seed_checks=seed_checks, primary_hypothesis_passed=passed,
        interpretation='Three-seed diagnostic; return rates are conditioned on different encountered opportunities and do not establish remembered navigation, lifelong learning, or causal mediation.')
    (out/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
