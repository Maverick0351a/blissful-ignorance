"""Independent accounting for interface-based PPO learning curves."""
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
from agents.sequence_ppo import Brain, Config
from experiments.audit_scarcity import recount
from experiments.replay_comparison import digest_tree, learning_digest


def audit(directory, archive_only=False):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    manifest, rows, complete, curve = map(read, ('preregistration.json', 'results.json', 'complete.json', 'learning-curves.json'))
    h = hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    assert h == (out/'preregistration.sha256').read_text()
    with zipfile.ZipFile(out/'source.zip') as archive:
        for file, sha in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(file)).hexdigest() == sha
            if not archive_only:
                assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == sha
    expected = set()
    for seed, splits in manifest['maps'].items():
        for split, maps in splits.items():
            for world in maps:
                for budget in manifest['budgets'] if split == 'validation' else manifest['budgets'][-1:]:
                    expected.add((int(seed), split, world, budget, 'trained'))
                for condition in ('initial', 'random', 'oracle'):
                    expected.add((int(seed), split, world, 0, condition))
    assert {(r['seed'], r['split'], r['world_seed'], r['budget'], r['condition']) for r in rows} == expected
    assert len(rows) == len(expected)
    training = set(sum(manifest['training_maps'].values(), []))
    validation = {w for s in manifest['maps'].values() for w in s['validation']}
    final = {w for s in manifest['maps'].values() for w in s['final']}
    assert not training & validation and not training & final and not validation & final
    ticks = manifest['evaluation_ticks']
    for row in rows:
        assert row['frozen_before'] == row['frozen_after']
        assert all(r['arenas'] == row['arenas'] for r in rows if r['world_seed'] == row['world_seed'])
        totals = recount(out/row['trace'], row['arenas'], ticks)
        for rid, values in totals.items():
            for key, value in values.items():
                assert abs(row['metrics'][rid][key]-value) < 1e-7, (row['trace'], rid, key)
    for seed in manifest['seeds']:
        entries = [e for e in curve if e['seed'] == seed]
        assert [e['world_seed'] for e in entries] == manifest['training_maps'][str(seed)]
        for i, e in enumerate(entries):
            assert e['episode'] == i and e['stage'] == manifest['stages'][i % len(manifest['stages'])]
            assert e['decisions'] == (i+1)*manifest['episode_decisions']
            for m in e['metrics'].values():
                assert m['remaining_food_stock'] == 4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions']
        for budget in manifest['budgets']:
            name = f'{seed}-{budget}-checkpoint.pt'
            assert hashlib.sha256((out/name).read_bytes()).hexdigest() == complete['checkpoints'][name]
            data = torch.load(out/name, map_location='cpu', weights_only=True)
            env = LearningEnv(tuple(data['brains']), max_cycles=manifest['episode_decisions'])
            env.restore(data['environment'])
            assert not env.agents
            assert data['next_episode'] == budget//manifest['episode_decisions']
            pointers = set()
            for rid, state in data['brains'].items():
                b = Brain(state['seed'], Config(**state['config']))
                b.restore(state)
                assert b.decisions == budget and b.updates > 0
                assert not b.buffer and b.pending is None
                initial = Brain(b.seed, b.config)
                assert digest_tree(b.model.state_dict()) != digest_tree(initial.model.state_dict())
                for parameter in list(b.model.parameters())+list(b.predictor.parameters()):
                    assert bool(torch.isfinite(parameter).all())
                    assert parameter.data_ptr() not in pointers
                    pointers.add(parameter.data_ptr())
                b.freeze()
                for row in rows:
                    if row['seed'] == seed and row['budget'] == budget:
                        assert learning_digest(b) == row['frozen_before'][rid]
        for rid in ('r0', 'r1'):
            b = Brain(seed+int(rid[1:]), Config(**manifest['config']))
            b.freeze()
            for row in rows:
                if row['seed'] == seed and row['budget'] == 0:
                    assert learning_digest(b) == row['frozen_before'][rid]
    summary, gates = {}, {}
    for split in ('validation', 'final'):
        levels = manifest['budgets'] if split == 'validation' else manifest['budgets'][-1:]
        for budget in levels:
            group = [r for r in rows if r['split'] == split and r['budget'] == budget]
            metrics = [m for r in group for m in r['metrics'].values()]
            summary[f'{split}-{budget}'] = dict(lives=len(metrics),
                zero_food_fraction=statistics.mean(m['zero_food_ticks'] for m in metrics)/ticks,
                fullness=statistics.mean(m['mean_fullness'] for m in metrics),
                safe_meals=statistics.mean(m['ordinary_eaten'] for m in metrics),
                thorn_contacts=statistics.mean(m['thorn_contacts'] for m in metrics),
                reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in metrics))
            checks, reliable, safe, n = {}, 0, 0, 0
            for seed in manifest['seeds']:
                groups = {c: [m for r in rows if r['seed'] == seed and r['split'] == split and r['condition'] == c
                              and r['budget'] == (budget if c == 'trained' else 0) for m in r['metrics'].values()]
                          for c in ('trained', 'initial')}
                avg = lambda c, k: statistics.mean(m[k] for m in groups[c])
                rate = lambda c: sum(m['thorn_contacts'] for m in groups[c])/max(1, sum(m['conscious_decisions'] for m in groups[c]))
                checks[str(seed)] = dict(less_starvation=avg('trained', 'zero_food_ticks') < avg('initial', 'zero_food_ticks'),
                    more_fullness=avg('trained', 'mean_fullness') > avg('initial', 'mean_fullness'),
                    lower_contact_rate=rate('trained') <= .5*rate('initial'))
                for m in groups['trained']:
                    n += 1
                    reliable += m['zero_food_ticks'] < .01*ticks
                    safe += m['foodward_crossings'] > 0 and m['safe_foodward_crossings'] > 0 and m['safe_crossings']/m['crossings'] >= .8
            gates[f'{split}-{budget}'] = dict(seed_checks=checks, reliable_lives=reliable, safe_active_lives=safe, lives=n,
                passed=all(all(c.values()) for c in checks.values()) and reliable/n >= .8 and safe/n >= .8)
        for condition in ('initial', 'random', 'oracle'):
            metrics = [m for r in rows if r['split'] == split and r['condition'] == condition for m in r['metrics'].values()]
            summary[f'{split}-{condition}'] = dict(lives=len(metrics),
                zero_food_fraction=statistics.mean(m['zero_food_ticks'] for m in metrics)/ticks,
                fullness=statistics.mean(m['mean_fullness'] for m in metrics),
                safe_meals=statistics.mean(m['ordinary_eaten'] for m in metrics),
                reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in metrics))
    assert gates == complete['gates']
    total = 4*manifest['budgets'][-1]*len(manifest['seeds'])+len(rows)*ticks
    assert total == complete['total_ticks']
    result = dict(preregistration_sha256=h, hashes_verified=True, traces_recounted=True,
        food_conserved=True, disjoint_train_validation_final=True, frozen_learning=True,
        checkpoint_provenance_verified=True, private_finite_parameters=True,
        summary=summary, gates=gates, total_ticks=total, seconds=complete['seconds'])
    (out/('audit-archive.json' if archive_only else 'audit.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.archive_only), indent=2))
