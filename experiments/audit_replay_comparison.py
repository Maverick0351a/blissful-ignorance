"""Independently recount matched comparisons and retain negative results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_scarcity import recount


def audit(directory, archive_only=False):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    manifest = read('preregistration.json')
    rows = read('results.json')
    complete = read('complete.json')
    prereg_hash = hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    assert prereg_hash == (out/'preregistration.sha256').read_text(encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip') as archive:
        for file, sha in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(file)).hexdigest() == sha
            if not archive_only:
                assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == sha
    expected = {(int(s), w, c) for s, worlds in manifest['evaluation_seeds'].items()
                for w in worlds for c in manifest['conditions']}
    actual = {(r['seed'], r['evaluation_seed'], r['condition']) for r in rows}
    assert expected == actual and len(rows) == len(expected)
    training_worlds = set(sum(manifest['training_worlds'].values(), []))
    evaluation_worlds = set(sum(manifest['evaluation_seeds'].values(), []))
    assert not training_worlds & evaluation_worlds
    for seed in manifest['seeds']:
        for family in manifest['families']:
            curve = read(f'{seed}-{family}-learning-curve.json')
            assert [(r['stage'], r['episode']) for r in curve] == [
                (stage, i) for stage, n, _ in manifest['training_schedule'] for i in range(n)]
            assert [r['world_seed'] for r in curve] == manifest['training_worlds'][str(seed)]
            for rid in ('r0', 'r1'):
                assert curve[-1]['inspector'][rid]['decisions'] == manifest['decisions_per_agent']
                assert curve[-1]['inspector'][rid]['updates'] > 0
            for entry in curve:
                for m in entry['metrics'].values():
                    assert m['remaining_food_stock'] == 4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions']
    behaviors = {}
    ticks = manifest['evaluation_ticks']
    for row in rows:
        if row['family'] != 'control':
            assert row['frozen_before'] == row['frozen_after']
        peers = [r for r in rows if r['evaluation_seed'] == row['evaluation_seed']]
        assert all(r['arenas'] == row['arenas'] for r in peers)
        trace = out/f"{row['evaluation_seed']}-{row['condition']}-trace.jsonl"
        totals = recount(trace, row['arenas'], ticks)
        for rid, values in totals.items():
            for key, value in values.items():
                assert abs(row['metrics'][rid][key]-value) < 1e-7, (row['condition'], rid, key)
        counter = Counter()
        repeats = decisions = moves = tones = 0
        previous = {}
        tiles = {rid: {tuple(a['origin'])} for rid, a in row['arenas'].items()}
        with trace.open(encoding='utf-8') as handle:
            for line in handle:
                for rid, event in json.loads(line)['agents'].items():
                    tiles[rid].add(tuple(event['position']))
                    if not event['conscious']:
                        previous.pop(rid, None)
                        continue
                    symbol = json.dumps(event['action'], sort_keys=True)
                    decisions += 1
                    repeats += previous.get(rid) == symbol
                    previous[rid] = symbol
                    verb = event['action']['verb']
                    counter[verb] += 1
                    tones += verb == 'tone'
                    moves += verb == 'move' and event['success']
        behaviors[(row['evaluation_seed'], row['condition'])] = dict(
            action_counts=dict(counter), repeated_choices=repeats, conscious_decisions=decisions,
            successful_moves=moves, tones=tones, unique_tiles=list(map(len, tiles.values())))
    summaries = {}
    for condition in manifest['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        metrics = [m for r in selected for m in r['metrics'].values()]
        behavior = [behaviors[(r['evaluation_seed'], condition)] for r in selected]
        mean = lambda key: statistics.mean(m[key] for m in metrics)
        count = sum(b['conscious_decisions'] for b in behavior)
        summaries[condition] = dict(
            lives=len(metrics), zero_food_fraction=mean('zero_food_ticks')/ticks,
            safe_meals=mean('ordinary_eaten'), harmful_meals=mean('amber_eaten'),
            unconscious_fraction=mean('unconscious_ticks')/ticks, nutrition=mean('nutrition'),
            fullness=mean('mean_fullness'), thorn_contacts=mean('thorn_contacts'),
            plantings=mean('planted'), seed_conversions=mean('seed_conversions'),
            matured_food=mean('matured_food'), safe_foodward_crossings=mean('safe_foodward_crossings'),
            reliable_lives=sum(m['zero_food_ticks'] < .01*ticks for m in metrics),
            tones_fraction=sum(b['tones'] for b in behavior)/max(1, count),
            repeated_choice_fraction=sum(b['repeated_choices'] for b in behavior)/max(1, count),
            unique_tiles=statistics.mean(n for b in behavior for n in b['unique_tiles']))
    # Recompute the preregistered gate separately from the runner implementation.
    gates = {}
    for family in manifest['families']:
        checks = {}
        reliable = safe = lives = 0
        for seed in manifest['seeds']:
            groups = {c: [m for r in rows if r['seed'] == seed and r['condition'] == f'{family}-{c}'
                           for m in r['metrics'].values()] for c in ('trained', 'initial')}
            avg = lambda c, k: statistics.mean(m[k] for m in groups[c])
            rate = lambda c: sum(m['thorn_contacts'] for m in groups[c])/max(1, sum(m['conscious_decisions'] for m in groups[c]))
            checks[str(seed)] = dict(less_starvation=avg('trained', 'zero_food_ticks') < avg('initial', 'zero_food_ticks'),
                more_fullness=avg('trained', 'mean_fullness') > avg('initial', 'mean_fullness'),
                lower_contact_rate=rate('trained') <= .5*rate('initial'))
            for m in groups['trained']:
                lives += 1
                reliable += m['zero_food_ticks'] < .01*ticks
                safe += m['foodward_crossings'] > 0 and m['safe_foodward_crossings'] > 0 and m['safe_crossings']/m['crossings'] >= .8
        gates[family] = dict(seed_checks=checks, reliable_lives=reliable, safe_active_lives=safe, lives=lives,
            passed=all(all(c.values()) for c in checks.values()) and reliable/lives >= .8 and safe/lives >= .8)
    assert gates == complete['gates']
    total_ticks = 4*manifest['decisions_per_agent']*len(manifest['seeds'])*len(manifest['families']) + ticks*len(rows)
    assert total_ticks == complete['total_ticks']
    report = dict(preregistration_sha256=prereg_hash, verified_hashes=True, source_archive_verified=True,
        matched_arenas=True, held_out_worlds=True, trace_totals_match=True, food_conserved=True,
        frozen_learning_verified=True, gates=gates, summary=summaries,
        seconds=complete['seconds'], timing=complete['timing'], total_ticks=total_ticks,
        harmful_food_note='No amber fruit in this route arena: zero harmful meals does not establish avoidance.')
    (out/('audit-archive.json' if archive_only else 'audit.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.archive_only), indent=2))


if __name__ == '__main__':
    main()
