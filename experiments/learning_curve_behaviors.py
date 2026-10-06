"""Post-hoc action descriptions for an already audited learning-curve run.

This changes no policy, checkpoint selection or preregistered gate. Frequencies
describe behavior; they do not establish communication or a cause of failure.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def describe(directory):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    audit, rows = read('audit.json'), read('results.json')
    assert audit['traces_recounted'] and audit['checkpoint_provenance_verified']
    lives, trace_hashes = [], {}
    for row in rows:
        trace = out/row['trace']
        trace_hashes[row['trace']] = hashlib.sha256(trace.read_bytes()).hexdigest()
        counts = {rid: Counter() for rid in row['metrics']}
        repeated = dict.fromkeys(counts, 0)
        previous = dict.fromkeys(counts)
        first_meal = dict.fromkeys(counts)
        tiles = {rid: {tuple(a['origin'])} for rid, a in row['arenas'].items()}
        ticks = 0
        with trace.open(encoding='utf-8') as handle:
            for line in handle:
                event = json.loads(line)
                ticks = event['tick']
                for rid, agent in event['agents'].items():
                    tiles[rid].add(tuple(agent['position']))
                    if not agent['conscious']:
                        previous[rid] = None
                        continue
                    command = json.dumps(agent['action'], sort_keys=True)
                    repeated[rid] += command == previous[rid]
                    previous[rid] = command
                    counts[rid][agent['action']['verb']] += 1
                    if first_meal[rid] is None and agent['success'] and agent['action']['verb'] == 'eat':
                        first_meal[rid] = ticks
        for rid, verbs in counts.items():
            m = row['metrics'][rid]
            conscious = sum(verbs.values())
            assert conscious == m['conscious_decisions']
            lives.append(dict(seed=row['seed'], agent=rid, split=row['split'],
                world_seed=row['world_seed'], condition=row['condition'], budget=row['budget'],
                zero_food_fraction=m['zero_food_ticks']/ticks, fullness=m['mean_fullness'],
                safe_meals=m['ordinary_eaten'], seed_conversions=m['seed_conversions'],
                plantings=m['planted'], thorn_contacts=m['thorn_contacts'],
                first_meal_tick=first_meal[rid], unique_tiles=len(tiles[rid]),
                conscious_choices=conscious, repeated_choices=repeated[rid],
                verb_counts=dict(verbs)))
    groups = defaultdict(list)
    for life in lives:
        groups[f"{life['split']}-{life['budget']}-{life['condition']}"].append(life)
    summary = {}
    for name, group in groups.items():
        verbs = Counter()
        for life in group:
            verbs.update(life['verb_counts'])
        choices = sum(verbs.values())
        summary[name] = dict(lives=len(group), conscious_choices=choices,
            verb_fractions={v: n/max(1, choices) for v, n in verbs.most_common()},
            repeated_choice_fraction=sum(l['repeated_choices'] for l in group)/max(1, choices),
            mean_unique_tiles=statistics.mean(l['unique_tiles'] for l in group),
            mean_plantings=statistics.mean(l['plantings'] for l in group),
            mean_seed_conversions=statistics.mean(l['seed_conversions'] for l in group))
    result = dict(analysis='post_hoc_descriptive_only',
        time_reference='First meal timestamps are the end of the four-tick observation interval.',
        preregistration_sha256=audit['preregistration_sha256'],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trace_hashes=trace_hashes, lives=lives, summary=summary)
    (out/'behavior-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(describe(args.directory)['summary'], indent=2))
