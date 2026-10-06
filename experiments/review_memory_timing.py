"""Post-hoc chronological check of boundaries before food-return outcomes.

Pre-action position at decision d results from action d-1. A boundary occurring
inside decide(d) cannot have helped that already-completed movement. Keep the
preregistered inclusive marker counts, but report strict before-outcome counts
separately. No learned checkpoint or primary feeding hypothesis is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def before_outcome(event, boundary_decisions):
    return any(event['opened'] <= d < event['closed'] for d in boundary_decisions)


def review(directory):
    out = Path(directory)
    read = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    audit = read('audit.json')
    assert audit['context_events_recounted'] and audit['trace_hashes_verified']
    rows = read('training.json')+read('results.json')
    complete = read('complete.json')
    groups, details = {}, []
    for row in rows:
        path = out/row['senses']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == complete['evidence_hashes'][row['senses']]
        boundaries = {rid: set() for rid in row['events']}
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                frame = json.loads(line)
                for rid, item in frame['agents'].items():
                    if item['boundary']:
                        boundaries[rid].add(frame['decision'])
        group = 'training-'+row['mode'] if 'episode' in row else 'evaluation-'+row['condition']
        total = groups.setdefault(group, dict(opportunities=0, returns=0,
            boundaries_before_outcome=0, returns_after_boundary=0,
            inclusive_markers_removed=0, inclusive_return_markers_removed=0))
        for rid, events in row['events'].items():
            for event in events:
                crossed = before_outcome(event, boundaries[rid])
                returned = event['outcome'] == 'returned'
                assert not crossed or event['crossed_boundary']
                total['opportunities'] += 1
                total['returns'] += returned
                total['boundaries_before_outcome'] += crossed
                total['returns_after_boundary'] += crossed and returned
                removed = event['crossed_boundary'] and not crossed
                total['inclusive_markers_removed'] += removed
                total['inclusive_return_markers_removed'] += removed and returned
                if removed:
                    details.append(dict(senses=row['senses'], resident=rid, event=event))
    source = Path(__file__).read_bytes()
    (out/'timing-review-source.py').write_bytes(source)
    result = dict(post_hoc=True, source_sha256=hashlib.sha256(source).hexdigest(),
        preregistration_sha256=audit['preregistration_sha256'], groups=groups,
        excluded_boundary_at_closure=details,
        primary_feeding_hypothesis_unchanged=True,
        interpretation='Strict markers exclude an update/reset after the movement producing the outcome. Return counts remain observational and do not demonstrate remembered navigation.')
    (out/'timing-review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(review(args.directory), indent=2))
