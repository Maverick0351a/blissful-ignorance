"""Export an offline public progress snapshot from already audited evidence.

Does not train, load weights, contact a service, or edit the source run.
The committed export works without the private runs directory.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs' / 'practice-amount-20261006'
OUTPUT = ROOT / 'docs' / 'progress'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((SOURCE / name).read_text(encoding='utf-8'))


def main():
    complete, audit, verified = read('complete.json'), read('audit.json'), read('verification.json')
    if not (audit['passed'] and verified['passed'] and complete['source_hashes_valid']):
        raise ValueError('A completed, audited, source-verified run is required')
    expected = {
        'analysis.json': complete['evidence_hashes']['analysis.json'],
        'preregistration.json': complete['evidence_hashes']['preregistration.json'],
        'audit.json': verified['derived_hashes']['audit.json'],
        'replay-adjacent.html': verified['derived_hashes']['replay-adjacent.html'],
    }
    for name, digest in expected.items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f'Source evidence changed: {name}')
    analysis = read('analysis.json')
    protocol = read('preregistration.json')
    keep = ('lives', 'prompt_successes', 'full_life_successes', 'ordinary_meals',
            'amber_meals', 'zero_food_ticks', 'unconscious_ticks', 'resident_ticks',
            'mean_unique_tiles', 'actions', 'damage', 'invalid')

    def scores(raw):
        return {arm: {key: row[key] for key in keep} for arm, row in raw.items()}

    fixtures = {}
    for task, data in analysis['fixtures'].items():
        fixtures[task] = {'arms': scores(data['arms']),
                          'candidates': {cid: scores(rows) for cid, rows in data['candidates'].items()}}
    exported = {
        'schema': 1, 'date': '2026-10-06', 'experiment': 'retained-practice-amount',
        'fixtures': fixtures, 'checks': analysis['checks'],
        'pilot_promising': analysis['pilot_promising'], 'milestone_confirmed': analysis['milestone_confirmed'],
        'audit': {key: audit[key] for key in ('passed', 'seconds', 'episodes', 'replayed_ticks',
                  'replayed_training_decisions', 'replayed_frozen_decisions',
                  'reproduced_updates', 'full_checkpoint_states', 'shared_code')},
        'tests': {'count': verified['tests'], 'optional_pettingzoo_passed': verified['optional_pettingzoo_passed'],
                  'scope': 'Recorded full suite from the practice experiment, not rerun by this exporter'},
        'protocol': {'seeds': protocol['seeds'], 'config': protocol['config'],
                     'starting_decisions_per_brain': 6144, 'additional_practice_lives': [0, 16, 64],
                     'practice_life_ticks': 512, 'adjacent_deadline_exclusive': 64,
                     'carried_deadline_exclusive': 16, 'evaluation_frozen': True},
        'source_hashes': expected,
        'limits': ['Six retained brains in three historical training groups; cases are clustered.',
                   'Starting policies were already trained. No random-policy arm in this comparison.',
                   'Fresh seeded instances of the same nearby-food fixture; not navigation transfer.',
                   'GT-01 remains open. No experimental weights were imported into live residents.',
                   'Public summary and selected replay only. Full traces and checkpoints remain local.'],
    }
    # Read inert JSON from the verified prior replay; never execute its JavaScript.
    original = (SOURCE / 'replay-adjacent.html').read_text(encoding='utf-8')
    match = re.search(r'const DATA=', original)
    if not match:
        raise ValueError('Recorded replay payload missing')
    replay, _ = json.JSONDecoder().raw_decode(original[match.end():])
    if set(replay) != {'starting', 'after-16', 'after-64'}:
        raise ValueError('Unexpected replay conditions')
    metric_keys = ('first_meal_tick', 'first_gather_food_tick', 'first_zero_tick', 'safe_eaten')
    for episode in replay.values():
        episode['metrics'] = {rid: {key: row[key] for key in metric_keys}
                              for rid, row in episode['metrics'].items()}
        if len(episode['frames']) != 129 or episode['frames'][-1]['tick'] != 512:
            raise ValueError('Incomplete selected replay')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'results.json').write_text(json.dumps(exported, indent=2) + '\n', encoding='utf-8')
    for name, global_name, data in [('results.js', 'PROGRESS', exported), ('replay.js', 'RECORDED', replay)]:
        body = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
        (OUTPUT / name).write_text(f'"use strict";\nwindow.{global_name} = {body};\n', encoding='utf-8')
    print(json.dumps({'exported': 'docs/progress', 'source_hashes_verified': len(expected),
                      'candidates': 6, 'replay_conditions': len(replay), 'frames_per_condition': 129}))


if __name__ == '__main__':
    main()
