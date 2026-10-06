"""Recount outcomes, replay physics, and re-run frozen evaluation decisions."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.affordances import action_mask
from agents.network import ACTIONS
from agents.sequence_ppo import Brain, Config
from sim.world import World


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def expected_view(world, rid, condition, mapping):
    o = world.observe(rid)
    if condition == 'muted':
        o['hearing'] = [s for s in o['hearing'] if not s['kind'].startswith('tone_')]
        o['auditory_memory'] = []
    elif condition == 'shuffled':
        for event in o['hearing']:
            if event['kind'].startswith('tone_'):
                event['kind'] = 'tone_' + mapping[event['kind'][5:]]
        for event in o['auditory_memory']:
            event['tone'] = mapping[event['tone']]
    else:
        assert condition in ('intact', 'initial')
    return o


def sees_food(o):
    return any(t['terrain'] >= 0 and t.get('resource', {}).get('kind') in ('food', 'berry')
               and t['resource'].get('amount', 0) > 0 for t in o['tiles'])


def audit(folder, wall_seconds=300):
    started = time.perf_counter(); deadline = started + wall_seconds
    complete = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder / 'preregistration.json').read_text(encoding='utf-8'))
    assert digest(folder / 'preregistration.json') == (folder / 'preregistration.sha256').read_text().strip()
    for name, expected in complete['evidence_hashes'].items():
        assert digest(folder / name) == expected, name
    with zipfile.ZipFile(folder / 'source.zip') as archive:
        for name, expected in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
            assert digest(ROOT / name) == expected, 'Replay requires recorded source: ' + name
    recorded_results = {row['episode']: row for row in json.loads((folder / 'results.json').read_text())}
    checkpoints = {}; starts = {}; scores = {}; evaluated = 0; steps = 0; decisions = 0; episode_count = 0
    last_world = None
    with gzip.open(folder / 'trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if steps % 64 == 0 and time.perf_counter() > deadline:
                raise TimeoutError('Audit wall-time budget reached; evidence retained')
            if row['type'] == 'start':
                assert last_world is None, 'Missing episode end'
                header = row; meta = row['metadata']; episode = row['episode']
                w = World.from_dict(copy.deepcopy(row['world'])); last_world = w
                assert set(w.residents) == {'r0', 'r1'}
                measures = {rid: dict(nutrition=0., safe_eaten=0, amber_eaten=0, zero_food_ticks=0,
                    unconscious_ticks=0, first_meal_tick=None, first_food_seen_tick=None,
                    heard_hidden_decisions=0, tone_memory_decisions=0, uniform_tone_expected=0.,
                    actions={}, successful_actions={}) for rid in w.residents}
                positions = {rid: {(a.x, a.y)} for rid, a in w.residents.items()}
                brains = {}
                if meta['stage'] == 'evaluation':
                    seed, case, arm = meta['training_seed'], meta['case'], meta['arm']
                    key = (seed, case)
                    physical_start = canonical(row['world'])
                    assert starts.setdefault(key, physical_start) == physical_start, 'Unmatched starting world'
                    checkpoint = (seed, 'initial' if arm == 'initial' else 'trained')
                    if checkpoint not in checkpoints:
                        checkpoints[checkpoint] = torch.load(folder / f'{checkpoint[0]}-{checkpoint[1]}.pt', weights_only=True)
                    for i, rid in enumerate(w.residents):
                        brain = Brain(seed + i, Config(**manifest['config']))
                        brain.restore(copy.deepcopy(checkpoints[checkpoint][rid])); brain.freeze()
                        brain.rng = random.Random(w.seed + 900000 + i)
                        brains[rid] = brain
            elif row['type'] == 'step':
                assert row['episode'] == episode and row['tick'] == w.tick
                before = {rid: a.food for rid, a in w.residents.items()}
                if w.tick % 4 == 0:
                    for rid in w.residents:
                        o = expected_view(w, rid, header['conditions'][rid], header['permutations'][rid])
                        assert canonical(o) == canonical(row['observations'][rid]), ('Private view', episode, w.tick, rid)
                        command = row['commands'][rid]
                        if brains:
                            assert brains[rid].decide(o) == command, ('Policy replay', episode, w.tick, rid)
                            decisions += 1
                        visible = sees_food(o); m = measures[rid]
                        if visible and m['first_food_seen_tick'] is None:
                            m['first_food_seen_tick'] = w.tick
                        if o['auditory_memory']:
                            m['tone_memory_decisions'] += 1
                            m['heard_hidden_decisions'] += not visible
                        if not o['body']['unconscious']:
                            mask = action_mask(o, ACTIONS)
                            m['uniform_tone_expected'] += sum(ok and a['verb'] == 'tone' for a, ok in zip(ACTIONS, mask)) / sum(mask)
                        verb = command['verb']
                        m['actions'][verb] = m['actions'].get(verb, 0) + 1
                else:
                    assert all(c == {'verb': 'wait'} for c in row['commands'].values())
                result = w.step(row['commands'], scripted=False)
                assert canonical(result) == canonical(row['results']), ('Physics result', episode, row['tick'])
                for rid, a in w.residents.items():
                    actual = {key: getattr(a, key) for key in row['after'][rid]}
                    assert actual == row['after'][rid], ('Body replay', episode, row['tick'], rid)
                    m = measures[rid]; positions[rid].add((a.x, a.y))
                    # A zero-clipped body cannot gain nutrition just from adding
                    # the normal decay constant back to a zero difference.
                    gain = max(0., a.food + .008 - before[rid]) if a.food > 0 else 0.
                    m['nutrition'] += gain if gain > 1e-7 else 0.
                    m['zero_food_ticks'] += a.food <= 0
                    m['unconscious_ticks'] += a.unconscious
                    if row['tick'] % 4 == 0 and result[rid][0]:
                        command = row['commands'][rid]; verb = command['verb']
                        m['successful_actions'][verb] = m['successful_actions'].get(verb, 0) + 1
                        if verb == 'eat':
                            m['amber_eaten' if command.get('item') == 'amber_fruit' else 'safe_eaten'] += 1
                            if m['first_meal_tick'] is None:
                                m['first_meal_tick'] = row['tick']
                steps += 1
            else:
                assert row['type'] == 'end' and row['episode'] == episode
                assert w.tick == row['ticks'] == manifest['ticks_per_episode']
                assert canonical(w.to_dict()) == canonical(row['world']), ('Ending world', episode)
                for rid, m in measures.items():
                    m['unique_tiles'] = len(positions[rid])
                    for name, value in m.items():
                        actual = row['metrics'][rid][name]
                        if isinstance(value, float):
                            assert abs(actual - value) < 1e-6, (episode, rid, name, actual, value)
                        else:
                            assert actual == value, (episode, rid, name, actual, value)
                if meta['stage'] == 'evaluation':
                    expected = dict(row); expected.pop('type'); expected.pop('world')
                    assert expected == recorded_results[episode]
                    assert row['weights_before'] == row['weights_after']
                    for rid, brain in brains.items():
                        for module, expected_hash in row['weights_before'][rid].items():
                            actual_hash = hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in getattr(brain, module).state_dict().values())).hexdigest()
                            assert actual_hash == expected_hash
                    target = measures[meta['initially_uninformed']]
                    score = scores.setdefault((meta['training_seed'], meta['arm']), {'n': 0, 'fed': 0})
                    score['n'] += 1; score['fed'] += target['safe_eaten'] > 0
                    evaluated += 1
                episode_count += 1; last_world = None
    assert last_world is None
    assert steps == manifest['total_ticks'] == complete['total_ticks']
    assert evaluated == len(manifest['seeds']) * manifest['evaluation_cases'] * len(manifest['arms'])
    gates = []
    for s in complete['result']['seeds']:
        seed = s['seed']
        for arm in manifest['arms']:
            assert scores[seed, arm] == {'n': s['arms'][arm]['episodes'], 'fed': s['arms'][arm]['fed']}
        rate = lambda arm: scores[seed, arm]['fed'] / scores[seed, arm]['n']
        communication = rate('intact') >= .75 and all(rate('intact') - rate(a) >= .25 for a in ('muted', 'shuffled'))
        acquisition = rate('intact') > rate('initial')
        assert s['communication_gate'] == communication and s['acquisition_gate'] == acquisition
        gates.append(communication and acquisition)
    assert complete['result']['passed'] == all(gates)
    return dict(passed=True, seconds=time.perf_counter() - started, episodes=episode_count,
                replayed_ticks=steps, evaluations=evaluated, replayed_frozen_decisions=decisions,
                checks=['source and evidence hashes', 'matched worlds', 'private sensory views',
                        'frozen model decisions', 'all physics and ending worlds', 'food and behavior recount', 'preregistered gates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--wall-seconds', type=int, default=300)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_relative_to(ROOT / 'runs'):
        parser.error('Audit only a project runs directory')
    try:
        result = audit(folder, args.wall_seconds)
        (folder / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    except Exception as exc:
        (folder / 'audit-failed.json').write_text(json.dumps(dict(error=type(exc).__name__, reason=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
