"""Independent outcome/reward recount, physics and frozen-policy replay for GT-01."""
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
from agents.sequence_ppo import Config
from experiments.category_trial import MeasuredFlat, MeasuredCategory
from sim.world import World


def canonical(value):return json.dumps(value, sort_keys=True, separators=(',', ':'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def weights(state):
    return {name: hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in state[name].values())).hexdigest()
            for name in ('model', 'predictor')}


def audit(folder, wall_seconds=600):
    started = time.perf_counter(); deadline = started+wall_seconds
    receipt = json.loads((folder/'complete.json').read_text())
    manifest = json.loads((folder/'preregistration.json').read_text())
    assert digest(folder/'preregistration.json') == (folder/'preregistration.sha256').read_text().strip()
    for name, value in receipt['evidence_hashes'].items():assert digest(folder/name) == value, name
    with zipfile.ZipFile(folder/'source.zip') as archive:
        for name, value in manifest['hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == value, name
            assert digest(ROOT/name) == value, 'Use recorded source for replay: '+name
    saved_rows = {r['episode']: r for name in ('results.json', 'training.json')
                  for r in json.loads((folder/name).read_text())}
    checkpoints = {}
    for seed in manifest['seeds']:
        for kind in ('flat', 'category'):
            for variant in ('initial', 'trained'):
                key = (seed, kind, variant)
                checkpoints[key] = torch.load(folder/f'{seed}-{kind}-{variant}.pt', weights_only=True)
                assert set(checkpoints[key]) == {'r0', 'r1'}
                for rid, state in checkpoints[key].items():
                    assert state['config'] == manifest['config']
                    assert state['seed'] == seed + int(rid[1:])
                    for module in ('model', 'predictor'):
                        assert all(torch.isfinite(t).all() for t in state[module].values())
        for rid in ('r0', 'r1'):
            flat = checkpoints[seed, 'flat', 'initial'][rid]
            category = checkpoints[seed, 'category', 'initial'][rid]
            for name, tensor in flat['model'].items():
                other = name.replace('actor.', 'actor.arguments.') if name.startswith('actor.') else name
                assert torch.equal(tensor, category['model'][other]), 'Unmatched initial core'
            for name, tensor in flat['predictor'].items():assert torch.equal(tensor, category['predictor'][name])
    starts = {}; training_ends = {}; scores = {}; episode_ids = set()
    steps = decisions = evaluations = 0; w = None
    with gzip.open(folder/'trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            if steps % 64 == 0 and time.perf_counter() > deadline:raise TimeoutError('Audit budget exhausted; files retained')
            row = json.loads(line)
            if row['type'] == 'start':
                assert w is None, 'Missing episode end'
                header = row; episode = row['episode']; meta = row['metadata']
                assert episode not in episode_ids; episode_ids.add(episode)
                w = World.from_dict(copy.deepcopy(row['world']))
                assert set(w.residents) == {'r0', 'r1'} and w.tick == 0
                for actor in w.residents.values():
                    assert actor.food == 4 and not any(actor.inventory.values())
                    assert actor.health == actor.water == actor.energy == actor.warmth == 100
                physical = canonical(row['world']); key = (meta['training_seed'], meta['stage'], meta['case'])
                assert starts.setdefault(key, physical) == physical, 'Unmatched starting worlds'
                m = {rid: dict(first_meal_tick=None, first_gather_food_tick=None, first_zero_tick=None,
                     first_food_seen_tick=None, actions={}, successful_actions={}, conscious_tones=0,
                     nutrition=0., zero_food_ticks=0, unconscious_ticks=0, safe_eaten=0, amber_eaten=0,
                     invalid=0, conscious_invalid=0, conscious_decisions=0, damage=0., reward_totals={}) for rid in w.residents}
                positions = {rid: {(a.x, a.y)} for rid, a in w.residents.items()}
                brains = {}; seed = meta['training_seed']; kind, variant = meta['arm'].split('-')
                if meta['stage'] == 'evaluation':
                    saved = checkpoints[seed, kind, 'trained' if variant == 'trained' else 'initial']
                    for i, rid in enumerate(w.residents):
                        cls = MeasuredCategory if kind == 'category' else MeasuredFlat
                        brain = cls(seed+i, Config(**manifest['config']))
                        brain.restore(copy.deepcopy(saved[rid])); brain.freeze()
                        brain.random_policy = variant == 'random'; brain.rng = random.Random(w.seed+900000+i)
                        brains[rid] = brain
                        assert weights(saved[rid]) == row['weights'][rid]
                else:
                    key = (seed, kind)
                    previous = training_ends.get(key, {rid: weights(b) for rid, b in checkpoints[seed, kind, 'initial'].items()})
                    assert previous == row['weights'], 'Training weights lost across lives'
            elif row['type'] == 'step':
                assert row['episode'] == episode and row['tick'] == w.tick
                before = {rid: (a.food, a.health, a.pain, a.unconscious, a.inventory['food']) for rid, a in w.residents.items()}
                deciding = w.tick % 4 == 0
                if deciding:
                    for rid in w.residents:
                        obs = w.observe(rid); command = row['commands'][rid]; s = m[rid]
                        assert canonical(obs) == canonical(row['observations'][rid]), 'Private view mismatch'
                        assert action_mask(obs, ACTIONS)[ACTIONS.index(command)], 'Physically masked action selected'
                        if brains:
                            assert brains[rid].decide(obs) == command, (episode, w.tick, rid, 'Policy mismatch')
                            decisions += 1
                        seen_food = any(t['terrain'] >= 0 and t.get('resource', {}).get('kind') in ('berry', 'food')
                                        and t['resource'].get('amount', 0) > 0 for t in obs['tiles'])
                        if seen_food and s['first_food_seen_tick'] is None:s['first_food_seen_tick'] = w.tick
                        verb = command['verb']; s['actions'][verb] = s['actions'].get(verb, 0)+1
                        s['conscious_tones'] += verb == 'tone' and not obs['body']['unconscious']
                else:assert all(c == {'verb': 'wait'} for c in row['commands'].values())
                actual = w.step(row['commands'], scripted=False)
                assert canonical(actual) == canonical(row['results']), 'Physics action result mismatch'
                for rid, actor in w.residents.items():
                    s = m[rid]; food, health, pain, unconscious, inventory_food = before[rid]
                    assert {k: getattr(actor, k) for k in row['after'][rid]} == row['after'][rid]
                    gain = max(0., actor.food+.008-food) if actor.food > 0 else 0.
                    if gain < 1e-7:gain = 0.
                    injury = max(0., health-actor.health, actor.pain-pain)
                    expected_reward = dict(food=gain/25, hunger=-min(1., max(0., (60-actor.food)/60))*.005, injury=-injury/20)
                    for name, value in expected_reward.items():
                        assert abs(value-row['rewards'][rid][name]) < 1e-9, 'Reward changed'
                        s['reward_totals'][name] = s['reward_totals'].get(name, 0.)+value
                    assert set(expected_reward) == set(row['rewards'][rid])
                    s['nutrition'] += gain; s['damage'] += injury
                    s['zero_food_ticks'] += actor.food <= 0; s['unconscious_ticks'] += actor.unconscious
                    if actor.food <= 0 and s['first_zero_tick'] is None:s['first_zero_tick'] = row['tick']
                    positions[rid].add((actor.x, actor.y))
                    if deciding:
                        command = row['commands'][rid]; verb = command['verb']; success = actual[rid][0]
                        s['invalid'] += not success
                        if not unconscious:
                            s['conscious_decisions'] += 1; s['conscious_invalid'] += not success
                        if success:
                            s['successful_actions'][verb] = s['successful_actions'].get(verb, 0)+1
                            if verb == 'gather' and actor.inventory['food'] > inventory_food and s['first_gather_food_tick'] is None:
                                s['first_gather_food_tick'] = row['tick']
                            if verb == 'eat':
                                s['amber_eaten' if command.get('item') == 'amber_fruit' else 'safe_eaten'] += 1
                                if command.get('item', 'food') == 'food' and s['first_meal_tick'] is None:s['first_meal_tick'] = row['tick']
                steps += 1
            else:
                assert row['type'] == 'end' and row['episode'] == episode
                assert w.tick == row['ticks'] == manifest['ticks_per_episode']
                assert canonical(w.to_dict()) == canonical(row['world']), 'Ending world mismatch'
                saved = dict(row); saved.pop('type'); saved.pop('world')
                assert saved == saved_rows[episode], 'Result table mismatch'
                for rid, s in m.items():
                    s['unique_tiles'] = len(positions[rid])
                    s['timely_acquisition'] = (s['first_gather_food_tick'] is not None and s['first_meal_tick'] is not None
                        and s['first_gather_food_tick'] < s['first_meal_tick']
                        and (s['first_zero_tick'] is None or s['first_meal_tick'] < s['first_zero_tick']))
                    for name, value in s.items():
                        observed = row['metrics'][rid][name]
                        if isinstance(value, float):assert abs(value-observed) < 1e-6, (episode, rid, name)
                        elif name == 'reward_totals':
                            for k, v in value.items():assert abs(v-observed[k]) < 1e-6
                        else:assert value == observed, (episode, rid, name)
                if meta['stage'] == 'evaluation':
                    assert row['weights_before'] == row['weights_after'] == header['weights']
                    for rid, brain in brains.items():assert weights(brain.state()) == row['weights_after'][rid]
                    key = (meta['training_seed'], meta['arm'])
                    score = scores.setdefault(key, dict(lives=0, successes=0, layouts={}))
                    score['lives'] += len(m); score['successes'] += sum(s['timely_acquisition'] for s in m.values())
                    layout = score['layouts'].setdefault(meta['layout'], dict(lives=0, successes=0))
                    layout['lives'] += len(m); layout['successes'] += sum(s['timely_acquisition'] for s in m.values())
                    evaluations += 1
                else:training_ends[meta['training_seed'], kind] = row['weights_after']
                w = None
    assert w is None and steps == manifest['total_ticks'] == receipt['total_ticks']
    assert len(episode_ids) == len(saved_rows)
    assert evaluations == len(manifest['seeds'])*manifest['evaluation_cases']*len(manifest['arms'])
    for (seed, kind), end in training_ends.items():
        assert end == {rid: weights(b) for rid, b in checkpoints[seed, kind, 'trained'].items()}
    deltas = []; beat_controls = []
    for group in receipt['result']['seeds']:
        seed = group['seed']
        for arm, stats in group['arms'].items():
            for key in ('lives', 'successes', 'layouts'):assert stats[key] == scores[seed, arm][key]
        rate = lambda arm: scores[seed, arm]['successes']/scores[seed, arm]['lives']
        delta = rate('category-trained')-rate('flat-trained'); deltas.append(delta)
        good = all(rate('category-trained')-rate(control) >= .1-1e-12 for control in ('category-initial', 'flat-random', 'category-random'))
        beat_controls.append(good)
        assert group['category_minus_flat'] == delta and group['category_beats_controls'] == good
        assert group['category_reaches_90'] == (rate('category-trained') >= .9)
    passed = sum(d > 0 for d in deltas) >= 2 and sum(deltas) > 0 and all(beat_controls)
    assert receipt['result']['pilot_promising'] == passed
    assert receipt['result']['milestone_confirmed'] is False
    return dict(passed=True, seconds=time.perf_counter()-started, replayed_ticks=steps,
                episodes=len(episode_ids), evaluations=evaluations, replayed_frozen_decisions=decisions,
                checks=['source and evidence hashes', 'matched initial tensors and worlds', 'training weight continuity',
                        'all physics transitions', 'private views and feasible actions', 'all frozen decisions',
                        'individual reward and behavioral metric recount', 'food deadlines and pilot gate'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path); parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args(); folder = args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local runs directory')
    try:
        result = audit(folder, args.wall_seconds)
        (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    except Exception as exc:
        (folder/'audit-failed.json').write_text(json.dumps(dict(error=type(exc).__name__, reason=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':main()
