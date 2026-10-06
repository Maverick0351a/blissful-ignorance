"""Voluntary primitive-action coordination pilot, isolated from saved residents.

The environment creates unequal information, never a leader, destination action,
tone meaning, route, shared reward, or a forced response to another resident.
"""
import argparse
import copy
from dataclasses import asdict
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
from agents.sequence_ppo import Brain, Config, Population
from sim.social import TONES
from sim.world import SIZE, World

SEEDS = (65113, 65124, 65135)
ARMS = ('intact', 'muted', 'shuffled', 'initial')
RESIDENTS = ('r0', 'r1')
MAX_BYTES = 256 * 1024 * 1024


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weight_hashes(brain):
    return {name: hashlib.sha256(b''.join(
        value.detach().numpy().tobytes() for value in getattr(brain, name).state_dict().values()
    )).hexdigest() for name in ('model', 'predictor')}


def permutation(seed):
    """Episode-specific symbol derangement: no symbol retains its usual label."""
    rng = random.Random(seed)
    shuffled = list(TONES)
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(TONES, shuffled)):
            return dict(zip(TONES, shuffled))


def channel_view(observation, condition, mapping):
    if condition in ('intact', 'initial'):
        return observation
    result = dict(observation)
    if condition == 'muted':
        result['hearing'] = [s for s in observation['hearing'] if not s['kind'].startswith('tone_')]
        result['auditory_memory'] = []
    elif condition == 'shuffled':
        result['hearing'] = [dict(s, kind='tone_' + mapping[s['kind'][5:]])
                             if s['kind'].startswith('tone_') else dict(s)
                             for s in observation['hearing']]
        result['auditory_memory'] = [dict(s, tone=mapping[s['tone']]) for s in observation['auditory_memory']]
    else:
        raise ValueError('Unknown tone intervention')
    return result


class MeasuredBrain(Brain):
    """Unchanged PPO policy with a recorded, optionally intervened sensory input."""
    def __init__(self, seed, config):
        super().__init__(seed, config)
        self.condition = 'intact'
        self.mapping = permutation(seed)
        self.last_observation = None
        self.last_action = None

    def decide(self, observation):
        self.last_observation = channel_view(observation, self.condition, self.mapping)
        self.last_action = super().decide(self.last_observation)
        return self.last_action


def food_visible(observation):
    return any(t['terrain'] >= 0 and t.get('resource', {}).get('kind') in ('berry', 'food')
               and t['resource'].get('amount', 0) > 0 for t in observation['tiles'])


def make_world(seed, case, held_out=False, warmup=False):
    """Rotate/translate a small room with an opaque divider and two open ends.

    Test rooms have a longer divider and food farther to either side. Only the
    experiment's recorder receives the case/role metadata; brains get observe().
    """
    w = World(seed)
    w.resources.clear(); w.structures.clear(); w.sounds.clear()
    w.events.clear(); w.event_serial = 0
    w.residents = {rid: w.residents[rid] for rid in RESIDENTS}
    w.terrain = [[4] * SIZE for _ in range(SIZE)]
    rng = random.Random(seed)
    cx, cy = rng.randint(15, 45), rng.randint(15, 45)
    rotation, side = (case // 2) % 4, (-1 if case % 2 == 0 else 1)

    def point(x, y):
        for _ in range(rotation):
            x, y = -y, x
        return cx + x, cy + y

    for x in range(-4, 5):
        for y in range(-4, 5):
            px, py = point(x, y)
            w.terrain[py][px] = 0
    half_wall = 2 if held_out else 1
    for x in range(-half_wall, half_wall + 1):
        px, py = point(x, 0)
        w.terrain[py][px] = 4
    first = RESIDENTS[(rotation + case % 2) % 2]
    other = next(rid for rid in RESIDENTS if rid != first)
    for rid, location in ((first, (-1, -1) if warmup else (0, -1)),
                          (other, (1, -1) if warmup else (0, 1))):
        a = w.residents[rid]
        a.x, a.y = point(*location)
        a.food = 2.; a.water = a.energy = a.warmth = a.health = 100.
        a.inventory = {key: 0 for key in a.inventory}
        a.facing = ('north', 'east', 'south', 'west')[rotation]
    food_at = point(side * (3 if held_out else 2), -2)
    w.resources[w.key(*food_at)] = {'kind': 'berry', 'amount': 4}
    assert food_visible(w.observe(first))
    assert food_visible(w.observe(other)) == warmup
    return w, dict(initially_informed=first, initially_uninformed=other,
                   food_at=food_at, rotation=rotation, side=side, held_out=held_out, warmup=warmup)


def population_for(seed, world, config):
    p = Population(seed, config, world=world, resident_ids=RESIDENTS)
    p.brains = {rid: MeasuredBrain(seed + i, config) for i, rid in enumerate(RESIDENTS)}
    p.neighbors_scripted = False
    return p


def bodies(world):
    return {rid: {key: copy.deepcopy(getattr(a, key)) for key in
                  ('x', 'y', 'food', 'water', 'energy', 'health', 'pain', 'unconscious', 'inventory')}
            for rid, a in world.residents.items()}


def log_line(stream, row):
    stream.write(json.dumps(row, separators=(',', ':')) + '\n')


def check_budget(output, deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('Declared wall-time budget reached; partial evidence preserved')
    if (output / 'STOP').exists():
        raise InterruptedError('STOP requested; partial evidence preserved')
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) >= MAX_BYTES:
        raise OSError('256 MiB evidence cap reached; all files preserved')


def run_episode(p, metadata, episode_id, ticks, stream, output, deadline):
    w = p.world
    start = dict(type='start', episode=episode_id, metadata=metadata, world=w.to_dict(),
                 conditions={rid: b.condition for rid, b in p.brains.items()},
                 permutations={rid: b.mapping for rid, b in p.brains.items()},
                 weights={rid: weight_hashes(b) for rid, b in p.brains.items()})
    log_line(stream, start)
    stats = {rid: dict(first_meal_tick=None, first_food_seen_tick=None, heard_hidden_decisions=0,
                      tone_memory_decisions=0, actions={}, successful_actions={}, unique_tiles=0,
                      uniform_tone_expected=0.) for rid in RESIDENTS}
    visited = {rid: set() for rid in RESIDENTS}
    for tick in range(ticks):
        if tick % 64 == 0:
            stream.flush(); check_budget(output, deadline)
        deciding = w.tick % 4 == 0
        before = bodies(w)
        results = p.step()
        commands = {rid: dict(b.last_action) if deciding else {'verb': 'wait'} for rid, b in p.brains.items()}
        row = dict(type='step', episode=episode_id, tick=tick, commands=commands,
                   results=results, after=bodies(w))
        if deciding:
            row['observations'] = {rid: b.last_observation for rid, b in p.brains.items()}
        for rid, command in commands.items():
            a = w.residents[rid]; s = stats[rid]
            visited[rid].add((before[rid]['x'], before[rid]['y']))
            visited[rid].add((a.x, a.y))
            if not deciding:
                continue
            o = p.brains[rid].last_observation
            if food_visible(o) and s['first_food_seen_tick'] is None:
                s['first_food_seen_tick'] = tick
            if o['auditory_memory']:
                s['tone_memory_decisions'] += 1
                s['heard_hidden_decisions'] += not food_visible(o)
            label = command['verb']
            s['actions'][label] = s['actions'].get(label, 0) + 1
            if results[rid][0]:
                s['successful_actions'][label] = s['successful_actions'].get(label, 0) + 1
                if label == 'eat' and s['first_meal_tick'] is None:
                    s['first_meal_tick'] = tick
            if not o['body']['unconscious']:
                mask = action_mask(o, ACTIONS)
                s['uniform_tone_expected'] += sum(ok and a['verb'] == 'tone' for a, ok in zip(ACTIONS, mask)) / sum(mask)
        log_line(stream, row)
    for rid, b in p.brains.items():
        b.finish(channel_view(w.observe(rid), b.condition, b.mapping), terminal=True)
        stats[rid].update(p.metrics[rid])
        stats[rid]['unique_tiles'] = len(visited[rid])
    summary = dict(episode=episode_id, metadata=metadata, ticks=ticks, metrics=stats,
                   weights_before=start['weights'], weights_after={rid: weight_hashes(b) for rid, b in p.brains.items()})
    if all(not b.training for b in p.brains.values()):
        assert summary['weights_before'] == summary['weights_after'], 'Evaluation modified weights'
    log_line(stream, dict(type='end', **summary, world=w.to_dict()))
    stream.flush()
    return summary


def summarize(rows, seeds):
    summaries = []
    for seed in seeds:
        scores = {}
        for arm in ARMS:
            selected = [r for r in rows if r['metadata']['training_seed'] == seed and r['metadata']['arm'] == arm]
            targets = [r['metrics'][r['metadata']['initially_uninformed']] for r in selected]
            scores[arm] = dict(episodes=len(selected), fed=sum(m['safe_eaten'] > 0 for m in targets),
                both_fed=sum(all(m['safe_eaten'] > 0 for m in r['metrics'].values()) for r in selected),
                target_nutrition=sum(m['nutrition'] for m in targets),
                target_zero_food_ticks=sum(m['zero_food_ticks'] for m in targets),
                target_unconscious_ticks=sum(m['unconscious_ticks'] for m in targets),
                heard_hidden_decisions=sum(m['heard_hidden_decisions'] for m in targets))
        n = scores['intact']['episodes']; intact = scores['intact']['fed'] / n
        communication_gate = intact >= .75 and all(intact - scores[arm]['fed'] / n >= .25 for arm in ('muted', 'shuffled'))
        acquisition_gate = scores['intact']['fed'] > scores['initial']['fed']
        summaries.append(dict(seed=seed, arms=scores, communication_gate=communication_gate,
                              acquisition_gate=acquisition_gate, passed=communication_gate and acquisition_gate))
    return dict(seeds=summaries, passed=all(s['passed'] for s in summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true', help='Infrastructure check, not a behavioral trial')
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / 'runs') or out == ROOT / 'runs':
        parser.error('Use a new directory inside this project runs folder')
    if not 1 <= args.wall_seconds <= 900:
        parser.error('Wall time must be between 1 and 900 seconds')
    out.mkdir(parents=True, exist_ok=False)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    warmups, training_episodes, ticks, cases = (1, 1, 32, 2) if args.smoke else (4, 16, 512, 8)
    config = Config()
    files = sorted({str(p.relative_to(ROOT)).replace('\\', '/') for folder in ('agents', 'sim')
                    for p in (ROOT / folder).glob('*.py')} | {
        'experiments/coordination_trial.py', 'experiments/audit_coordination.py',
        'tests/test_coordination_trial.py'})
    manifest = dict(protocol=1, seeds=seeds, warmup_episodes=warmups, training_episodes=training_episodes,
        ticks_per_episode=ticks, evaluation_cases=cases, arms=ARMS, config=asdict(config), smoke=args.smoke,
        total_ticks=len(seeds) * (warmups + training_episodes + cases * len(ARMS)) * ticks,
        wall_seconds=args.wall_seconds, evidence_cap_bytes=MAX_BYTES, python=sys.version, torch=torch.__version__,
        reward='Unchanged individual PPO reward: own food relief/25, own hunger distress, own injury; no team, signaling, following, leader, curiosity or route bonus.',
        inputs='Unchanged local sensory encoder and feasible primitive actions, including all ten tones. No case, role, global position, food destination or assigned tone meaning enters a policy.',
        environment='Two disposable residents; 2 initial fullness each, empty packs, four shared berries, no hazards, ecological replenishment disabled. Warmup: both see food. Coordination: one sees food across an opaque divider, rotating and exchanging starting positions. Held-out divider is longer and food farther sideways.',
        interventions='Frozen matched weights and policy RNGs in identical worlds. Muted removes tones from instantaneous hearing AND private auditory memory only. Shuffled applies a fresh per-episode/per-recipient derangement of the ten labels, stable for each memory; timing, bearing, strength and repeated patterns remain. It does not preserve per-label histogram and is not a temporal sequence shuffle. Initial is matching initialization with intact hearing.',
        gate='Every training seed: initially uninformed resident eats ordinary food in >=75% intact episodes, >=25 percentage points more than each muted/shuffled arm, and more episodes than matching initial weights. This is a pilot advancement gate, not a significance test or evidence of leadership.',
        limitations='Small repeated rooms, short episodes and small training budget. Alternate visual, footsteps, tap, timing and bearing cues remain. Individual reward may offer weak credit for helping. Successful signaling would not establish leadership, general language, survival, Clef/Laya ability, or a self-organized society.',
        hashes={name: sha(ROOT / name) for name in files})
    write(out / 'preregistration.json', manifest)
    (out / 'preregistration.sha256').write_text(sha(out / 'preregistration.json'), encoding='utf-8')
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, name)
    start = time.perf_counter(); deadline = start + args.wall_seconds
    evaluations = []; training = []
    try:
        with gzip.open(out / 'trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as trace:
            for seed in seeds:
                w, _ = make_world(seed, 0)
                p = population_for(seed, w, config)
                initial = {rid: copy.deepcopy(b.state()) for rid, b in p.brains.items()}
                torch.save(initial, out / f'{seed}-initial.pt')
                schedule = list(range(training_episodes))
                random.Random(seed).shuffle(schedule)
                for index, case in enumerate(list(range(warmups)) + schedule):
                    warmup = index < warmups
                    w, meta = make_world(seed * 1000 + index, case, warmup=warmup)
                    p.world = w; p.metrics = {rid: p.new_metrics() for rid in RESIDENTS}
                    meta.update(training_seed=seed, stage='warmup' if warmup else 'coordination', arm='training')
                    row = run_episode(p, meta, f'{seed}-train-{index}', ticks, trace, out, deadline)
                    training.append(row)
                    write(out / 'training.partial.json', training)
                    if (index + 1) % 4 == 0 or index == warmups + training_episodes - 1:
                        print(json.dumps(dict(seed=seed, training_episodes=index + 1,
                            meals={rid: p.metrics[rid]['safe_eaten'] for rid in RESIDENTS},
                            seconds=round(time.perf_counter() - start, 2))), flush=True)
                trained = {rid: copy.deepcopy(b.state()) for rid, b in p.brains.items()}
                torch.save(trained, out / f'{seed}-trained.pt')
                for case in range(cases):
                    for arm in ARMS:
                        evaluation_seed = seed * 1000 + 500 + case
                        w, meta = make_world(evaluation_seed, case, held_out=True)
                        q = population_for(seed, w, config)
                        for i, (rid, b) in enumerate(q.brains.items()):
                            b.restore((initial if arm == 'initial' else trained)[rid]); b.freeze()
                            b.rng = random.Random(evaluation_seed + 900000 + i)
                            b.condition = arm; b.mapping = permutation(evaluation_seed + 800000 + i)
                        meta.update(training_seed=seed, stage='evaluation', arm=arm, case=case)
                        evaluations.append(run_episode(q, meta, f'{seed}-eval-{case}-{arm}', ticks, trace, out, deadline))
                        write(out / 'results.partial.json', evaluations)
                    if (case + 1) % 2 == 0:
                        print(json.dumps(dict(seed=seed, evaluated_cases=case + 1,
                            seconds=round(time.perf_counter() - start, 2))), flush=True)
        assert all(sha(ROOT / name) == value for name, value in manifest['hashes'].items()), 'Source changed during trial'
        result = summarize(evaluations, seeds)
        write(out / 'results.json', evaluations)
        write(out / 'complete.json', dict(seconds=time.perf_counter() - start, smoke=args.smoke,
              source_hashes_valid=True, total_ticks=manifest['total_ticks'], result=result,
              evidence_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'complete.json'}))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write(out / 'incomplete.json', dict(error=type(exc).__name__, reason=str(exc),
              seconds=time.perf_counter() - start, completed_evaluations=len(evaluations), completed_training=len(training)))
        raise


if __name__ == '__main__':
    main()
