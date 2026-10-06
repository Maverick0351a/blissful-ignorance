"""GT-01 development pilot: learned verb/argument versus flat primitive PPO.

Runs only fresh disposable worlds, with matching initialization and two random
controls. All ten tones remain. No survival rule or teacher supplies actions.
"""
import argparse
import copy
import ctypes
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.category_ppo import Brain as CategoryBrain, CATEGORIES
from agents.network import ACTIONS
from agents.sequence_ppo import Brain as FlatBrain, Config, Population
from experiments.coordination_trial import write, sha, weight_hashes, bodies, log_line, food_visible
from sim.world import SIZE, World

SEEDS = (76411, 76423, 76439)
RESIDENTS = ('r0', 'r1')
ARMS = ('flat-trained', 'category-trained', 'flat-initial', 'category-initial', 'flat-random', 'category-random')
MAX_BYTES = 512 * 1024 * 1024


class Measurement:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_policy = False
        self.last_observation = self.last_action = None
        self.last_reward = {}
        self.reward_totals = {}
        self.decision_seconds = self.learning_seconds = 0.

    def distribution(self, logits, mask):
        return super().distribution(torch.zeros_like(logits) if self.random_policy else logits, mask)

    def decide(self, observation):
        start = time.perf_counter()
        self.last_observation = observation
        self.last_action = super().decide(observation)
        self.decision_seconds += time.perf_counter() - start
        return self.last_action

    def _learn(self, bootstrap):
        start = time.perf_counter()
        super()._learn(bootstrap)
        self.learning_seconds += time.perf_counter() - start

    def add_reward(self, parts):
        self.last_reward = dict(parts)
        for name, value in parts.items():
            self.reward_totals[name] = self.reward_totals.get(name, 0.) + value
        super().add_reward(parts)


class MeasuredFlat(Measurement, FlatBrain):pass
class MeasuredCategory(Measurement, CategoryBrain):pass


def make_world(seed, case):
    """Two separated 7x7 rooms; rotated open/occluded access to finite food.

    Rooms remove food theft and social assistance as explanations of first
    feeding. No coordinates, case labels or routes are passed to controllers.
    Held-out pilot maps use fresh seeds; they are not confirmation maps.
    """
    w = World(seed)
    for name in ('resources', 'structures', 'sounds', 'events', 'visual_memories',
                 'auditory_memories', 'attention_events', 'exploration', 'caches', 'soils'):
        getattr(w, name).clear()
    w.event_serial = 0; w.ecology_enabled = False
    w.residents = {rid: w.residents[rid] for rid in RESIDENTS}
    w.terrain = [[4]*SIZE for _ in range(SIZE)]
    rng = random.Random(seed)
    layout = 'occluded' if case % 2 else 'open'
    info = {}
    for i, rid in enumerate(RESIDENTS):
        cx, cy = 16 + 28*i + rng.randint(-3, 3), 16 + 28*i + rng.randint(-3, 3)
        rotation = rng.randrange(4)
        def point(x, y):
            for _ in range(rotation):x, y = -y, x
            return cx+x, cy+y
        for x in range(-3, 4):
            for y in range(-3, 4):
                px, py = point(x, y); w.terrain[py][px] = 0
        if layout == 'occluded':
            for x in range(-1, 2):
                px, py = point(x, 0); w.terrain[py][px] = 4
        a = w.residents[rid]
        a.x, a.y = point(0, 2 if layout == 'occluded' else 0)
        a.food = 4.; a.water = a.energy = a.warmth = a.health = 100.
        a.inventory = {key: 0 for key in a.inventory}
        a.facing = ('north', 'east', 'south', 'west')[rotation]
        side = rng.choice((-1, 1))
        food_at = point(side*rng.choice((1, 2)), -2 if layout == 'occluded' else rng.choice((-2, 2)))
        w.resources[w.key(*food_at)] = dict(kind='berry', amount=4)
        info[rid] = dict(food_at=food_at, rotation=rotation, food_visible_at_start=food_visible(w.observe(rid)))
    return w, dict(layout=layout, locations=info, world_seed=seed)


def population_for(seed, kind, world, config):
    p = Population(seed, config, world=world, resident_ids=RESIDENTS)
    cls = MeasuredCategory if kind == 'category' else MeasuredFlat
    p.brains = {rid: cls(seed+i, config) for i, rid in enumerate(RESIDENTS)}
    p.neighbors_scripted = False
    return p


def check_budget(out, deadline):
    if time.perf_counter() >= deadline:raise TimeoutError('Declared wall cap reached; partial evidence preserved')
    if (out/'STOP').exists():raise InterruptedError('STOP requested; partial evidence preserved')
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) >= MAX_BYTES:
        raise OSError('512 MiB evidence cap reached; all files preserved')


def run_episode(p, meta, episode, ticks, stream, out, deadline):
    start = time.perf_counter(); w = p.world
    weights = {rid: weight_hashes(b) for rid, b in p.brains.items()}
    log_line(stream, dict(type='start', episode=episode, metadata=meta, world=w.to_dict(), weights=weights))
    stats = {rid: dict(first_meal_tick=None, first_gather_food_tick=None, first_zero_tick=None,
             first_food_seen_tick=None, actions={}, successful_actions={}, conscious_tones=0) for rid in RESIDENTS}
    visited = {rid: set() for rid in RESIDENTS}
    counters = {rid: (b.updates, b.decisions, b.decision_seconds, b.learning_seconds) for rid, b in p.brains.items()}
    for b in p.brains.values():b.reward_totals = {}
    for tick in range(ticks):
        if tick % 64 == 0:
            stream.flush(); check_budget(out, deadline)
        deciding = w.tick % 4 == 0; before = bodies(w)
        result = p.step()
        commands = {rid: dict(b.last_action) if deciding else {'verb': 'wait'} for rid, b in p.brains.items()}
        row = dict(type='step', episode=episode, tick=tick, commands=commands, results=result,
                   after=bodies(w), rewards={rid: b.last_reward for rid, b in p.brains.items()})
        if deciding:row['observations'] = {rid: b.last_observation for rid, b in p.brains.items()}
        for rid, command in commands.items():
            a = w.residents[rid]; s = stats[rid]
            visited[rid].update(((before[rid]['x'], before[rid]['y']), (a.x, a.y)))
            if a.food <= 0 and s['first_zero_tick'] is None:s['first_zero_tick'] = tick
            if not deciding:continue
            o = p.brains[rid].last_observation; verb = command['verb']
            if food_visible(o) and s['first_food_seen_tick'] is None:s['first_food_seen_tick'] = tick
            s['actions'][verb] = s['actions'].get(verb, 0) + 1
            s['conscious_tones'] += verb == 'tone' and not o['body']['unconscious']
            if result[rid][0]:
                s['successful_actions'][verb] = s['successful_actions'].get(verb, 0) + 1
                if verb == 'gather' and a.inventory['food'] > before[rid]['inventory']['food'] and s['first_gather_food_tick'] is None:
                    s['first_gather_food_tick'] = tick
                if verb == 'eat' and command.get('item', 'food') == 'food' and s['first_meal_tick'] is None:
                    s['first_meal_tick'] = tick
        log_line(stream, row)
    timings = {}
    for rid, b in p.brains.items():
        b.finish(w.observe(rid), terminal=True)
        s = stats[rid]; s.update(p.metrics[rid]); s['unique_tiles'] = len(visited[rid])
        s['timely_acquisition'] = (s['first_gather_food_tick'] is not None and s['first_meal_tick'] is not None
            and s['first_gather_food_tick'] < s['first_meal_tick']
            and (s['first_zero_tick'] is None or s['first_meal_tick'] < s['first_zero_tick']))
        s['reward_totals'] = dict(b.reward_totals)
        updates, decisions, decision_time, learning_time = counters[rid]
        timings[rid] = dict(updates=b.updates-updates, decisions=b.decisions-decisions,
            decision_seconds=b.decision_seconds-decision_time, learning_seconds=b.learning_seconds-learning_time)
    row = dict(episode=episode, metadata=meta, ticks=ticks, metrics=stats, timings=timings,
               seconds=time.perf_counter()-start, weights_before=weights,
               weights_after={rid: weight_hashes(b) for rid, b in p.brains.items()})
    if all(not b.training for b in p.brains.values()):assert row['weights_before'] == row['weights_after']
    log_line(stream, dict(type='end', **row, world=w.to_dict())); stream.flush()
    return row


def summarize(rows, seeds):
    result = []
    for seed in seeds:
        arms = {}
        for arm in ARMS:
            episodes = [r for r in rows if r['metadata']['training_seed'] == seed and r['metadata']['arm'] == arm]
            lives = [m for r in episodes for m in r['metrics'].values()]
            arms[arm] = dict(lives=len(lives), successes=sum(m['timely_acquisition'] for m in lives),
                ordinary_meals=sum(m['safe_eaten'] for m in lives), amber_meals=sum(m['amber_eaten'] for m in lives),
                zero_food_ticks=sum(m['zero_food_ticks'] for m in lives),
                unconscious_ticks=sum(m['unconscious_ticks'] for m in lives),
                layouts={layout: dict(lives=sum(len(r['metrics']) for r in episodes if r['metadata']['layout'] == layout),
                    successes=sum(m['timely_acquisition'] for r in episodes if r['metadata']['layout'] == layout for m in r['metrics'].values()))
                    for layout in ('open', 'occluded')})
        rate = lambda arm: arms[arm]['successes']/arms[arm]['lives']
        delta = rate('category-trained') - rate('flat-trained')
        learned = all(rate('category-trained') - rate(control) >= .1-1e-12
                      for control in ('category-initial', 'flat-random', 'category-random'))
        result.append(dict(seed=seed, arms=arms, category_minus_flat=delta,
                           category_beats_controls=learned, category_reaches_90=rate('category-trained') >= .9))
    # Pilot advancement only: a positive matched effect in >=2/3 seeds, positive
    # pooled effect, and >=10 points above own initialization and BOTH randoms
    # in every seed. Confirmation still requires five fresh training seeds.
    promising = (sum(r['category_minus_flat'] > 0 for r in result) >= 2
                 and sum(r['category_minus_flat'] for r in result) > 0
                 and all(r['category_beats_controls'] for r in result))
    return dict(seeds=result, pilot_promising=promising, milestone_confirmed=False)


def memory_usage():
    if sys.platform != 'win32':return {'unavailable': 'Windows process accounting only'}
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
             'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage',
             'PeakPagefileUsage', 'PrivateUsage')]
    data = Counters(); data.cb = ctypes.sizeof(data)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not get_info(current(), ctypes.byref(data), data.cb):raise ctypes.WinError()
    return dict(peak_working_set_bytes=data.PeakWorkingSetSize, private_bytes=data.PrivateUsage,
                peak_commit_bytes=data.PeakPagefileUsage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-episodes', type=int, default=32)
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args(); out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or out == ROOT/'runs':parser.error('Use a NEW directory under runs')
    if not 1 <= args.wall_seconds <= 600 or not 2 <= args.train_episodes <= 48:parser.error('Pilot budget out of bounds')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (76101,) if args.smoke else SEEDS
    episodes, cases, ticks = (2, 2, 512) if args.smoke else (args.train_episodes, 8, 512)
    config = Config()
    files = sorted({str(p.relative_to(ROOT)).replace('\\', '/') for folder in ('agents', 'sim') for p in (ROOT/folder).glob('*.py')}
                   | {'experiments/category_trial.py', 'experiments/audit_category.py', 'experiments/coordination_trial.py',
                      'tests/test_category_ppo.py', 'tests/test_category_trial.py'})
    sizes = {}
    for kind, cls in (('flat', FlatBrain), ('category', CategoryBrain)):
        b = cls(1, config)
        sizes[kind] = {name: sum(p.numel() for p in module.parameters()) for name, module in
                      (('policy_and_value', b.model), ('actor_head', b.model.actor), ('predictor', b.predictor))}
    manifest = dict(protocol=1, milestone='GT-01', smoke=args.smoke, seeds=seeds, config=asdict(config),
        training_episodes=episodes, evaluation_cases=cases, ticks_per_episode=ticks, arms=ARMS,
        total_ticks=len(seeds)*(2*episodes + len(ARMS)*cases)*ticks,
        wall_seconds=args.wall_seconds, evidence_cap_bytes=MAX_BYTES, parameters=sizes,
        python=sys.version, torch=torch.__version__, categories=CATEGORIES, actions=ACTIONS,
        primary='Own ordinary-food gathering followed by eating before ever reaching zero fullness in that life; empty starting inventory. Report each training seed and each layout.',
        environment='Two separate 7x7 rooms, four berries per resident, fullness 4, other needs/health 100, no hazards or neighboring residents in reach. Open and occluded layouts alternate, patch side/rotation/translation vary with world seed. Training worlds seed*1000+episode; held-out pilot worlds seed*1000+500+case. Not confirmation worlds. No automatic wild refill, ecology off; ordinary seed-making/planting remains possible.',
        matched='Same sensory encoder, CNN/LSTM, initial trunk/value/argument/predictor weights, private optimizer/RNG, Config, individual reward, terminal resets and interaction counts. Category adds a verb head with separate seeded initialization. Joint primitive entropy uses the same coefficient; its maximum still favors equal primitive probabilities, not equal category probabilities.',
        controls='Each trained arm versus its frozen initial weights; flat-uniform and verb-uniform random feasible actions. Random controls ignore logits. Common per-life action RNG seed world_seed+900000+resident_index; one joint-categorical draw per decision. All models evaluated frozen. No controller sees case, coordinates, experiment counters or food destination.',
        reward='Unchanged own food relief/25, own hunger distress and own injury; curiosity coefficient zero; no gather, navigation, tone, farming, category or team bonus.',
        gate='Pilot promising only if category-trained exceeds flat-trained in at least 2/3 seeds and pooled rate, AND exceeds own initialization and BOTH random controls by at least 10 percentage points in every seed. This is a development screen, not milestone confirmation or statistical significance.',
        limitations='Short narrow assay; no competitive economy, hazards or social opportunities. Three clustered training seeds and eight development evaluation maps each; no survival, language, leadership or transfer claim.',
        hashes={name: sha(ROOT/name) for name in files})
    write(out/'preregistration.json', manifest)
    (out/'preregistration.sha256').write_text(sha(out/'preregistration.json'), encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(ROOT/name, name)
    start = time.perf_counter(); deadline = start+args.wall_seconds
    training = []; evaluations = []
    try:
        with gzip.open(out/'trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as trace:
            for seed in seeds:
                states = {}
                # Alternate which architecture runs first across seeds.
                kinds = ('flat', 'category') if seeds.index(seed) % 2 == 0 else ('category', 'flat')
                for kind in kinds:
                    p = population_for(seed, kind, make_world(seed*1000, 0)[0], config)
                    initial = {rid: copy.deepcopy(b.state()) for rid, b in p.brains.items()}
                    torch.save(initial, out/f'{seed}-{kind}-initial.pt')
                    for index in range(episodes):
                        w, meta = make_world(seed*1000+index, index)
                        p.world = w; p.metrics = {rid: p.new_metrics() for rid in RESIDENTS}
                        meta.update(training_seed=seed, stage='training', arm=kind+'-training', case=index)
                        training.append(run_episode(p, meta, f'{seed}-{kind}-train-{index}', ticks, trace, out, deadline))
                        write(out/'training.partial.json', training)
                        if (index+1) % 8 == 0 or index+1 == episodes:
                            print(json.dumps(dict(seed=seed, kind=kind, trained_episodes=index+1,
                                  seconds=round(time.perf_counter()-start, 2))), flush=True)
                    trained = {rid: copy.deepcopy(b.state()) for rid, b in p.brains.items()}
                    torch.save(trained, out/f'{seed}-{kind}-trained.pt')
                    states[kind] = dict(initial=initial, trained=trained)
                for case in range(cases):
                    for arm in ARMS:
                        kind, variant = arm.split('-')
                        w, meta = make_world(seed*1000+500+case, case)
                        q = population_for(seed, kind, w, config)
                        for i, (rid, b) in enumerate(q.brains.items()):
                            b.restore(copy.deepcopy(states[kind]['trained' if variant == 'trained' else 'initial'][rid]))
                            b.freeze(); b.random_policy = variant == 'random'
                            b.rng = random.Random(w.seed+900000+i)
                        meta.update(training_seed=seed, stage='evaluation', arm=arm, case=case)
                        evaluations.append(run_episode(q, meta, f'{seed}-eval-{case}-{arm}', ticks, trace, out, deadline))
                        write(out/'results.partial.json', evaluations)
                    if (case+1) % 2 == 0:
                        print(json.dumps(dict(seed=seed, evaluated_cases=case+1,
                              seconds=round(time.perf_counter()-start, 2))), flush=True)
        check_budget(out, deadline)
        assert all(sha(ROOT/name) == value for name, value in manifest['hashes'].items()), 'Source changed during pilot'
        write(out/'results.json', evaluations); write(out/'training.json', training)
        result = summarize(evaluations, seeds)
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, total_ticks=manifest['total_ticks'],
            source_hashes_valid=True, smoke=args.smoke, result=result, memory=memory_usage(),
            evidence_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'complete.json'}))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write(out/'incomplete.json', dict(error=type(exc).__name__, reason=str(exc),
              seconds=time.perf_counter()-start, completed_training=len(training), completed_evaluations=len(evaluations)))
        raise


if __name__ == '__main__':main()
