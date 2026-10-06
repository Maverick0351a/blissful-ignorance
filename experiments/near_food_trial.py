"""GT-01 diagnostic: can unchanged learners acquire the short gather/eat chain?

Food starts within gathering reach. A separate, frozen carried-food probe
isolates consumption. Neither setup supplies a command or trains on a demo.
"""
import argparse
import copy
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
from agents.affordances import action_mask
from agents.network import ACTIONS, encode_observation
from agents.sequence_ppo import Config
from experiments.category_trial import (ARMS, RESIDENTS, MAX_BYTES, population_for,
    run_episode, check_budget, memory_usage, write, sha)
from sim.world import SIZE, World

SEEDS = (77501, 77513, 77527)
FIXTURES = ('adjacent', 'carried')
HORIZONS = dict(adjacent=512, carried=64)
WINDOWS = dict(adjacent=64, carried=16)


def make_world(seed, case, fixture='adjacent'):
    if fixture not in FIXTURES:raise ValueError('Unknown food diagnostic fixture')
    w = World(seed)
    for name in ('resources', 'structures', 'sounds', 'events', 'visual_memories',
                 'auditory_memories', 'attention_events', 'exploration', 'caches', 'soils'):
        getattr(w, name).clear()
    w.event_serial = 0; w.ecology_enabled = False
    w.residents = {rid: w.residents[rid] for rid in RESIDENTS}
    w.terrain = [[4]*SIZE for _ in range(SIZE)]
    rng = random.Random(seed); locations = {}
    for i, rid in enumerate(RESIDENTS):
        cx, cy = 16+28*i+rng.randint(-3, 3), 16+28*i+rng.randint(-3, 3)
        rotation = (case+2*i) % 4
        for x in range(cx-3, cx+4):
            for y in range(cy-3, cy+4):w.terrain[y][x] = 0
        actor = w.residents[rid]; actor.x = cx; actor.y = cy
        actor.food = 4.; actor.water = actor.energy = actor.warmth = actor.health = 100.
        actor.inventory = {item: 0 for item in actor.inventory}
        actor.facing = ('north', 'east', 'south', 'west')[rotation]
        dx, dy = (0, -1)
        for _ in range(rotation):dx, dy = -dy, dx
        food_at = (cx+dx, cy+dy)
        if fixture == 'adjacent':w.resources[w.key(*food_at)] = dict(kind='berry', amount=4)
        else:actor.inventory['food'] = 1
        locations[rid] = dict(food_at=food_at if fixture == 'adjacent' else None, rotation=rotation)
    return w, dict(fixture=fixture, layout=fixture, locations=locations, world_seed=seed)


def choice_probe(brain, observation):
    """Probability of each primitive on the given private view; no state changes.

    Values are policy probabilities, not calibrated probabilities of success.
    The observation and these diagnostics never become training examples.
    """
    patch, features = encode_observation(observation)
    mask = torch.tensor([action_mask(observation, ACTIONS)])
    with torch.no_grad():
        logits, _, _ = brain.model(patch, features, brain.hidden)
        return brain.distribution(logits, mask).probs[0].tolist()


def quick_success(metric, fixture):
    meal = metric['first_meal_tick']
    if meal is None or meal >= WINDOWS[fixture]:return False
    if metric['first_zero_tick'] is not None and meal >= metric['first_zero_tick']:return False
    if fixture == 'adjacent':
        gather = metric['first_gather_food_tick']
        return gather is not None and gather < meal
    return True


def summarize(rows, seeds):
    groups = []
    for seed in seeds:
        fixtures = {}
        for fixture in FIXTURES:
            arms = {}
            for arm in ARMS:
                episodes = [r for r in rows if r['metadata']['training_seed'] == seed
                            and r['metadata']['fixture'] == fixture and r['metadata']['arm'] == arm]
                metrics = [m for r in episodes for m in r['metrics'].values()]
                probabilities = [p for r in episodes for p in r['metadata']['choice_probes'].values()]
                n = len(metrics)
                arms[arm] = dict(lives=n, quick_successes=sum(quick_success(m, fixture) for m in metrics),
                    gathered_lives=sum(m['first_gather_food_tick'] is not None for m in metrics),
                    ate_lives=sum(m['first_meal_tick'] is not None for m in metrics),
                    full_life_acquisition=sum(m['timely_acquisition'] for m in metrics) if fixture == 'adjacent' else None,
                    ordinary_meals=sum(m['safe_eaten'] for m in metrics), amber_meals=sum(m['amber_eaten'] for m in metrics),
                    zero_food_ticks=sum(m['zero_food_ticks'] for m in metrics),
                    unconscious_ticks=sum(m['unconscious_ticks'] for m in metrics),
                    starting_probabilities={verb: sum(sum(p[i] for i, a in enumerate(ACTIONS) if a['verb'] == verb)
                                                     for p in probabilities)/n
                                            for verb in ('gather', 'eat', 'move', 'tone', 'rest', 'make_seeds')})
            gates = {}
            for kind in ('flat', 'category'):
                rate = lambda arm: arms[arm]['quick_successes']/arms[arm]['lives']
                gates[kind] = (rate(kind+'-trained') >= (.8 if fixture == 'adjacent' else .9)
                    and all(rate(kind+'-trained')-rate(control) >= .2-1e-12
                            for control in (kind+'-initial', 'flat-random', 'category-random')))
            fixtures[fixture] = dict(arms=arms, gates=gates)
        groups.append(dict(seed=seed, fixtures=fixtures))
    return dict(seeds=groups,
        adjacent_learning_gate={kind: all(s['fixtures']['adjacent']['gates'][kind] for s in groups) for kind in ('flat','category')},
        carried_consumption_gate={kind: all(s['fixtures']['carried']['gates'][kind] for s in groups) for kind in ('flat','category')},
        milestone_confirmed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args(); out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or out == ROOT/'runs':parser.error('Use a NEW directory under runs')
    if not 1 <= args.wall_seconds <= 600:parser.error('Diagnostic wall cap must be <=600 seconds')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (77401,) if args.smoke else SEEDS
    training_episodes, cases = (2, 2) if args.smoke else (32, 8)
    config = Config()
    sources = sorted({str(p.relative_to(ROOT)).replace('\\', '/') for folder in ('agents', 'sim') for p in (ROOT/folder).glob('*.py')}
        | {'experiments/near_food_trial.py', 'experiments/audit_near_food.py', 'experiments/category_trial.py',
           'experiments/coordination_trial.py', 'experiments/near_food_report.py',
           'experiments/category_replay.py', 'tests/test_near_food.py'})
    manifest = dict(protocol=1, milestone='GT-01', diagnostic='near-food-chain', seeds=seeds, smoke=args.smoke,
        config=asdict(config), arms=ARMS, fixtures=FIXTURES, horizons=HORIZONS, quick_windows=WINDOWS,
        training_episodes=training_episodes, evaluation_cases=cases,
        total_ticks=len(seeds)*(2*training_episodes*HORIZONS['adjacent'] + len(ARMS)*cases*sum(HORIZONS.values())),
        wall_seconds=args.wall_seconds, audit_wall_seconds=600, evidence_cap_bytes=MAX_BYTES,
        python=sys.version, torch=torch.__version__, actions=ACTIONS,
        hypothesis='With food already within gathering reach, unchanged PPO can learn a prompt gather-then-eat chain; the carried-food check separates consumption from acquisition.',
        primary='Adjacent empty-pack lives: successful own ordinary-food gathering then eating in the first 64 ticks (16 decisions), before any zero fullness. First-meal latency and full 512-tick life success are secondary.',
        secondary='Carried-food lives: one initial fruit, no wild patch, frozen evaluation only; eating in the first 16 ticks (4 decisions). This measures consumption, never foraging.',
        choice_probe='All 68 primitive probabilities on each exact starting private observation, computed without advancing hidden state, world, RNG or optimizer. Compare trained versus its own initialization and both randoms. These probabilities are not calibrated success confidence.',
        gate='For each architecture, adjacent diagnostic passes only if EVERY training seed reaches >=80% prompt feeding and >=20 percentage points above its own initialization and BOTH random controls. Separate carried gate requires >=90% prompt consumption and the same margins in every seed. Neither gate completes GT-01.',
        environment='Separate 7x7 rooms, fullness 4, other needs and health 100. Adjacent: empty packs, four berries one cardinal tile away, balanced rotations, varied translations. Carried: one ordinary fruit in pack, no wild food, otherwise same starting state. All ten tones and every existing primitive remain; the unmodified physical mask determines feasibility. Ecology and hazards off; no neighbors in reach, no automatic wild refill.',
        training='Fresh private brains only. 32 adjacent-food lives of 512 ticks per resident and architecture; no carried-food training. Same core, observations, reward, entropy, optimizer, rollout, terminal reset and interaction budget as the previous category pilot. No teacher commands, movement lock, food priority, demonstration, outcome bonus or imported live weights.',
        schedule='Train worlds seed*1000+episode, evaluate worlds seed*1000+500+case. Each resident sees two balanced cardinal rotations per eight held-out cases. Same per-life action RNG world_seed+900000+resident_index across all arms; independent brains. Seed 77401 is smoke-only.',
        limits='Three training seeds, eight development maps each, two private residents. Clustered descriptive pilot, not a statistical confirmation, navigation benchmark, social test, crop-production proof or live deployment. Do not pool carried and empty-pack success.',
        hashes={name:sha(ROOT/name) for name in sources})
    write(out/'preregistration.json', manifest)
    (out/'preregistration.sha256').write_text(sha(out/'preregistration.json'), encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name, name)
    start = time.perf_counter(); deadline = start+args.wall_seconds
    training = []; evaluations = []
    try:
        with gzip.open(out/'trace.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as trace:
            for seed in seeds:
                states = {}
                kinds = ('flat', 'category') if seeds.index(seed) % 2 == 0 else ('category', 'flat')
                for kind in kinds:
                    p = population_for(seed, kind, make_world(seed*1000, 0)[0], config)
                    initial = {rid:copy.deepcopy(b.state()) for rid,b in p.brains.items()}
                    torch.save(initial, out/f'{seed}-{kind}-initial.pt')
                    for case in range(training_episodes):
                        p.world, meta = make_world(seed*1000+case, case)
                        p.metrics = {rid:p.new_metrics() for rid in RESIDENTS}
                        meta.update(training_seed=seed, stage='training', arm=kind+'-training', case=case)
                        training.append(run_episode(p, meta, f'{seed}-{kind}-train-{case}', HORIZONS['adjacent'], trace, out, deadline))
                        write(out/'training.partial.json', training)
                        if (case+1)%8 == 0 or case+1 == training_episodes:
                            print(json.dumps(dict(seed=seed, kind=kind, trained_lives=case+1, seconds=round(time.perf_counter()-start,2))), flush=True)
                    trained = {rid:copy.deepcopy(b.state()) for rid,b in p.brains.items()}
                    torch.save(trained, out/f'{seed}-{kind}-trained.pt')
                    states[kind] = dict(initial=initial, trained=trained)
                for case in range(cases):
                    for fixture in FIXTURES:
                        for arm in ARMS:
                            kind, variant = arm.split('-')
                            w, meta = make_world(seed*1000+500+case, case, fixture)
                            q = population_for(seed, kind, w, config)
                            for i, (rid, brain) in enumerate(q.brains.items()):
                                brain.restore(copy.deepcopy(states[kind]['trained' if variant == 'trained' else 'initial'][rid]))
                                brain.freeze(); brain.random_policy = variant == 'random'
                                brain.rng = random.Random(w.seed+900000+i)
                            meta.update(training_seed=seed, stage='evaluation', arm=arm, case=case,
                                        choice_probes={rid:choice_probe(b,w.observe(rid)) for rid,b in q.brains.items()})
                            evaluations.append(run_episode(q,meta,f'{seed}-eval-{case}-{fixture}-{arm}',HORIZONS[fixture],trace,out,deadline))
                            write(out/'results.partial.json', evaluations)
                    if (case+1)%2 == 0:
                        print(json.dumps(dict(seed=seed, evaluated_maps=case+1, seconds=round(time.perf_counter()-start,2))), flush=True)
        check_budget(out, deadline)
        assert all(sha(ROOT/name) == value for name,value in manifest['hashes'].items()), 'Source changed during diagnostic'
        write(out/'results.json', evaluations); write(out/'training.json', training)
        result = summarize(evaluations, seeds)
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, total_ticks=manifest['total_ticks'],
            source_hashes_valid=True, smoke=args.smoke, result=result, memory=memory_usage(),
            evidence_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
        print(json.dumps({k:v for k,v in result.items() if k!='seeds'}), flush=True)
    except Exception as exc:
        write(out/'incomplete.json', dict(error=type(exc).__name__, reason=str(exc), seconds=time.perf_counter()-start,
              completed_training=len(training), completed_evaluations=len(evaluations)))
        raise


if __name__ == '__main__':main()
