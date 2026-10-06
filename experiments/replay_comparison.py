"""Matched local-senses PPO / feedforward replay / recurrent replay experiment.

Runs only disposable worlds. No server, live save, Laya inference, downloads,
shared learner data, recipe reward or pretrained policy is used.
"""
import argparse
from dataclasses import asdict
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
from agents.sequence_ppo import Brain as PPOBrain, Config as PPOConfig, Population
from agents.sequence_replay import Brain as ReplayBrain, Config as ReplayConfig
from experiments.scarcity_routes import configure, safe_oracle
from experiments.scarcity_trial import arena_record, gate, run

FAMILIES = ('ppo', 'feedforward', 'recurrent')
SEEDS = (69071, 69082, 69093)
SCHEDULE = (('feeding', 2, 512), ('routes', 2, 1024), ('scarcity', 2, 1536))
FILES = ('agents/sequence_replay.py', 'agents/sequence_ppo.py', 'agents/preservation.py', 'agents/network.py',
         'agents/affordances.py', 'agents/physiology.py', 'sim/world.py', 'sim/ecology.py',
         'sim/social.py', 'sim/visual_memory.py', 'experiments/scarcity_routes.py',
         'experiments/scarcity_trial.py', 'experiments/sequence_trial.py',
         'experiments/development.py', 'experiments/replay_comparison.py',
         'experiments/audit_replay_comparison.py', 'experiments/audit_scarcity.py')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest_tree(value):
    """Stable content digest, including optimizer and private replay state."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(f'tensor:{item.dtype}:{tuple(item.shape)}:'.encode())
            h.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b'dict{')
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
            h.update(b'}')
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode() + b'[')
            for entry in item:
                visit(entry)
            h.update(b']')
        else:
            h.update((type(item).__name__ + ':' + repr(item) + ';').encode())
    visit(value)
    return h.hexdigest()


def learning_digest(brain):
    state = brain.state()
    # Runtime sensing, hidden carry, action RNG and counters can evolve in evaluation.
    keys = ('model', 'target', 'optimizer', 'replay', 'priorities', 'replay_rng', 'updates',
            'predictor', 'predictor_optimizer', 'buffer', 'journal', 'progress_fast', 'progress_slow')
    return digest_tree({k: state[k] for k in keys if k in state})


def make_population(seed, family):
    p = Population(seed, PPOConfig())
    if family != 'ppo':
        c = ReplayConfig(recurrent=family == 'recurrent')
        p.brains = {rid: ReplayBrain(seed+i, c) for i, rid in enumerate(p.brains)}
    return p


def evaluation_population(world_seed, family, source, condition):
    p = make_population(world_seed, family)
    arenas = configure(p, world_seed)
    for i, (rid, b) in enumerate(p.brains.items()):
        original = source.brains[rid]
        b.model.load_state_dict(original.model.state_dict())
        if family == 'ppo':
            b.predictor.load_state_dict(original.predictor.state_dict())
        else:
            b.target.load_state_dict(original.target.state_dict())
        b.freeze()
        b.rng = random.Random(world_seed + 900000 + i)
        if condition == 'oracle':
            b.decide = lambda observation, rid=rid: safe_oracle(p.world, rid)
        elif condition == 'random':
            def choose(observation, brain=b):
                mask = action_mask(observation, ACTIONS)
                return dict(ACTIONS[brain.rng.choice([j for j, ok in enumerate(mask) if ok])])
            b.decide = choose
    return p, arenas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=2400)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / 'runs') or args.wall_seconds <= 0:
        parser.error('Use a new runs subdirectory and a positive wall budget')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (90211,) if args.pilot else SEEDS
    schedule = tuple((s, 1, 64) for s, _, _ in SCHEDULE) if args.pilot else SCHEDULE
    ticks = 512 if args.pilot else 12000
    evaluation_seeds = {s: [s+100000] if args.pilot else [s+100000, s+200000] for s in seeds}
    conditions = [f'{f}-{c}' for f in FAMILIES for c in ('trained', 'initial')] + ['random', 'oracle']
    training_worlds = {s: [s+1000*stage_i+episode for stage_i, (_, n, _) in enumerate(schedule)
                           for episode in range(n)] for s in seeds}
    assert not set(sum(training_worlds.values(), [])) & set(sum(evaluation_seeds.values(), []))
    manifest = dict(protocol=1, pilot=args.pilot, seeds=seeds, families=FAMILIES, conditions=conditions,
        training_schedule=schedule, training_worlds=training_worlds, evaluation_seeds=evaluation_seeds,
        decisions_per_agent=sum(n*d for _, n, d in schedule), evaluation_ticks=ticks,
        ppo_config=asdict(PPOConfig()), replay_config=asdict(ReplayConfig()),
        wall_seconds=args.wall_seconds, device='CPU; one Torch thread; no downloads',
        interface='Same encode_observation, 9x9 local vision/private recall, body/hearing/touch, ACTIONS and physical mask. No absolute location, goal, route or hidden state of world.',
        architecture='PPO CNN+LSTM; feedforward replay CNN+64x64 MLP; recurrent replay CNN+LSTM. Private weights, optimizer, RNG, replay/carry per resident. No experience sharing.',
        replay='One-step Double-DQN; prioritized contiguous chunks; beta=1; current/target independent burn-in from approximate stored behavior carry. Not full R2D2. Feedforward is a full-senses adaptation, not the old 23-input nine-action checkpoint.',
        reward='Identical Population.step food relief/25, hunger -0.005/tick, injury -1/20. Curiosity zero. No movement, route, farming, communication or skill bonus.',
        exploration='Flat legal-action sampling. PPO samples policy at train/eval; replay epsilon .25 train/.05 eval. All ten tones remain. Exploration differs by algorithm and is not independently controlled.',
        gate='Per family: all three training-seed groups lower zero-food ticks and raise fullness versus own initial weights; >=80% held-out lives <1% zero food and >=80% make a safe foodward crossing with >=80% safe completed crossings; halve thorn contact rate. Exact scarcity_trial.gate.',
        timing='Equal environment decisions and evaluation horizons. Wall time and parameter counts reported, not equal wall-compute budgets. Concurrent live play can affect timings.',
        limits='Small finite-stock isolated fork arenas, two independent residents per world. No competition, society, language or farming-planning conclusion. Oracle sees global resources only as feasibility control, never learner/teacher. No live imports.',
        hashes={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out / 'preregistration.json', manifest)
    (out / 'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest(), encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in FILES:
            archive.write(ROOT/file, file)
    start = time.perf_counter()
    deadline = start + args.wall_seconds
    rows = []
    timings = []
    try:
        for seed in seeds:
            # Rotate order to reduce systematic thermal/time-of-day bias.
            order = FAMILIES[seeds.index(seed):] + FAMILIES[:seeds.index(seed)]
            for family in order:
                trained = make_population(seed, family)
                initial = make_population(seed, family)
                curves = []
                family_start = time.perf_counter()
                for stage_i, (stage, episodes, decisions) in enumerate(schedule):
                    for episode in range(episodes):
                        world_seed = seed + 1000*stage_i + episode
                        fresh = make_population(world_seed, family)
                        fresh.brains = trained.brains
                        trained = fresh
                        arenas = configure(trained, world_seed, stage)
                        metrics = run(trained, arenas, decisions*4, deadline)
                        for rid, brain in trained.brains.items():
                            brain.finish(trained.world.observe(rid), terminal=True)
                        curves.append(dict(stage=stage, episode=episode, world_seed=world_seed,
                            arenas=arena_record(arenas), metrics=metrics,
                            inspector={rid: b.inspector() for rid, b in trained.brains.items()}))
                        write(out/f'{seed}-{family}-learning-curve.json', curves)
                        print(json.dumps(dict(phase='training', seed=seed, family=family, stage=stage,
                            episode=episode, seconds=round(time.perf_counter()-start, 1))), flush=True)
                timings.append(dict(seed=seed, family=family, training_seconds=time.perf_counter()-family_start,
                    decisions_per_agent=manifest['decisions_per_agent'],
                    parameters=sum(x.numel() for x in trained.brains['r0'].model.parameters())))
                trained.save(out/f'{seed}-{family}-trained.pt')
                for world_seed in evaluation_seeds[seed]:
                    for condition in ('trained', 'initial'):
                        source = trained if condition == 'trained' else initial
                        p, arenas = evaluation_population(world_seed, family, source, condition)
                        before = {rid: learning_digest(b) for rid, b in p.brains.items()}
                        label = f'{family}-{condition}'
                        eval_start = time.perf_counter()
                        metrics = run(p, arenas, ticks, deadline, out/f'{world_seed}-{label}-trace.jsonl')
                        after = {rid: learning_digest(b) for rid, b in p.brains.items()}
                        assert before == after, 'Evaluation changed learned state'
                        rows.append(dict(seed=seed, evaluation_seed=world_seed, condition=label,
                            family=family, arenas=arena_record(arenas), metrics=metrics,
                            frozen_before=before, frozen_after=after, seconds=time.perf_counter()-eval_start))
                        write(out/'results.partial.json', rows)
                        print(json.dumps(dict(phase='evaluation', seed=seed, condition=label,
                            world=world_seed, zero_food={rid:m['zero_food_ticks']/ticks for rid,m in metrics.items()},
                            seconds=round(time.perf_counter()-start, 1))), flush=True)
                del trained, initial, p, source
            for world_seed in evaluation_seeds[seed]:
                for condition in ('random', 'oracle'):
                    source = make_population(seed, 'ppo')
                    p, arenas = evaluation_population(world_seed, 'ppo', source, condition)
                    metrics = run(p, arenas, ticks, deadline, out/f'{world_seed}-{condition}-trace.jsonl')
                    rows.append(dict(seed=seed, evaluation_seed=world_seed, condition=condition,
                                     family='control', arenas=arena_record(arenas), metrics=metrics))
                    write(out/'results.partial.json', rows)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in manifest['hashes'].items())
        verdicts = {}
        for family in FAMILIES:
            selected = [{**r, 'condition': r['condition'].split('-')[-1]} for r in rows if r['family'] == family]
            verdicts[family] = gate(selected, ticks)
        write(out/'results.json', rows)
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, timing=timings,
            source_hashes_valid=True, gates=verdicts,
            total_ticks=4*manifest['decisions_per_agent']*len(seeds)*len(FAMILIES)+ticks*len(rows)))
        print(json.dumps(dict(phase='complete', seconds=round(time.perf_counter()-start, 1), gates=verdicts)), flush=True)
    except TimeoutError as error:
        write(out/'budget-stop.json', dict(reason=str(error), seconds=time.perf_counter()-start,
                                           completed_evaluations=len(rows)))
        raise


if __name__ == '__main__':
    main()
