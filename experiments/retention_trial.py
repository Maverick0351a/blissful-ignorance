"""Fresh-seed 2x2 retention/exploration screen in disposable private worlds."""
import argparse
import copy
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

from agents.affordances import action_mask
from agents.network import ACTIONS
from agents.retained_replay import Brain, Config
from agents.sequence_ppo import Population
from experiments.replay_comparison import FILES as BASE_FILES, digest_tree, write
from experiments.scarcity_routes import configure, safe_oracle
from experiments.scarcity_trial import arena_record, gate, run

ARMS = ('fifo_flat', 'retained_flat', 'fifo_balanced', 'retained_balanced')
SEEDS = (70131, 70142, 70153)
SCHEDULE = (('feeding', 1, 512), ('routes', 1, 512), ('scarcity', 1, 1024))
FILES = BASE_FILES + ('agents/retained_replay.py', 'experiments/retention_trial.py',
                      'experiments/audit_retention.py')


def config(arm):
    return Config(retain_consequences=arm.startswith('retained'),
                  balanced_exploration=arm.endswith('balanced'))


def make_population(seed, arm):
    p = Population(seed)
    p.brains = {rid: Brain(seed+i, config(arm)) for i, rid in enumerate(p.brains)}
    return p


def learning_digest(brain):
    state = brain.state()
    return digest_tree({k: state[k] for k in ('model', 'target', 'optimizer', 'replay',
        'priorities', 'replay_rng', 'retention_rng', 'experiment', 'updates')})


def evaluate_population(world_seed, source, condition):
    p = Population(world_seed)
    arenas = configure(p, world_seed)
    for i, (rid, original) in enumerate(source.brains.items()):
        b = Brain(original.seed, original.config)
        b.restore(copy.deepcopy(original.state()))
        b.freeze()
        b.rng = random.Random(world_seed+900000+i)
        p.brains[rid] = b
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
    parser.add_argument('--wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or args.wall_seconds <= 0:
        parser.error('Use a new runs subdirectory and a positive wall budget')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (91231,) if args.pilot else SEEDS
    schedule = tuple((s, 1, 64) for s, _, _ in SCHEDULE) if args.pilot else SCHEDULE
    ticks = 512 if args.pilot else 12000
    eval_seeds = {s: [s+100000] if args.pilot else [s+100000, s+200000] for s in seeds}
    training_worlds = {s: [s+1000*i for i in range(len(schedule))] for s in seeds}
    assert not set(sum(training_worlds.values(), [])) & set(sum(eval_seeds.values(), []))
    manifest = dict(protocol=1, pilot=args.pilot, seeds=seeds, arms=ARMS,
        conditions=ARMS+('initial', 'random', 'oracle'),
        training_schedule=schedule, training_worlds=training_worlds,
        evaluation_seeds=eval_seeds, evaluation_ticks=ticks,
        decisions_per_agent=sum(n*d for _, n, d in schedule),
        configs={arm: asdict(config(arm)) for arm in ARMS}, wall_seconds=args.wall_seconds,
        hypotheses='For each seed report main retention and exploration effects and their interaction on zero-food fraction and fullness. Directional support requires negative deprivation and positive fullness effects in every seed; three seeds are a mechanism screen, not significance or promotion evidence.',
        retention='Same 64 chunks, up to 16 protected by a uniform reservoir over chunks with own observed nutrition or injury. Unprotected oldest chunk evicted. No duplicates; available unreserved slots stay recent. Pure FIFO in controls.',
        sampling='All retained chunks share one priority distribution and base beta=1 correction to the uniform retained-chunk objective. Retention changes the dataset, not the correction. No oversampling of protected chunks.',
        exploration='Only training epsilon draws change: flat feasible actions versus uniform available verb then uniform feasible variant. All directions, harmful physical actions and ten tones remain. Evaluation always flat 5% epsilon.',
        matching='Same fresh weights per seed, architecture/parameter count, 2048 training decisions, optimizer/update count, memory capacity, body reward, local senses, worlds and flat evaluation exploration. Only retention and training epsilon distribution vary. Buffer occupancy/context lengths and wall time reported, not forced equal.',
        reward='Existing own nutrition/25, hunger -0.005/tick, injury -1/20; curiosity zero; no supplied actions or skill/communication/route reward.',
        initial_control='One shared initial-weight evaluation per seed/world because every arm has identical initialization and frozen exploration. These are evaluation controls, never shared training data.',
        gate='Existing scarcity reliable-feeding gate versus matched initial weights is also reported. No automatic promotion regardless of outcome; confirm useful factors with longer training and new seeds.',
        limits='Two private residents in separate finite-stock fork arenas; no interaction, competition, learned language or general intelligence conclusion. No amber food here. No live save/server/model accessed.',
        hashes={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out/'preregistration.json', manifest)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest(), encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in FILES:
            archive.write(ROOT/file, file)
    start = time.perf_counter()
    deadline = start+args.wall_seconds
    rows, timings, checkpoints = [], [], {}
    try:
        for seed in seeds:
            initial = make_population(seed, 'fifo_flat')
            initial_digests = {rid: digest_tree(b.model.state_dict()) for rid, b in initial.brains.items()}
            order = ARMS[seeds.index(seed):]+ARMS[:seeds.index(seed)]
            for arm in order:
                trained = make_population(seed, arm)
                assert initial_digests == {rid: digest_tree(b.model.state_dict()) for rid, b in trained.brains.items()}
                curves = []
                train_start = time.perf_counter()
                for stage_i, (stage, _, decisions) in enumerate(schedule):
                    world_seed = training_worlds[seed][stage_i]
                    fresh = Population(world_seed)
                    fresh.brains = trained.brains
                    trained = fresh
                    arenas = configure(trained, world_seed, stage)
                    metrics = run(trained, arenas, decisions*4, deadline)
                    for rid, brain in trained.brains.items():
                        brain.finish(trained.world.observe(rid), terminal=True)
                    curves.append(dict(stage=stage, world_seed=world_seed,
                        arenas=arena_record(arenas), metrics=metrics,
                        inspector={rid: b.inspector() for rid, b in trained.brains.items()}))
                    write(out/f'{seed}-{arm}-learning-curve.json', curves)
                timings.append(dict(seed=seed, arm=arm, training_seconds=time.perf_counter()-train_start,
                    parameters=sum(p.numel() for p in trained.brains['r0'].model.parameters()),
                    initial_model_digests=initial_digests,
                    inspectors={rid: b.inspector() for rid, b in trained.brains.items()}))
                name = f'{seed}-{arm}-trained.pt'
                trained.save(out/name)
                checkpoints[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
                print(json.dumps(dict(phase='training', seed=seed, arm=arm,
                    memory={rid: {k: v for k, v in b.inspector().items() if k.startswith('retained_') or k == 'protected_chunks'} for rid, b in trained.brains.items()},
                    seconds=round(time.perf_counter()-start, 1))), flush=True)
                for world_seed in eval_seeds[seed]:
                    p, arenas = evaluate_population(world_seed, trained, arm)
                    before = {rid: learning_digest(b) for rid, b in p.brains.items()}
                    metrics = run(p, arenas, ticks, deadline, out/f'{world_seed}-{arm}-trace.jsonl')
                    after = {rid: learning_digest(b) for rid, b in p.brains.items()}
                    assert before == after, 'Evaluation modified learned state'
                    rows.append(dict(seed=seed, evaluation_seed=world_seed, condition=arm,
                        arenas=arena_record(arenas), metrics=metrics, frozen_before=before, frozen_after=after))
                    write(out/'results.partial.json', rows)
                    print(json.dumps(dict(phase='evaluation', seed=seed, condition=arm, world=world_seed,
                        zero_food={rid: m['zero_food_ticks']/ticks for rid, m in metrics.items()},
                        seconds=round(time.perf_counter()-start, 1))), flush=True)
                del trained, p
            for world_seed in eval_seeds[seed]:
                for condition in ('initial', 'random', 'oracle'):
                    p, arenas = evaluate_population(world_seed, initial, condition)
                    before = {rid: learning_digest(b) for rid, b in p.brains.items()}
                    metrics = run(p, arenas, ticks, deadline, out/f'{world_seed}-{condition}-trace.jsonl')
                    after = {rid: learning_digest(b) for rid, b in p.brains.items()}
                    assert before == after
                    rows.append(dict(seed=seed, evaluation_seed=world_seed, condition=condition,
                        arenas=arena_record(arenas), metrics=metrics, frozen_before=before, frozen_after=after))
                    write(out/'results.partial.json', rows)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in manifest['hashes'].items())
        gates = {arm: gate([{**r, 'condition': 'trained' if r['condition'] == arm else 'initial'}
                 for r in rows if r['condition'] in (arm, 'initial')], ticks) for arm in ARMS}
        write(out/'results.json', rows)
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, timing=timings,
            checkpoints=checkpoints, source_hashes_valid=True, gates=gates,
            total_ticks=4*manifest['decisions_per_agent']*len(seeds)*len(ARMS)+ticks*len(rows)))
        print(json.dumps(dict(phase='complete', seconds=round(time.perf_counter()-start, 1), gates=gates)), flush=True)
    except TimeoutError as error:
        write(out/'budget-stop.json', dict(reason=str(error), seconds=time.perf_counter()-start,
                                           completed_evaluations=len(rows)))
        raise


if __name__ == '__main__':
    main()
