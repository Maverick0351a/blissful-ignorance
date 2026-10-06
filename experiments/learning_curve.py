"""Fixed-budget recurrent-PPO learning curves through the parallel interface."""
import argparse
import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.environment import LearningEnv
from agents.network import ACTIONS
from agents.sequence_ppo import Brain, Config
from experiments.replay_comparison import FILES as BASE_FILES, learning_digest, write
from experiments.scarcity_routes import Routes, configure, food_stock, safe_oracle
from experiments.scarcity_trial import arena_record, gate
from sim.world import World

SEEDS = (71201, 71212)
BUDGETS = (2048, 8192, 16384)
STAGES = ('feeding', 'feeding', 'routes', 'routes', 'scarcity', 'scarcity', 'scarcity', 'scarcity')
FILES = BASE_FILES + ('agents/environment.py', 'adapters/pettingzoo_env.py',
                      'experiments/learning_curve.py', 'experiments/audit_learning_curve.py',
                      'scripts/rl-dependencies.json')


class ForkFactory:
    def __init__(self, stage):
        self.stage = stage
        self.arenas = None

    def __call__(self, seed, options):
        world = World(seed)
        shell = SimpleNamespace(world=world, brains={'r0': None, 'r1': None})
        self.arenas = configure(shell, seed, self.stage)
        return world


def make_env(seed, stage, decisions, interface):
    factory = ForkFactory(stage)
    cls = LearningEnv
    if interface == 'pettingzoo':
        from adapters.pettingzoo_env import GodhoodParallelEnv
        cls = GodhoodParallelEnv
    env = cls(('r0', 'r1'), max_cycles=decisions, world_factory=factory)
    env.reset(seed=seed)
    return env, factory.arenas


def core(env):
    return env.core if hasattr(env, 'core') else env


def make_brains(seed):
    return {rid: Brain(seed+i, Config()) for i, rid in enumerate(('r0', 'r1'))}


def run_life(env, arenas, brains, deadline, condition='policy', trace_path=None):
    engine = core(env)
    trackers = {rid: Routes(a) for rid, a in arenas.items()}
    names = ('nutrition', 'damage', 'zero_food_ticks', 'unconscious_ticks', 'ordinary_eaten',
             'amber_eaten', 'seed_conversions', 'planted', 'matured_food', 'mean_fullness_sum',
             'conscious_decisions', 'thorn_contacts', 'thorn_opportunities', 'invalid', 'conscious_invalid')
    metrics = {rid: dict.fromkeys(names, 0) for rid in brains}
    handle = trace_path.open('w', encoding='utf-8') if trace_path else None
    try:
        while env.agents:
            if engine.cycles % 32 == 0 and time.perf_counter() > deadline:
                raise TimeoutError('Declared learning-curve wall budget reached')
            world = engine.world
            before = {rid: dict(conscious=not a.unconscious, thorns=a.thorn_contacts,
                                position=(a.x, a.y))
                      for rid in brains for a in (world.residents[rid],)}
            commands = {}
            for rid, brain in brains.items():
                obs = engine.raw_observation(rid)
                if condition == 'oracle':
                    commands[rid] = safe_oracle(world, rid)
                elif condition == 'random':
                    from agents.affordances import action_mask
                    legal = [i for i, ok in enumerate(action_mask(obs, ACTIONS)) if ok]
                    commands[rid] = dict(ACTIONS[brain.rng.choice(legal)])
                else:
                    commands[rid] = brain.decide(obs)
            crops = {k for k, r in world.resources.items() if r['kind'] == 'crop'}
            _, _, _, _, infos = env.step({rid: ACTIONS.index(c) for rid, c in commands.items()})
            matured = {rid: 0 for rid in brains}
            for key in crops:
                resource = world.resources.get(key, {})
                if resource.get('kind') == 'berry':
                    point = tuple(map(int, key.split(',')))
                    for rid, arena in arenas.items():
                        if point in arena.cells():
                            matured[rid] += resource['amount']
            trace = dict(tick=engine.cycles*4, agents={})
            for rid, brain in brains.items():
                for parts in infos[rid]['reward_parts_by_tick']:
                    brain.add_reward(parts)
                a, info, m, command = world.residents[rid], infos[rid], metrics[rid], commands[rid]
                conscious, success = before[rid]['conscious'], info['action_success']
                verb = command['verb']
                m['nutrition'] += info['nutrition']
                m['damage'] += info['injury']
                m['zero_food_ticks'] += info['zero_food_ticks']
                m['unconscious_ticks'] += info['unconscious_ticks']
                m['mean_fullness_sum'] += info['fullness_sum']
                m['conscious_decisions'] += conscious
                m['thorn_opportunities'] += conscious and sum(abs(x-y) for x, y in
                    zip(before[rid]['position'], arenas[rid].hazard)) <= 1
                m['thorn_contacts'] += a.thorn_contacts-before[rid]['thorns']
                m['invalid'] += not success
                m['conscious_invalid'] += conscious and not success
                m['ordinary_eaten'] += success and verb == 'eat' and command.get('item', 'food') == 'food'
                m['amber_eaten'] += success and verb == 'eat' and command.get('item') == 'amber_fruit'
                m['seed_conversions'] += success and verb == 'make_seeds'
                m['planted'] += success and verb == 'plant'
                m['matured_food'] += matured[rid]
                trackers[rid].observe((a.x, a.y))
                actual = food_stock(world, arenas[rid])
                assert actual == 4+m['matured_food']-m['ordinary_eaten']-m['seed_conversions']
                trace['agents'][rid] = dict(position=[a.x, a.y], food=a.food, health=a.health, water=a.water,
                    conscious=conscious, unconscious=a.unconscious, action=command, success=success,
                    nutrition=info['nutrition'], damage=info['injury'], zero_food_ticks=info['zero_food_ticks'],
                    unconscious_ticks=info['unconscious_ticks'], thorn_contacts=a.thorn_contacts-before[rid]['thorns'],
                    food_stock=actual, mean_fullness_sum=m['mean_fullness_sum'], matured_food=m['matured_food'])
            if handle:
                handle.write(json.dumps(trace, separators=(',', ':'))+'\n')
    finally:
        if handle:
            handle.close()
    for rid, m in metrics.items():
        m.update(trackers[rid].summary(), mean_fullness=m['mean_fullness_sum']/(engine.cycles*4),
                 initial_food_stock=4, remaining_food_stock=food_stock(engine.world, arenas[rid]))
    return metrics


def evaluate(seed, world_seed, source, budget, split, condition, interface, decisions, out, deadline):
    env, arenas = make_env(world_seed, 'scarcity', decisions, interface)
    brains = {}
    for i, (rid, original) in enumerate(source.items()):
        b = Brain(original.seed, original.config)
        b.restore(copy.deepcopy(original.state()))
        b.freeze()
        b.rng = random.Random(world_seed+900000+i)
        brains[rid] = b
    before = {rid: learning_digest(b) for rid, b in brains.items()}
    name = f'{world_seed}-{budget}-{condition}-trace.jsonl'
    metrics = run_life(env, arenas, brains, deadline, condition, out/name)
    after = {rid: learning_digest(b) for rid, b in brains.items()}
    assert before == after
    return dict(seed=seed, world_seed=world_seed, budget=budget, split=split, condition=condition,
        arenas=arena_record(arenas), metrics=metrics, frozen_before=before, frozen_after=after, trace=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--interface', choices=('native', 'pettingzoo'), default='native')
    parser.add_argument('--wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or args.wall_seconds <= 0:
        parser.error('Use a new runs subdirectory and positive wall budget')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (92321,) if args.pilot else SEEDS
    budgets = (128, 256) if args.pilot else BUDGETS
    episode_decisions = 64 if args.pilot else 512
    eval_decisions = 128 if args.pilot else 3000
    maps = {s: {'validation': [s+1000000] if args.pilot else [s+1000000, s+2000000],
                'final': [s+3000000] if args.pilot else [s+3000000, s+4000000]} for s in seeds}
    training_maps = {s: [s+1000*i for i in range(budgets[-1]//episode_decisions)] for s in seeds}
    assert not set(sum(training_maps.values(), [])) & {w for m in maps.values() for ws in m.values() for w in ws}
    manifest = dict(protocol=1, pilot=args.pilot, seeds=seeds, budgets=budgets,
        episode_decisions=episode_decisions, stages=STAGES, maps=maps, training_maps=training_maps,
        evaluation_ticks=eval_decisions*4, config=asdict(Config()), interface=args.interface,
        runtime=dict(python=sys.version, device='cpu', torch=str(torch.__version__),
            numpy=version('numpy'), **({name: version(name) for name in
                ('pettingzoo', 'gymnasium', 'cloudpickle', 'farama-notifications')}
                if args.interface == 'pettingzoo' else {})),
        wall_seconds=args.wall_seconds,
        objective='Learning-curve benchmark and exact environment-interface validation. Two seeds/four private learners in full run; not a conclusive architecture ranking.',
        training='Same private recurrent PPO and body reward; no teacher or scripted learner; no curiosity bonus. Fixed repeated curriculum feeding/feeding/routes/routes/scarcity x4. Independent world and brain seeds. Four physical ticks per action.',
        truncation='Episode time limits bootstrap the value target (finish terminal=False), then reset recurrent carry. Mortality stays off. This differs from earlier terminal-cut experiments.',
        selection='Validate each fixed budget. Evaluate the maximum budget on fresh final maps regardless of validation results; no test-guided checkpoint selection. Initial/random/oracle controls never teach.',
        physics='Original World.step with scripted=False; batched actions resolved by original seeded random ordering. Same local sensory encoder/action masks. Conservation audited every four-tick action interval.',
        limits='Small isolated finite-stock forks, no competition/language conclusion. Fixed train/validation/final layouts; no live import or promotion. Longer budget alone is not expected to guarantee success.',
        hashes={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out/'preregistration.json', manifest)
    (out/'preregistration.sha256').write_text(hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest(), encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for f in FILES:
            archive.write(ROOT/f, f)
    start = time.perf_counter()
    deadline = start+args.wall_seconds
    rows, curves, checkpoints = [], [], {}
    try:
        for seed in seeds:
            brains, initial = make_brains(seed), make_brains(seed)
            episode = 0
            for budget in budgets:
                train_start = time.perf_counter()
                while brains['r0'].decisions < budget:
                    world_seed = training_maps[seed][episode]
                    stage = STAGES[episode % len(STAGES)]
                    env, arenas = make_env(world_seed, stage, episode_decisions, args.interface)
                    metrics = run_life(env, arenas, brains, deadline)
                    for rid, b in brains.items():
                        b.finish(core(env).raw_observation(rid), terminal=False)
                    curves.append(dict(seed=seed, episode=episode, stage=stage, world_seed=world_seed,
                        metrics=metrics, decisions=brains['r0'].decisions,
                        inspectors={rid: b.inspector() for rid, b in brains.items()}))
                    episode += 1
                name = f'{seed}-{budget}-checkpoint.pt'
                torch.save(dict(schema=1, environment=core(env).state(),
                                brains={rid: b.state() for rid, b in brains.items()}, next_episode=episode), out/name)
                checkpoints[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
                write(out/'learning-curves.json', curves)
                print(json.dumps(dict(phase='trained', seed=seed, budget=budget,
                    training_seconds=round(time.perf_counter()-train_start, 2),
                    updates={rid: b.updates for rid, b in brains.items()}, elapsed=round(time.perf_counter()-start, 1))), flush=True)
                for world_seed in maps[seed]['validation']:
                    row = evaluate(seed, world_seed, brains, budget, 'validation', 'trained',
                                   args.interface, eval_decisions, out, deadline)
                    rows.append(row)
                    write(out/'results.partial.json', rows)
                    print(json.dumps(dict(phase='validation', seed=seed, budget=budget,
                        world=world_seed, zero_food={rid: m['zero_food_ticks']/(eval_decisions*4)
                        for rid, m in row['metrics'].items()}, elapsed=round(time.perf_counter()-start, 1))), flush=True)
            for split, world_seeds in maps[seed].items():
                for world_seed in world_seeds:
                    conditions = ('trained', 'initial', 'random', 'oracle') if split == 'final' else ('initial', 'random', 'oracle')
                    for condition in conditions:
                        rows.append(evaluate(seed, world_seed, brains if condition == 'trained' else initial,
                            budgets[-1] if condition == 'trained' else 0, split, condition,
                            args.interface, eval_decisions, out, deadline))
                        write(out/'results.partial.json', rows)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in manifest['hashes'].items())
        gates = {}
        for split in ('validation', 'final'):
            levels = budgets if split == 'validation' else (budgets[-1],)
            for budget in levels:
                selected = [dict(r, seed=r['seed']) for r in rows if r['split'] == split
                    and (r['condition'] == 'trained' and r['budget'] == budget or r['condition'] == 'initial')]
                gates[f'{split}-{budget}'] = gate(selected, eval_decisions*4)
        write(out/'results.json', rows)
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, checkpoints=checkpoints,
            gates=gates, total_ticks=4*budgets[-1]*len(seeds)+len(rows)*eval_decisions*4,
            source_hashes_valid=True))
        print(json.dumps(dict(phase='complete', elapsed=round(time.perf_counter()-start, 1), gates=gates)), flush=True)
    except TimeoutError as error:
        write(out/'budget-stop.json', dict(reason=str(error), completed_evaluations=len(rows)))
        raise


if __name__ == '__main__':
    main()
