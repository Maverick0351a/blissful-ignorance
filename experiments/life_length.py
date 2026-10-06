"""Matched short/long training lives in disposable worlds; live saves untouched."""
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
import torch
from agents.affordances import action_mask
from agents.environment import LearningEnv
from agents.network import ACTIONS
from agents.sequence_ppo import Brain, Config
from experiments.learning_curve import FILES as BASE_FILES, make_env, run_life
from experiments.replay_comparison import digest_tree, learning_digest, write
from experiments.scarcity_trial import arena_record

SEEDS = (75101, 75112, 75123)
BLOCKS = ('feeding', 'routes', 'scarcity', 'scarcity')
CONDITIONS = ('short', 'long', 'initial', 'random', 'oracle')
FILES = tuple(dict.fromkeys(BASE_FILES + ('experiments/life_length.py',
    'experiments/audit_life_length.py', 'tests/test_life_length.py')))


def schedule(seed, arm, pilot=False):
    if arm not in ('short', 'long'):
        raise ValueError('Unknown training arm')
    block_decisions = 256 if pilot else 4096
    life_decisions = (128 if pilot else 512) if arm == 'short' else block_decisions
    return [dict(block=block, stage=stage, repetition=repetition,
                 world_seed=seed+1000*block, decisions=life_decisions)
            for block, stage in enumerate(BLOCKS)
            for repetition in range(block_decisions//life_decisions)]


class ObservedEnv(LearningEnv):
    """Record local opportunities without supplying any new input to a brain."""
    def attach(self, handle):
        self.handle = handle
        self.visited = {rid: {(a.x, a.y)} for rid, a in self.world.residents.items()}
        self.opportunities = {rid: dict(visible_food_decisions=0,
            adjacent_food_decisions=0, food_carried_decisions=0,
            hungry_food_carried_decisions=0, tones=0, moves=0)
            for rid in self.possible_agents}

    def step(self, actions):
        row = dict(decision=self.cycles, agents={})
        for rid in self.possible_agents:
            obs, a = self._raw[rid], self.world.residents[rid]
            visible = [[t['dx'], t['dy'], t['resource']['amount']] for t in obs['tiles']
                       if t.get('resource', {}).get('kind') in ('berry', 'food')
                       and t['resource'].get('amount', 0) > 0]
            mask = action_mask(obs, ACTIONS)
            record = dict(position=[a.x, a.y], conscious=not a.unconscious,
                fullness=obs['needs'][0], food_carried=obs['inventory'].get('food', 0),
                visible_food=visible, legal=[i for i, yes in enumerate(mask) if yes],
                action=actions[rid])
            row['agents'][rid] = record
            m = self.opportunities[rid]
            m['visible_food_decisions'] += not a.unconscious and bool(visible)
            m['adjacent_food_decisions'] += not a.unconscious and any(abs(t[0])+abs(t[1]) <= 1 for t in visible)
            m['food_carried_decisions'] += not a.unconscious and record['food_carried'] > 0
            m['hungry_food_carried_decisions'] += not a.unconscious and record['food_carried'] > 0 and a.food < 60
            m['tones'] += not a.unconscious and ACTIONS[actions[rid]]['verb'] == 'tone'
            m['moves'] += not a.unconscious and ACTIONS[actions[rid]]['verb'] == 'move'
        result = super().step(actions)
        for rid, a in self.world.residents.items():
            self.visited[rid].add((a.x, a.y))
        self.handle.write(json.dumps(row, separators=(',', ':'))+'\n')
        return result


def observe_life(world_seed, stage, brains, decisions, condition, out, label, deadline):
    original, arenas = make_env(world_seed, stage, decisions, 'native')
    env = ObservedEnv(tuple(brains), max_cycles=decisions)
    env.restore(original.state())
    start = dict(world_digest=digest_tree(env.world.to_dict()),
        residents={rid: copy.deepcopy(vars(a)) for rid, a in env.world.residents.items()})
    trace, senses = label+'-trace.jsonl', label+'-senses.jsonl'
    with (out/senses).open('w', encoding='utf-8') as handle:
        env.attach(handle)
        metrics = run_life(env, arenas, brains, deadline, condition, out/trace)
    for rid in brains:
        metrics[rid].update(env.opportunities[rid], unique_tiles=len(env.visited[rid]))
    return env, dict(world_seed=world_seed, stage=stage, decisions=decisions,
        arenas=arena_record(arenas), start=start, metrics=metrics, trace=trace, senses=senses)


def make_brains(seed):
    return {rid: Brain(seed+i, Config()) for i, rid in enumerate(('r0', 'r1'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=2400)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or args.wall_seconds <= 0:
        parser.error('Use a new runs subdirectory and positive wall budget')
    out.mkdir(parents=True, exist_ok=False)
    seeds = (95641,) if args.pilot else SEEDS
    plans = {seed: {arm: schedule(seed, arm, args.pilot) for arm in ('short', 'long')}
             for seed in seeds}
    maps = {seed: [seed+1000000] if args.pilot else [seed+1000000, seed+2000000] for seed in seeds}
    eval_decisions = 512 if args.pilot else 2048
    budget = 1024 if args.pilot else 16384
    manifest = dict(protocol=1, pilot=args.pilot, seeds=seeds, plans=plans,
        test_maps=maps, conditions=CONDITIONS, decisions_per_resident=budget,
        evaluation_ticks=eval_decisions*4, config=asdict(Config()), wall_seconds=args.wall_seconds,
        runtime=dict(python=sys.version, torch=str(torch.__version__), device='cpu', threads=torch.get_num_threads()),
        hypothesis='Long training lives reduce mean zero-fullness time in every paired seed group and raise overall mean fullness versus short training lives, at equal physical decisions and PPO update counts.',
        competence_gate='Separately for each trained arm: lower zero-fullness and higher fullness than both initial and random controls in every seed; at least 80% of evaluation lives below 1% zero-fullness; at least 80% make a safe foodward crossing and at least 80% of their crossings are safe; thorn contacts per conscious decision no greater than initial controls in every seed. No automatic live promotion.',
        treatment='Four equal blocks: feeding/routes/scarcity/scarcity. Each block uses one identical map seed across arms. Short repeats that same fresh world eight times for 512 decisions; long continues it once for 4096. Private weights, optimizer and learned journal persist across lives. Both bootstrap time limits, reset hidden carry after each 128-decision update and at life end. Frozen evaluation is continuous.',
        fresh_food_units_per_resident={arm: 4*len(plans[seeds[0]][arm]) for arm in ('short', 'long')},
        confounds='Reset frequency includes fresh bodies, inventories, world resources, sensory memories and repeated early-state exposure. Fresh food totals differ by arm as recorded separately. These are treatment components, not matched resource supply. Equal map identities/stage durations/experience do not equate visited states, gradient steps after KL stopping or elapsed compute. This cannot isolate memory or elapsed time alone.',
        controls='All primitive actions and ten tones, original local senses/body rewards, curiosity zero. Initial/random/oracle never teach. Two private residents in isolated arenas, not a social community. No amber food: avoidance is unmeasured. Crop maturation is measured output, not proof of useful farming plans.',
        selection='Three fresh seed groups; two held-out maps each; final checkpoint only; no hyperparameter sweep or test-guided choice. Twelve evaluated lives per condition are repeated brains over maps, not twelve independent training replicates.',
        live='No live save loads/writes, server commands, downloads, imports, Laya changes, publication or spending.',
        hashes={file: hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES})
    write(out/'preregistration.json', manifest)
    prereg = hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    (out/'preregistration.sha256').write_text(prereg, encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in FILES:
            archive.write(ROOT/file, file)
    print(json.dumps(dict(phase='preregistered', sha256=prereg)), flush=True)
    start_time = time.perf_counter()
    deadline = start_time+args.wall_seconds
    training, results, checkpoints = [], [], {}
    try:
        for seed_index, seed in enumerate(seeds):
            sources = {}
            for arm in (('short', 'long') if seed_index % 2 == 0 else ('long', 'short')):
                brains = make_brains(seed)
                for life, plan in enumerate(plans[seed][arm]):
                    before = {rid: learning_digest(b) for rid, b in brains.items()}
                    env, row = observe_life(plan['world_seed'], plan['stage'], brains,
                        plan['decisions'], 'policy', out, f'{seed}-{arm}-train-{life}', deadline)
                    for rid, brain in brains.items():
                        brain.finish(env.raw_observation(rid), terminal=False)
                    row.update(seed=seed, arm=arm, life=life, block=plan['block'], repetition=plan['repetition'],
                        learning_before=before, learning_after={rid: learning_digest(b) for rid, b in brains.items()},
                        inspectors={rid: b.inspector() for rid, b in brains.items()})
                    training.append(row)
                    write(out/'training.json', training)
                    next_life = plans[seed][arm][life+1:life+2]
                    if not next_life or next_life[0]['block'] != plan['block']:
                        print(json.dumps(dict(phase='trained-block', seed=seed, arm=arm,
                            block=plan['block'], decisions=brains['r0'].decisions,
                            updates=brains['r0'].updates, elapsed=round(time.perf_counter()-start_time, 1))), flush=True)
                name = f'{seed}-{arm}-checkpoint.pt'
                torch.save(dict(brains={rid: b.state() for rid, b in brains.items()}, environment=env.state()), out/name)
                checkpoints[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
                sources[arm] = brains
            initial = make_brains(seed)
            for world_seed in maps[seed]:
                for condition in CONDITIONS:
                    source = sources.get(condition, initial)
                    brains = {}
                    for i, (rid, original) in enumerate(source.items()):
                        brain = Brain(original.seed, original.config)
                        brain.restore(copy.deepcopy(original.state()))
                        brain.freeze()
                        brain.rng = random.Random(world_seed+900000+i)
                        brains[rid] = brain
                    before = {rid: learning_digest(b) for rid, b in brains.items()}
                    _, row = observe_life(world_seed, 'scarcity', brains, eval_decisions,
                        condition if condition in ('random', 'oracle') else 'policy',
                        out, f'{world_seed}-{condition}', deadline)
                    after = {rid: learning_digest(b) for rid, b in brains.items()}
                    assert before == after
                    row.update(seed=seed, condition=condition, frozen_before=before, frozen_after=after)
                    results.append(row)
                    write(out/'results.partial.json', results)
                print(json.dumps(dict(phase='evaluated', seed=seed, world=world_seed,
                    elapsed=round(time.perf_counter()-start_time, 1))), flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in manifest['hashes'].items())
        write(out/'results.json', results)
        files = [row[key] for row in training+results for key in ('trace', 'senses')]
        write(out/'complete.json', dict(seconds=time.perf_counter()-start_time,
            total_ticks=sum(r['decisions']*4 for r in training+results), checkpoints=checkpoints,
            evidence_hashes={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files},
            source_hashes_valid=True, live_population_touched=False))
        print(json.dumps(dict(phase='complete', seconds=round(time.perf_counter()-start_time, 2))), flush=True)
    except Exception as error:
        write(out/'incomplete.json', dict(error=repr(error), training_lives=len(training),
            evaluation_lives=len(results), elapsed=time.perf_counter()-start_time))
        raise


if __name__ == '__main__':
    main()
