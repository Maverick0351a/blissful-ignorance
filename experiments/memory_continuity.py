"""Matched private recurrent-context diagnostic in disposable physical worlds."""
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
from agents.environment import LearningEnv
from agents.memory_ppo import Brain
from agents.sequence_ppo import Config
from experiments.learning_curve import FILES as BASE_FILES, STAGES, make_env, run_life
from experiments.replay_comparison import digest_tree, learning_digest, write
from experiments.scarcity_trial import arena_record

SEEDS = (73401, 73412, 73423)
CONDITIONS = ('reset-continuous', 'rebuild-continuous', 'reset-reset', 'rebuild-reset',
              'initial-continuous', 'initial-reset', 'random', 'oracle')
FILES = tuple(dict.fromkeys(BASE_FILES + ('agents/memory_ppo.py',
    'experiments/memory_continuity.py', 'experiments/audit_memory_continuity.py',
    'tests/test_memory_ppo.py', 'tests/test_memory_diagnostic.py')))


class FoodReturns:
    """Observer-only counts. No target or event is supplied to an actor."""
    def __init__(self):
        self.previous = set()
        self.active = None
        self.events = []
        self.tiles = set()

    def observe(self, row, decision):
        visible = set(map(tuple, row['visible_food']))
        point = tuple(row['position'])
        self.tiles.add(point)
        available = {tuple(p) for p in row['previous_food_remaining']}
        if self.active:
            event = self.active
            event['crossed_boundary'] |= row['boundary']
            distance = sum(abs(a-b) for a, b in zip(point, event['target']))
            outcome = ('depleted' if tuple(event['target']) not in available else
                       'returned' if distance <= 1 else
                       'expired' if decision-event['opened'] >= 128 else None)
            if outcome:
                event.update(outcome=outcome, closed=decision)
                self.events.append(event)
                self.active = None
        if self.active is None and row['conscious']:
            candidates = sorted(p for p in self.previous-visible if p in available
                                and sum(abs(a-b) for a, b in zip(point, p)) > 1)
            if candidates:
                self.active = dict(target=list(candidates[0]), cue=decision-1, opened=decision,
                                   crossed_boundary=row['boundary'])
        self.previous = visible if row['conscious'] else set()

    def finish(self, decision):
        if self.active:
            self.active.update(outcome='censored', closed=decision)
            self.events.append(self.active)
            self.active = None

    def summary(self):
        return dict(food_loss_opportunities=len(self.events),
            food_returns=sum(e['outcome'] == 'returned' for e in self.events),
            boundary_opportunities=sum(e['crossed_boundary'] for e in self.events),
            boundary_returns=sum(e['crossed_boundary'] and e['outcome'] == 'returned' for e in self.events),
            expired=sum(e['outcome'] == 'expired' for e in self.events),
            censored=sum(e['outcome'] == 'censored' for e in self.events),
            depleted=sum(e['outcome'] == 'depleted' for e in self.events), unique_tiles=len(self.tiles))


class MonitoredEnv(LearningEnv):
    def attach(self, brains, condition, handle):
        self.brains, self.condition, self.handle = brains, condition, handle
        self.trackers = {rid: FoodReturns() for rid in brains}

    def step(self, actions):
        row = dict(decision=self.cycles, agents={})
        for rid, brain in self.brains.items():
            a, tracker = self.world.residents[rid], self.trackers[rid]
            obs = self._raw[rid]
            visible = [(a.x+t['dx'], a.y+t['dy']) for t in obs['tiles']
                       if t.get('resource', {}).get('kind') in ('berry', 'food')
                       and t['resource'].get('amount', 0) > 0]
            candidates = tracker.previous | ({tuple(tracker.active['target'])} if tracker.active else set())
            available = [p for p in sorted(candidates)
                         if self.world.resources.get(self.world.key(*p), {}).get('kind') in ('berry', 'food')
                         and self.world.resources[self.world.key(*p)].get('amount', 0) > 0]
            observed = dict(position=[a.x, a.y], visible_food=sorted(visible),
                conscious=not a.unconscious, previous_food_remaining=available,
                boundary=bool(brain.last_boundary) if self.condition == 'policy' else False,
                action=actions[rid])
            tracker.observe(observed, self.cycles)
            row['agents'][rid] = observed
        result = super().step(actions)
        self.handle.write(json.dumps(row, separators=(',', ':'))+'\n')
        return result


def observe_life(seed, stage, brains, decisions, condition, out, label, deadline):
    original, arenas = make_env(seed, stage, decisions, 'native')
    env = MonitoredEnv(tuple(brains), max_cycles=decisions)
    env.restore(original.state())
    trace, senses = label+'-trace.jsonl', label+'-senses.jsonl'
    with (out/senses).open('w', encoding='utf-8') as handle:
        env.attach(brains, condition, handle)
        metrics = run_life(env, arenas, brains, deadline, condition, out/trace)
    events = {}
    for rid, tracker in env.trackers.items():
        tracker.finish(decisions)
        metrics[rid].update(tracker.summary())
        events[rid] = tracker.events
    return env, dict(world_seed=seed, stage=stage, arenas=arena_record(arenas),
                     metrics=metrics, events=events, trace=trace, senses=senses)


def make_brains(seed, mode):
    return {rid: Brain(seed+i, Config(), mode, 128) for i, rid in enumerate(('r0', 'r1'))}


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
    seeds = (94531,) if args.pilot else SEEDS
    episode_decisions = 256 if args.pilot else 512
    episodes = 2 if args.pilot else 8
    eval_decisions = 256 if args.pilot else 2048
    training_maps = {s: [s+1000*i for i in range(episodes)] for s in seeds}
    test_maps = {s: [s+1000000] if args.pilot else [s+1000000, s+2000000] for s in seeds}
    manifest = dict(protocol=1, pilot=args.pilot, seeds=seeds, conditions=CONDITIONS,
        training_maps=training_maps, test_maps=test_maps, stages=STAGES,
        episode_decisions=episode_decisions, episodes=episodes,
        decisions_per_resident=episodes*episode_decisions, evaluation_ticks=eval_decisions*4,
        config=asdict(Config()), memory_capacity=128, wall_seconds=args.wall_seconds,
        runtime=dict(python=sys.version, torch=str(torch.__version__), device='cpu', threads=torch.get_num_threads()),
        hypothesis='At equal physical decisions and update budgets, rebuilding context lowers mean zero-fullness time in every seed and raises overall fullness versus reset training, both evaluated continuously.',
        secondary='Frozen periodic-reset versus continuous evaluation of each identical checkpoint diagnoses boundary sensitivity. Food returns are descriptive opportunity-conditioned behavior, not proof of memory use.',
        returns='One active target per resident: ordinary food seen on the preceding conscious decision, now out of sight and still present beyond gathering reach. Return is pre-action Manhattan distance <=1 within 128 decisions. Depleted, expired and end-censored opportunities remain in denominator. A boundary is an update/reset between cue and outcome. Observer world positions/availability never reach actor.',
        controls='Initial weights with continuous/reset evaluation, random physically feasible actions, privileged feasibility oracle. Controls never teach. All primitive actions and tones retained.',
        limits='Three seeds, two private residents each, eight 512-decision episodes and two test maps per seed. No hyperparameter sweep or test-based selection. Bounded reconstruction does not propagate gradients through earlier rollouts. Existing visual recall stays enabled; return counts alone cannot isolate recurrent memory. No amber fruit or social interaction in these arenas.',
        live='No server writes, live save loading, policy promotion, downloads or Laya changes.',
        hashes={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES})
    write(out/'preregistration.json', manifest)
    prereg = hashlib.sha256((out/'preregistration.json').read_bytes()).hexdigest()
    (out/'preregistration.sha256').write_text(prereg, encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in FILES:
            archive.write(ROOT/file, file)
    print(json.dumps(dict(phase='preregistered', sha256=prereg)), flush=True)
    start = time.perf_counter()
    deadline = start+args.wall_seconds
    training, results, checkpoints = [], [], {}
    try:
        for seed in seeds:
            sources = {}
            for mode in ('reset', 'rebuild'):
                brains = make_brains(seed, mode)
                initial_digests = {rid: learning_digest(b) for rid, b in brains.items()}
                for episode, world_seed in enumerate(training_maps[seed]):
                    stage = STAGES[episode % len(STAGES)]
                    env, row = observe_life(world_seed, stage, brains, episode_decisions, 'policy',
                        out, f'{seed}-{mode}-train-{episode}', deadline)
                    for rid, b in brains.items():
                        b.finish(env.raw_observation(rid), terminal=False)
                    row.update(seed=seed, mode=mode, episode=episode, initial_digests=initial_digests,
                               inspectors={rid: b.inspector() for rid, b in brains.items()})
                    training.append(row)
                    write(out/'training.json', training)
                name = f'{seed}-{mode}-checkpoint.pt'
                torch.save(dict(brains={rid: b.state() for rid, b in brains.items()},
                                environment=env.state()), out/name)
                checkpoints[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
                sources[mode] = brains
                print(json.dumps(dict(phase='trained', seed=seed, mode=mode,
                    updates={rid: b.updates for rid, b in brains.items()},
                    elapsed=round(time.perf_counter()-start, 1))), flush=True)
            initial = make_brains(seed, 'reset')
            for world_seed in test_maps[seed]:
                for condition in CONDITIONS:
                    source_name = condition.split('-')[0]
                    source = sources.get(source_name, initial)
                    brains = {}
                    for i, (rid, original) in enumerate(source.items()):
                        b = Brain(original.seed, original.config, original.memory_mode, original.memory_capacity)
                        b.restore(copy.deepcopy(original.state()))
                        b.freeze()
                        b.evaluation_boundary = 'reset' if condition.endswith('-reset') else 'continuous'
                        b.rng = random.Random(world_seed+900000+i)
                        brains[rid] = b
                    frozen = {rid: learning_digest(b) for rid, b in brains.items()}
                    _, row = observe_life(world_seed, 'scarcity', brains, eval_decisions,
                        condition if condition in ('random', 'oracle') else 'policy',
                        out, f'{world_seed}-{condition}', deadline)
                    after = {rid: learning_digest(b) for rid, b in brains.items()}
                    assert frozen == after
                    row.update(seed=seed, condition=condition, frozen_before=frozen, frozen_after=after)
                    results.append(row)
                    write(out/'results.partial.json', results)
                print(json.dumps(dict(phase='evaluated', seed=seed, world=world_seed,
                    elapsed=round(time.perf_counter()-start, 1))), flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f, h in manifest['hashes'].items())
        write(out/'results.json', results)
        files = [row[k] for row in training+results for k in ('trace', 'senses')]
        write(out/'complete.json', dict(seconds=time.perf_counter()-start, checkpoints=checkpoints,
            total_ticks=len(training)*episode_decisions*4+len(results)*eval_decisions*4,
            evidence_hashes={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files},
            source_hashes_valid=True, live_population_touched=False))
        print(json.dumps(dict(phase='complete', seconds=round(time.perf_counter()-start, 2))), flush=True)
    except Exception as error:
        write(out/'incomplete.json', dict(error=repr(error), training_lives=len(training),
                                        evaluation_lives=len(results)))
        raise


if __name__ == '__main__':
    main()
