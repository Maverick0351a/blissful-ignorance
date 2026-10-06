"""Bounded observation of continuing lives, without changing policies or physics.

Stop the playable server first. Use scripts/Watch-Population.ps1 for save,
observation, atomic continuation, and restart. Raw evidence stays under runs/.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.network import ACTIONS
from agents.sequence_ppo import Population
from sim.checkpoint_lease import checkpoint_lease

FILES = ('agents/sequence_ppo.py', 'agents/preservation.py', 'agents/network.py', 'agents/affordances.py', 'agents/laya_resident.py',
         'agents/physiology.py', 'sim/world.py', 'sim/ecology.py', 'sim/social.py',
         'sim/visual_memory.py', 'sim/checkpoint_lease.py', 'experiments/population_watch.py')
BODY_FIELDS = ('name', 'x', 'y', 'food', 'water', 'energy', 'warmth', 'health',
               'pain', 'unconscious', 'starvation_ticks', 'fainted', 'thorn_contacts',
               'damage_taken', 'gathered', 'shared')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(resident):
    return {**{k: getattr(resident, k) for k in BODY_FIELDS},
            'inventory': dict(resident.inventory)}


def snapshot(population):
    w = population.world
    return dict(tick=w.tick, residents={rid: body(w.residents[rid]) for rid in population.brains},
                metrics={rid: dict(m) for rid, m in population.metrics.items()},
                brains={rid: dict(backend=b.backend, decisions=b.decisions, updates=b.updates,
                                  policy_version=b.policy_version, buffer=len(b.buffer),
                                  pending=b.pending is not None, config=asdict(b.config))
                        for rid, b in population.brains.items()},
                resources=dict(Counter(r['kind'] for r in w.resources.values())),
                ground_food=sum(r['amount'] for r in w.resources.values() if r['kind'] in ('berry', 'food')),
                inventory_food=sum(a.inventory.get('food', 0) for a in w.residents.values()),
                cache_food=sum(c['food'] for c in w.caches.values()) +
                           sum(a.carried_cache['food'] for a in w.residents.values() if a.carried_cache is not None),
                structures=len(w.structures), caches=len(w.caches), event_serial=w.event_serial)


def observe(population, output, ticks, wall_seconds):
    """Use each actual policy observation once; never request extra observations."""
    if population.neighbors_scripted or set(population.brains) != set(population.world.residents) - {'player'}:
        raise ValueError('Every autonomous resident must have an independent brain')
    if ticks <= 0 or wall_seconds <= 0:
        raise ValueError('Positive tick and wall budgets required')
    start = snapshot(population)
    write(output / 'start.json', start)
    commands = {}
    originals = {rid: b.decide for rid, b in population.brains.items()}

    def recorder(rid):
        def decide(observation):
            command = originals[rid](observation)
            brain = population.brains[rid]
            commands[rid] = dict(action=dict(command), observation=observation,
                                 allowed=[i for i, enabled in enumerate(brain.pending['mask'][0].tolist()) if enabled],
                                 chosen=brain.pending['chosen'], version=brain.policy_version)
            return command
        return decide

    for rid, b in population.brains.items():
        b.decide = recorder(rid)
    started = time.perf_counter()
    reason = 'tick budget'
    event_serial = population.world.event_serial
    try:
        with gzip.open(output / 'trace.jsonl.gz', 'wt', encoding='utf-8') as trace:
            for offset in range(ticks):
                if time.perf_counter() - started >= wall_seconds:
                    reason = 'wall budget'
                    break
                if (output / 'STOP').exists():
                    reason = 'stop marker'
                    break
                commands.clear()
                results = population.step()
                w = population.world
                events = [dict(e) for e in w.events if e['id'] > event_serial]
                event_serial = w.event_serial
                row = dict(tick=w.tick, residents={rid: body(w.residents[rid]) for rid in population.brains},
                           decisions={rid: {**d, 'success': results[rid][0], 'message': results[rid][1]}
                                      for rid, d in commands.items()}, events=events)
                if (offset + 1) % 256 == 0:
                    row['sample'] = snapshot(population)
                    trace.flush()
                    print(json.dumps(dict(tick=w.tick, advanced=offset+1,
                                          seconds=round(time.perf_counter()-started, 2))), flush=True)
                trace.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    finally:
        for rid, b in population.brains.items():
            b.decide = originals[rid]
    end = snapshot(population)
    write(output / 'end.json', end)
    summary = summarize(output, start, end)
    summary.update(seconds=time.perf_counter()-started, stop_reason=reason)
    write(output / 'summary.json', summary)
    return summary


def summarize(output, start, end):
    elapsed = end['tick'] - start['tick']
    ids = list(start['residents'])
    stats = {rid: dict(actions=Counter(), successes=Counter(), failures=Counter(),
                      conscious_decisions=0, zero_food_ticks=0, hungry_ticks=0, unconscious_ticks=0,
                      thirsty_ticks=0, visible_food_decisions=0, adjacent_food_decisions=0,
                      hungry_visible_food_decisions=0, hungry_adjacent_food_decisions=0,
                      hungry_adjacent_gathers=0, edible_inventory_decisions=0,
                      eat_when_available=0, fullness_sum=0., positions=set(),
                      uniform_tone_mass_sum=0., tones_heard_decisions=0)
             for rid in ids}
    windows = {}
    count = 0
    with gzip.open(output / 'trace.jsonl.gz', 'rt', encoding='utf-8') as trace:
        for line in trace:
            row = json.loads(line)
            count += 1
            if row['tick'] != start['tick'] + count or set(row['residents']) != set(ids):
                raise ValueError('Trace continuity or identity mismatch')
            expected = set(ids) if (row['tick'] - 1) % 4 == 0 else set()
            if set(row['decisions']) != expected:
                raise ValueError('Incorrect decision schedule')
            index = str((count-1)//1024)
            win = windows.setdefault(index, dict(first_tick=row['tick'], last_tick=row['tick'],
                resident_ticks=0, zero_food_ticks=0, unconscious_ticks=0,
                conscious_decisions=0, tones=0, moves=0, gathers=0, safe_eaten=0))
            win['last_tick'] = row['tick']
            for rid, a in row['residents'].items():
                s = stats[rid]
                s['positions'].add((a['x'], a['y']))
                s['fullness_sum'] += a['food']
                win['resident_ticks'] += 1
                for key, value in [('zero_food_ticks', a['food'] <= 0),
                                   ('hungry_ticks', a['food'] < 60),
                                   ('thirsty_ticks', a['water'] <= 0),
                                   ('unconscious_ticks', bool(a['unconscious']))]:
                    s[key] += value
                    if key in win:
                        win[key] += value
            for rid, decision in row['decisions'].items():
                s = stats[rid]
                observation = decision['observation']
                if observation['body']['unconscious']:
                    continue
                action = decision['action']; verb = action['verb']
                s['conscious_decisions'] += 1
                s['actions'][verb] += 1
                win['conscious_decisions'] += 1
                for key, target in [('tones', 'tone'), ('moves', 'move'), ('gathers', 'gather')]:
                    win[key] += verb == target
                if decision['success']:
                    s['successes'][verb] += 1
                    win['safe_eaten'] += verb == 'eat' and action.get('item', 'food') == 'food'
                else:
                    s['failures'][decision['message']] += 1
                food = [t for t in observation['tiles'] if t.get('resource', {}).get('kind') in ('berry', 'food')
                        and t['resource'].get('amount', 0) > 0 or t.get('cache', {}).get('food', 0) > 0]
                adjacent = any(abs(t['dx']) + abs(t['dy']) <= 1 for t in food)
                hungry = observation['needs'][0] < 60
                s['visible_food_decisions'] += bool(food)
                s['adjacent_food_decisions'] += adjacent
                s['hungry_visible_food_decisions'] += hungry and bool(food)
                s['hungry_adjacent_food_decisions'] += hungry and adjacent
                s['hungry_adjacent_gathers'] += hungry and adjacent and verb in ('gather', 'take_food')
                edible = any(ACTIONS[i]['verb'] == 'eat' and ACTIONS[i].get('item', 'food') == 'food'
                             for i in decision['allowed'])
                s['edible_inventory_decisions'] += edible
                s['eat_when_available'] += edible and verb == 'eat' and action.get('item', 'food') == 'food'
                s['uniform_tone_mass_sum'] += sum(ACTIONS[i]['verb'] == 'tone' for i in decision['allowed']) / len(decision['allowed'])
                s['tones_heard_decisions'] += any(h.get('kind', '').startswith('tone_') for h in observation['hearing'])
    if count != elapsed:
        raise ValueError('Trace length mismatch')
    for rid, s in stats.items():
        s['unique_tiles'] = len(s.pop('positions'))
        s['mean_fullness'] = s.pop('fullness_sum') / max(1, elapsed)
        s['uniform_tone_share'] = s.pop('uniform_tone_mass_sum') / max(1, s['conscious_decisions'])
        s['tone_share'] = s['actions']['tone'] / max(1, s['conscious_decisions'])
        s['metric_deltas'] = {k: v-start['metrics'][rid][k] for k, v in end['metrics'][rid].items()}
        for key in ('zero_food_ticks', 'unconscious_ticks', 'conscious_decisions'):
            if s[key] != s['metric_deltas'][key]:
                raise ValueError(f'{rid}: independent recount mismatch for {key}')
        expected_decisions = sum(t % 4 == 0 for t in range(start['tick'], end['tick']))
        if end['brains'][rid]['decisions'] - start['brains'][rid]['decisions'] != expected_decisions:
            raise ValueError('Checkpoint decision count mismatch')
        s['updates'] = end['brains'][rid]['updates'] - start['brains'][rid]['updates']
        s['name'] = end['residents'][rid]['name']
        s['model'] = end['brains'][rid].get('backend', 'recurrent-ppo')
    return dict(start_tick=start['tick'], end_tick=end['tick'], ticks=elapsed,
                agents=stats, windows=list(windows.values()), audited=True,
                interpretation='Observational continuation of one shared world. No causal learning, language, or society claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ticks', type=int, default=4096)
    parser.add_argument('--wall-seconds', type=float, default=120)
    parser.add_argument('--continue-live', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'runs') or output == ROOT / 'runs':
        parser.error('Use a new directory beneath runs')
    if not 0 < args.ticks <= 50000 or not 0 < args.wall_seconds <= 600:
        parser.error('Limits: 1..50000 ticks and 0..600 seconds')
    runs = ROOT / 'runs'
    with checkpoint_lease(runs):
        if (runs / 'server-process.json').exists():
            raise RuntimeError('Stop the launcher-owned server before observing its checkpoint')
        source = runs / 'population.pt'
        source_hash = sha(source)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, output / 'start.pt')
        manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), source=str(source),
                        source_sha256=source_hash, requested_ticks=args.ticks, wall_seconds=args.wall_seconds,
                        continue_live=args.continue_live, python=platform.python_version(), torch=torch.__version__,
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        source_hashes={f: sha(ROOT/f) for f in FILES},
                        changes='No policy, reward, resource, architecture, or player interventions. Learning continues.')
        write(output / 'manifest.json', manifest)
        with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for f in FILES:
                archive.write(ROOT/f, f)
        population = Population.load(output / 'start.pt', require_all=True)
        summary = observe(population, output, args.ticks, args.wall_seconds)
        population.save(output / 'end.pt')
        if any(sha(ROOT/f) != h for f, h in manifest['source_hashes'].items()):
            raise RuntimeError('Experiment source changed during the run; live save was not replaced')
        if sha(source) != source_hash:
            raise RuntimeError('Live save changed during the run; it was not replaced')
        if args.continue_live:
            temporary = runs / 'watch-continuation.tmp'
            shutil.copy2(output / 'end.pt', temporary)
            temporary.replace(source)
            write(runs / 'checkpoint-status.json', dict(saved_at=datetime.now(timezone.utc).isoformat(),
                  tick=population.world.tick, checkpoint='population.pt', bytes=source.stat().st_size,
                  residents=list(population.brains), backend='recurrent-ppo', scripted_residents=0))
        receipt = dict(end_sha256=sha(output / 'end.pt'), trace_sha256=sha(output / 'trace.jsonl.gz'),
                       source_hashes_valid=True, continued_live=args.continue_live, **summary)
        write(output / 'complete.json', receipt)
        write(runs / 'latest-population-watch.json', dict(output=str(output), tick=population.world.tick,
              ticks=summary['ticks'], seconds=summary['seconds'], continued_live=args.continue_live))
        print(json.dumps(dict(output=str(output), ticks=summary['ticks'], end_tick=population.world.tick,
                              seconds=round(summary['seconds'], 2), continued_live=args.continue_live)), flush=True)


if __name__ == '__main__':
    main()
