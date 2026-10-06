"""Descriptive summaries of audited GT-01 evidence; no additional training."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.affordances import action_mask
from agents.network import ACTIONS


def analyze(folder):
    assert json.loads((folder/'audit.json').read_text())['passed']
    protocol = json.loads((folder/'preregistration.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    rows = json.loads((folder/'results.json').read_text())
    training = json.loads((folder/'training.json').read_text())
    aggregate = {}
    for arm in protocol['arms']:
        chosen = [r for r in rows if r['metadata']['arm'] == arm]
        lives = [m for r in chosen for m in r['metrics'].values()]
        ticks = sum(r['ticks']*len(r['metrics']) for r in chosen)
        actions = Counter(); successful = Counter(); rewards = Counter()
        for m in lives:
            actions.update(m['actions']); successful.update(m['successful_actions']); rewards.update(m['reward_totals'])
        aggregate[arm] = dict(lives=len(lives), successes=sum(m['timely_acquisition'] for m in lives),
            resident_ticks=ticks, safe_eaten=sum(m['safe_eaten'] for m in lives),
            amber_eaten=sum(m['amber_eaten'] for m in lives), nutrition=sum(m['nutrition'] for m in lives),
            zero_food_ticks=sum(m['zero_food_ticks'] for m in lives),
            unconscious_ticks=sum(m['unconscious_ticks'] for m in lives), damage=sum(m['damage'] for m in lives),
            invalid=sum(m['invalid'] for m in lives), mean_unique_tiles=sum(m['unique_tiles'] for m in lives)/len(lives),
            conscious_decisions=sum(m['conscious_decisions'] for m in lives),
            conscious_tones=sum(m['conscious_tones'] for m in lives), actions=dict(actions),
            successful_actions=dict(successful), reward_totals=dict(rewards),
            lives_saw_food=sum(m['first_food_seen_tick'] is not None for m in lives),
            lives_gathered_food=sum(m['first_gather_food_tick'] is not None for m in lives),
            lives_ate=sum(m['first_meal_tick'] is not None for m in lives))
    # This diagnostic is post hoc, descriptive, and on each policy's OWN visited
    # states. Different opportunity counts cannot establish a causal mechanism.
    opportunities = defaultdict(Counter)
    gather = ACTIONS.index({'verb': 'gather'}); eat = ACTIONS.index({'verb': 'eat'})
    with gzip.open(folder/'trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['type'] == 'start':meta = row['metadata']
            elif row['type'] == 'step' and meta['stage'] == 'evaluation' and 'observations' in row:
                for rid, o in row['observations'].items():
                    count = opportunities[meta['arm']]; mask = action_mask(o, ACTIONS)
                    action = row['commands'][rid]
                    if mask[gather]:
                        count['gather_opportunities'] += 1
                        count['gather_selected'] += action['verb'] == 'gather'
                    if mask[eat]:
                        count['ordinary_eat_opportunities'] += 1
                        count['ordinary_eat_selected'] += action == {'verb': 'eat'}
                    if o['inventory']['food'] > 0:count['decisions_carrying_food'] += 1
    timing = {}
    for kind in ('flat', 'category'):
        selected = [r for r in training if r['metadata']['arm'] == kind+'-training']
        timing[kind] = dict(episodes=len(selected), seconds=sum(r['seconds'] for r in selected),
            updates=sum(t['updates'] for r in selected for t in r['timings'].values()),
            decisions=sum(t['decisions'] for r in selected for t in r['timings'].values()),
            learning_seconds=sum(t['learning_seconds'] for r in selected for t in r['timings'].values()),
            meals=sum(m['safe_eaten'] for r in selected for m in r['metrics'].values()))
    return dict(primary=complete['result'], aggregate=aggregate, training=timing,
                post_hoc_opportunities={k: dict(v) for k, v in opportunities.items()},
                opportunity_caveat='Own-policy visited states differ; descriptive, not a matched causal test.',
                seconds=complete['seconds'], memory=complete['memory'],
                evidence_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
                parameters=protocol['parameters'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local runs directory')
    result = analyze(folder)
    (folder/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
