"""Summarize audited nearby-food evidence and export recorded local replays."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.category_replay import export
from experiments.near_food_trial import FIXTURES, ARMS, WINDOWS, quick_success


def report(folder):
    assert json.loads((folder/'audit.json').read_text())['passed']
    rows = json.loads((folder/'results.json').read_text())
    training = json.loads((folder/'training.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    result = dict(fixtures={}, training={}, gates={k:v for k,v in complete['result'].items() if k!='seeds'},
                  seconds=complete['seconds'], memory=complete['memory'])
    for fixture in FIXTURES:
        arms = {}
        for arm in ARMS:
            selected = [r for r in rows if r['metadata']['fixture']==fixture and r['metadata']['arm']==arm]
            lives = [m for r in selected for m in r['metrics'].values()]
            n = len(lives); actions = Counter(); rewards = Counter()
            for metric in lives:
                actions.update(metric['successful_actions']); rewards.update(metric['reward_totals'])
            arms[arm] = dict(lives=n, successes=sum(quick_success(m,fixture) for m in lives),
                fed_before_zero=sum(m['timely_acquisition'] for m in lives) if fixture=='adjacent' else None,
                gathered_lives=sum(m['first_gather_food_tick'] is not None for m in lives),
                ate_lives=sum(m['first_meal_tick'] is not None for m in lives),
                ordinary_meals=sum(m['safe_eaten'] for m in lives), amber_meals=sum(m['amber_eaten'] for m in lives),
                nutrition=sum(m['nutrition'] for m in lives), zero_food_ticks=sum(m['zero_food_ticks'] for m in lives),
                resident_ticks=sum(r['ticks']*len(r['metrics']) for r in selected),
                unconscious_ticks=sum(m['unconscious_ticks'] for m in lives), damage=sum(m['damage'] for m in lives),
                invalid=sum(m['invalid'] for m in lives), mean_unique_tiles=sum(m['unique_tiles'] for m in lives)/n,
                tones=sum(m['conscious_tones'] for m in lives), decisions=sum(m['conscious_decisions'] for m in lives),
                actions=dict(actions), rewards=dict(rewards),
                starting_probabilities={verb:sum(g['fixtures'][fixture]['arms'][arm]['starting_probabilities'][verb]*
                                                   g['fixtures'][fixture]['arms'][arm]['lives'] for g in complete['result']['seeds'])/n
                                        for verb in ('gather','eat','move','tone','rest','make_seeds')})
        result['fixtures'][fixture] = arms
    for kind in ('flat','category'):
        selected = [r for r in training if r['metadata']['arm']==kind+'-training']
        result['training'][kind] = dict(episodes=len(selected), seconds=sum(r['seconds'] for r in selected),
            decisions=sum(t['decisions'] for r in selected for t in r['timings'].values()),
            updates=sum(t['updates'] for r in selected for t in r['timings'].values()),
            learning_seconds=sum(t['learning_seconds'] for r in selected for t in r['timings'].values()),
            ordinary_meals=sum(m['safe_eaten'] for r in selected for m in r['metrics'].values()),
            last_episode_rewards={str(r['metadata']['training_seed']):{rid:m['reward_totals'] for rid,m in r['metrics'].items()}
                                  for r in selected if r['metadata']['case']==max(x['metadata']['case'] for x in selected)})
    (folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for fixture in FIXTURES:
        export(folder, fixture=fixture, success_window=WINDOWS[fixture],
               title='Godhood Trials · '+('nearby food: gather then eat' if fixture=='adjacent' else 'carried fruit: consumption only'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local runs directory')
    print(json.dumps(report(folder)),flush=True)
