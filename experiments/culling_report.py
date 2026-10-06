"""Summarize audited restart evidence and export a local recorded comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.category_replay import export


def report(folder):
    assert json.loads((folder/'audit.json').read_text())['passed']
    rows=json.loads((folder/'results.json').read_text());training=json.loads((folder/'training.json').read_text())
    complete=json.loads((folder/'complete.json').read_text());decision=json.loads((folder/'retirement.json').read_text())
    result=dict(result=complete['result'],retirements=decision,seconds=complete['seconds'],memory=complete['memory'],activity={},training={})
    for arm in complete['result']['arms']:
        selected=[r for r in rows if r['metadata']['arm']==arm]
        metrics=[m for r in selected for m in r['metrics'].values()];actions=Counter();rewards=Counter()
        for m in metrics:actions.update(m['successful_actions']);rewards.update(m['reward_totals'])
        result['activity'][arm]=dict(actions=dict(actions),rewards=dict(rewards),
            mean_unique_tiles=sum(m['unique_tiles'] for m in metrics)/len(metrics),
            decisions=sum(m['conscious_decisions'] for m in metrics),tones=sum(m['conscious_tones'] for m in metrics),
            invalid=sum(m['invalid'] for m in metrics),damage=sum(m['damage'] for m in metrics),
            nutrition=sum(m['nutrition'] for m in metrics),resident_ticks=sum(r['ticks']*2 for r in selected),
            identities={cid:identity for r in selected for rid,cid in r['metadata']['candidates'].items()
                        for identity in [r['metadata']['identities'][rid]]})
        selected=[r for r in training if r['metadata']['arm']==arm]
        result['training'][arm]=dict(seconds=sum(r['seconds'] for r in selected),
            decisions=sum(t['decisions'] for r in selected for t in r['timings'].values()),
            updates=sum(t['updates'] for r in selected for t in r['timings'].values()),
            learning_seconds=sum(t['learning_seconds'] for r in selected for t in r['timings'].values()))
    (folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    replay=export(folder,fixture='adjacent',success_window=64,title='Godhood Trials · archived restart comparison')
    html=replay.read_text(encoding='utf-8').replace('six conditions','three conditions')
    html=html.replace('These are fresh experimental residents.',
        'These are experimental copies in slots c0/r0 and c1/r1. Replacement identities and the full six-slot population are recorded in the report.')
    replay.write_text(html,encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('Use a local run directory')
    print(json.dumps(report(folder)),flush=True)
