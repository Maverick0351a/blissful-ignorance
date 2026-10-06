"""Post-run checkpoint provenance, independence and experience-retention audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.network import ACTIONS
from agents.sequence_ppo import Population
from experiments.replay_comparison import evaluation_population, learning_digest, make_population


def audit(out):
    manifest = json.loads((out/'preregistration.json').read_text())
    results = json.loads((out/'results.json').read_text())
    checkpoints = []
    for seed in manifest['seeds']:
        for family in manifest['families']:
            path = out/f'{seed}-{family}-trained.pt'
            p = Population.load(path)
            initial = make_population(seed, family)
            curve = json.loads((out/f'{seed}-{family}-learning-curve.json').read_text())
            brains = list(p.brains.values())
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(brains[0].model.parameters(), brains[1].model.parameters()))
            entry = dict(seed=seed, family=family, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), residents={})
            for rid, brain in p.brains.items():
                assert brain.decisions == manifest['decisions_per_agent']
                assert all(bool(torch.isfinite(t).all()) for t in brain.model.state_dict().values())
                learned = any(not torch.equal(t, initial.brains[rid].model.state_dict()[name])
                              for name, t in brain.model.state_dict().items())
                assert learned
                data = dict(decisions=brain.decisions, updates=brain.updates, changed_from_initial=learned,
                            training_safe_meals=sum(e['metrics'][rid]['safe_eaten'] for e in curve))
                if hasattr(brain, 'replay'):
                    assert brain.context == [] and brain.segment == [] and brain.pending is None
                    assert all(not any(r['terminal'] for r in c['rows'][:-1]) for c in brain.replay)
                    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(brain.model.parameters(), brain.target.parameters()))
                    tail = [r for c in brain.replay for r in c['rows'][c['burn_in']:]]
                    data.update(retained_transitions=len(tail), positive_rewards=sum(r['reward'] > 0 for r in tail),
                                eating_choices=sum(ACTIONS[r['chosen']]['verb'] == 'eat' for r in tail))
                entry['residents'][rid] = data
            for row in results:
                if row['seed'] != seed or row['family'] != family:
                    continue
                condition = row['condition'].split('-')[-1]
                source = p if condition == 'trained' else initial
                evaluation, _ = evaluation_population(row['evaluation_seed'], family, source, condition)
                expected = {rid: learning_digest(b) for rid, b in evaluation.brains.items()}
                assert expected == row['frozen_before'] == row['frozen_after']
            checkpoints.append(entry)
    replay_residents = [b for c in checkpoints if c['family'] != 'ppo' for b in c['residents'].values()]
    report = dict(post_hoc=True, audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        finite_distinct_models=True, trained_weights_match_evaluations=True,
        changed_from_initial=True, replay_episode_boundaries=True,
        previously_fed_without_retained_eating=sum(b['training_safe_meals'] > 0 and b['eating_choices'] == 0 for b in replay_residents),
        replay_residents=len(replay_residents), checkpoints=checkpoints)
    (out/'checkpoint-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    print(json.dumps({k: v for k, v in report.items() if k != 'checkpoints'}, indent=2))
