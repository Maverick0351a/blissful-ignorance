import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from agents.network import ACTIONS
from agents.retained_replay import Brain, Config
from agents.sequence_replay import Brain as BaseBrain, Config as BaseConfig
from agents.sequence_ppo import Population
from experiments.scarcity_routes import configure
from tests.test_sequence_replay import equal


def population(retention=False, exploration=False, baseline=False):
    p = Population(9137)
    args = dict(hidden_size=16, unroll=4, burn_in=2, capacity=8,
                batch_size=2, warmup=2, target_interval=2)
    c = (BaseConfig(**args) if baseline else Config(**args, reserved_chunks=2,
         retain_consequences=retention, balanced_exploration=exploration))
    cls = BaseBrain if baseline else Brain
    p.brains = {rid: cls(9137+i, c) for i, rid in enumerate(p.brains)}
    configure(p, 9137, 'feeding')
    return p


class RetentionTests(unittest.TestCase):
    def test_no_intervention_matches_original_learner(self):
        p, q = population(baseline=True), population()
        for _ in range(257):
            self.assertEqual(p.step(), q.step())
        self.assertEqual(p.world.to_dict(), q.world.to_dict())
        for rid in p.brains:
            a, b = p.brains[rid].state(), q.brains[rid].state()
            for key in ('model', 'target', 'optimizer', 'rng', 'replay_rng',
                        'priorities', 'decisions', 'transitions', 'updates', 'hidden'):
                equal(self, a[key], b[key])

    def test_reserve_survives_fifo_eviction_without_extra_capacity_or_duplicates(self):
        b = population(retention=True).brains['r0']
        for serial in range(1, 41):
            b.store_chunk(dict(rows=[dict(chosen=4, nutrition=serial == 1)], burn_in=0))
            b.priorities[-1] = serial
            self.assertLessEqual(len(b.replay), 8)
            self.assertEqual([c['serial'] for c in b.replay], b.priorities)
            self.assertEqual(len({c['serial'] for c in b.replay}), len(b.replay))
        self.assertEqual([c['serial'] for c in b.replay], [1, *range(34, 41)])
        self.assertEqual(b.inspector()['retained_nutrition'], 1)
        for _ in range(50):
            b.store_chunk(dict(rows=[dict(chosen=4, injury=True)], burn_in=0))
        self.assertEqual(sum(c['protected'] for c in b.replay), 2)
        self.assertEqual(b.experiment['salient_chunks_seen'], 51)
        self.assertEqual(len(b.replay), 8)

    def test_exploration_preserves_all_options_and_balances_verbs_only_in_training(self):
        b = population(exploration=True).brains['r0']
        allowed = [i for i, a in enumerate(ACTIONS) if a['verb'] in ('tone', 'move', 'rest', 'eat')]
        draws = Counter(b.explore(allowed) for _ in range(20000))
        self.assertEqual(set(draws), set(allowed))
        verbs = Counter()
        for i, n in draws.items():
            verbs[ACTIONS[i]['verb']] += n
        for count in verbs.values():
            self.assertAlmostEqual(count/20000, .25, delta=.015)
        b.freeze()
        flat = population().brains['r0']
        flat.freeze()
        flat.rng.setstate(b.rng.getstate())
        self.assertEqual([b.explore(allowed) for _ in range(100)],
                         [flat.explore(allowed) for _ in range(100)])

    def test_exact_resume_all_four_conditions_with_pending_rewards_and_evictions(self):
        for retention in (False, True):
            for exploration in (False, True):
                with self.subTest(retention=retention, exploration=exploration):
                    p = population(retention, exploration)
                    for _ in range(173):
                        p.step()
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder)/'checkpoint.pt'
                        p.save(path)
                        q = Population.load(path)
                        for _ in range(63):
                            self.assertEqual(p.step(), q.step())
                        self.assertEqual(p.world.to_dict(), q.world.to_dict())
                        self.assertEqual(p.metrics, q.metrics)
                        for rid in p.brains:
                            equal(self, p.brains[rid].state(), q.brains[rid].state())

    def test_private_learning_and_frozen_retention_state(self):
        p = population(True, True)
        b, other = p.brains.values()
        untouched = copy.deepcopy(other.state())
        observation = p.world.observe('r0')
        for _ in range(41):
            b.decide(observation)
            b.add_reward({'food': 1.})
        self.assertGreater(b.experiment['sampled_nutrition'], 0)
        equal(self, untouched, other.state())
        b.freeze()
        frozen = copy.deepcopy(b.state())
        for _ in range(20):
            b.decide(observation)
            b.add_reward({'injury': -1.})
        for key in ('model', 'target', 'optimizer', 'replay', 'priorities', 'replay_rng',
                    'retention_rng', 'experiment', 'updates'):
            equal(self, frozen[key], b.state()[key])


if __name__ == '__main__':
    unittest.main()
