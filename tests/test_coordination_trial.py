import copy
import io
from pathlib import Path
import random
import tempfile
import time
import unittest

from agents.network import ACTIONS
from agents.sequence_ppo import Config
from experiments.coordination_trial import (RESIDENTS, channel_view, food_visible, make_world,
    permutation, population_for, run_episode, summarize, weight_hashes)
from sim.world import DIRECTIONS


class CoordinationTests(unittest.TestCase):
    def test_equal_bodies_private_information_reachable_food_and_roles(self):
        for held_out in (False, True):
            roles = []
            for case in range(8):
                w, meta = make_world(77100 + case, case, held_out)
                first, other = meta['initially_informed'], meta['initially_uninformed']
                roles.append(first)
                self.assertTrue(food_visible(w.observe(first)))
                self.assertFalse(food_visible(w.observe(other)))
                self.assertEqual(w.residents[first].food, w.residents[other].food)
                self.assertEqual(w.residents[first].inventory, w.residents[other].inventory)
                self.assertFalse({'leader', 'role', 'food_at', 'case', 'position'} & set(w.observe(other)))
                # Test geometry reachability only; no path is supplied to a brain.
                actor = w.residents[other]; seen = {(actor.x, actor.y)}; pending = list(seen)
                while pending:
                    x, y = pending.pop()
                    for dx, dy in DIRECTIONS.values():
                        nxt = (x + dx, y + dy)
                        if nxt not in seen and w.walkable(*nxt):
                            seen.add(nxt); pending.append(nxt)
                self.assertIn(tuple(meta['food_at']), seen)
            self.assertEqual(roles.count('r0'), 4)
            self.assertEqual(roles.count('r1'), 4)

    def test_controls_remove_both_tone_paths_and_preserve_other_senses(self):
        w, meta = make_world(77881, 0)
        sender = meta['initially_informed']; receiver = meta['initially_uninformed']
        w.step({sender: {'verb': 'tone', 'tone': '7'}, receiver: {'verb': 'wait'}}, scripted=False)
        original = w.observe(receiver); pristine = copy.deepcopy(original)
        self.assertTrue(original['auditory_memory'])
        self.assertTrue(any(s['kind'] == 'tone_7' for s in original['hearing']))
        mapping = permutation(55)
        muted = channel_view(original, 'muted', mapping)
        shuffled = channel_view(original, 'shuffled', mapping)
        self.assertEqual(muted['auditory_memory'], [])
        self.assertFalse(any(s['kind'].startswith('tone_') for s in muted['hearing']))
        self.assertEqual(shuffled['auditory_memory'][0]['tone'], mapping['7'])
        for field in ('bearing', 'age', 'strength'):
            self.assertEqual(shuffled['auditory_memory'][0][field], original['auditory_memory'][0][field])
        for field in set(original) - {'hearing', 'auditory_memory'}:
            self.assertEqual(muted[field], original[field]); self.assertEqual(shuffled[field], original[field])
        self.assertEqual(original, pristine)
        self.assertEqual(w.observe(receiver), pristine)
        self.assertEqual(channel_view(original, 'shuffled', mapping), shuffled)

    def test_derangement_and_all_ten_unassigned_symbols(self):
        for seed in range(20):
            mapping = permutation(seed)
            self.assertEqual(set(mapping), set(mapping.values()))
            self.assertEqual(len(mapping), 10)
            self.assertTrue(all(a != b for a, b in mapping.items()))
        self.assertEqual(sum(a['verb'] == 'tone' for a in ACTIONS), 10)
        self.assertFalse(any(a['verb'] in ('lead', 'follow', 'obey', 'route') for a in ACTIONS))

    def test_separate_optimizers_and_unchanged_individual_reward(self):
        w, _ = make_world(78114, 0)
        p = population_for(61, w, Config(rollout=8, epochs=1))
        other = weight_hashes(p.brains['r1'])
        self.assertIsNot(p.brains['r0'].optimizer, p.brains['r1'].optimizer)
        p.step()
        self.assertEqual(other, weight_hashes(p.brains['r1']))
        self.assertEqual(set(p.brains['r0'].reward_parts), {'food', 'hunger', 'injury'})
        self.assertEqual(p.brains['r0'].config.curiosity_coefficient, 0)
        self.assertFalse(p.neighbors_scripted)
        self.assertEqual(set(w.residents), set(RESIDENTS))

    def test_identical_frozen_replay_and_no_weight_updates(self):
        rows = []
        with tempfile.TemporaryDirectory() as temporary:
            for _ in range(2):
                w, meta = make_world(78115, 2, held_out=True)
                p = population_for(62, w, Config(rollout=8, epochs=1))
                for i, b in enumerate(p.brains.values()):
                    b.freeze(); b.rng = random.Random(555 + i); b.condition = 'shuffled'
                trace = io.StringIO()
                result = run_episode(p, meta, 'test', 32, trace, Path(temporary), time.perf_counter() + 20)
                self.assertEqual(result['weights_before'], result['weights_after'])
                rows.append(trace.getvalue())
        self.assertEqual(rows[0], rows[1])

    def test_gate_rejects_activity_without_channel_benefit(self):
        rows = []
        for arm in ('intact', 'muted', 'shuffled', 'initial'):
            for case in range(8):
                m = dict(safe_eaten=1, nutrition=25., zero_food_ticks=0, unconscious_ticks=0, heard_hidden_decisions=20)
                rows.append(dict(metadata=dict(training_seed=1, arm=arm, initially_uninformed='r1'),
                                 metrics={'r0': dict(m), 'r1': dict(m)}))
        result = summarize(rows, (1,))
        self.assertFalse(result['passed'])
        self.assertFalse(result['seeds'][0]['communication_gate'])
        self.assertFalse(result['seeds'][0]['acquisition_gate'])


if __name__ == '__main__':
    unittest.main()
