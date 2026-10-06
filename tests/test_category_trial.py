import io
from pathlib import Path
import tempfile
import time
import unittest

from agents.network import ACTIONS
from agents.sequence_ppo import Config
from experiments.category_trial import make_world, population_for, run_episode, summarize, ARMS
from sim.world import DIRECTIONS


class CategoryTrialTests(unittest.TestCase):
    def test_empty_packs_reachable_varied_food_and_private_information(self):
        locations = set()
        for case in range(12):
            w, meta = make_world(71500+case, case)
            for rid, actor in w.residents.items():
                self.assertFalse(any(actor.inventory.values()))
                self.assertEqual(actor.food, 4.)
                target = tuple(meta['locations'][rid]['food_at']); locations.add(target)
                seen = {(actor.x, actor.y)}; pending = list(seen)
                while pending:
                    x, y = pending.pop()
                    for dx, dy in DIRECTIONS.values():
                        nxt = (x+dx, y+dy)
                        if nxt not in seen and w.walkable(*nxt):seen.add(nxt); pending.append(nxt)
                self.assertIn(target, seen)
                other = w.residents['r1' if rid == 'r0' else 'r0']
                self.assertNotIn((other.x, other.y), seen)
                observation = w.observe(rid)
                self.assertFalse({'layout', 'food_at', 'world_seed', 'role'} & set(observation))
                self.assertEqual(meta['locations'][rid]['food_visible_at_start'], meta['layout'] == 'open')
        self.assertGreater(len(locations), 12)

    def test_random_controls_keep_ten_tones_and_ignore_logits(self):
        import torch
        from agents.affordances import action_mask
        for kind in ('flat', 'category'):
            w, _ = make_world(41, 0); p = population_for(41, kind, w, Config())
            b = p.brains['r0']; b.freeze(); b.random_policy = True
            from agents.network import encode_observation
            o = w.observe('r0'); x, f = encode_observation(o)
            logits, _, _ = b.model(x, f); mask = torch.tensor([action_mask(o, ACTIONS)])
            original = b.distribution(logits, mask).probs
            self.assertTrue(torch.equal(original, b.distribution(logits*100+15, mask).probs))
            self.assertEqual(sum(float(original[0, i]) > 0 for i, a in enumerate(ACTIONS) if a['verb'] == 'tone'), 10)

    def test_measured_training_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            w, meta = make_world(731, 0); p = population_for(61, 'category', w, Config(rollout=8, epochs=1))
            result = run_episode(p, meta, 'train', 40, io.StringIO(), Path(temporary), time.perf_counter()+20)
            self.assertTrue(all(b.updates == 2 for b in p.brains.values()))
            self.assertTrue(all(set(m['reward_totals']) == {'food', 'hunger', 'injury'} for m in result['metrics'].values()))
            for b in p.brains.values():b.freeze()
            p.world, meta = make_world(732, 1); p.metrics = {rid: p.new_metrics() for rid in p.brains}
            result = run_episode(p, meta, 'eval', 16, io.StringIO(), Path(temporary), time.perf_counter()+20)
            self.assertEqual(result['weights_before'], result['weights_after'])
            self.assertTrue(all(m['timely_acquisition'] is False for m in result['metrics'].values()))

    def test_random_exploration_gain_cannot_pass_learning_gate(self):
        rows = []
        for seed in (1, 2, 3):
            for arm in ARMS:
                for case in range(2):
                    metric = dict(timely_acquisition=arm.startswith('category'), safe_eaten=1,
                                  amber_eaten=0, zero_food_ticks=0, unconscious_ticks=0)
                    rows.append(dict(metadata=dict(training_seed=seed, arm=arm, layout='open' if case == 0 else 'occluded'),
                                     metrics={'r0': dict(metric), 'r1': dict(metric)}))
        result = summarize(rows, (1, 2, 3))
        self.assertFalse(result['pilot_promising'])
        self.assertFalse(result['milestone_confirmed'])


if __name__ == '__main__':unittest.main()
