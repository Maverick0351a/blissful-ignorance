import copy
from pathlib import Path
import tempfile
import unittest

import torch
from agents.affordances import action_mask
from agents.category_ppo import Brain, CATEGORIES, ACTION_CATEGORY, joint_distribution
from agents.network import ACTIONS, encode_observation
from agents.sequence_ppo import Brain as FlatBrain, Config, Population
from experiments.development import configure


class CategoryTests(unittest.TestCase):
    def test_normalized_joint_probabilities_and_all_tones(self):
        allowed = [a['verb'] in ('move', 'rest', 'tone') for a in ACTIONS]
        mask = torch.tensor([allowed])
        logits = torch.zeros(1, len(CATEGORIES) + len(ACTIONS))
        d = joint_distribution(logits, mask)
        self.assertAlmostEqual(float(d.probs.sum()), 1., places=6)
        for i, action in enumerate(ACTIONS):
            expected = {'move': 1/12, 'rest': 1/3, 'tone': 1/30}.get(action['verb'], 0.)
            self.assertAlmostEqual(float(d.probs[0, i]), expected, places=6)
        self.assertEqual(sum(a['verb'] == 'tone' for a in ACTIONS), 10)
        # Joint entropy is H(category) + E[H(argument | category)].
        expected_entropy = torch.log(torch.tensor(3.)) + (torch.log(torch.tensor(4.)) + torch.log(torch.tensor(10.))) / 3
        self.assertAlmostEqual(float(d.entropy()), float(expected_entropy), places=6)

    def test_one_legal_action_empty_categories_and_gradients(self):
        logits = torch.randn(2, len(CATEGORIES) + len(ACTIONS), requires_grad=True)
        mask = torch.zeros(2, len(ACTIONS), dtype=torch.bool)
        rest = ACTIONS.index({'verb': 'rest'})
        mask[0, rest] = True
        mask[1] = torch.tensor([a['verb'] in ('tone', 'move', 'rest') for a in ACTIONS])
        d = joint_distribution(logits, mask)
        self.assertEqual(float(d.probs[0, rest]), 1.)
        self.assertTrue(torch.equal(d.probs[~mask], torch.zeros_like(d.probs[~mask])))
        loss = -d.log_prob(torch.tensor([rest, 0])).mean() - .01*d.entropy().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[1, :len(CATEGORIES)].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[1, len(CATEGORIES):].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.)
        with self.assertRaises(ValueError):joint_distribution(logits, torch.zeros_like(mask))

    def test_matches_explicit_two_stage_probability_and_masked_gradients(self):
        torch.manual_seed(917)
        raw = torch.randn(3, len(CATEGORIES) + len(ACTIONS), requires_grad=True)
        mask = torch.rand(3, len(ACTIONS)) > .5
        actual = joint_distribution(raw, mask)
        for row in range(3):
            cats = sorted({ACTION_CATEGORY[i] for i in range(len(ACTIONS)) if mask[row, i]})
            pcat = raw[row, cats].softmax(0)
            for k, category in enumerate(cats):
                indices = [i for i, c in enumerate(ACTION_CATEGORY) if c == category and mask[row, i]]
                conditional = raw[row, [len(CATEGORIES) + i for i in indices]].softmax(0)
                self.assertTrue(torch.allclose(actual.probs[row, indices], pcat[k]*conditional, atol=1e-7))
        actual.entropy().sum().backward()
        self.assertEqual(float(raw.grad[:, len(CATEGORIES):][~mask].abs().sum()), 0.)

    def test_matched_core_predictor_and_unchanged_flat_head(self):
        a, b = FlatBrain(612), Brain(612)
        for name in ('patch_encoder', 'feature_encoder', 'recurrent', 'value'):
            for key, value in getattr(a.model, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(b.model, name).state_dict()[key]))
        for key, value in a.predictor.state_dict().items():
            self.assertTrue(torch.equal(value, b.predictor.state_dict()[key]))
        self.assertTrue(torch.equal(a.model.actor.weight, b.model.actor.arguments.weight))
        p = Population(8)
        observation = p.world.observe('r0')
        patch, features = encode_observation(observation)
        logits, _, _ = a.model(patch, features)
        mask = torch.tensor([action_mask(observation, ACTIONS)])
        old = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
        self.assertTrue(torch.equal(a.distribution(logits, mask).logits, old.logits))
        self.assertNotEqual(a.model.recurrent.weight_hh.data_ptr(), b.model.recurrent.weight_hh.data_ptr())

    def test_exact_resume_across_update_and_independence(self):
        config = Config(rollout=8, epochs=2)
        p = configure(Population(713, config), 713)
        p.brains = {rid: Brain(713+i, config) for i, rid in enumerate(p.brains)}
        for _ in range(21):p.step()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'experimental.pt'
            p.save(checkpoint)
            data = torch.load(checkpoint, weights_only=True)
            from sim.world import World
            q = Population(713, config, world=World.from_dict(data['world']))
            q.neighbors_scripted = p.neighbors_scripted
            q.brains = {rid: Brain(saved['seed'], config) for rid, saved in data['brains'].items()}
            for rid, saved in data['brains'].items():q.brains[rid].restore(saved)
            q.metrics = data['metrics']
            # The production loader must not silently promote this experiment.
            with self.assertRaises(ValueError):Population.load(checkpoint)
            for _ in range(25):self.assertEqual(p.step(), q.step())
            self.assertEqual(p.world.to_dict(), q.world.to_dict())
            self.assertEqual(p.metrics, q.metrics)
            for rid in p.brains:
                self.assertGreater(p.brains[rid].updates, 0)
                self.assertEqual(p.brains[rid].diagnostics, q.brains[rid].diagnostics)
                for module in ('model', 'predictor'):
                    for key, value in getattr(p.brains[rid], module).state_dict().items():
                        self.assertTrue(torch.equal(value, getattr(q.brains[rid], module).state_dict()[key]))
            self.assertIsNot(p.brains['r0'].optimizer, p.brains['r1'].optimizer)
            self.assertNotEqual(p.brains['r0'].model.actor.category.weight.data_ptr(),
                                p.brains['r1'].model.actor.category.weight.data_ptr())

    def test_checkpoint_schema_rejects_reordered_actions(self):
        brain = Brain(9)
        saved = copy.deepcopy(brain.state())
        saved['category_schema']['actions'].reverse()
        with self.assertRaises(ValueError):brain.restore(saved)
        with self.assertRaises(ValueError):FlatBrain(9).restore(brain.state())


if __name__ == '__main__':unittest.main()
