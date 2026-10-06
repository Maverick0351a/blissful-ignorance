import copy
from io import BytesIO
import unittest
from unittest.mock import patch

import torch

from agents.laya_resident import Config, LayaBrain, compact_observation
from agents.network import ACTIONS
from sim.world import World


class FakeTok:
    mask_token = '[MASK]'

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


class FakeEngine:
    tok = FakeTok()

    def __init__(self):
        self.offered = []

    def system_one(self, state, questions):
        self.offered.append(copy.deepcopy((state, questions)))
        criteria = questions['action']['criteria']
        return {'answers': {'action': {'choice': list(criteria)[0]}}}


class LayaResidentTests(unittest.TestCase):
    def setUp(self):
        self.observation = World(731).observe('r0')

    def test_all_actions_including_tones_reach_model(self):
        engine = FakeEngine(); brain = LayaBrain(9, engine=engine)
        with patch('agents.laya_resident.action_mask', return_value=[True]*len(ACTIONS)):
            command = brain.decide(self.observation)
        offered = [value for _, qs in engine.offered for value in qs['action']['criteria'].values()]
        for action in ACTIONS:
            self.assertIn(' '.join(str(v) for v in action.values()), offered)
        self.assertIn(command, ACTIONS)
        self.assertGreater(brain.diagnostics['calls'], 1)
        self.assertLessEqual(brain.diagnostics['max_input_tokens'], 512)
        self.assertEqual(brain.pending['mask'].shape, (1, len(ACTIONS)))

    def test_restore_preserves_order_and_real_history(self):
        engine = FakeEngine(); brain = LayaBrain(3, engine=engine)
        brain.decide(self.observation); brain.add_reward({'food': .4})
        changed = copy.deepcopy(self.observation); changed['needs'][0] += 10
        state = copy.deepcopy(brain.state()); stream = BytesIO()
        torch.save(state, stream); stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        resumed = LayaBrain(3, engine=FakeEngine()); resumed.restore(loaded)
        self.assertEqual(brain.decide(changed), resumed.decide(changed))
        self.assertEqual(brain.journal, resumed.journal)
        self.assertEqual(brain.journal[-1]['observed']['food'], 10)
        self.assertEqual(brain.journal[-1]['reward'], {'food': .4})
        self.assertEqual(brain.observed_transitions, 1)
        self.assertEqual(brain.updates, 0); self.assertFalse(brain.training)
        self.assertNotIn('_engine', loaded)

    def test_local_projection_ignores_privileged_fields(self):
        observation = copy.deepcopy(self.observation)
        expected = compact_observation(observation)
        observation.update(x=1000, y=1000, god=True, global_resources={'secret': 20})
        observation['tiles'].append(dict(dx=2, dy=2, terrain=-1,
                                        resource={'kind': 'secret', 'amount': 20}))
        self.assertEqual(expected, compact_observation(observation))

    def test_no_fallback_on_model_failure(self):
        engine = FakeEngine(); brain = LayaBrain(3, engine=engine)
        with patch.object(engine, 'system_one', side_effect=RuntimeError('NPU unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'NPU unavailable'):
                brain.decide(self.observation)
        self.assertEqual(brain.decisions, 0)
        self.assertIsNone(brain.pending)

    def test_unconscious_only_allowed_rest_still_uses_model(self):
        observation = copy.deepcopy(self.observation)
        observation['body']['unconscious'] = True
        engine = FakeEngine(); brain = LayaBrain(3, engine=engine)
        self.assertEqual(brain.decide(observation), {'verb': 'rest'})
        self.assertEqual(len(engine.offered), 1)
        self.assertEqual(int(brain.pending['mask'].sum()), 1)

    def test_reject_nonreducing_tournament(self):
        with self.assertRaises(ValueError):
            Config(group_size=1)

    def test_conservative_head_budget_accepts_installed_256_token_configuration(self):
        engine=FakeEngine();engine.cfg={'head_max_len':256}
        brain=LayaBrain(3,engine=engine)
        self.assertIn(brain.decide(self.observation),ACTIONS)
        engine.cfg={'head_max_len':128}
        with self.assertRaisesRegex(ValueError,'smaller than'):
            brain.decide(self.observation)


if __name__ == '__main__':
    unittest.main()
