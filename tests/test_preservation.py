"""Fault injection in disposable populations; never opens a live save or NPU."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from agents.lifelong import Population as LegacyPopulation
from agents.sequence_ppo import Config, Population
from experiments.add_laya_resident import add_laya
from experiments.living_population import LearningRuntime


class StubTokenizer:
    mask_token = '[MASK]'

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


class StubLayaEngine:
    tok = StubTokenizer()

    def system_one(self, state, questions):
        criteria = questions['action']['criteria']
        return {'answers': {'action': {'choice': next(iter(criteria))}}}


def fresh_population(*, laya=False, legacy=False):
    population = (LegacyPopulation(204) if legacy else
                  Population(204, Config(hidden_size=8, rollout=2, epochs=1)))
    population.world.residents = {rid: body for rid, body in population.world.residents.items()
                                  if rid in ('r0', 'r1', 'player')}
    population.neighbors_scripted = False
    if laya:
        add_laya(population)
        population.brains['laya']._engine = StubLayaEngine()
    return population


def snapshot(population):
    return copy.deepcopy(dict(world=population.world.to_dict(), metrics=population.metrics,
                              brains={rid: brain.state() for rid, brain in population.brains.items()}))


class PreservationTests(unittest.TestCase):
    def assertStateEqual(self, actual, expected, path='population'):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype, path)
            self.assertTrue(torch.equal(actual, expected), path)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys(), path)
            for key in expected:
                self.assertStateEqual(actual[key], expected[key], f'{path}.{key}')
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self.assertStateEqual(actual[index], value, f'{path}[{index}]')
        else:
            self.assertEqual(actual, expected, path)

    def test_incomplete_controlled_saves_fail_before_creating_brains_or_writing(self):
        for legacy, missing in ((False, 'r0'), (False, 'laya'), (True, 'r0')):
            with self.subTest(legacy=legacy, missing=missing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); runs = root/'runs'; runs.mkdir()
                path = runs/'population.pt'
                population = fresh_population(laya=not legacy, legacy=legacy)
                population.save(path)
                data = torch.load(path, weights_only=True)
                del data['brains'][missing]; del data['metrics'][missing]
                torch.save(data, path)
                before = path.read_bytes()
                module = 'agents.lifelong' if legacy else 'agents.sequence_ppo'
                with patch(module+'.Brain') as constructor:
                    with self.assertRaisesRegex(ValueError, 'Incomplete controlled population'):
                        LearningRuntime(root)
                    constructor.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(runs.iterdir()), [path])

    def test_laya_identity_rejects_a_different_backend_even_in_legacy_migration(self):
        for scripted in (False, True):
            with self.subTest(scripted=scripted), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'population.pt'
                population = fresh_population(laya=True)
                population.save(path)
                data = torch.load(path, weights_only=True)
                data['brains']['laya'] = copy.deepcopy(data['brains']['r0'])
                data['neighbors_scripted'] = scripted
                torch.save(data, path)
                with patch('agents.sequence_ppo.Brain') as constructor:
                    with self.assertRaisesRegex(ValueError, 'Laya must retain'):
                        Population.load(path)
                    constructor.assert_not_called()

    def test_rejected_manual_load_does_not_change_the_current_population_or_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); runs = root/'runs'
            fresh_population().save(runs/'population.pt')
            runtime = LearningRuntime(root)
            try:
                runtime._save('world.json')
                path = runs/'manual-population.pt'
                data = torch.load(path, weights_only=True)
                del data['brains']['r0']; del data['metrics']['r0']
                torch.save(data, path)
                before = snapshot(runtime.population)
                files = {file.name: file.read_bytes() for file in runs.iterdir()}
                with self.assertRaisesRegex(ValueError, 'Incomplete controlled population'):
                    runtime.control({'command':'load'})
                self.assertStateEqual(snapshot(runtime.population), before)
                self.assertEqual({file.name: file.read_bytes() for file in runs.iterdir()}, files)
            finally:
                runtime.close()

    def test_migration_and_save_cannot_recreate_a_missing_controlled_brain(self):
        population = fresh_population(laya=True)
        del population.brains['laya']; del population.metrics['laya']
        with self.assertRaisesRegex(ValueError, 'Incomplete controlled population'):
            population.enable_all_residents()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'population.pt'
            with self.assertRaisesRegex(ValueError, 'Laya must retain'):
                population.save(path)
            self.assertFalse(path.exists())
        self.assertNotIn('laya', population.brains)

    def test_scripted_legacy_migration_preserves_existing_brains_and_backup(self):
        for cls in (Population, LegacyPopulation):
            with self.subTest(backend=cls.__module__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); path = root/'runs/population.pt'
                population = cls(52)
                for _ in range(5): population.step()
                before = snapshot(population)
                population.save(path); original_bytes = path.read_bytes()
                runtime = LearningRuntime(root)
                try:
                    self.assertEqual(runtime.world.to_dict(), before['world'])
                    for rid, state in before['brains'].items():
                        self.assertStateEqual(runtime.population.brains[rid].state(), state)
                    self.assertEqual(set(runtime.population.brains), set(runtime.world.residents)-{'player'})
                    self.assertFalse(runtime.population.neighbors_scripted)
                    self.assertEqual((root/'runs/before-all-residents-population.pt').read_bytes(), original_bytes)
                finally:
                    runtime.close()

    def test_complete_mixed_save_loads_without_fresh_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); population = fresh_population(laya=True)
            for _ in range(9): population.step()
            before = snapshot(population)
            population.save(root/'runs/population.pt')
            with patch('agents.laya_resident.LayaBrain._load_engine', return_value=StubLayaEngine()):
                runtime = LearningRuntime(root)
            try:
                self.assertStateEqual(snapshot(runtime.population), before)
                self.assertEqual(runtime.population.brains['laya'].backend, 'laya-npu')
                self.assertTrue(runtime.manual_pause)
                self.assertFalse((root/'runs/before-all-residents-population.pt').exists())
            finally:
                runtime.close()

    def check_decision_rollback(self, ticks, *, optimizer_failure=False, legacy=False):
        population = fresh_population(laya=not legacy, legacy=legacy)
        control = fresh_population(laya=not legacy, legacy=legacy)
        for _ in range(ticks):
            self.assertEqual(population.step(), control.step())
        before = snapshot(population)
        first = population.brains['r0']; last = population.brains['r1']
        module = last.optimizer if optimizer_failure else last.model
        method = 'step' if optimizer_failure else 'forward'
        original = getattr(module, method)
        reached = []

        def fail_after_earlier_decisions(*args, **kwargs):
            reached.append((first.decisions, first.updates))
            if not legacy:
                self.assertEqual(population.brains['laya'].decisions, before['brains']['laya']['decisions']+1)
            if optimizer_failure:
                original(*args, **kwargs)  # Fault after an actual in-place Adam update.
            raise RuntimeError('injected later controller failure')

        with patch.object(module, method, side_effect=fail_after_earlier_decisions):
            with self.assertRaisesRegex(RuntimeError, 'injected later controller failure'):
                population.step({'verb':'rest'})
        self.assertEqual(len(reached), 1)
        self.assertEqual(reached[0][0], before['brains']['r0']['decisions']+1)
        if optimizer_failure:
            self.assertGreater(reached[0][1], before['brains']['r0']['updates'])
        self.assertStateEqual(snapshot(population), before)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rolled-back.pt'; population.save(path)
            cls = LegacyPopulation if legacy else Population
            self.assertStateEqual(snapshot(cls.load(path)), before)
        self.assertEqual(population.step({'verb':'rest'}), control.step({'verb':'rest'}))
        self.assertStateEqual(snapshot(population), snapshot(control))

    def test_later_decision_failure_restores_every_state_and_retries_exactly(self):
        for ticks in (0, 4, 8, 16):
            with self.subTest(ticks=ticks):
                self.check_decision_rollback(ticks)

    def test_partial_optimizer_update_is_rolled_back(self):
        self.check_decision_rollback(16, optimizer_failure=True)

    def test_legacy_controller_decision_is_also_transactional(self):
        self.check_decision_rollback(4, optimizer_failure=True, legacy=True)

    def test_max_speed_stops_at_first_and_mid_batch_failure_until_explicit_retry(self):
        for fail_on in (1, 3):
            with self.subTest(fail_on=fail_on), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); fresh_population().save(root/'runs/population.pt')
                runtime = LearningRuntime(root)
                try:
                    runtime.speed = 'max'
                    runtime.manual_pause = runtime.paused = False
                    runtime.last_heartbeat = time.monotonic()
                    step = runtime.population.step; calls = []

                    def maybe_fail(command=None):
                        calls.append(command)
                        if len(calls) == fail_on: raise RuntimeError('injected batch failure')
                        return step(command)

                    # One complete clock iteration; deterministic even if a regression
                    # incorrectly continues after failure. No timing-sensitive sleeps.
                    with patch.object(runtime.population, 'step', side_effect=maybe_fail), \
                         patch.object(runtime.stop_event, 'wait', side_effect=lambda delay: runtime.stop_event.set()):
                        runtime._run()
                    self.assertEqual(len(calls), fail_on)
                    self.assertEqual(runtime.world.tick, fail_on-1)
                    self.assertTrue(runtime.paused); self.assertTrue(runtime.manual_pause)
                    self.assertIn('injected batch failure', runtime.controller_error)
                    self.assertTrue(runtime.control({'command':'step'})[0])
                    self.assertEqual(runtime.world.tick, fail_on)
                    self.assertIsNone(runtime.controller_error)
                finally:
                    runtime.close()

    def test_queued_player_action_survives_failure_and_executes_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fresh_population().save(root/'runs/population.pt')
            runtime = LearningRuntime(root)
            try:
                self.assertTrue(runtime.action({'verb':'rest'})[0])
                before = snapshot(runtime.population)
                with patch.object(runtime.population.brains['r1'].model, 'forward', side_effect=RuntimeError('retry me')):
                    self.assertFalse(runtime.control({'command':'step'})[0])
                self.assertEqual(runtime.pending_action, {'verb':'rest'})
                self.assertStateEqual(snapshot(runtime.population), before)
                with patch.object(runtime.world, 'apply_action', wraps=runtime.world.apply_action) as apply:
                    self.assertTrue(runtime.control({'command':'step'})[0])
                player_calls = [call for call in apply.call_args_list if call.args[0] == 'player']
                self.assertEqual(len(player_calls), 1)
                self.assertEqual(player_calls[0].args[1], {'verb':'rest'})
                self.assertIsNone(runtime.pending_action); self.assertIsNone(runtime.controller_error)
                self.assertEqual(runtime.world.tick, 1)
            finally:
                runtime.close()

    def test_failed_rollback_blocks_steps_and_saves_without_overwriting_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'runs/population.pt'
            fresh_population().save(path); original = path.read_bytes()
            runtime = LearningRuntime(root)
            try:
                with patch.object(runtime.population.brains['r1'].model, 'forward', side_effect=RuntimeError('decision fault')), \
                     patch.object(runtime.population.brains['r0'], 'restore', side_effect=RuntimeError('restore fault')):
                    self.assertFalse(runtime._tick())
                self.assertIn('rollback failed for r0', runtime.controller_error)
                for command in ('step','resume','save','speed'):
                    self.assertFalse(runtime.control({'command':command})[0])
                with self.assertRaisesRegex(RuntimeError, 'rollback failed'):
                    runtime.population.save(path)
                self.assertFalse(runtime._save('autosave.json'))
            finally:
                runtime.close()
            self.assertEqual(path.read_bytes(), original)

    def test_partial_world_transition_requires_reload_and_preserves_disk_save(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'runs/population.pt'
            fresh_population().save(path); original = path.read_bytes()
            runtime = LearningRuntime(root)
            try:
                runtime.control({'command':'save'})
                before = snapshot(runtime.population)
                with patch.object(runtime.population.brains['r1'], 'add_reward', side_effect=RuntimeError('reward fault')):
                    self.assertFalse(runtime._tick())
                self.assertEqual(runtime.world.tick, 1)
                self.assertIn('Saving and advancing are blocked', runtime.controller_error)
                self.assertFalse(runtime.control({'command':'resume'})[0])
                self.assertFalse(runtime._save('autosave.json'))
                self.assertEqual(path.read_bytes(), original)
                # Loading must not try to save the now-inconsistent population.
                self.assertTrue(runtime.control({'command':'load'})[0])
                self.assertStateEqual(snapshot(runtime.population), before)
                self.assertIsNone(runtime.controller_error)
            finally:
                runtime.close()


if __name__ == '__main__':
    unittest.main()
