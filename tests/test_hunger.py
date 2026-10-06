"""Deterministic hunger physiology, saved recovery, and scripted curiosity."""
import copy
import json
import unittest
from unittest.mock import patch
from sim.world import SIZE, World


class HungerTests(unittest.TestCase):
    def lab(self):
        w = World(42)
        w.terrain = [[0] * SIZE for _ in range(SIZE)]
        w.resources.clear()
        w.structures.clear()
        w.residents = {'player': w.residents['player']}
        a = w.residents['player']
        a.x, a.y = 20, 20
        return w, a

    def test_boundaries_and_derived_views(self):
        w, a = self.lab()
        for food, stage, discomfort in ((100, 'satiated', 0), (60, 'satiated', 0),
                                         (30, 'hungry', 50), (20, 'hungry', 200/3),
                                         (19, 'weak', 410/6), (0, 'starving', 100)):
            a.food = food
            self.assertEqual(w.hunger_state(a)['hunger_stage'], stage)
            self.assertAlmostEqual(w.hunger_state(a)['hunger_discomfort'], discomfort)
            self.assertEqual(w.observe(a.id)['schema'], 9)
            self.assertEqual(w.observe(a.id)['body']['hunger_stage'], stage)
            self.assertEqual(w.state()['residents'][0]['hunger_stage'], stage)
        a.starvation_ticks = 120
        self.assertEqual(w.hunger_state(a)['starvation_strain'], 50)
        self.assertNotIn('hunger_discomfort', w.to_dict()['residents'][0])

    def test_rest_and_movement_weakness(self):
        w, a = self.lab()
        for food, gain in ((60, .25), (30, .125), (0, 0)):
            a.food, a.energy = food, 50
            w.apply_action(a.id, {'verb': 'rest'})
            self.assertAlmostEqual(a.energy, 50 + gain)
        for food, cooldown in ((60, 1), (19, 2), (0, 4)):
            a.food, a.energy, a.next_move_tick = food, 50, w.tick
            self.assertTrue(w.apply_action(a.id, {'verb': 'move', 'direction': 'east'})[0])
            expected = .035 * (1 + 2 * w.hunger_state(a)['hunger_discomfort']/100)
            self.assertAlmostEqual(a.energy, 50-expected)
            self.assertFalse(w.apply_action(a.id, {'verb': 'move', 'direction': 'east'})[0])
            w.tick += cooldown
            self.assertTrue(w.apply_action(a.id, {'verb': 'move', 'direction': 'west'})[0])

    def test_starvation_faint_deadline_and_grace(self):
        w, a = self.lab()
        a.food, a.energy = 0, 0
        for _ in range(239): w.step()
        self.assertFalse(a.unconscious)
        self.assertEqual(a.starvation_ticks, 239)
        w.step()
        self.assertTrue(a.unconscious)
        self.assertEqual(a.hunger_faint_until, 320)
        self.assertEqual(a.fainted, 1)
        for _ in range(79): w.step()
        self.assertTrue(a.unconscious)
        w.step()
        self.assertFalse(a.unconscious)
        self.assertEqual((a.food, a.energy, a.health, a.pain), (0, 10, 100, 0))
        self.assertEqual(a.starvation_ticks, 0)
        for _ in range(240): w.step()
        self.assertTrue(a.unconscious)
        self.assertEqual(a.fainted, 2)
        self.assertFalse(w.state()['mortality'])

    def test_hunger_wake_still_requires_health_and_injury_wake_unchanged(self):
        w, a = self.lab()
        a.unconscious, a.hunger_faint_until, a.health = True, 1, 25
        a.last_hurt = 0
        w.step()
        self.assertTrue(a.unconscious)
        a.health = 30
        w.step()
        self.assertFalse(a.unconscious)
        a.unconscious, a.health, a.hunger_faint_until = True, 30, 0
        w.step()
        self.assertFalse(a.unconscious)

    def test_eating_and_supportive_revival(self):
        w, a = self.lab()
        a.food, a.starvation_ticks, a.inventory['food'] = 0, 230, 1
        self.assertTrue(w.apply_action(a.id, {'verb': 'eat'})[0])
        self.assertEqual(a.starvation_ticks, 0)
        self.assertEqual(w.hunger_state(a)['hunger_stage'], 'hungry')
        b = World(1).residents['r0']
        b.x, b.y, b.food, b.unconscious = 21, 20, 0, True
        b.hunger_faint_until, b.starvation_ticks = 90, 50
        w.residents[b.id] = b
        a.inventory['food'] = 1
        ok, message = w.apply_action(a.id, {'verb': 'revive'})
        self.assertTrue(ok)
        self.assertIn('1 ordinary food', message)
        self.assertEqual((b.food, b.hunger_faint_until, b.starvation_ticks, a.inventory['food']), (25, 0, 0, 0))
        b.food, b.unconscious = 0, True
        self.assertTrue(w.apply_action(a.id, {'verb': 'revive'})[0])
        self.assertEqual(b.food, 0)

    def test_save_defaults_replay_and_transition_journal(self):
        w, a = self.lab()
        a.food = 20.004
        w.step()
        events = [e for e in w.events if e['kind'] == 'hunger']
        self.assertEqual(len(events), 1)
        for _ in range(3): w.step()
        self.assertEqual(len([e for e in w.events if e['kind'] == 'hunger']), 1)
        a.food, a.starvation_ticks, a.next_move_tick = 0, 238, w.tick+4
        restored = World.from_dict(json.loads(json.dumps(w.to_dict())))
        for _ in range(90):
            w.step(); restored.step()
        self.assertEqual(w.to_dict(), restored.to_dict())
        old = copy.deepcopy(w.to_dict())
        old.pop('exploration', None)
        for resident in old['residents']:
            for field in ('starvation_ticks', 'hunger_faint_until', 'next_move_tick'):
                resident.pop(field)
        loaded = World.from_dict(old)
        self.assertEqual(loaded.exploration, {})
        self.assertEqual(loaded.residents[a.id].starvation_ticks, 0)

    def test_curiosity_can_open_a_closed_exit(self):
        w,a=self.lab()
        for dx,dy in ((0,-1),(0,1),(-1,0)):
            w.structures[w.key(a.x+dx,a.y+dy)]={'kind':'wall'}
        w.structures[w.key(a.x+1,a.y)]={'kind':'door','open':False}
        with patch.object(w.rng,'random',return_value=.9):
            self.assertEqual(w.baseline_action(a.id), {'verb':'toggle_door','direction':'east'})

    def test_scripted_curiosity_independent_saved_and_no_unconscious_exploration(self):
        w, a = self.lab()
        b = World(1).residents['r0']
        b.x, b.y = 30, 30
        w.residents[b.id] = b
        # Skip optional baseline gather/give/rest branches.
        with patch.object(w.rng, 'random', return_value=.9):
            directions = [w.baseline_action(a.id)['direction'] for _ in range(4)]
            self.assertEqual(set(directions), {'north', 'east', 'south', 'west'})
            self.assertNotIn(b.id, w.exploration)
            w.baseline_action(b.id)
            self.assertIsNot(w.exploration[a.id], w.exploration[b.id])
            saved = copy.deepcopy(w.exploration)
            a.unconscious = True
            self.assertEqual(w.baseline_action(a.id), {'verb': 'rest'})
            self.assertEqual(w.exploration, saved)
        restored = World.from_dict(json.loads(json.dumps(w.to_dict())))
        self.assertEqual(restored.exploration, w.exploration)
        a.unconscious, a.food, a.energy = False, 0, 0
        w.resources[w.key(21,20)] = {'kind': 'berry', 'amount': 1}
        self.assertEqual(w.baseline_action(a.id)['verb'], 'gather')


if __name__ == '__main__':
    unittest.main()
