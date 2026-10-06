"""Controlled consequences for later learning experiments; no learning claim."""
import copy
import unittest
from sim.world import World, SIZE
from sim.visual_memory import impression


class HazardTests(unittest.TestCase):
    def setUp(self):
        self.w = World(713)
        self.w.terrain = [[0] * SIZE for _ in range(SIZE)]
        self.w.resources.clear()
        self.w.residents = {"player": self.w.residents["player"]}
        self.a = self.w.residents["player"]
        self.a.x, self.a.y = 20, 20
        self.a.food = 50

    def test_harmful_food_is_a_choice_and_harvesting_does_not_hurt(self):
        self.w.resources["20,19"] = {"kind": "amber_bush", "amount": 2}
        self.assertTrue(self.w.apply_action("player", {"verb": "gather"})[0])
        self.assertEqual(self.a.health, 100)
        self.assertEqual(self.a.inventory["amber_fruit"], 1)
        self.a.inventory["food"] = 1
        self.w.apply_action("player", {"verb": "eat", "item": "food"})
        self.assertEqual(self.a.health, 100)
        self.w.apply_action("player", {"verb": "eat", "item": "amber_fruit"})
        self.assertEqual(self.a.health, 80)
        self.assertEqual(self.a.pain, 20)
        self.assertEqual(self.a.food, 87)
        self.assertEqual(self.a.amber_eaten, 1)
        self.assertEqual(self.a.inventory["amber_fruit"], 0)

    def test_empty_or_inedible_choice_cannot_damage_or_consume(self):
        self.a.inventory["wood"] = 1
        for item in ("amber_fruit", "wood", "bogus"):
            self.assertFalse(self.w.apply_action("player", {"verb": "eat", "item": item})[0])
        self.assertEqual(self.a.health, 100)
        self.assertEqual(self.a.inventory["wood"], 1)
        self.assertEqual(self.a.amber_eaten, 0)

    def test_adjacent_thorns_are_visible_but_only_contact_hurts(self):
        self.w.resources["21,20"] = {"kind": "thorns", "amount": 1}
        before = self.w.observe("player")
        self.assertFalse(next(t for t in before["touch"] if t["direction"] == "east")["blocked"])
        self.w.step()
        self.assertEqual(self.a.health, 100)
        self.w.step({"player": {"verb": "move", "direction": "east"}})
        self.assertEqual(self.a.health, 92)
        self.assertEqual(self.a.thorn_contacts, 1)
        self.assertGreater(self.w.observe("player")["body"]["pain"], 0)
        for _ in range(14):
            self.w.step()
        self.assertEqual(self.a.thorn_contacts, 1)
        self.w.step()
        self.assertEqual(self.a.thorn_contacts, 2)

    def test_handling_thorns_hurts_without_creating_inventory_or_removing_plant(self):
        self.w.resources["20,19"] = {"kind": "thorns", "amount": 1}
        self.assertFalse(self.w.apply_action("player", {"verb": "gather"})[0])
        self.assertEqual(self.a.health, 92)
        self.assertNotIn("thorns", self.a.inventory)
        self.assertEqual(self.w.resources["20,19"]["amount"], 1)

    def test_recovery_has_delay_and_mortality_stays_off(self):
        self.w.hurt(self.a, 200, "test")
        self.assertEqual(self.a.health, 1)
        for _ in range(39):
            self.w.step()
        self.assertEqual(self.a.health, 1)
        self.w.step()
        self.assertAlmostEqual(self.a.health, 1.2)
        self.assertFalse(self.w.state()["mortality"])
        self.assertEqual(self.a.damage_taken, 99)

    def test_new_visual_cues_and_body_values_are_local_without_danger_labels(self):
        self.w.resources["20,19"] = {"kind": "amber_bush", "amount": 3}
        first = self.w.observe("player")
        tile = next(t for t in first["tiles"] if (t["dx"], t["dy"]) == (0, -1))
        self.assertEqual(set(tile["resource"]), {"kind", "amount"})
        self.w.resources["20,19"]["kind"] = "thorns"
        self.assertNotEqual(impression(first), impression(self.w.observe("player")))
        self.w.hurt(self.a, 10, "test")
        self.assertEqual(self.w.observe("player")["body"]["health"], 90)

    def test_older_save_migration_preserves_objects_rng_and_is_idempotent(self):
        data = copy.deepcopy(self.w.to_dict())
        data.pop("hazard_version")
        data["resources"]["33,35"] = {"kind": "tree", "amount": 3}
        data["structures"]["32,37"] = {"kind": "shelter", "builder": "player", "tick": 0}
        old = data["residents"][0]
        old["inventory"].pop("amber_fruit")
        for field in ("health", "pain", "last_hurt", "last_thorn_contact", "amber_eaten", "thorn_contacts", "damage_taken"):
            old.pop(field)
        loaded = World.from_dict(data)
        self.assertEqual(loaded.resources["33,35"], {"kind": "tree", "amount": 3})
        self.assertEqual(loaded.structures, data["structures"])
        self.assertEqual(loaded.rng.getstate(), self.w.rng.getstate())
        self.assertEqual(loaded.residents["player"].inventory["amber_fruit"], 0)
        self.assertEqual(loaded.to_dict(), World.from_dict(copy.deepcopy(loaded.to_dict())).to_dict())

    def test_save_continuation_preserves_injury_and_cooldown(self):
        self.w.resources["21,20"] = {"kind": "thorns", "amount": 1}
        self.w.step({"player": {"verb": "move", "direction": "east"}})
        restored = World.from_dict(copy.deepcopy(self.w.to_dict()))
        for _ in range(50):
            self.w.step()
            restored.step()
        self.assertEqual(self.w.to_dict(), restored.to_dict())

    def test_faint_blocks_actions_survives_save_and_recovers_on_thorns(self):
        self.w.resources["20,20"] = {"kind": "thorns", "amount": 1}
        self.w.hurt(self.a, 80, "test")
        self.assertTrue(self.a.unconscious)
        self.assertEqual(self.a.fainted, 1)
        restored = World.from_dict(copy.deepcopy(self.w.to_dict()))
        self.assertTrue(restored.residents["player"].unconscious)
        for verb in ("move", "eat", "gather", "build", "drop"):
            self.assertFalse(self.w.apply_action("player", {"verb": verb, "direction": "east"})[0])
        for _ in range(100):
            self.w.step()
        self.assertFalse(self.a.unconscious)
        self.assertGreaterEqual(self.a.health, 30)
        self.assertEqual(self.a.thorn_contacts, 0)
        self.assertTrue(self.w.apply_action("player", {"verb": "move", "direction": "east"})[0])

    def test_neighbor_revives_without_healing_fully_and_rejects_invalid_help(self):
        from sim.world import Resident
        b = Resident("r1", "Test", 21, 20, "#ffffff")
        self.w.residents[b.id] = b
        self.w.hurt(b, 99, "test")
        self.assertTrue(self.w.observe("player")["others"][0]["unconscious"])
        self.a.energy = 9
        self.assertFalse(self.w.apply_action("player", {"verb": "revive"})[0])
        self.a.energy = 100
        self.assertEqual(self.w.baseline_action("player"), {"verb": "revive"})
        self.assertTrue(self.w.apply_action("player", {"verb": "revive"})[0])
        self.assertFalse(b.unconscious)
        self.assertEqual(b.health, 30)
        self.assertEqual(self.a.energy, 90)
        self.assertGreater(b.pain, 0)
        self.assertFalse(self.w.apply_action("player", {"verb": "revive"})[0])
        self.w.hurt(b, 20, "test")
        b.x = 23
        self.assertFalse(self.w.apply_action("player", {"verb": "revive"})[0])
