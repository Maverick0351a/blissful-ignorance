"""Construction, embodied perception, and private lossy memory regression checks."""
import copy
import unittest

from sim.world import World, SIZE, RECIPES
from sim.visual_memory import impression, pack, unpack, similarity, remember


class ConstructionSensesTests(unittest.TestCase):
    def setUp(self):
        self.world = World(100)
        self.world.terrain = [[0] * SIZE for _ in range(SIZE)]
        self.world.resources.clear()
        self.world.residents = {k: self.world.residents[k] for k in ("r0", "r1", "player")}
        for a, point in zip(self.world.residents.values(), ((20, 20), (24, 20), (40, 40))):
            a.x, a.y = point
        self.a = self.world.residents["r0"]

    def action(self, verb, **kwargs):
        return self.world.apply_action("r0", {"verb": verb, **kwargs})

    def quiet(self, count=1):
        for _ in range(count):
            self.world.step({rid: {"verb": "wait"} for rid in self.world.residents})

    def test_build_wall_blocks_passage_sight_and_recovers_exact_materials(self):
        self.a.inventory["wood"] = 2
        self.assertTrue(self.action("build", kind="wall", x=21, y=20)[0])
        self.assertFalse(self.action("move", direction="east")[0])
        self.assertFalse(self.world.visible(self.a, 22, 20))
        self.assertTrue(self.world.visible(self.a, 21, 20))
        self.assertTrue(self.action("dismantle", x=21, y=20)[0])
        self.assertEqual(self.a.inventory["wood"], 2)
        self.assertTrue(self.action("move", direction="east")[0])

    def test_illegal_builds_do_not_spend_materials(self):
        self.a.inventory.update(wood=6, stone=3)
        before = dict(self.a.inventory)
        for fields in ({"x": 20, "y": 20}, {"x": 21, "y": 21}, {"x": 50, "y": 50}, {"x": True, "y": 20}):
            self.assertFalse(self.action("build", kind="wall", **fields)[0])
        self.world.resources["21,20"] = {"kind": "berry", "amount": 0}
        self.assertFalse(self.action("build", kind="wall", x=21, y=20)[0])
        self.assertEqual(before, self.a.inventory)

    def test_floor_layer_survives_top_piece_removal_and_pack_cannot_overflow(self):
        self.a.inventory.update(wood=3)
        self.assertTrue(self.action("build", kind="floor", x=21, y=20)[0])
        self.assertTrue(self.action("build", kind="wall", x=21, y=20)[0])
        self.a.inventory["food"] = 12
        self.assertFalse(self.action("dismantle", x=21, y=20)[0])
        self.assertEqual(self.world.structures["21,20"]["kind"], "wall")
        self.a.inventory["food"] = 0
        self.assertTrue(self.action("dismantle", x=21, y=20)[0])
        self.assertEqual(self.world.structures["21,20"]["kind"], "floor")
        self.assertTrue(self.action("dismantle", x=21, y=20)[0])
        self.assertEqual(self.a.inventory["wood"], 3)

    def test_door_must_open_to_pass_or_see_and_cannot_close_on_resident(self):
        self.a.inventory["wood"] = 3
        self.assertTrue(self.action("build", kind="door", x=21, y=20)[0])
        self.assertFalse(self.world.visible(self.a, 24, 20))
        self.assertTrue(self.action("toggle_door", direction="east")[0])
        self.assertTrue(self.world.visible(self.a, 24, 20))
        self.assertTrue(self.action("move", direction="east")[0])
        self.assertFalse(self.action("toggle_door")[0])
        self.world.tick += 1
        self.assertTrue(self.action("move", direction="west")[0])
        self.assertTrue(self.action("toggle_door", direction="east")[0])
        self.assertFalse(self.world.walkable(21, 20))

    def test_enclosed_floors_protect_from_rain_but_wall_or_open_floor_does_not(self):
        self.world.structures["20,20"] = {"kind": "floor"}
        self.assertNotIn("20,20", self.world.covered_tiles())
        for x, y in ((19, 20), (21, 20), (20, 19)):
            self.world.structures[f"{x},{y}"] = {"kind": "wall"}
        self.world.structures["20,21"] = {"kind": "door", "open": True}
        self.assertIn("20,20", self.world.covered_tiles())
        self.world.tick = 2400
        self.a.warmth = 50
        self.quiet()
        self.assertGreater(self.a.warmth, 50)
        self.assertTrue(self.world.observe("r0")["body"]["sheltered"])
        del self.world.structures["19,20"]
        self.assertFalse(self.world.observe("r0")["body"]["sheltered"])

    def test_distinct_perspectives_and_hidden_changes_do_not_reach_senses_or_memory(self):
        self.world.structures["21,20"] = {"kind": "wall"}
        self.world.resources["22,20"] = {"kind": "berry", "amount": 2}
        first = self.world.observe("r0")
        other = self.world.observe("r1")
        self.assertFalse(any(t.get("resource", {}).get("kind") == "berry" for t in first["tiles"]))
        self.assertTrue(any(t.get("resource", {}).get("kind") == "berry" for t in other["tiles"]))
        self.world.resources["22,20"]["amount"] = 900
        self.world.residents["r1"].food = 0
        self.assertEqual(first, self.world.observe("r0"))
        self.assertNotEqual(first["visual_memory"]["current"], other["visual_memory"]["current"])

    def test_touch_and_hearing_are_relative_local_occluded_and_transient(self):
        other = self.world.residents["r1"]
        self.world.emit_sound(other, "building", 6)
        heard = self.world.observe("r0")["hearing"]
        self.assertEqual(heard[0]["bearing"], "east")
        self.assertEqual(set(heard[0]), {"kind", "bearing", "strength", "age"})
        self.assertEqual(self.world.observe("player")["hearing"], [])
        self.world.structures["21,20"] = {"kind": "wall"}
        quieter = self.world.observe("r0")
        self.assertLess(quieter["hearing"][0]["strength"], heard[0]["strength"])
        self.assertTrue(next(t["blocked"] for t in quieter["touch"] if t["direction"] == "east"))
        self.quiet(8)
        self.assertEqual(self.world.observe("r0")["hearing"], [])

    def test_no_sight_through_diagonal_wall_corner(self):
        self.world.structures.update({"21,20": {"kind": "wall"}, "20,21": {"kind": "wall"}})
        self.assertFalse(self.world.visible(self.a, 21, 21))

    def test_memory_stores_only_lossy_personal_impressions_and_survives_save(self):
        self.quiet(16)
        mine = self.world.visual_memories["r0"]
        self.assertIsNot(mine, self.world.visual_memories["r1"])
        self.assertEqual(set(mine[0]), {"sketch", "fingerprint", "formed", "last_seen", "visits"})
        self.assertEqual(len(bytes.fromhex(mine[0]["sketch"])), 41)
        restored = World.from_dict(copy.deepcopy(self.world.to_dict()))
        self.assertEqual(restored.observe("r0"), self.world.observe("r0"))
        self.quiet(16)
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["visits"], 2)

    def test_sketch_discards_resource_amount_and_out_of_sight_payloads(self):
        o = self.world.observe("r0")
        tile = o["tiles"][0]
        tile.update(terrain=0, resource={"kind": "berry", "amount": 1})
        first = impression(o)
        tile["resource"]["amount"] = 999
        self.assertEqual(first, impression(o))
        tile["terrain"] = -1
        first = impression(o)
        tile["resource"] = {"kind": "tree", "amount": 400}
        self.assertEqual(first, impression(o))

    def test_similarity_and_capacity_are_bounded_and_near_scenes_match(self):
        first = impression(self.world.observe("r0"))
        self.world.resources["20,19"] = {"kind": "berry", "amount": 3}
        near = impression(self.world.observe("r0"))
        far = {"sketch": pack([2]*81), "fingerprint": "ffffffffffffffff"}
        self.assertEqual(similarity(first, first), 1)
        self.assertGreater(similarity(first, near), similarity(first, far))
        memories = []
        for tick, scene in enumerate((first, far)):
            remember(memories, scene, tick, capacity=1)
        self.assertEqual(len(memories), 1)
        self.assertEqual(unpack(memories[0]["sketch"]), [2]*81)

    def test_old_save_without_senses_loads_without_changing_player_or_structures(self):
        data = copy.deepcopy(self.world.to_dict())
        data["structures"]["40,40"] = {"kind": "shelter", "builder": "player", "tick": 0}
        for key in ("sounds", "visual_memories", "memory_capacity"):
            data.pop(key)
        loaded = World.from_dict(data)
        self.assertEqual(loaded.residents["player"].x, 40)
        self.assertEqual(loaded.structures, data["structures"])
        self.assertEqual(loaded.observe("player")["visual_memory"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
