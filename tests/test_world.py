"""Simulation checks run with Python's standard library; no renderer or ML needed."""
import json
from pathlib import Path
import tempfile
import unittest

from sim.world import ITEMS, SIZE, RECIPES, World


class WorldTests(unittest.TestCase):
    def laboratory(self):
        world = World(42)
        world.terrain = [[0] * SIZE for _ in range(SIZE)]
        world.resources.clear()
        world.structures.clear()
        world.residents = {rid: world.residents[rid] for rid in ("r0", "r1", "player")}
        for a, position in zip(world.residents.values(), ((20, 20), (21, 20), (25, 25))):
            a.x, a.y = position
        return world

    def quiet(self, world):
        return {rid: {"verb": "wait"} for rid in world.residents}

    def total_items(self, world):
        totals = {item: sum(a.inventory[item] for a in world.residents.values()) for item in ITEMS}
        for resource in world.resources.values():
            item = {"berry": "food", "tree": "wood"}.get(resource["kind"], resource["kind"])
            totals[item] += resource["amount"]
        for structure in world.structures.values():
            for item, amount in RECIPES[structure["kind"]]["cost"].items():
                totals[item] += amount
            if structure.get("floor"):
                totals["wood"] += 1
        return totals

    def test_generation_and_scripted_replay_are_seed_deterministic(self):
        first, second = World(123), World(123)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.resources, World(124).resources)
        directions = ("north", "east", "south", "west")
        for tick in range(160):
            command = {"player": {"verb": "move", "direction": directions[tick % 4]}}
            self.assertEqual(first.step(command), second.step(command))
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_json_round_trip_and_save_resume_preserve_rng_continuation(self):
        uninterrupted = World(725)
        for _ in range(89):
            uninterrupted.step()
        decoded = World.from_dict(json.loads(json.dumps(uninterrupted.to_dict())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "save.json"
            uninterrupted.save(path)
            restored = World.load(path)
            self.assertFalse(path.with_suffix(".tmp").exists())
            for _ in range(100):
                expected = uninterrupted.step()
                self.assertEqual(decoded.step(), expected)
                self.assertEqual(restored.step(), expected)
            self.assertEqual(uninterrupted.to_dict(), restored.to_dict())
            self.assertEqual(uninterrupted.rng.random(), restored.rng.random())

    def test_unsupported_save_schema_is_rejected(self):
        data = World(12).to_dict()
        data["schema"] = -1
        with self.assertRaises(ValueError):
            World.from_dict(data)

    def test_observation_is_local_and_has_no_player_or_identity_flag(self):
        world = self.laboratory()
        world.residents["player"].x, world.residents["player"].y = 19, 20
        observation = world.observe("r0")
        self.assertEqual(set(observation), {"schema", "tiles", "others", "needs", "inventory", "hearing", "attention", "auditory_memory", "ecology", "touch", "facing", "body", "visual_memory"})
        self.assertEqual(len(observation["tiles"]), 81)
        self.assertEqual({(t["dx"], t["dy"]) for t in observation["tiles"]},
                         {(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5)})
        self.assertEqual(len(observation["others"]), 2)
        for other in observation["others"]:
            self.assertEqual(set(other), {"dx", "dy", "color", "unconscious"})
            self.assertLessEqual(max(abs(other["dx"]), abs(other["dy"])), 4)
        before = json.dumps(observation, sort_keys=True)
        # Outside the sensory patch, world changes must not reach the observation.
        world.resources["50,50"] = {"kind": "berry", "amount": 100}
        world.terrain[51][50] = 1
        self.assertEqual(before, json.dumps(world.observe("r0"), sort_keys=True))
        # Observation dictionaries do not mutate the simulation when an encoder uses them.
        observation["inventory"]["food"] = 100
        self.assertEqual(world.residents["r0"].inventory["food"], 0)

    def test_blocker_is_visible_but_occludes_resources_and_residents(self):
        world = self.laboratory()
        world.resources["21,20"] = {"kind": "tree", "amount": 3}
        world.resources["22,20"] = {"kind": "berry", "amount": 4}
        world.residents["r1"].x = 23
        observation = world.observe("r0")
        tiles = {(t["dx"], t["dy"]): t for t in observation["tiles"]}
        self.assertEqual(tiles[1, 0]["resource"]["kind"], "tree")
        self.assertEqual(tiles[2, 0], {"dx": 2, "dy": 0, "terrain": -1})
        self.assertEqual(tiles[3, 0]["terrain"], -1)
        self.assertEqual(observation["others"], [])
        world.resources.pop("21,20")
        self.assertEqual(world.observe("r0")["others"][0]["dx"], 3)

    def test_edge_observation_keeps_fixed_patch_and_unknown_outside(self):
        world = self.laboratory()
        world.residents["r0"].x, world.residents["r0"].y = 0, 0
        tiles = world.observe("r0")["tiles"]
        self.assertEqual(len(tiles), 81)
        self.assertTrue(all(t["terrain"] == -1 for t in tiles if t["dx"] < 0 or t["dy"] < 0))

    def test_gather_give_drop_pickup_conserve_items(self):
        world = self.laboratory()
        world.resources["20,19"] = {"kind": "berry", "amount": 2}
        world.residents["r0"].facing = "north"
        original = self.total_items(world)
        for rid, command in (("r0", {"verb": "gather"}),
                             ("r0", {"verb": "give", "item": "food", "target": "r1"}),
                             ("r1", {"verb": "drop", "item": "food"}),
                             ("r1", {"verb": "pickup"})):
            self.assertTrue(world.apply_action(rid, command)[0])
            self.assertEqual(self.total_items(world), original)
        self.assertNotIn("21,20", world.resources)
        self.assertEqual(world.residents["r1"].inventory["food"], 1)

    def test_full_pack_and_rejected_transfers_do_not_change_items(self):
        world = self.laboratory()
        world.resources["20,19"] = {"kind": "berry", "amount": 2}
        a, b = world.residents["r0"], world.residents["r1"]
        a.inventory["food"] = 12
        b.inventory["wood"] = 12
        original = self.total_items(world)
        self.assertFalse(world.apply_action("r0", {"verb": "gather"})[0])
        self.assertFalse(world.apply_action("r0", {"verb": "give", "target": "r1"})[0])
        self.assertFalse(world.apply_action("r0", {"verb": "drop", "item": "unknown"})[0])
        self.assertEqual(self.total_items(world), original)

    def test_gather_preserves_each_resource_type_and_pack_capacity(self):
        for kind, item in (("tree", "wood"), ("stone", "stone"), ("berry", "food"),
                           ("food", "food"), ("wood", "wood"), ("seed", "seed")):
            with self.subTest(kind=kind):
                world = self.laboratory()
                world.resources["20,19"] = {"kind": kind, "amount": 1}
                a = world.residents["r0"]
                a.facing = "north"
                a.inventory["stone"] = 11
                original = self.total_items(world)
                self.assertTrue(world.apply_action("r0", {"verb": "gather"})[0])
                self.assertEqual(world.capacity(a), 0)
                self.assertEqual(self.total_items(world), original)
                self.assertEqual(a.inventory[item], 12 if item == "stone" else 1)
                self.assertFalse(world.apply_action("r0", {"verb": "gather"})[0])
                self.assertEqual(self.total_items(world), original)

    def test_eating_consumes_one_food_and_restores_bounded_need(self):
        world = self.laboratory()
        a = world.residents["r0"]
        a.inventory["food"], a.food = 2, 90
        before = self.total_items(world)
        self.assertTrue(world.apply_action("r0", {"verb": "eat"})[0])
        self.assertEqual(a.food, 100)
        self.assertEqual(self.total_items(world)["food"], before["food"] - 1)
        self.assertFalse(world.apply_action("r0", {"verb": "eat"})[0])
        self.assertEqual(a.inventory["food"], 1)

    def test_shelter_embodies_construction_cost_and_duplicate_build_is_rejected(self):
        world = self.laboratory()
        a = world.residents["r0"]
        a.inventory.update(wood=8, stone=4)
        original = self.total_items(world)
        self.assertTrue(world.apply_action("r0", {"verb": "build"})[0])
        self.assertEqual(a.inventory["wood"], 4)
        self.assertEqual(a.inventory["stone"], 2)
        self.assertEqual(self.total_items(world), original)
        self.assertFalse(world.apply_action("r0", {"verb": "build"})[0])
        self.assertEqual(self.total_items(world), original)

    def test_drop_cannot_overwrite_a_different_resource(self):
        world = self.laboratory()
        a = world.residents["r0"]
        a.inventory["food"] = 1
        world.resources["20,20"] = {"kind": "berry", "amount": 3}
        original = self.total_items(world)
        self.assertFalse(world.apply_action("r0", {"verb": "drop"})[0])
        self.assertEqual(self.total_items(world), original)

    def test_simultaneous_contenders_cannot_occupy_same_tile(self):
        world = self.laboratory()
        world.residents["r1"].x = 22
        results = world.step({"r0": {"verb": "move", "direction": "east"},
                              "r1": {"verb": "move", "direction": "west"},
                              "player": {"verb": "wait"}})
        self.assertEqual(sum(results[rid][0] for rid in ("r0", "r1")), 1)
        self.assertEqual(len({(a.x, a.y) for a in world.residents.values()}), len(world.residents))

    def test_occupancy_and_needs_remain_valid_over_a_scripted_run(self):
        world = World(138)
        for _ in range(600):
            world.step()
            self.assertEqual(len({(a.x, a.y) for a in world.residents.values()}), len(world.residents))
            for a in world.residents.values():
                self.assertTrue(world.inside(a.x, a.y))
                for need in (a.food, a.water, a.energy, a.warmth):
                    self.assertGreaterEqual(need, 0)
                    self.assertLessEqual(need, 100)
                self.assertGreaterEqual(world.capacity(a), 0)
                self.assertTrue(all(amount >= 0 for amount in a.inventory.values()))

    def test_needs_have_lower_bounds_and_rest_upper_bound(self):
        world = self.laboratory()
        world.tick = 2400
        for a in world.residents.values():
            a.food = a.water = a.warmth = .001
            a.energy = 99.99
        for _ in range(10):
            world.step(self.quiet(world))
        for a in world.residents.values():
            self.assertEqual((a.food, a.water, a.warmth), (0, 0, 0))
            self.assertLess(a.energy, 99.991)  # Almost no recovery before fullness reaches zero.
            a.food = 100
            world.apply_action(a.id, {"verb": "rest"})
            self.assertEqual(a.energy, 100)

    def test_shelter_improves_warmth_during_same_rain_tick(self):
        world = self.laboratory()
        world.tick = 2400
        sheltered, exposed = world.residents["r0"], world.residents["r1"]
        sheltered.warmth = exposed.warmth = 50
        world.structures["20,20"] = {"kind": "shelter", "builder": "r0", "tick": 0}
        world.step(self.quiet(world))
        self.assertEqual(world.weather, "Rain")
        self.assertAlmostEqual(sheltered.warmth, 50.06)
        self.assertAlmostEqual(exposed.warmth, 49.965)

    def test_existing_bush_and_dropped_stock_stays_finite(self):
        world = self.laboratory()
        world.resources.update({"10,10": {"kind": "berry", "amount": 0},
                                "11,10": {"kind": "berry", "amount": 5},
                                "12,10": {"kind": "berry", "amount": 8},
                                "13,10": {"kind": "food", "amount": 2}})
        world.tick = 598
        world.step(self.quiet(world))
        self.assertEqual(world.resources["10,10"]["amount"], 0)
        world.step(self.quiet(world))
        self.assertEqual(world.resources["10,10"]["amount"], 0)
        for _ in range(8):
            world.tick = ((world.tick // 600) + 1) * 600 - 1
            world.step(self.quiet(world))
        self.assertEqual(world.resources["10,10"]["amount"], 0)
        self.assertEqual(world.resources["11,10"]["amount"], 5)
        self.assertEqual(world.resources["12,10"]["amount"], 8)
        self.assertEqual(world.resources["13,10"]["amount"], 2)


class OptionalNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from agents.network import Controller, ResidentNetwork, encode_observation
        except ImportError:
            raise unittest.SkipTest("Optional PyTorch package is unavailable")
        cls.torch = torch
        cls.Controller = Controller
        cls.Network = ResidentNetwork
        cls.encode = staticmethod(encode_observation)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def test_encoder_ignores_privileged_fields_and_hidden_tile_payloads(self):
        import copy
        world = World(72)
        original = world.observe("r0")
        altered = copy.deepcopy(original)
        altered.update(player=True, resident_id="player", world=world.to_dict(), tick=999)
        altered["tiles"][0] = {"dx": -4, "dy": -4, "terrain": -1}
        original["tiles"][0] = dict(altered["tiles"][0])
        altered["tiles"][0].update(resource={"kind": "food", "amount": 12}, shelter=True)
        altered["others"].append({"dx": -4, "dy": -4, "color": "#ffffff", "player": True})
        first, second = self.encode(original), self.encode(altered)
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(first, second)))
        from agents.network import FEATURES, PATCH_CHANNELS
        self.assertEqual(tuple(first[0].shape), (1, PATCH_CHANNELS, 9, 9))
        self.assertEqual(tuple(first[1].shape), (1, FEATURES))

    def test_controllers_have_separate_weights_memory_and_seeded_initialization(self):
        first, second = self.Controller(seed=29), self.Controller(seed=29)
        different = self.Controller(seed=30)
        first_parameter = next(first.model.parameters())
        second_parameter = next(second.model.parameters())
        self.assertTrue(self.torch.equal(first_parameter, second_parameter))
        self.assertNotEqual(first_parameter.data_ptr(), second_parameter.data_ptr())
        self.assertFalse(self.torch.equal(first_parameter, next(different.model.parameters())))
        observation = World(12).observe("r0")
        command, metadata = first.decide(observation)
        self.assertIn("verb", command)
        self.assertFalse(metadata["trained"])
        self.assertEqual(first.decisions, 1)
        self.assertEqual(second.decisions, 0)
        self.assertTrue(self.torch.equal(second.state[0], self.torch.zeros_like(second.state[0])))
        self.assertFalse(self.torch.equal(first.state[0], second.state[0]))
        first.reset()
        self.assertEqual(first.decisions, 0)
        self.assertTrue(self.torch.equal(first.state[0], second.state[0]))

    def test_actor_value_and_recurrent_shapes_are_finite(self):
        from agents.network import ACTIONS
        model = self.Network()
        patch, features = self.encode(World(83).observe("r0"))
        logits, value, state = model(patch, features)
        self.assertEqual(tuple(logits.shape), (1, len(ACTIONS)))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual([tuple(tensor.shape) for tensor in state], [(1, 128), (1, 128)])
        self.assertTrue(all(self.torch.isfinite(tensor).all() for tensor in (logits, value, *state)))

    def test_capacity_is_configurable_and_described_in_manifest(self):
        small = self.Controller(seed=83, hidden_size=64)
        larger = self.Controller(seed=83, hidden_size=256)
        observation = World(83).observe("r0")
        for controller, width in ((small, 64), (larger, 256)):
            command, metadata = controller.decide(observation)
            self.assertIn("verb", command)
            self.assertEqual(tuple(controller.state[0].shape), (1, width))
            self.assertEqual(controller.manifest()["hidden_size"], width)
            self.assertFalse(metadata["trained"])
        self.assertGreater(larger.manifest()["parameters"], small.manifest()["parameters"])

    def test_new_senses_and_visual_recall_reach_the_neural_encoder(self):
        import copy
        observation = World(83).observe("r0")
        changed = copy.deepcopy(observation)
        changed["hearing"] = [{"kind": "building", "bearing": "east", "strength": .8, "age": 0}]
        changed["touch"] = [{"direction": "north", "blocked": True}]
        changed["body"] = {"rain": True, "sheltered": False}
        changed["visual_memory"]["recalled"] = [{"sketch": changed["visual_memory"]["current"], "similarity": .8}]
        original, updated = self.encode(observation), self.encode(changed)
        self.assertFalse(self.torch.equal(original[0], updated[0]))
        self.assertFalse(self.torch.equal(original[1], updated[1]))
        self.assertGreater(int(self.torch.count_nonzero(updated[0][0, 23])), 0)
        _, value, _ = self.Network()(*updated)
        self.assertTrue(self.torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
