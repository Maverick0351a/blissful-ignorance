"""Authoritative simulation. No wall clock, renderer, networking, or ML dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from sim import ecology
from sim.visual_memory import impression, recall, remember
from sim.social import tap, tone, attention, expire_attention, sound_lifetime, remember_tones, auditory_memory, LEGACY_TONES

SCHEMA = 1
SIZE = 64
DAY_TICKS = 2400  # Ten real minutes at the normal four ticks/second.
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
ITEMS = ("food", "wood", "stone", "seed", "amber_fruit")
NAMES = ("Moss", "Pip", "Ember", "Fern", "Cove", "Ochre", "Luma", "Reed")
COLORS = ("#89ac82", "#d9a67c", "#b97c63", "#9baf69", "#81b0ae", "#d5b86b", "#b9a0c6", "#879bc1")
RECIPES = {
    "wall": {"label": "Wood wall", "cost": {"wood": 2}},
    "stone_wall": {"label": "Stone wall", "cost": {"stone": 2}},
    "floor": {"label": "Wood floor", "cost": {"wood": 1}},
    "door": {"label": "Door", "cost": {"wood": 3}},
    "shelter": {"label": "Shelter", "cost": {"wood": 4, "stone": 2}},
}
VISION_RADIUS = 4
SOUND_LIFETIME = 8
CROP_TICKS = 480  # Two minutes at 1x; simulation time, including fast-forward.
CROP_YIELD = 3


@dataclass
class Resident:
    id: str
    name: str
    x: int
    y: int
    color: str
    variant: int = 0
    facing: str = "south"
    food: float = 100.0
    water: float = 100.0
    energy: float = 100.0
    warmth: float = 100.0
    inventory: dict = field(default_factory=lambda: {item: 0 for item in ITEMS})
    action: str = "Looking around"
    gathered: int = 0
    shared: int = 0
    steps: int = 0
    health: float = 100.0
    pain: float = 0.0
    last_hurt: int = -1000
    last_thorn_contact: int = -1000
    amber_eaten: int = 0
    thorn_contacts: int = 0
    damage_taken: float = 0.0
    unconscious: bool = False
    fainted: int = 0
    starvation_ticks: int = 0
    hunger_faint_until: int = 0
    next_move_tick: int = 0
    last_tap_tick: int = -1000
    last_tone_tick: int = -1000
    irrigation_water: int = 0
    carried_cache: dict | None = None


class World:
    def __init__(self, seed: int = 1729, memory_capacity: int = 64):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0
        self.revision = 0
        self.terrain = [[0] * SIZE for _ in range(SIZE)]
        self.resources: dict[str, dict] = {}
        self.structures: dict[str, dict] = {}
        self.sounds: list[dict] = []
        self.attention_events: dict[str, list] = {}
        self.auditory_memories: dict[str, list] = {}
        self.ecology_enabled = False
        self.soils = {}
        self.caches = {}
        self.irrigation_supply = 500.0
        self.memory_capacity = max(1, int(memory_capacity))
        self.visual_memories: dict[str, list] = {}
        self.exploration: dict[str, dict] = {}
        self.events: list[dict] = []
        self.event_serial = 0
        self.residents: dict[str, Resident] = {}
        self._generate()
        self.seed_hazards()

    def seed_hazards(self):
        """Add plants once, without replacing existing resources, buildings or bodies.

        A separate generator leaves the running world's action RNG untouched.
        """
        rng = random.Random(self.seed ^ 0xA8BE)
        occupied = {(a.x, a.y) for a in self.residents.values()}
        sites = [(33, 35, "amber_bush"), (32, 37, "thorns")]
        sites += [(rng.randrange(3, SIZE-3), rng.randrange(3, SIZE-3), kind)
                  for kind in ("amber_bush", "thorns") for _ in range(28)]
        for x, y, kind in sites:
            key = self.key(x, y)
            if self.terrain[y][x] == 0 and key not in self.resources and key not in self.structures and (x, y) not in occupied:
                self.resources[key] = {"kind": kind, "amount": 4 if kind == "amber_bush" else 1}

    def hurt(self, a, amount, cause):
        actual = min(amount, max(0, a.health - 1))  # Development world: mortality remains off.
        a.health -= actual
        a.pain = min(100, a.pain + amount)
        a.last_hurt = self.tick
        a.damage_taken += actual
        a.action = "Recoiling in pain"
        self.record(f"{a.name} {'were' if a.name == 'You' else 'was'} hurt by {cause} ({actual:g} health).", "injury", a.id)
        self.emit_sound(a, "cry", 6)
        if a.health <= 20 and not a.unconscious:
            a.unconscious = True
            a.fainted += 1
            a.action = "Passed out - recovering"
            self.record(f"{a.name} passed out.", "injury", a.id)

    @staticmethod
    def hunger_state(a):
        """Internal hunger is derived from fullness; it is separate from injury pain."""
        return {"starvation_strain": min(100.0, max(0.0, a.starvation_ticks / 240 * 100)),
                "hunger_discomfort": max(0.0, min(100.0, (60 - a.food) * 100 / 60)),
                "hunger_stage": "starving" if a.food <= 0 else "weak" if a.food < 20
                else "hungry" if a.food < 60 else "satiated"}

    def hunger_transition(self, a, previous_food):
        previous = "starving" if previous_food <= 0 else "weak" if previous_food < 20 else "hungry" if previous_food < 60 else "satiated"
        stage = self.hunger_state(a)["hunger_stage"]
        if stage != previous:
            owner = "Your" if a.name == "You" else a.name + "'s"
            self.record(f"{owner} hunger changed: {previous} to {stage}.", "hunger", a.id)

    def thorn_contact(self, a):
        if a.unconscious or self.tick - a.last_thorn_contact < 16:
            return False
        a.last_thorn_contact = self.tick
        a.thorn_contacts += 1
        self.hurt(a, 8, "thorns")
        return True

    @staticmethod
    def key(x, y):
        return f"{x},{y}"

    def _generate(self):
        for y in range(SIZE):
            river = 42 + round(math.sin(y / 8) * 3)
            for x in range(SIZE):
                if x < 2 or y < 2 or x >= SIZE - 2 or y >= SIZE - 2:
                    self.terrain[y][x] = 4
                elif abs(x - river) <= 1:
                    self.terrain[y][x] = 1
                elif abs(x - river) == 2:
                    self.terrain[y][x] = 2
                elif self.rng.random() < 0.1:
                    kind = self.rng.choices(("tree", "berry", "stone"), (6, 3, 2))[0]
                    self.resources[self.key(x, y)] = {"kind": kind, "amount": 5 if kind == "tree" else 3}
        # An open clearing and two river crossings are terrain, not resident-built structures.
        for y in range(28, 38):
            for x in range(27, 38):
                self.resources.pop(self.key(x, y), None)
        for y in (18, 34, 49):
            for x in range(36, 49):
                if self.terrain[y][x] in (1, 2):
                    self.terrain[y][x] = 3
                self.resources.pop(self.key(x, y), None)
        for x, y, kind in ((31, 35, "berry"), (29, 32, "berry"), (35, 36, "berry"),
                           (28, 35, "tree"), (34, 29, "tree"), (35, 33, "stone"), (30, 37, "stone")):
            self.resources[self.key(x, y)] = {"kind": kind, "amount": 8}
        positions = ((30, 30), (32, 30), (34, 31), (36, 32), (36, 35), (34, 37), (29, 36), (28, 33))
        for i, ((x, y), name, color) in enumerate(zip(positions, NAMES, COLORS)):
            self.residents[f"r{i}"] = Resident(f"r{i}", name, x, y, color, i % 3)
        self.residents["player"] = Resident("player", "You", 32, 34, "#e6d8af", 2)
        self.record("Eight residents arrived in the valley.", "arrival")

    @property
    def weather(self):
        return "Rain" if (self.tick // 800) % 5 == 3 else "Clear"

    def record(self, text, kind="activity", actor=None):
        self.event_serial += 1
        self.events.append({"id": self.event_serial, "tick": self.tick, "text": text, "kind": kind, "actor": actor})
        self.events = self.events[-80:]

    def inside(self, x, y):
        return 0 <= x < SIZE and 0 <= y < SIZE

    def walkable(self, x, y):
        return self.inside(x, y) and self.terrain[y][x] not in (1, 4) and not self.solid(x, y)

    def solid(self, x, y):
        key = self.key(x, y)
        structure = self.structures.get(key, {})
        return (self.resources.get(key, {}).get("kind") in ("tree", "stone") or
                structure.get("kind") in ("wall", "stone_wall") or
                (structure.get("kind") == "door" and not structure.get("open", False)))

    def opaque(self, x, y):
        return not self.inside(x, y) or self.terrain[y][x] == 4 or self.solid(x, y)

    def covered_tiles(self):
        """A shelter or a fully enclosed floored room has a roof, up to 64 floor tiles.

        Doors form part of the roof's perimeter even when swung open. This is an
        explicit construction rule, not a claim that walls alone keep rain out.
        """
        covered = {key for key, s in self.structures.items() if s["kind"] == "shelter"}
        floors = {key for key, s in self.structures.items() if s["kind"] == "floor"}
        visited = set()
        for start in floors:
            if start in visited:
                continue
            room, pending, enclosed = set(), [start], True
            while pending:
                key = pending.pop()
                if key in room:
                    continue
                room.add(key)
                x, y = map(int, key.split(","))
                for dx, dy in DIRECTIONS.values():
                    neighbor = self.key(x + dx, y + dy)
                    if neighbor in floors and neighbor not in room:
                        pending.append(neighbor)
                    elif neighbor not in floors and self.structures.get(neighbor, {}).get("kind") not in ("wall", "stone_wall", "door"):
                        enclosed = False
            visited.update(room)
            if enclosed and len(room) <= 64:
                covered.update(room)
        return covered

    def emit_sound(self, resident, kind, radius=6):
        self.sounds.append({"x": resident.x, "y": resident.y, "actor": resident.id,
                            "kind": kind, "radius": radius, "tick": self.tick})
        self.sounds = self.sounds[-128:]

    def hearing(self, resident, internal=False):
        heard = []
        for sound in self.sounds:
            age = self.tick - sound["tick"]
            lifetime = sound_lifetime(sound["kind"])
            dx, dy = sound["x"] - resident.x, sound["y"] - resident.y
            distance = abs(dx) + abs(dy)
            if sound["actor"] == resident.id or not 0 <= age < lifetime or distance > sound["radius"]:
                continue
            strength = (1 - distance / (sound["radius"] + 1)) * (1 - age / lifetime)
            if not self.visible(resident, sound["x"], sound["y"]):
                strength *= .35
            if strength < .08:
                continue
            bearing = ("north" if dy < 0 else "south" if dy > 0 else "") + ("west" if dx < 0 else "east" if dx > 0 else "")
            heard.append({"kind": sound["kind"], "bearing": bearing or "here", "strength": round(strength, 3), "age": age})
            if internal:
                heard[-1]["key"] = [sound["tick"], sound["actor"], sound["kind"]]
        return heard[-128:] if internal else heard[-8:]

    def capacity(self, resident):
        return 12 - sum(resident.inventory.values()) - (4 + resident.carried_cache["food"] if resident.carried_cache is not None else 0)

    def nearby(self, resident, include_self=True):
        offsets = [(0, 0)] if include_self else []
        front = DIRECTIONS[resident.facing]
        offsets += [front] + [d for d in DIRECTIONS.values() if d != front]
        return [(resident.x + dx, resident.y + dy) for dx, dy in offsets if self.inside(resident.x + dx, resident.y + dy)]

    def visible(self, resident, x, y):
        # Occluders themselves are visible. Check corner crossings too.
        dx, dy = x - resident.x, y - resident.y
        distance = max(abs(dx), abs(dy))
        for i in range(1, distance):
            px = round(resident.x + dx * i / distance)
            py = round(resident.y + dy * i / distance)
            if self.opaque(px, py):
                return False
        for i in range(distance):
            px = round(resident.x + dx * i / max(1, distance))
            py = round(resident.y + dy * i / max(1, distance))
            nx = round(resident.x + dx * (i + 1) / max(1, distance))
            ny = round(resident.y + dy * (i + 1) / max(1, distance))
            if px != nx and py != ny and self.opaque(px, ny) and self.opaque(nx, py):
                return False
        return True

    def observe(self, rid, include_memory=True):
        a = self.residents[rid]
        tiles = []
        for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
            for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
                x, y = a.x + dx, a.y + dy
                tile = {"dx": dx, "dy": dy, "terrain": -1}
                if self.inside(x, y) and self.visible(a, x, y):
                    structure = self.structures.get(self.key(x, y), {})
                    tile.update(terrain=self.terrain[y][x], resource=dict(self.resources.get(self.key(x, y), {})),
                                shelter=structure.get("kind") == "shelter",
                                structure={k: structure[k] for k in ("kind", "open", "floor") if k in structure},
                                blocked=not self.walkable(x, y))
                if tile["terrain"] >= 0 and self.ecology_enabled:
                    tile["soil"] = ecology.soil(self, x, y) if tile["terrain"] == 0 else {}
                    tile["cache"] = dict(self.caches.get(self.key(x,y), {}))
                tiles.append(tile)
        others = [{"dx": b.x - a.x, "dy": b.y - a.y, "color": b.color, "unconscious": b.unconscious}
                  for b in self.residents.values() if b.id != rid and max(abs(b.x - a.x), abs(b.y - a.y)) <= 4 and self.visible(a, b.x, b.y)]
        touch = []
        for direction, (dx, dy) in DIRECTIONS.items():
            x, y = a.x + dx, a.y + dy
            occupied = any(b.id != rid and (b.x, b.y) == (x, y) for b in self.residents.values())
            touch.append({"direction": direction, "blocked": not self.walkable(x, y) or occupied})
        covered = self.key(a.x, a.y) in self.covered_tiles()
        # Growth is visible in broad stages, not as privileged clock metadata.
        for tile in tiles:
            resource = tile.get("resource", {})
            if resource.get("kind") == "crop":
                tile["resource"] = {"kind": "crop", "amount": 0,
                                    "stage": resource.get("stage", min(2, (self.tick - resource["planted"]) * 3 // CROP_TICKS))}
        observation = {"schema": 9, "tiles": tiles, "others": others,
                "needs": [a.food, a.water, a.energy, a.warmth], "inventory": dict(a.inventory),
                "hearing": self.hearing(a), "attention": attention(self, rid), "auditory_memory": auditory_memory(self, rid), "touch": touch, "facing": a.facing,
                "ecology": {"enabled": self.ecology_enabled, "water_carried": a.irrigation_water, "carried_cache": dict(a.carried_cache) if a.carried_cache is not None else None},
                "body": {"sheltered": covered, "rain": self.weather == "Rain" and not covered,
                         "health": a.health, "pain": a.pain, "unconscious": a.unconscious,
                         **self.hunger_state(a)}}
        if include_memory:
            scene = impression(observation)
            memories = self.visual_memories.get(rid, [])
            observation["visual_memory"] = {"current": scene["sketch"], "count": len(memories),
                                             "capacity": self.memory_capacity, "recalled": recall(memories, scene, self.tick)}
        return observation

    def baseline_action(self, rid):
        """Explicitly scripted, local-observation baseline. Never described as learning."""
        if self.residents[rid].unconscious:
            return {"verb": "rest"}
        observation = self.observe(rid)
        food, water, energy, _ = observation["needs"]
        inventory = observation["inventory"]
        if energy >= 10 and any(b["unconscious"] and abs(b["dx"]) + abs(b["dy"]) == 1 for b in observation["others"]):
            return {"verb": "revive"}
        if food < 75:
            edible = [item for item in ("food", "amber_fruit") if inventory[item]]
            if edible:
                return {"verb": "eat", "item": self.rng.choice(edible)}
        if energy < 45 and food >= 20:
            return {"verb": "rest"}
        seeking = "water" if water < 65 else "food" if food < 80 else None
        if seeking:
            goals = [t for t in observation["tiles"] if
                     (seeking == "water" and t["terrain"] == 1) or
                     (seeking == "food" and t.get("resource", {}).get("kind") in ("berry", "amber_bush") and t["resource"].get("amount", 0) > 0)]
            if goals:
                goal = min(goals, key=lambda t: abs(t["dx"]) + abs(t["dy"]))
                if abs(goal["dx"]) + abs(goal["dy"]) <= 1:
                    return {"verb": "drink" if seeking == "water" else "gather"}
                choices = list(DIRECTIONS)
                self.rng.shuffle(choices)
                choices.sort(key=lambda d: abs(goal["dx"] - DIRECTIONS[d][0]) + abs(goal["dy"] - DIRECTIONS[d][1]))
                for d in choices:
                    dx, dy = DIRECTIONS[d]
                    tile = next(t for t in observation["tiles"] if (t["dx"], t["dy"]) == (dx, dy))
                    if tile.get("structure", {}).get("kind") == "door" and tile.get("blocked"):
                        return {"verb": "toggle_door", "direction": d}
                    if not tile.get("blocked", True):
                        return {"verb": "move", "direction": d}
        # Demonstration farming is scripted. A future learner must discover this tradeoff.
        if food >= 80 and water >= 65:
            sites = [t for t in observation["tiles"] if t["terrain"] == 0
                     and abs(t["dx"]) + abs(t["dy"]) <= 1
                     and not t.get("resource") and not t.get("structure")]
            if sites and inventory["seed"]:
                site = self.rng.choice(sites)
                direction = next((d for d, delta in DIRECTIONS.items() if delta == (site["dx"], site["dy"])), None)
                return {"verb": "plant", **({"direction": direction} if direction else {})}
            if sites and inventory["food"] >= 2 and sum(inventory.values()) < 12:
                return {"verb": "make_seeds"}
        roll = self.rng.random()
        if roll < .10:
            return {"verb": "gather"}
        if roll < .14 and inventory["food"] > 1:
            return {"verb": "give", "item": "food"}
        if roll < .38:
            return {"verb": "rest"}
        # Scripted curiosity: independently remember less-tried visible routes.
        adjacent = {(t["dx"], t["dy"]): t for t in observation["tiles"]}
        doors = {d for d, delta in DIRECTIONS.items()
                 if adjacent.get(delta, {}).get("structure", {}).get("kind") == "door"
                 and not adjacent[delta]["structure"].get("open", False)}
        choices = [t["direction"] for t in observation["touch"] if not t["blocked"] or t["direction"] in doors]
        if not choices:
            return {"verb": "rest"}
        scenes = self.exploration.setdefault(rid, {})
        fingerprint = impression(observation)["fingerprint"]
        if fingerprint not in scenes:
            if len(scenes) >= 2048:
                del scenes[next(iter(scenes))]
            scenes[fingerprint] = {}
        counts = scenes[fingerprint]
        least = min(counts.get(d, 0) for d in choices)
        direction = self.rng.choice([d for d in choices if counts.get(d, 0) == least])
        counts[direction] = counts.get(direction, 0) + 1
        return {"verb": "toggle_door" if direction in doors else "move", "direction": direction}

    def action_tile(self, a, command):
        if "x" in command or "y" in command:
            x, y = command.get("x"), command.get("y")
            if type(x) is not int or type(y) is not int:
                return None
        elif command.get("direction") in DIRECTIONS:
            dx, dy = DIRECTIONS[command["direction"]]
            x, y = a.x + dx, a.y + dy
        else:
            x, y = a.x, a.y
        return (x, y) if self.inside(x, y) and abs(x - a.x) + abs(y - a.y) <= 1 else None

    def apply_action(self, rid, command):
        a = self.residents[rid]
        if a.unconscious:
            return False, "Passed out. Advance time to recover."
        verb = command.get("verb", "rest")
        if verb in ecology.VERBS:
            ok,message=ecology.action(self,rid,command)
            if ok:
                self.revision+=1
                a.action=message
            return ok,message
        if verb == "tap":
            return tap(self, rid, command)
        if verb == "tone":
            return tone(self, rid, command)
        if verb == "make_seeds":
            if not a.inventory["food"]:
                return False, "Need 1 ordinary food to make 2 seeds. Amber fruit cannot be planted."
            if self.capacity(a) < 1:
                return False, "Make one space in your pack: 1 food becomes 2 seeds."
            a.inventory["food"] -= 1
            a.inventory["seed"] += 2
            a.action = "Saving seeds instead of eating"
            self.record(f"{a.name} turned 1 food into 2 seeds.", "farm", rid)
            return True, "Used 1 food to make 2 seeds."
        if verb == "plant":
            tile = self.action_tile(a, command)
            if tile is None:
                return False, "Plant on your tile or one tile beside you."
            x, y = tile
            key = self.key(x, y)
            if self.terrain[y][x] != 0 or key in self.resources or key in self.structures or key in self.caches:
                return False, "Plant on empty grass."
            if not a.inventory["seed"]:
                return False, "Need 1 seed. Make seeds from ordinary food first."
            a.inventory["seed"] -= 1
            self.resources[key] = {"kind": "crop", "amount": 0, "planted": self.tick,
                                   "ready": self.tick + CROP_TICKS}
            if self.ecology_enabled:
                self.soils.setdefault(key, ecology.soil(self,x,y))
                self.resources[key].update(ecological=True,growth=0.,stage=0)
            self.revision += 1
            a.action = "Planting a seed"
            self.record(f"{a.name} planted a seed. Anyone can harvest its fruit.", "farm", rid)
            return True, "Planted. Growth depends on soil and water; anyone can harvest." if self.ecology_enabled else "Planted. 3 food after 2 minutes at 1x; anyone can harvest."
        if verb == "revive":
            if a.energy < 10:
                return False, "Need at least 10 energy to help someone up."
            candidates = [b for b in self.residents.values() if b.id != rid and b.unconscious
                          and abs(b.x-a.x) + abs(b.y-a.y) == 1 and self.visible(a, b.x, b.y)]
            if not candidates:
                return False, "No unconscious neighbor within reach."
            b = candidates[0]
            b.unconscious = False
            b.hunger_faint_until = 0
            b.starvation_ticks = 0
            supportive_food = b.food < 20 and a.inventory["food"] > 0
            if supportive_food:
                previous_food = b.food
                a.inventory["food"] -= 1
                b.food = min(100, b.food + 25)
                self.hunger_transition(b, previous_food)
            b.health = max(30, b.health)
            b.action = "Helped back to their feet"
            b.last_thorn_contact = self.tick
            a.energy -= 10
            a.action = "Helping someone up"
            self.record(f"{a.name} helped {b.name} regain consciousness.", "recovery", a.id)
            return True, f"Helped {b.name} up." + (" Used 1 ordinary food to relieve their hunger (+25 fullness)." if supportive_food else " They still need recovery.")
        if verb == "move":
            direction = command.get("direction")
            if direction not in DIRECTIONS:
                return False, "Choose a direction."
            if self.tick < a.next_move_tick:
                a.action = "Too weak to move yet"
                return False, "Hunger weakness slows movement. Rest or eat while recovering your stride."
            a.facing = direction
            dx, dy = DIRECTIONS[direction]
            x, y = a.x + dx, a.y + dy
            if not self.walkable(x, y) or any(b.x == x and b.y == y for b in self.residents.values()):
                a.action = "Looking for a way through"
                return False, "That way is blocked."
            a.x, a.y = x, y
            a.steps += 1
            discomfort = self.hunger_state(a)["hunger_discomfort"]
            a.energy = max(0, a.energy - .035 * (1 + 2 * discomfort / 100))
            a.next_move_tick = self.tick + (4 if a.food <= 0 else 2 if a.food < 20 else 1)
            a.action = "Walking"
            self.emit_sound(a, "footsteps", 4)
            if self.resources.get(self.key(x, y), {}).get("kind") == "thorns" and self.thorn_contact(a):
                return True, "Thorns hurt you. Move off the plant."
            return True, "Walking"
        if verb in ("rest", "wait"):
            a.energy = min(100, a.energy + .25 * (1 - self.hunger_state(a)["hunger_discomfort"] / 100))
            a.action = "Resting"
            return True, "Resting"
        if verb == "eat":
            item = command.get("item", "food")
            if item not in ("food", "amber_fruit"):
                return False, "Choose food or amber fruit to eat."
            if not a.inventory[item]:
                return False, "Gather the selected food first."
            if a.food > 95:
                return False, "You are already full."
            a.inventory[item] -= 1
            previous_food = a.food
            a.food = min(100, a.food + (12 if item == "amber_fruit" else 25))
            a.starvation_ticks = 0
            self.hunger_transition(a, previous_food)
            if item == "amber_fruit":
                a.amber_eaten += 1
                self.hurt(a, 20, "eating amber fruit")
                return True, "The amber fruit hurts your stomach. Lost 20 health (minimum 1)."
            a.action = "Eating berries"
            self.record(f"{a.name} ate some berries.", actor=rid)
            return True, "A little food goes a long way."
        if verb == "drink":
            if any(self.terrain[y][x] == 1 for x, y in self.nearby(a)):
                a.water = min(100, a.water + 35)
                a.action = "Drinking"
                return True, "Drank from the river."
            return False, "Stand beside the river to drink."
        if verb in ("gather", "pickup"):
            if self.capacity(a) <= 0:
                return False, "Your pack is full. Give or drop something."
            for x, y in self.nearby(a):
                key = self.key(x, y)
                resource = self.resources.get(key)
                if resource and resource["amount"] > 0:
                    if resource["kind"] == "thorns":
                        self.thorn_contact(a)
                        return False, "The thorny plant scratches you when touched. It cannot be gathered."
                    item = {"berry": "food", "tree": "wood", "stone": "stone", "amber_bush": "amber_fruit"}.get(resource["kind"], resource["kind"])
                    a.inventory[item] += 1
                    a.gathered += 1
                    resource["amount"] -= 1
                    if resource["amount"] == 0:
                        del self.resources[key]
                    self.revision += 1
                    a.action = f"Gathering {item}"
                    self.emit_sound(a, "gathering")
                    if rid == "player" or a.gathered % 4 == 1:
                        self.record(f"{a.name} gathered {item}.", "gather", rid)
                    return True, f"Added 1 {item} to your pack."
            return False, "Stand beside a tree, berry bush, rock, or dropped item."
        if verb == "give":
            item = command.get("item", "food")
            if item not in ITEMS or not a.inventory[item]:
                return False, f"You don't have any {item}."
            candidates = [b for b in self.residents.values() if b.id != rid and abs(b.x-a.x)+abs(b.y-a.y) <= 1 and self.capacity(b) > 0]
            target = command.get("target")
            if target:
                candidates = [b for b in candidates if b.id == target]
            if not candidates:
                return False, "Stand next to a resident with room in their pack."
            b = candidates[0]
            a.inventory[item] -= 1
            b.inventory[item] += 1
            a.shared += 1
            a.action = f"Sharing with {b.name}"
            self.record(f"{a.name} gave {item} to {b.name}.", "share", rid)
            return True, f"Gave 1 {item} to {b.name}."
        if verb == "drop":
            item = command.get("item", "food")
            key = self.key(a.x, a.y)
            existing = self.resources.get(key)
            if item not in ITEMS or not a.inventory[item]:
                return False, "You aren't carrying that item."
            if existing and existing["kind"] != item:
                return False, "Move onto an empty tile first."
            self.resources[key] = {"kind": item, "amount": (existing or {}).get("amount", 0) + 1}
            a.inventory[item] -= 1
            self.revision += 1
            a.action = f"Putting down {item}"
            return True, f"Dropped 1 {item}."
        if verb == "build":
            tile = self.action_tile(a, command)
            if tile is not None and self.key(*tile) in self.caches:
                return False, "Move the basket before building here."
            kind = command.get("kind", "shelter")
            if kind not in RECIPES:
                return False, "Choose a construction piece."
            tile = self.action_tile(a, command)
            if tile is None:
                return False, "Build on your tile or one tile beside you."
            x, y = tile
            key = self.key(x, y)
            existing = self.structures.get(key)
            if key in self.resources or (existing and (existing["kind"] != "floor" or kind == "floor")) or self.terrain[y][x] != 0:
                return False, "Build on clear grass or an existing wooden floor."
            if kind in ("wall", "stone_wall", "door") and any((b.x, b.y) == tile for b in self.residents.values()):
                return False, "Someone is standing there. Build beside them."
            recipe = RECIPES[kind]
            if any(a.inventory[item] < amount for item, amount in recipe["cost"].items()):
                return False, recipe["label"] + " needs " + " + ".join(f"{amount} {item}" for item, amount in recipe["cost"].items()) + "."
            for item, amount in recipe["cost"].items():
                a.inventory[item] -= amount
            self.structures[key] = {"kind": kind, "builder": rid, "tick": self.tick}
            if existing:
                self.structures[key]["floor"] = True
            if kind == "door":
                self.structures[key]["open"] = False
            a.action = f"Building {recipe['label'].lower()}"
            self.revision += 1
            self.emit_sound(a, "building")
            self.record(f"{a.name} built a {recipe['label'].lower()}.", "build", rid)
            return True, recipe["label"] + " built."
        if verb in ("toggle_door", "dismantle"):
            tile = self.action_tile(a, command)
            if tile is None:
                return False, "Stand on or beside the structure."
            if verb == "toggle_door" and not any(k in command for k in ("x", "y", "direction")):
                tile = next(((x, y) for x, y in self.nearby(a) if self.structures.get(self.key(x, y), {}).get("kind") == "door"), tile)
            key = self.key(*tile)
            structure = self.structures.get(key)
            if not structure:
                # Old saves can contain exhausted trees and rocks.
                resource = self.resources.get(key)
                if verb == "dismantle" and resource and resource["amount"] == 0 and resource["kind"] != "crop":
                    del self.resources[key]
                    self.revision += 1
                    return True, "Cleared the exhausted resource."
                return False, "There is no structure there."
            if verb == "toggle_door":
                if structure["kind"] != "door":
                    return False, "Choose a door."
                if structure.get("open") and any((b.x, b.y) == tile for b in self.residents.values()):
                    return False, "Someone is in the doorway."
                structure["open"] = not structure.get("open", False)
                a.action = "Opening a door" if structure["open"] else "Closing a door"
                self.emit_sound(a, "door")
                self.revision += 1
                return True, "Door opened." if structure["open"] else "Door closed."
            cost = RECIPES[structure["kind"]]["cost"]
            if self.capacity(a) < sum(cost.values()):
                return False, "Make room in your pack to recover the materials."
            for item, amount in cost.items():
                a.inventory[item] += amount
            if structure.get("floor"):
                self.structures[key] = {"kind": "floor", "builder": structure["builder"], "tick": structure["tick"]}
            else:
                del self.structures[key]
            self.revision += 1
            self.emit_sound(a, "dismantling")
            a.action = "Dismantling"
            self.record(f"{a.name} dismantled a {RECIPES[structure['kind']]['label'].lower()}.", "build", rid)
            return True, "Dismantled. All materials recovered."
        return False, "Unknown action."

    def step(self, commands=None, *, scripted=True):
        commands = commands or {}
        if not scripted:
            missing = {rid for rid in self.residents if rid != "player" and not commands.get(rid)}
            if missing:
                raise ValueError(f"Autonomous residents need explicit controllers: {sorted(missing)}")
        # All policies observe the same pre-transition state; order changes each tick.
        actions = {rid: commands.get(rid) or (self.baseline_action(rid) if rid != "player" else {"verb": "rest"}) for rid in self.residents}
        order = list(self.residents)
        self.rng.shuffle(order)
        results = {rid: self.apply_action(rid, actions[rid]) for rid in order}
        # Finish a new touch response after all simultaneous actions. No movement or compliance.
        for rid, actor in self.residents.items():
            fresh = [e for e in attention(self, rid) if e["tick"] == self.tick]
            if fresh and not actor.unconscious:
                actor.facing = fresh[-1]["bearing"]
                actor.action = "Noticing a tap"
        remember_tones(self)
        self.tick += 1
        expire_attention(self)
        self.sounds = [s for s in self.sounds if self.tick - s["tick"] < sound_lifetime(s["kind"])]
        covered = self.covered_tiles()
        for a in self.residents.values():
            if self.resources.get(self.key(a.x, a.y), {}).get("kind") == "thorns":
                self.thorn_contact(a)
            a.pain = max(0, a.pain - .25)
            if self.tick - a.last_hurt >= 40:
                a.health = min(100, a.health + (.2 if a.unconscious and not a.hunger_faint_until else .04))
            woke_from_hunger = False
            if a.unconscious and a.health >= 30 and (not a.hunger_faint_until or self.tick >= a.hunger_faint_until):
                a.unconscious = False
                if a.hunger_faint_until:
                    woke_from_hunger = True
                    a.energy = max(10, a.energy)
                    a.starvation_ticks = 0
                    a.hunger_faint_until = 0
                a.action = "Waking up"
                a.last_thorn_contact = self.tick  # Four seconds to leave thorns after waking.
                self.record(f"{a.name} woke up.", "recovery", a.id)
            previous_food = a.food
            a.food = max(0, a.food - .008)
            self.hunger_transition(a, previous_food)
            if a.food > 0:
                a.starvation_ticks = 0
            elif not a.unconscious and not woke_from_hunger:
                a.starvation_ticks += 1
                if a.starvation_ticks >= 240:
                    a.unconscious = True
                    a.fainted += 1
                    a.hunger_faint_until = self.tick + 80
                    a.starvation_ticks = 0
                    a.action = "Passed out from hunger"
                    self.record(f"{a.name} passed out from prolonged hunger.", "hunger", a.id)
            a.water = max(0, a.water - .012)
            sheltered = self.key(a.x, a.y) in covered
            change = -.035 if self.weather == "Rain" and not sheltered else .06
            a.warmth = max(0, min(100, a.warmth + change))
        ecology.tick(self)
        for key, resource in list(self.resources.items()):
            if resource["kind"] == "crop" and not resource.get("ecological") and self.tick >= resource["ready"]:
                self.resources[key] = {"kind": "berry", "amount": CROP_YIELD}
                self.revision += 1
        if self.tick % 800 == 0:
            self.record("Rain arrived in the valley." if self.weather == "Rain" else "The valley skies are clear.", "weather")
        if self.tick % 16 == 0:
            for rid in self.residents:
                if self.residents[rid].unconscious:
                    continue
                scene = impression(self.observe(rid, include_memory=False))
                remember(self.visual_memories.setdefault(rid, []), scene, self.tick, self.memory_capacity)
        return results

    def state(self, terrain=False):
        result = {"seed": self.seed, "tick": self.tick, "revision": self.revision, "size": SIZE,
                  "day": 1 + self.tick // DAY_TICKS, "dayProgress": (self.tick % DAY_TICKS) / DAY_TICKS,
                  "weather": self.weather, "residents": [{**asdict(a), **self.hunger_state(a)} for a in self.residents.values()],
                  "resources": self.resources, "structures": self.structures, "events": self.events[-20:],
                  "caches": self.caches, "ecology_enabled": self.ecology_enabled, "irrigation_supply": self.irrigation_supply, "recipes": RECIPES, "covered": sorted(self.covered_tiles()),
                  "economy": {"foodAvailable": sum(r["amount"] for r in self.resources.values() if r["kind"] in ("berry", "food")),
                              "growingCrops": sum(r["kind"] == "crop" for r in self.resources.values()),
                              "cropTicks": CROP_TICKS, "cropYield": CROP_YIELD},
                  "controller": "Scripted survival and private exploration", "learning": False, "mortality": False}
        if terrain:
            result["terrain"] = self.terrain
        return result

    def to_dict(self):
        return {"schema": SCHEMA, "hazard_version": 1, "seed": self.seed, "tick": self.tick, "revision": self.revision,
                "rng": self.rng.getstate(), "terrain": self.terrain, "resources": self.resources,
                "ecology_enabled": self.ecology_enabled, "soils": self.soils, "caches": self.caches, "irrigation_supply": self.irrigation_supply,
                "structures": self.structures, "sounds": self.sounds, "visual_memories": self.visual_memories,
                "exploration": self.exploration, "attention_events": self.attention_events,
                "auditory_memories": self.auditory_memories,
                "memory_capacity": self.memory_capacity, "residents": [asdict(a) for a in self.residents.values()],
                "events": self.events, "event_serial": self.event_serial}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != SCHEMA:
            raise ValueError("Unsupported save version")
        world = cls(data["seed"])
        for key in ("tick", "revision", "terrain", "resources", "structures", "events", "event_serial"):
            setattr(world, key, data[key])
        world.residents = {a["id"]: Resident(**a) for a in data["residents"]}
        for a in world.residents.values():
            for item in ITEMS:
                a.inventory.setdefault(item, 0)
        if not data.get("hazard_version"):
            world.seed_hazards()
        world.attention_events = data.get("attention_events", {})
        expire_attention(world)
        world.sounds = [dict(s) for s in data.get("sounds", [])]
        for sound in world.sounds:
            if sound["kind"].startswith("tone_"):
                old = sound["kind"][5:]
                sound["kind"] = "tone_" + LEGACY_TONES.get(old, old)
        for field in ("ecology_enabled", "soils", "caches", "irrigation_supply"):
            if field in data:setattr(world,field,data[field])
        world.auditory_memories = data.get("auditory_memories", {})
        world.visual_memories = data.get("visual_memories", {})
        world.exploration = data.get("exploration", {})
        world.memory_capacity = data.get("memory_capacity", 64)
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
        world.rng.setstate(tuples(data["rng"]))
        return world

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
