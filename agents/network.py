"""Optional untrained resident policy; no privileged simulation state is consumed.

Each Controller owns separate parameters and recurrent memory. This is an
inference/training scaffold, not a learning controller connected to the game.
"""
from __future__ import annotations

import math
from sim.ecology import VERBS as ECOLOGY_VERBS
from sim.social import TONES, AUDITORY_CAPACITY, AUDITORY_WINDOW

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("The optional resident network requires an existing PyTorch installation.") from exc

ITEMS = ("food", "wood", "stone", "seed", "amber_fruit")
RESOURCE_KINDS = ("tree", "berry", "stone", "food", "wood", "seed")
# Six terrain channels (unknown plus five terrain types), six resource kinds,
# resource amount, shelter, local occupancy, three visible resident colors,
# five structure states, and three recalled lossy scenes.
STRUCTURE_KINDS = ("wall", "stone_wall", "floor", "closed_door", "open_door")
BEARINGS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest", "here")
SOUND_KINDS = ("footsteps", "gathering", "building", "door", "dismantling", "cry") + tuple("tone_" + t for t in TONES)
DIRECTIONS = ("north", "east", "south", "west")
PATCH_CHANNELS = 36  # Local vision, recall, hazard cues, unconscious neighbors and crop growth.
FEATURES = 9 + 4 + len(BEARINGS) * len(SOUND_KINDS) + 4 + 7 + 3 + 4 + AUDITORY_CAPACITY * (len(TONES) + len(BEARINGS) + 3) + 3
HIDDEN_SIZE = 128
ACTIONS = (
    {"verb": "move", "direction": "north"},
    {"verb": "move", "direction": "east"},
    {"verb": "move", "direction": "south"},
    {"verb": "move", "direction": "west"},
    {"verb": "rest"}, {"verb": "gather"}, {"verb": "eat"}, {"verb": "drink"},
    {"verb": "give", "item": "food"}, {"verb": "drop", "item": "food"},
    {"verb": "build"}, {"verb": "give", "item": "wood"},
    {"verb": "give", "item": "stone"}, {"verb": "drop", "item": "wood"},
    {"verb": "drop", "item": "stone"},
) + tuple({"verb": "build", "kind": kind, "direction": direction}
          for kind in ("wall", "stone_wall", "floor", "door") for direction in DIRECTIONS) + (
    {"verb": "build", "kind": "floor"},
) + tuple({"verb": verb, "direction": direction}
          for verb in ("toggle_door", "dismantle") for direction in DIRECTIONS) + ({"verb": "dismantle"},) + tuple(
              {"verb": verb, "item": "amber_fruit"} for verb in ("eat", "give", "drop")) + ({"verb": "revive"}, {"verb": "make_seeds"}, {"verb": "plant"}) + tuple(
    {"verb": "plant", "direction": d} for d in DIRECTIONS) + ({"verb": "tap"},) + tuple({"verb": "tone", "tone": pitch} for pitch in TONES) + tuple({"verb": verb} for verb in ECOLOGY_VERBS)


def _unit(value, scale):
    value = float(value)
    return max(0.0, min(1.0, value / scale)) if math.isfinite(value) else 0.0


def encode_observation(observation, device="cpu"):
    """Encode local vision, hearing, touch, own body and private visual recall.

    Absolute coordinates, resident IDs, player status, world tick, unseen tiles,
    and global resources are neither required nor read. Unknown tiles carry no
    resource/structure/occupancy information, even if malformed data adds it.
    """
    patch = torch.zeros((1, PATCH_CHANNELS, 9, 9), dtype=torch.float32)
    patch[:, 0] = 1  # Missing observations are unknown, not empty grass.
    visible = set()
    for tile in observation.get("tiles", ()):
        dx, dy = int(tile["dx"]), int(tile["dy"])
        if not (-4 <= dx <= 4 and -4 <= dy <= 4):
            continue
        row, col = dy + 4, dx + 4
        terrain = int(tile.get("terrain", -1))
        if terrain not in range(5):
            continue
        visible.add((dx, dy))
        row, col = dy + 4, dx + 4
        patch[0,32,row,col]=_unit(tile.get("soil",{}).get("fertility",0),1)
        patch[0,33,row,col]=_unit(tile.get("soil",{}).get("moisture",0),1)
        patch[0,34,row,col]=float(bool(tile.get("cache")))
        patch[0,35,row,col]=_unit(tile.get("cache",{}).get("food",0),12)
        patch[0, 0, row, col] = 0
        patch[0, terrain + 1, row, col] = 1
        resource = tile.get("resource", {})
        kind = resource.get("kind")
        if kind in RESOURCE_KINDS:
            patch[0, 6 + RESOURCE_KINDS.index(kind), row, col] = 1
            patch[0, 12, row, col] = _unit(resource.get("amount", 0), 12)
        if kind in ("amber_bush", "amber_fruit", "thorns"):
            patch[0, 26 + ("amber_bush", "amber_fruit", "thorns").index(kind), row, col] = 1
            patch[0, 12, row, col] = _unit(resource.get("amount", 0), 12)
        if kind == "crop":
            patch[0, 30, row, col] = 1
            patch[0, 31, row, col] = _unit(resource.get("stage", 0), 2)
        patch[0, 13, row, col] = float(bool(tile.get("shelter", False)))
        structure = tile.get("structure", {})
        kind = structure.get("kind")
        if kind == "door":
            kind = "open_door" if structure.get("open", False) else "closed_door"
        if kind in STRUCTURE_KINDS:
            patch[0, 18 + STRUCTURE_KINDS.index(kind), row, col] = 1
        if structure.get("floor"):
            patch[0, 20, row, col] = 1
    for other in observation.get("others", ()):
        dx, dy = int(other["dx"]), int(other["dy"])
        if (dx, dy) not in visible:
            continue
        row, col = dy + 4, dx + 4
        patch[0, 14, row, col] = 1
        patch[0, 29, row, col] = float(bool(other.get("unconscious")))
        color = str(other.get("color", "")).lstrip("#")
        if len(color) == 6:
            try:
                for channel in range(3):
                    patch[0, 15 + channel, row, col] = int(color[2 * channel:2 * channel + 2], 16) / 255
            except ValueError:
                pass
    needs = list(observation.get("needs", ()))
    if len(needs) != 4:
        raise ValueError("Expected four local needs.")
    inventory = observation.get("inventory", {})
    values = [_unit(value, 100) for value in needs] + [_unit(inventory.get(item, 0), 12) for item in ITEMS]
    contact = {t["direction"]: bool(t["blocked"]) for t in observation.get("touch", ())}
    values += [float(contact.get(direction, False)) for direction in DIRECTIONS]
    auditory = [0.0] * (len(BEARINGS) * len(SOUND_KINDS))
    for sound in observation.get("hearing", ()):
        if sound.get("kind") in SOUND_KINDS and sound.get("bearing") in BEARINGS:
            index = SOUND_KINDS.index(sound["kind"]) * len(BEARINGS) + BEARINGS.index(sound["bearing"])
            auditory[index] = max(auditory[index], _unit(sound.get("strength", 0), 1))
    values += auditory
    body = observation.get("body", {})
    values += [float(bool(body.get(key))) for key in ("sheltered", "rain", "unconscious")]
    values += [_unit(body.get("health", 100), 100), _unit(body.get("pain", 0), 100)]
    values += [_unit(body.get("hunger_discomfort", 0), 100), _unit(body.get("starvation_strain", 0), 100)]
    values += [float(observation.get("facing") == direction) for direction in DIRECTIONS]
    from sim.visual_memory import unpack
    recalled = observation.get("visual_memory", {}).get("recalled", [])[:3]
    for index in range(3):
        values.append(_unit(recalled[index]["similarity"], 1) if index < len(recalled) else 0)
        if index < len(recalled):
            patch[0, 23 + index] = torch.tensor(unpack(recalled[index]["sketch"]), dtype=torch.float32).reshape(9, 9) / 15
    signals = observation.get("attention", ())
    values += [float(any(e.get("bearing") == d for e in signals)) for d in DIRECTIONS]
    # Oldest-to-newest slots preserve repeated symbols, gaps and direction.
    history = observation.get("auditory_memory", ())[-AUDITORY_CAPACITY:]
    for slot in range(AUDITORY_CAPACITY):
        event = history[slot] if slot < len(history) else {}
        values += [float(event.get("tone") == t) for t in TONES]
        values += [float(event.get("bearing") == b) for b in BEARINGS]
        values += [float(bool(event)), _unit(event.get("age", 0), AUDITORY_WINDOW), _unit(event.get("strength", 0), 1)]
    ecology = observation.get("ecology", {})
    basket = ecology.get("carried_cache")
    values += [_unit(ecology.get("water_carried",0),3), float(basket is not None), _unit((basket or {}).get("food",0),12)]
    features = torch.tensor([values], dtype=torch.float32)
    return patch.to(device), features.to(device)


class ResidentNetwork(nn.Module):
    """Small encoder with configurable recurrent capacity; 128 is the baseline."""
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(PATCH_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 5 * 5, 64), nn.ReLU(),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(FEATURES, 16), nn.ReLU())
        self.recurrent = nn.LSTMCell(80, hidden_size)
        self.actor = nn.Linear(hidden_size, len(ACTIONS))
        self.value = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size=1):
        parameter = next(self.parameters())
        zeros = parameter.new_zeros((batch_size, self.hidden_size))
        return zeros, zeros.clone()

    def forward(self, patch, features, state=None):
        encoded = torch.cat((self.patch_encoder(patch), self.feature_encoder(features)), dim=-1)
        if state is None:
            state = self.initial_state(patch.shape[0])
        hidden, cell = self.recurrent(encoded, state)
        return self.actor(hidden), self.value(hidden).squeeze(-1), (hidden, cell)


class Controller:
    """Independent untrained model and private recurrent state for one resident.

    The caller must pass World.observe(resident_id), never World.state(). No
    parameter sharing or pretrained checkpoint is performed by this class.
    """
    trained = False
    label = "Untrained recurrent policy (experimental, disconnected from game)"

    def __init__(self, seed=0, device="cpu", hidden_size=HIDDEN_SIZE):
        # Isolate initialization from the caller's global CPU random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = ResidentNetwork(hidden_size=hidden_size).to(device).eval()
        self.device = torch.device(device)
        self.state = self.model.initial_state()
        self.decisions = 0

    def manifest(self):
        """Architecture metadata; not a full trained-state checkpoint or migration."""
        return {"backend": "conv-lstm", "schema": 9, "observation_schema": 9,
                "action_schema": 8, "hidden_size": self.model.hidden_size,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trained": False}

    def reset(self):
        self.state = self.model.initial_state()
        self.decisions = 0

    @torch.inference_mode()
    def decide(self, observation):
        patch, features = encode_observation(observation, self.device)
        logits, value, self.state = self.model(patch, features, self.state)
        self.decisions += 1
        index = int(logits.argmax(dim=-1).item())
        return dict(ACTIONS[index]), {"value": float(value.item()), "trained": False}
