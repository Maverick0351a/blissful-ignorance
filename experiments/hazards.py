"""Bounded offline hazard experiments; never loads or saves a game world.

Run from the project root: python experiments/hazards.py
Only writes runs/hazard-experiments/{preregistration,results}.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.world import Resident, SIZE, World

SEEDS = [713, 1729, 20261004]
PLAN = {
    "seeds": SEEDS,
    "hypotheses": [
        "With both foods replenished, scripted amber choices will not systematically decline after injury.",
        "An adjacent helper shortens unconscious time; help does not remove the contact hazard.",
        "An idle revived body on thorns can faint repeatedly, unlike a body moved to safety.",
        "Short unmodified worlds may provide too little injury exposure to assess even baseline behavior.",
    ],
    "food": "80 choices per seed; replenish both foods, reset fullness to 50, allow natural recovery before each choice; retain pain, health, counters and memories.",
    "rescue": "Start at health 1 after controlled damage. Compare alone, adjacent helper, distant helper; wait on thorns versus move east once awake. Helpers remain in place and run the existing baseline help rule, otherwise rest. Horizon 400 ticks.",
    "population": "Unmodified generated worlds, eight scripted residents plus idle player, 600 ticks per seed. Count local visible-hazard resident-ticks and food choice opportunities separately from injury events.",
    "limits": "No neural controller or learning, no live world access, no downloads; roughly two minutes compute ceiling.",
}


def flat(seed):
    w = World(seed)
    w.terrain = [[0] * SIZE for _ in range(SIZE)]
    w.resources.clear()
    w.structures.clear()
    w.residents = {"player": w.residents["player"]}
    a = w.residents["player"]
    a.x, a.y = 20, 20
    return w, a


def food_choices(seed):
    w, a = flat(seed)
    choices = []
    full_before = 0
    for _ in range(80):
        while a.unconscious:
            w.step()
        a.food = 50
        a.inventory["food"] = a.inventory["amber_fruit"] = 1
        action = w.baseline_action(a.id)
        assert action["verb"] == "eat"
        full_before += a.food
        result = w.step({a.id: action})[a.id]
        assert result[0], result
        choices.append(action["item"])
    return {"seed": seed, "opportunities": len(choices),
            "amber_first_40": choices[:40].count("amber_fruit"),
            "amber_last_40": choices[40:].count("amber_fruit"),
            "amber_total": a.amber_eaten, "faints": a.fainted,
            "actual_damage": round(a.damage_taken, 3), "ticks": w.tick,
            "choices": choices}


def rescue(seed, helper_distance, move_safe):
    w, a = flat(seed)
    w.resources["20,20"] = {"kind": "thorns", "amount": 1}
    helper = None
    if helper_distance is not None:
        helper = Resident("helper", "Helper", 20-helper_distance, 20, "#ffffff")
        w.residents[helper.id] = helper
    w.hurt(a, 99, "controlled starting injury")
    first_awake = None
    down_ticks = revives = 0
    for _ in range(400):
        down_ticks += int(a.unconscious)
        commands = {a.id: {"verb": "move", "direction": "east"}
                    if move_safe and not a.unconscious and a.x == 20 else {"verb": "rest"}}
        if helper:
            action = w.baseline_action(helper.id)
            commands[helper.id] = action if action["verb"] == "revive" else {"verb": "rest"}
        result = w.step(commands)
        if helper and commands[helper.id]["verb"] == "revive" and result[helper.id][0]:
            revives += 1
        if not a.unconscious and first_awake is None:
            first_awake = w.tick
    return {"seed": seed, "helper_distance": helper_distance, "move_safe": move_safe,
            "first_awake_tick": first_awake, "unconscious_tick_starts": down_ticks,
            "revives": revives, "faints_including_initial": a.fainted,
            "thorn_contacts": a.thorn_contacts, "health_final": round(a.health, 3),
            "helper_energy_final": round(helper.energy, 3) if helper else None}


def population(seed):
    w = World(seed)
    metrics = {"seed": seed, "ticks": 600, "agents": 8,
               "thorn_visible_resident_ticks": 0, "amber_visible_resident_ticks": 0,
               "food_choice_opportunities": 0, "both_food_choice_opportunities": 0,
               "amber_choices": 0, "safe_choices": 0, "successful_revives": 0}
    for _ in range(600):
        commands = {}
        for rid, a in w.residents.items():
            if rid == "player":
                continue
            observation = w.observe(rid)
            kinds = {t.get("resource", {}).get("kind") for t in observation["tiles"]}
            metrics["thorn_visible_resident_ticks"] += int("thorns" in kinds and not a.unconscious)
            metrics["amber_visible_resident_ticks"] += int("amber_bush" in kinds and not a.unconscious)
            action = w.baseline_action(rid)
            commands[rid] = action
            if action["verb"] == "eat" and not a.unconscious:
                metrics["food_choice_opportunities"] += 1
                metrics["both_food_choice_opportunities"] += int(bool(a.inventory["food"] and a.inventory["amber_fruit"]))
                metrics["amber_choices" if action["item"] == "amber_fruit" else "safe_choices"] += 1
        results = w.step(commands)
        metrics["successful_revives"] += sum(action["verb"] == "revive" and results[rid][0] for rid, action in commands.items())
    residents = [a for rid, a in w.residents.items() if rid != "player"]
    metrics.update(thorn_contacts=sum(a.thorn_contacts for a in residents),
                   faints=sum(a.fainted for a in residents),
                   actual_damage=round(sum(a.damage_taken for a in residents), 3),
                   mean_health=round(sum(a.health for a in residents)/len(residents), 3),
                   gathered=sum(a.gathered for a in residents),
                   mean_food=round(sum(a.food for a in residents)/len(residents), 3))
    return metrics


def main():
    output = ROOT / "runs" / "hazard-experiments"
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(PLAN, source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [ROOT / "sim/world.py", Path(__file__).resolve()]})
    (output / "preregistration.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    start = time.perf_counter()
    result = {"food": [food_choices(seed) for seed in SEEDS],
              "rescue": [rescue(seed, distance, move) for seed in SEEDS
                         for distance in (None, 1, 2) for move in (False, True)],
              "population": [population(seed) for seed in SEEDS]}
    result["elapsed_seconds"] = round(time.perf_counter()-start, 3)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "food": [{k:v for k,v in row.items() if k != "choices"} for row in result["food"]]}, indent=2))


if __name__ == "__main__":
    main()
