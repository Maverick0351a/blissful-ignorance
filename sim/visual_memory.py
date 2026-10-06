"""Lossy visual impressions built ONLY from a resident's visible sensory patch.

Each scene is 9x9 four-bit palette values (41 bytes), plus a 64-bit difference
hash. It discards exact sprites, amounts, colors, identities and coordinates.
Similarity is a retrieval heuristic, never a probability of being in one place.
"""
PALETTE = ("#142b29", "#85965d", "#699d99", "#bbae7e", "#a28759", "#60744f",
           "#456945", "#8c5871", "#91977f", "#b59663", "#cdc087", "#756048",
           "#cfbb97", "#9d9272", "#6e5940", "#b5a26a")


def pack(pixels):
    values = list(pixels) + [0]
    return bytes((values[i] << 4) | values[i + 1] for i in range(0, 81, 2)).hex()


def unpack(sketch):
    raw = bytes.fromhex(sketch)
    if len(raw) != 41:
        raise ValueError("Expected a 41-byte visual impression")
    return [n for byte in raw for n in (byte >> 4, byte & 15)][:81]


def impression(observation):
    pixels = [0] * 81
    visible = set()
    for tile in observation["tiles"]:
        dx, dy = tile["dx"], tile["dy"]
        if tile["terrain"] < 0 or not (-4 <= dx <= 4 and -4 <= dy <= 4):
            continue
        visible.add((dx, dy))
        color = (1, 2, 3, 4, 5)[tile["terrain"]]
        resource = tile.get("resource", {})
        color = {"tree": 6, "berry": 7, "stone": 8, "wood": 9, "food": 7, "seed": 10}.get(resource.get("kind"), color)
        color = {"amber_bush": 10, "amber_fruit": 10, "thorns": 14, "crop": 10}.get(resource.get("kind"), color)
        structure = tile.get("structure", {})
        color = {"wall": 11, "stone_wall": 13, "floor": 9, "door": 9 if structure.get("open") else 14,
                 "shelter": 15}.get(structure.get("kind"), color)
        pixels[(dy + 4) * 9 + dx + 4] = color
    for other in observation.get("others", ()):
        dx, dy = other["dx"], other["dy"]
        if (dx, dy) in visible:
            pixels[(dy + 4) * 9 + dx + 4] = 12
    # The resident's own silhouette is omitted so it does not dominate matching.
    luminance = [sum(int(color[i:i+2], 16) * weight for i, weight in ((1, .2126), (3, .7152), (5, .0722))) for color in PALETTE]
    fingerprint = 0
    for y in range(8):
        for x in range(8):
            fingerprint = (fingerprint << 1) | int(luminance[pixels[y * 9 + x]] > luminance[pixels[y * 9 + x + 1]])
    return {"sketch": pack(pixels), "fingerprint": f"{fingerprint:016x}"}


def similarity(first, second):
    a, b = unpack(first["sketch"]), unpack(second["sketch"])
    union = [i for i in range(81) if a[i] or b[i]]
    if len(union) < 9:
        return 0.0
    agreement = sum(a[i] == b[i] for i in union) / len(union)
    hash_match = 1 - (int(first["fingerprint"], 16) ^ int(second["fingerprint"], 16)).bit_count() / 64
    return round(.8 * agreement + .2 * hash_match, 3)


def remember(memories, scene, tick, capacity):
    nearest = max(memories, key=lambda m: similarity(scene, m), default=None)
    if nearest is not None and similarity(scene, nearest) >= .93:
        nearest["last_seen"] = tick
        nearest["visits"] += 1
        return
    memories.append({**scene, "formed": tick, "last_seen": tick, "visits": 1})
    if len(memories) > capacity:
        memories.remove(min(memories, key=lambda m: m["last_seen"]))


def recall(memories, scene, tick):
    matches = sorted(((similarity(scene, memory), memory) for memory in memories), key=lambda pair: pair[0], reverse=True)
    return [{"sketch": m["sketch"], "similarity": score, "age": max(0, tick - m["last_seen"]), "visits": m["visits"]}
            for score, m in matches[:3] if score >= .35]
