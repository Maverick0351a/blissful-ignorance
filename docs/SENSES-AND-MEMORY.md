# Individual senses and visual memory

Implemented October 4, 2026. These are simulation inputs and a first memory mechanism, not a claim of consciousness or learned understanding.

## One perspective per resident

`World.observe(resident_id)` constructs the same input contract separately for every resident, including the player. It provides:

- **Sight:** a relative 9×9 tile patch. Trees, stone, walls and closed doors occlude tiles and creatures behind them. Two solid corner tiles prevent diagonal peeking. The blocking tile itself remains visible. This initial view is omnidirectional within four tiles, not a facing-dependent vision cone.
- **Hearing:** recent nearby footsteps, gathering, building, doors and dismantling. Signals have a coarse bearing and strength, with distance loss and occlusion attenuation. Footsteps reach four tiles; other sounds reach six. They fade over eight ticks. These are simulated environmental sounds, not microphone recordings or spoken-language understanding.
- **Touch:** whether each immediately adjacent cardinal tile is blocked by terrain, a solid object, or another resident.
- **Internal state:** own fullness, hydration, energy, warmth, carried materials and orientation, plus whether the body is under cover or exposed to rain.
- **Visual recall:** up to three approximate matches from that resident's own prior impressions.

Observation dictionaries contain no absolute coordinates, global journal, other creatures' needs or inventory, structure-builder identity, player/creator flag, future state, or hidden resources. The local browser is an observer tool and receives world state separately to draw the game. Its wrapper supplies a display origin; that wrapper is never a neural input. Through their eyes hides unseen surroundings and the global minimap. Inspector/journal information is for the human viewer.

Historical note: the October 4 first implementation used a scripted baseline
and an untrained optional neural encoder. The current playable population uses
eight independent recurrent PPO learners and a distinct Laya NPU controller.
PPO receives encoded local senses and recall; Laya receives a compressed subset
that omits the visual recall sketches. Providing these inputs does not prove
that a controller has learned to use memory. See [current status](STATUS.md)
for deployment and behavioral evidence.

## Compressed visual impressions

The first implementation avoids storing a full-resolution screenshot on every frame. It reconstructs a small visual impression only from the creature's visible patch, then quantizes it to a shared 16-color palette:

1. **9×9 pixels, four bits per pixel:** 41 bytes, padded at the final nibble. This keeps coarse color/layout cues while discarding detailed sprites, exact colors, resource counts, identities and coordinates. It is an abstract thumbnail of their view, not a browser screenshot or perfect photographic memory.
2. **64-bit difference hash:** eight additional bytes describe neighboring brightness changes.
3. **Local episode metadata:** formation tick, last-seen tick, and repeat count. The surrounding save format adds storage overhead beyond the raw 49 visual bytes.
4. **Similarity retrieval:** 80% palette-layout agreement across nonempty view locations plus 20% hash agreement. Return the top three matches above 0.35. A score is a heuristic resemblance, not calibrated confidence or proof that two scenes are the same place.

Every 16 ticks (four simulated seconds at normal speed), the resident considers storing its view. A match of at least 0.93 refreshes an existing impression; a sufficiently different scene creates a new one. A full collection evicts the least recently seen impression. Each resident's collection is separate and saved with the world. Fast-forward advances this process by simulation ticks; merely opening the inspector does not create memories.

The default budget is 64 impressions per resident. `World(memory_capacity=...)` and the save's `memory_capacity` field configure it. That is a resource budget, not a permanent intelligence limit. Reproduction and inheritance are not implemented, so no memories are currently passed between generations.

## What this can and cannot establish

A resident can now retain approximate visual episodes and retrieve scenes that resemble its present view. Feature extraction and matching are hand-designed. Similar layouts can collide; a moved camera center or changed foliage can lower similarity. Spatial relations, route planning, object permanence, identity recognition, and usefulness in decision-making still require learning/evaluation.

Recall merging refreshes when an impression was seen but keeps the original sketch; small scene changes can therefore remain represented by an older approximate impression until they exceed the novelty threshold. Hearing currently keeps the last eight audible events, so a newer quiet sound can displace an older stronger one. Salience-based sound selection and adaptive visual refresh remain possible improvements.

The next memory experiment should compare agents with no recall, this fixed visual sketch, and a small visual encoder trained only on that individual's experiences. Test revisiting locations after resource changes, confusingly similar scenes, retention after a new task, and whether recall improves behavior on held-out maps. A learned embedding can replace the sketch behind a versioned memory interface; larger disk-backed stores and associative cue/outcome memory can follow measured benefits. No pretrained language or vision model is needed for this initial design.

Schema 3 adds personal health, pain and unconscious state, nearby cries, amber bushes/fruit and thorn visual cues. New visual impressions stop forming while unconscious. Hazard names do not encode a safe/unsafe instruction. See [hazards](HAZARDS.md).
