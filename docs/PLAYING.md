# Playing Blissful Ignorance

The app opens directly on the playable map. Choose a portrait to follow a
resident; **Everyone** fits all characters in view, **Through their eyes**
shows the selected character's limited perception, and **Expand map** hides
the inspector until you restore it. Moving your traveler returns the camera
to your character. On narrow screens, arrow buttons also move the traveler.
The **Research** link opens the recorded results; its **The world** link
returns to the same saved population.

## Play locally

Requires **Python 3.11+, a compatible PyTorch installation**, and a modern browser. The host and renderer use Python's standard library and plain JavaScript. PPO resident brains train locally from fresh initialization; no pretrained model download, API key or paid service is needed for a fresh world. Optional Laya residents require existing local model assets. The launcher uses an existing PyTorch environment and does not install dependencies or substitute scripted residents.

On Windows, open a terminal in this folder:

```powershell
.\scripts\Start-World.ps1
```

If needed, pass `-Python` with the path to your existing PyTorch Python executable. The valley starts paused; resume when ready for residents to act and learn.

Or double-click `Start-Blissful-Ignorance.cmd` (`Stop-Blissful-Ignorance.cmd` stops and saves).

Or on any supported Python platform:

```sh
python server.py
```

Open **http://127.0.0.1:8788/**. The Windows launcher opens it automatically unless passed `-NoBrowser`. To stop and save:

```powershell
.\scripts\Stop-World.ps1
```

For a directly launched server, use Ctrl+C. The server pauses when no browser has polled it for three seconds; it is not an always-on background training service.

## Available now

- A seeded 64×64 valley, a river and crossings, eight distinct residents, and a player avatar.
- Movement, local resource gathering, inventory limits, eating, drinking, giving, and dropping.
- Modular wood/stone walls, wooden floors, working doors, shelters, placement previews, and dismantling with material recovery. Walls and closed doors block movement and sight. Fully enclosed floored rooms provide rain cover.
- Each resident's own 9×9 visual patch, relative nearby sounds, contact in four directions, exposure, and private compressed visual memories. The inspector can show the world through that resident's limited view.
- Hunger, thirst, energy, warmth, rain, day/night, and finite shared harvests and seed-based farming. Mortality is disabled for this development milestone.
- Original procedural pixel art, resident inspection, a minimap, camera pan/zoom, and an event journal.
- Pause, one-tick stepping, 1×/5×/20×/maximum speed, achieved-speed readout, and automatic return to 1× on player action.
- Manual save/load, autosaves, and backups before fast-forward or loading. World, each brain's weights, optimizer, recurrent state, rollout and RNG commit together for continuation.

### Controls

| Control | Action |
| --- | --- |
| WASD / arrows | Move |
| E | Gather from current or adjacent tile |
| F | Eat the selected carried food |
| R | Drink beside water |
| H | Help an unconscious adjacent resident up (10 energy) |
| G | Give selected item to an adjacent resident |
| Q | Drop selected item |
| B / Escape | Open / close the construction palette |
| O | Open or close a nearby door |
| Space | Pause / resume when map is focused |
| Drag / scroll | Pan / zoom |

Actions issued while paused are queued for the next step or resume. Click residents in the map or roster to inspect them. The player marker is visual UI only; agent observations contain no player/creator flag.

Choose **Build**, select a piece, and click your tile or an adjacent tile. The keyboard-accessible **Place ahead** and **At your feet** buttons use the same action. Wood walls cost 2 wood, stone walls 2 stone, floors 1 wood, doors 3 wood, and shelters 4 wood + 2 stone. Floors can sit under another piece; removing that piece leaves the floor. Dismantling needs pack space for all recovered materials. Gather trees/rocks completely to clear them; Remove also clears exhausted resources left by older saves.

Enclose a connected floor area with walls and doors to form a roofed room, up to 64 floor tiles per room. The roof is an automatic construction rule; a lone wall or exposed floor does not provide shelter. Select a resident and choose **Through their eyes** to inspect its current view. The minimap is hidden in this mode; the journal and needs inspector remain observer tools, never agent inputs.

## Hazards and recovery

Amber bushes provide fruit that costs 20 health when eaten; gathering it is safe. Thorns cost 8 health on contact, repeating every four simulated seconds. At 20 health or lower, inhabitants pass out and cannot act. They recover and wake at 30 health. Mortality stays off. Choose Food or Amber fruit in the item selector before eating. See [mechanics and evaluation](HAZARDS.md).

## Learning development

`agents/network.py` defines the shared encoder and primitive-action vocabulary for independent recurrent controllers. The sequence backend adds each resident's own training, consequence predictor and complete checkpoints; its initial recurrent width is 64. Capacity is configurable, not a permanent identity constraint. See [the learning architecture proposal](LEARNING-ARCHITECTURE.md), [sequence learning](SEQUENCE-LEARNING.md) and [senses and visual memory](SENSES-AND-MEMORY.md).

With the compatible PyTorch runtime:

```sh
python -m experiments.benchmark --quick
```

The benchmark reports inference timing, rendering-independent simulation throughput, and a **synthetic** recurrent PPO optimizer smoke test. Parameter changes on random data are not evidence of useful learning. No pretrained model weights are bundled.

## Verification

```sh
python -m unittest discover -s tests -v
node --check web/app.js
```

Run the suite in the PyTorch environment. Node is only needed for the optional JavaScript syntax check. Tests cover deterministic replay, resource conservation, observation boundaries, collision handling, independent learning, save continuation, server controls, and graceful lifecycle behavior.

## Project notes

- [Implementation status and next milestone](STATUS.md)
- [Individual senses and lossy visual memory](SENSES-AND-MEMORY.md)
- [Architecture research, fruit-fly brains, and lifelong learning](ARCHITECTURE-RESEARCH.md)
- [Matched PPO and private recurrent replay comparison](REPLAY-COMPARISON.md)
- [Small private learners and embodied swarm research](SMALL-AGENT-SWARMS-RESEARCH.md)
- [Private consequence retention and exploration experiment](RETENTION-EXPLORATION.md)
- [Parallel environment interface and compatibility checks](PARALLEL-ENVIRONMENT.md)
- [Standardized PPO learning curve](LEARNING-CURVE.md)
- [Recurrent working-context diagnostic](MEMORY-CONTINUITY.md)
- [Short versus long training lives](LIFE-LENGTH.md)
- [Approved rebuild direction](REBUILD-PROPOSAL.md)
- [ALIEN local installation and evaluation](ALIEN-EVALUATION.md)
- [NVIDIA and AMD comparison](VENDOR-ARCHITECTURES.md)
- [Chinese research/framework comparison](CHINESE-ARCHITECTURES.md)
- [Observed hardware benchmarks and NPU status](HARDWARE.md)
- [Publication and Hugging Face plan](PUBLISHING.md)

World saves, checkpoints, development machine settings, and private planning notes are excluded from Git. The code and original procedural graphics are offered under [Apache-2.0](../LICENSE). Research links do not imply that linked projects' code, models, or data are included or covered by this project's license.

### Farming

Press **T** to turn one ordinary food into two seeds, **P** to plant on empty grass at your feet, and **E** to harvest. New crops in the live ecology depend on soil fertility and moisture and yield two or three food; older crops retain their original timers. Anyone can harvest; wild bushes no longer refill. See [current resource rules](LIVING-POPULATION.md#resource-rules) and the [original farming milestone](RESOURCE-ECONOMY.md). Laya remains a distinct persistent model alongside the live PPO residents.

The [farming learning pilot](FARMING-TRIAL.md) reports a trained tabular learner, untrained/random controls, scripted reference and permanent Laya track in a fixed shared-plot task. These experimental controllers are separate from live residents.

Follow-up: [longer lives with movement and independent competitors](MOBILE-FARMING-TRIAL.md), plus [explicit-recipe frozen Laya](LAYA-RECIPE-TRIAL.md). These experiments retain negative results and do not switch live resident controllers.

### Hunger and curiosity

[Hunger and curiosity](MOTIVATION.md) describes internal hunger distress, reduced recovery, weakness and nonfatal starvation fainting. Eating provides relief. The live residents receive their own bodily and sensory inputs and choose actions through their individual models. The linked report also records an earlier scripted exploration baseline and a separate novelty-reward experiment; neither describes the current live controller. Useful curiosity and reliable survival remain under evaluation.

Taps and tones: **J** taps an adjacent conscious neighbor; choose one of ten tones (0–9) and press **K** to emit a local signal. See [social signals](SOCIAL-SIGNALS.md), [neural trial results](NEURAL-FARMING-TRIAL.md), and [experimental network growth](EVOLUTION.md).

[Resource ecology and persistent learning](LIVING-POPULATION.md): irrigation, soil and movable food baskets, plus the historical first persistent-learning prototype. The main valley now uses [independent resident controllers](UNSCRIPTED-POPULATION.md).

[Sequence learning and development worlds](SEQUENCE-LEARNING.md): independent recurrent PPO, private consequence prediction and an observable learning record. Run `scripts/Start-Development.ps1` to open the separate local population on port 8790. Its training and later farming/social competence remain subject to experimental acceptance checks.

[Scarcity and safe routes](SCARCITY-ROUTES.md): finite starting food, a thorny shortcut and a longer safe path, with paired initial-policy controls and a separate trace audit. For a fresh paused playable arena, run `scripts/Start-Development.ps1 -RunName scarcity-development -Stage scarcity -Port 8791`. Existing directories resume their saved inhabitants; trial-trained weights are not automatically imported.
