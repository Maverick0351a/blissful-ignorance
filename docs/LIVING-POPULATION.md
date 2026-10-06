# Persistent population and resource ecology

**Current main game:** http://127.0.0.1:8788/ uses independent resident models.
The saved population contains eight recurrent PPO learners and a distinct
local NPU Laya. Each PPO learner updates private weights and experience;
Laya uses frozen pretrained weights and her own saved context. See
[the controller transition](UNSCRIPTED-POPULATION.md) and
[current population record](POPULATION-WATCH.md). Ecology is enabled in the
preserved save; existing crops keep their original timers.

## Historical v0.8 prototype

At v0.8, the main game still used scripted residents. A separate experimental
world on port 8789 introduced online learning for Moss and Pip alongside six
scripted neighbors. The descriptions, settings and results in this section
belong to that older prototype, not the current main population.

Its launcher is `scripts/Start-Learning.ps1`, with `scripts/Stop-Learning.ps1`
for saving and stopping. It uses the existing laya-lab Python environment;
pass `-Python` to select another existing PyTorch environment. The service
binds to loopback and pauses without browser heartbeats. Its last recorded
test ended paused; this document does not assert that port 8789 is running now.

For a finite run, use the existing Python environment to run `experiments/living_population.py --directory runs/my-learning-trial --ticks 2400`. A reused directory resumes its checkpoint; use a new directory for a fresh run. Only directories within project runs are accepted.

### Historical learning and evidence

Each learner has independent convolutional weights, a 32-unit recurrent state, actor/value heads, Adam optimizer, action RNG and pending transition. Inputs come only from World.observe: local vision, resources, soil, baskets, private memories and own body. It chooses among the general primitive actions every four ticks, including ten tones and ecology actions. It has no learned route planner or hierarchical skills system.

Online actor-critic settings: gamma .99, learning rate .0003, entropy coefficient .005, gradient norm cap 1. Food relief earns reward; hunger distress and injury incur costs. No reward is given merely for farming, building, helping or talking. Entropy encourages exploration. Explicit curiosity prediction, a learned world model, relationships and reproduction are not implemented here. Learners start with 50 fullness, 2 food and 3 wood; scripted residents use normal starts, so this is not a matched architecture comparison.

Atomic checkpoints contain world, weights, optimizers, recurrent states, action RNGs, pending transitions, accumulated rewards and metrics. They load through the restricted tensor loader. Mid-transition save/load testing reproduced subsequent decisions, world state and model parameters exactly.

The 2,400-tick functional run took 25.58 seconds. Each learner made 600 decisions and 599 updates. Moss had 462 invalid choices (77.0%); Pip had 466 (77.7%). Neither ate food. Neither reached zero food or unconsciousness, but starting fullness alone covers this short horizon. This demonstrates working updates and persistence, not useful survival behavior. Action confidence calibration was not measured. No learner was promoted into the main world.

Evidence: `runs/learning-population/latest-report.json` and combined checkpoints under `runs/learning-population/runs/`. Interactive browser testing subsequently reached tick 2422 and successfully restored that tick and 605 updates from a manual checkpoint. Reports describe the fixed initial run; interactive use continues learning.

The next task at that milestone was to filter physically impossible choices,
then compare survival on longer lives and fresh seeds. Those follow-ups are
recorded in [physical actions](PHYSICAL-ACTIONS.md) and later experiment
reports. Lower invalid-action counts alone do not establish better survival.

## Resource rules

- Initial soil fertility is deterministic, .50 to 1.00; moisture starts at .70. Stored soil state is created on cultivation or watering. Untouched soil uses initial values; this is a simplified soil model.
- New crops need 480 growth units. With moisture above .10, growth per tick equals fertility. Dry crops stall. Moisture falls .001 per growing tick; rain adds .002 instead.
- Maturation yields 2 food below .70 fertility, otherwise 3, and removes .15 fertility with a .20 floor. Soil without a growing crop recovers .00002 fertility per tick.
- Each resident has a three-charge, currently weightless irrigation flask. Fill beside water, spending one unit from a shared 500-unit reserve. Rain restores .1 unit per tick up to 500. Drinking is unchanged. Watering adds .60 moisture, capped at 1.
- Three wood makes a basket on clear grass. Capacity is 12 ordinary food. Anyone can store or retrieve one food per action. Carrying needs four pack spaces plus the contents; placing requires clear grass. No ownership, theft rules or spoilage is predefined.
- Old crops retain their legacy timer. Old experiment worlds may leave ecology disabled.

Soil and basket contents enter senses only on visible tiles. Observation schema is 9; general neural action schema is 8. Old neural dimensions require explicit migration. Historical results belong to their original source versions.

## Historical v0.8 validation

105 tests passed in 17.98 seconds. Tests cover growth, water accounting, rain, fertility, basket conservation, simultaneous retrieval, local sensors, old crops and exact learning continuation. Browser checks verified irrigation, basket carrying, experimental save/load and no main-page horizontal overflow. Screenshots: `runs/ecology-storage.jpg`, `runs/learning-population.jpg`.

Main save was backed up to `runs/before-ecology-20261004.json` and checked field-by-field at tick 16904, with six structures and nine characters, before enabling ecology. Both local servers restarted successfully; experimental graceful shutdown and restart were tested.
