# GT-01: nearby-food learning diagnostic

Run completed October 5; audit completed October 6, 2026
(America/Los_Angeles). **Diagnostic complete and audited; both architectures
failed both behavioral gates.** No live migration or GT-01 completion.

The preceding [action-category pilot](ACTION-CATEGORY-PILOT.md) found 15/48
timely feeding lives for category PPO, compared with 18/48 for category-random.
Every category-trained life saw food, but only 19 gathered it. This diagnostic
tests the short gather-then-eat sequence when food is already within reach,
without changing actions, rewards, model capacity or the learning algorithm.

## Result and interpretation

The run completed **181,248 world ticks in 499.97 seconds**, within its
600-second budget. Category PPO gathered and ate within the first 64 ticks in
**23/48 lives (47.92%)**, versus **20/48 (41.67%)** at initialization and
**19/48 (39.58%)** for random category choices. Flat PPO achieved **17/48
(35.42%)**, versus 9/48 at initialization. Neither architecture met the
preregistered reliability and control-margin requirements in any seed.

| Frozen condition | Gather then eat within 64 ticks / 48 | Gather then eat before zero fullness over 512 ticks / 48 | Eat carried fruit within 16 ticks / 48 | Eat carried fruit over 64 ticks / 48 |
|---|---:|---:|---:|---:|
| Trained flat PPO | 17 | 28 | 12 | 26 |
| Trained category PPO | 23 | 41 | 22 | 34 |
| Initial flat policy | 9 | 21 | 10 | 18 |
| Initial category policy | 20 | 37 | 19 | 28 |
| Uniform feasible primitives | 8 | 17 | 9 | 18 |
| Uniform feasible categories/arguments | 19 | 34 | 17 | 23 |

The longer-horizon counts matter: missing the prompt deadline does not mean
never eating. The primary deadlines were declared before training and are not
extra reward terms. Carried-food scores measure consumption only; they never
count toward acquisition. Category PPO's 41/48 eventual acquisition lives are
encouraging activity, but do not replace its failed prompt-feeding gate or
establish reliable full-world foraging.

Each cell below is **adjacent-food prompt successes / carried-food prompt
successes**, with 16 lives in each fixture. These are three independent
training seeds, with two private brains per architecture in each seed.

| Training seed | Trained flat | Trained category | Initial flat | Initial category | Primitive random | Category random |
|---|---:|---:|---:|---:|---:|---:|
| 77501 | 9 / 6 | 8 / 10 | 5 / 4 | 7 / 9 | 6 / 4 | 8 / 12 |
| 77513 | 4 / 1 | 9 / 6 | 2 / 0 | 8 / 5 | 0 / 1 | 8 / 2 |
| 77527 | 4 / 5 | 6 / 6 | 2 / 6 | 5 / 5 | 2 / 4 | 3 / 3 |

The exact starting-state probes show modest increases in the food actions'
probability after training. They do not depend on which states each policy
later chose to visit.

| Model | Gather at adjacent start: initial -> trained | Eat at carried start: initial -> trained |
|---|---:|---:|
| Flat PPO | 6.17% -> 7.24% | 5.34% -> 6.52% |
| Category PPO | 24.94% -> 27.00% | 17.23% -> 19.82% |

These are action probabilities, not calibrated success confidence. The
measured shifts establish that the policy changed; they do not identify why
the changes are small. This test does not isolate reward attribution, update
strength, entropy pressure, generalization or forgetting. Carried-food starts
also begin with reset recurrent context and were evaluation-only, unlike a
training state immediately after gathering.

**Mechanism diagnostic recommendation:** audit one recorded feeding rollout and its PPO update
using copied experimental state. Verify the sensory encoding, which action
receives nutrition feedback, the advantages assigned to gathering/eating,
and the policy/value/entropy gradient contributions. Compare the same hungry,
food-carrying observation immediately before and after the update, then at
later recorded checkpoints where available. This distinguishes a weak or
misdirected learning update from a retention hypothesis before selecting a
change. It is proposed, not another completed experiment or permission for a
longer training run. The user subsequently requested a culling/restart test
and chose a separate experimental population. The resulting
[culling/restart pilot](CULLING-TRIAL.md) is now complete and audited, with
continued learning outperforming selective restart in that pool. The mechanism
audit remains a possible follow-up rather than an assumed memory fix.

More memory and mortality remain separate hypotheses. Immediate eating does
not require recalling a distant food location. Death changes the learning
horizon and could truncate future negative hunger costs; it does not by itself
teach which earlier action was useful. A future mortality comparison needs
explicit terminal accounting and a declared rule for retained learning or
inheritance. Mortality remains off for the live residents.

## Measured activity and limits

Each adjacent-food condition includes 24,576 resident-ticks and 6,144 conscious
decisions. Ordinary meals include repeat meals and meals after the primary
deadline. All conditions recorded **zero amber meals, injuries, unconscious
ticks and invalid actions**; no hazards were present.

| Adjacent-food condition | Ordinary meals | Zero-food ticks | Mean unique tiles | Tone decisions | Successful planting actions |
|---|---:|---:|---:|---:|---:|
| Trained flat PPO | 46 | 260 (1.06%) | 15.25 | 4,009 (65.25%) | 46 |
| Trained category PPO | 101 | 91 (0.37%) | 17.69 | 1,911 (31.10%) | 84 |
| Initial flat policy | 28 | 351 (1.43%) | 15.71 | 4,043 (65.80%) | 32 |
| Initial category policy | 79 | 135 (0.55%) | 17.35 | 1,820 (29.62%) | 109 |
| Primitive random | 23 | 403 (1.64%) | 16.02 | 4,049 (65.90%) | 36 |
| Category random | 68 | 174 (0.71%) | 18.69 | 1,820 (29.62%) | 132 |

Each carried-food condition includes 3,072 resident-ticks and 768 decisions,
with zero zero-food or unconscious ticks. It ends well before initial fullness
runs out. Its ordinary meals are respectively 26, 34, 18, 28, 18 and 23 in the
table's condition order. Dropping and re-gathering supplied fruit can occur;
those gathers are not new-food acquisition. Planting and visiting tiles are
measured activity, not established productive farming or useful exploration.
The rooms separate residents, so tones do not demonstrate communication,
leadership or cooperation. No direct ranking against the earlier navigation
pilot is valid because the fixtures and primary deadlines differ.

**Sampling limitation discovered during review:** the preregistered inherited
action RNG formula, `world_seed + 900000 + resident_index`, gives nine distinct
sampling seeds for the 16 evaluation lives within a training seed, condition
and fixture. Resident r1 in map k shares the seed used by r0 in map k+1. They
still have separate brains and RNG instances, but these observations have
additional dependence. The run and failed gates were preserved unchanged.
Future protocols should use a collision-free map/resident seed mapping while
retaining matched randomness across conditions. This is a clustered,
descriptive development diagnostic; there is no IID significance or causal
claim about the uncontrolled live population.

## Evidence and verification

Evidence is local in `runs/near-food-pilot-20261005/`. The preregistration's
SHA-256 is
`6919106a7fb34f36078300a297d3a0bbb4103251d1eac5b3dcadd87cae6d65d0`.
The completion record (local evidence; not bundled),
analysis (local evidence; not bundled) and
audit (local evidence; not bundled) retain the failed gates.
The audit passed in **209.22 seconds**, reproducing all **480 episodes,
181,248 transitions and 41,472 frozen evaluation decisions**, including the
starting probability vectors. Source/evidence hashes, private observations,
physical masks, individual rewards and training-weight continuity passed.

Recorded replays show the first map of the first seed in all six conditions:
adjacent food (local evidence; not bundled) and
carried fruit (local evidence; not bundled). Selection
was by index, not outcome. Neither page contacts the live game. Both final
pages passed a JavaScript DOM/canvas stub check for all conditions at their
first and last frames; see the check (local evidence; not bundled).
This does not claim visual browser inspection.

Both families received 24,576 training decisions and 192 PPO updates across
six private brains. Flat episode time was 141.24 seconds, including 33.70
seconds learning; category time was 160.94 seconds, including 42.36 seconds
learning. They experienced 145 and 312 ordinary training meals respectively.
These are interaction-matched CPU trials, not equal-wall-time comparisons.
Peak working set was **650.98 MiB** and peak private commit **1.75 GiB**.
Core evidence occupied 92.51 MiB before derived analysis/replays, below the
512 MiB cap. Initial/trained checkpoints and the full compressed trace remain.

Read-only `/health` checks retained PID 69328, all eight `recurrent-ppo`
resident IDs and the distinct `laya-npu` ID. The world advanced through its
own activity from tick 74,477 to 74,860 and was automatically paused at the
after-check (`manualPause=false`, public sharing off). No live state endpoint,
browser heartbeat, restart, policy import or resident reset was used. Health
does not expose controller error details; this is not an error-log inspection.

## Protocol frozen before the run

Use three fresh training seeds: **77501, 77513, 77527**. Train a flat PPO pair
and a category PPO pair, each resident with private weights, optimizer, RNG and
memory. Every resident gets **32 lives of 512 ticks** with food initially one
cardinal tile away: **4,096 decisions each**. Positions are translated between
lives, and food direction is balanced across the four cardinal bearings.
The same 7x7 room, four-berry budget, empty pack, fullness 4 and other needs/
health 100 are used for both models. Agents can freely walk away or choose any
physically possible action. No navigation lock, food priority, reward shaping,
demonstration or forced command is supplied to learners.

Frozen evaluation uses eight unseen development world seeds per training seed,
two independently trained residents and six conditions: trained flat, trained
category, their own matching initial weights, uniform feasible primitives and
uniform feasible categories/arguments. All six arms have matching starting
worlds and action RNG seeds. Training/evaluation seeds are declared in the
machine-readable preregistration; these are not the later confirmation maps.

There are **two separate fixtures**, with 48 lives per condition in each:

1. **Adjacent food, empty pack:** the primary outcome is own gathering followed
   by ordinary-food consumption within the first **64 ticks / 16 decisions**,
   before any zero fullness. Evaluation continues for 512 ticks to record
   latency, repeat meals and the previous full-life acquisition measure.
2. **One carried fruit:** an evaluation-only consumption check. The pack starts
   with one ordinary fruit; there is no wild food patch. The outcome is eating
   within **16 ticks / four decisions**. Evaluation lasts 64 ticks. This check
   cannot establish food acquisition, and its scores are never pooled with
   empty-pack lives. Neither family trains on these supplied-food lives.

For each exact starting observation, record the complete 68-action probability
vector for every condition. This forward pass changes no world, hidden state,
RNG, optimizer or choice. It allows matched-state comparisons of gathering and
eating preferences, avoiding the earlier diagnostic's different visited-state
denominators. Policy probabilities are not calibrated success confidence.

For each architecture, the adjacent-food diagnostic gate requires **at least
80% prompt success and a 20-percentage-point advantage over its initialization
and both random controls in every training seed**. A separate consumption gate
requires 90% prompt eating and the same margins in every seed. Neither gate
closes GT-01: reliable navigation and the roadmap's larger held-out confirmation
remain separate. Report per-seed results; repeated lives are clustered under
their learned brains. This is a development diagnostic, not a significance test.

Total budget: **181,248 world ticks**, including 98,304 training ticks, 73,728
adjacent evaluation ticks and 9,216 carried-food evaluation ticks. The run has
a **600-second wall cap and 512 MiB evidence cap**, followed by a separate
600-second replay-audit cap. The prior matched pilot took 328.13 seconds for
172,032 ticks; this adds short frozen probes, not another architecture or a
larger training budget. Budget exhaustion preserves partial evidence and does
not authorize an automatic retry or additional training.

## Implementation checks and boundaries

The new runner reuses the existing measured brains, physics and trace recorder.
The separate auditor recounts outcomes/rewards and replays world transitions,
private observations and frozen decisions. It independently verifies the
starting-state probability vectors, acquisition/consumption deadlines,
initial-weight matching, training-weight continuity and declared gates.
Before simulation, archive source, preregistration and hashes; retain initial
and trained checkpoints plus every decision input and body transition.

Focused tests check immediate physical gather/eat feasibility in every
direction, the distinct carried-food condition, all ten available tones,
probe side effects, deadline boundaries and rejection of a learning claim when
random controls perform equally well. Prescribed commands in a physics unit
test are not supplied to learners as demonstrations.

This short isolated assay has no resource competition, social opportunities,
hazards, architectural growth, crop-production claim or useful real-world
transfer test. Fullness lasts about 500 ticks without eating, and carried
probes end before deprivation; low zero-food or unconscious time cannot be
called survival competence. All ongoing residents, their histories, the eight
live PPO models and Laya's distinct local NPU model are preserved. The only
live-server inspection is the read-only `/health` endpoint.

The full regression suite passed **196 tests** in 101.65 seconds. The separate
8,960-tick smoke check completed in 24.90 seconds while regression tests were
running; its audit replayed all 28 episodes and 3,456 frozen decisions in
18.56 seconds. Both recorded replay fixtures passed a JavaScript/DOM/canvas
stub check through their final tick; this is not a visual browser inspection.
