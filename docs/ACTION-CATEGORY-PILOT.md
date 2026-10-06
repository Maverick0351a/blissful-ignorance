# GT-01: learned action categories

October 5, 2026 (America/Los_Angeles). **Pilot complete and audited.
The advancement gate failed. No live migration or milestone completion.**

The trained category policy fed before zero fullness in **15/48 lives (31.25%)**,
versus **6/48 (12.50%)** for trained flat PPO. However, **random category choices
fed in 18/48 (37.50%)**. The changed action distribution is useful experimental
evidence, but this result does not establish a learning advantage over the
category-random control or reliable food acquisition.

## Development result

The run completed 172,032 world ticks in **328.13 seconds**, within its 600-second
cap. These are 48 evaluated lives per condition, clustered under six private
brains and three independent training seeds. They are not 48 independently
trained policies, and differences are descriptive rather than a significance
claim.

| Frozen evaluation condition | Timely feeding / 48 | Rate | Open / 24 | Occluded / 24 | Ordinary meals |
|---|---:|---:|---:|---:|---:|
| Trained flat PPO | 6 | 12.50% | 6 | 0 | 9 |
| Trained category PPO | 15 | 31.25% | 9 | 6 | 29 |
| Initial flat policy | 5 | 10.42% | 5 | 0 | 7 |
| Initial category policy | 8 | 16.67% | 4 | 4 | 15 |
| Uniform feasible primitives | 2 | 4.17% | 2 | 0 | 4 |
| Uniform feasible categories/arguments | 18 | 37.50% | 11 | 7 | 29 |

Ordinary meals include repeats and meals after the deadline; they are not the
primary success count. Every condition had zero amber meals, injuries and
unconscious ticks in this short, hazard-free assay. Starting fullness lasts
about 500 of the 512 ticks without food, so low deprivation percentages cannot
be interpreted as sustained feeding competence.

| Training seed | Trained flat / 16 | Trained category / 16 | Initial category / 16 | Category random / 16 | Category minus flat |
|---|---:|---:|---:|---:|---:|
| 76411 | 1 | 6 | 2 | 5 | +31.25 points |
| 76423 | 1 | 5 | 2 | 5 | +25.00 points |
| 76439 | 4 | 4 | 4 | 8 | 0 points |

Category PPO exceeded its own initialization in two seeds and tied in one.
Against random categories it won one, tied one and lost one. No seed passed
the required ten-point advantage over all controls, and none reached 90%
absolute feeding reliability. The preregistered development gate therefore
failed. GT-01 remains open, and this run does not justify a confirmation claim
or a live policy migration.

The flat policy/value network has 89,173 parameters; the category version has
90,538 (**+1,365, or 1.53%**). Both have a separate 63,623-parameter consequence
predictor. There are 68 primitive actions and 21 verb categories. No larger
recurrent core or additional sensory input explains the comparison.

**Next bounded action:** isolate the gather-then-eat chain in a controlled
nearby-food diagnostic, keeping all physical actions and individual rewards.
Check whether the learner retains that short action sequence before adding
navigation difficulty or spending on a longer full-world run. This is a
proposal recorded with this pilot. **Follow-up completed October 5-6:** the
[nearby-food diagnostic](NEAR-FOOD-DIAGNOSTIC.md) passed its audit but failed
both behavioral gates; its different fixtures and deadlines are reported
separately.

Evidence directory: `runs/category-pilot-20261005/`. Preregistration SHA-256:
`8c2ac0609ef4c9f1c25b1499eeefd4e20bc04001443d94a17218392976fad24a`.

The separate audit passed in **183.59 seconds**, replaying all **336 episodes,
172,032 world transitions and 36,864 frozen evaluation decisions**. It verified
source/evidence hashes, matching initial tensors and worlds, private sensory
inputs, physical masks, individual rewards, training-weight continuity,
unchanged evaluation weights, food deadlines and the failed gate. All **191
tests passed** in 82.71 seconds, including optional PettingZoo compatibility.

The prerecorded interactive replay (local evidence; not bundled)
shows the first evaluation case of the first seed in all six conditions,
selected by index rather than success. It reconstructs physical frames from
the audited commands and does not contact the live server. JavaScript parsing
and a DOM/canvas stub check passed on the smoke replay; this is not a visual
browser inspection.

The experiment's peak working set was **646.65 MiB** and peak private commit
was **1.75 GiB**, including the Python/PyTorch runtime and checkpoints. Both
families received 24,576 training decisions and 192 PPO updates across their
six private brains. Recorded episode time was 99.74 seconds flat and 110.88
seconds category, of which learning used 26.10 and 32.32 seconds respectively.
These are interaction-matched results, not an equal-wall-time comparison.
Evidence occupied approximately **85 MiB**, below the 512 MiB cap.

Models and training used the CPU. `/health` before/after retained server PID
69328, all eight PPO IDs and the distinct `laya-npu` ID. The running world
advanced from tick 70,227 to 70,315 through its own activity and was automatically
paused at the final check (`manualPause=false`, public sharing off). No live
state API, browser heartbeat, restart or controller mutation was used. Health
does not expose controller error details, so this does not claim a fresh error
log inspection.

## Activity and the next diagnostic

Each condition includes 24,576 resident-ticks and 6,144 conscious decisions.
Fullness/health, empty packs, isolated rooms and food budgets were matched.

| Condition | Nutrition | Zero-food ticks | Mean unique tiles/life | Plantings | Tone decisions |
|---|---:|---:|---:|---:|---:|
| Trained flat | 225 | 542 (2.21%) | 13.81 | 4 | 4,316 (70.25%) |
| Trained category | 725 | 421 (1.71%) | 18.44 | 38 | 2,353 (38.30%) |
| Initial flat | 175 | 555 (2.26%) | 14.63 | 10 | 4,301 (70.00%) |
| Initial category | 375 | 516 (2.10%) | 17.63 | 51 | 2,103 (34.23%) |
| Primitive random | 100 | 586 (2.38%) | 14.94 | 7 | 4,294 (69.89%) |
| Category random | 725 | 386 (1.57%) | 17.69 | 42 | 2,004 (32.62%) |

All recorded actions were physically valid. No tap, gift, revival, construction
or social coordination occurred; neighbors were out of reach by design.
Planting is measured activity, not demonstrated productive farming: this
protocol does not track crop provenance through later meals and covers only
512 ticks against a 480-tick crop growth period. There was no hazard exposure.

In the trained category condition, 48/48 lives saw ordinary food, 19 gathered
it, 16 eventually ate and 15 ate before the deadline. The corresponding flat
counts were 47, 10, 7 and 6. This identifies observable points of failure without
establishing why a controller missed them.

A **post hoc descriptive diagnostic** counted physically available actions in
each policy's own visited states. Category PPO gathered in 78/340 available
decisions (22.94%) and ate in 29/184 (15.76%); category-random gathered in 70/340
(20.59%) and ate in 29/164 (17.68%). Flat PPO gathered in 20/333 (6.01%) and ate
in 9/120 (7.50%). These denominators come from different trajectories; they
cannot establish a causal explanation or be substituted for the preregistered
success measure. They support examining the short gather/eat sequence next.

Category PPO's tone share rose from 34.23% at initialization to 38.30% after
training. Joint entropy mathematically favors equal primitive probabilities,
which allocates more mass to verbs with many arguments. Whether that pressure
caused the observed change is an untested hypothesis. No entropy or reward
coefficient was changed after viewing the result. The full descriptive output
is analysis.json (local evidence; not bundled).

The category policy learns a distribution over existing action verbs, then a
conditional distribution over that verb's feasible arguments. The joint
primitive probability is used for PPO's probability ratio, clipping, KL and
entropy. No category receives a special reward or hunger-triggered rule.
All existing actions, including all ten tones, remain available under the
unchanged physical-feasibility mask.

`agents/category_ppo.py` is experimental. Its checkpoint records and validates
the category/action order. The live loader rejects this backend, so a saved
experimental policy cannot silently replace a continuing resident. The flat
PPO implementation gained only a model factory and distribution hook, plus
recognition of the experimental backend as local inference. Its default
distribution and checkpoint format are unchanged.

## Matched development protocol

Use three fresh training seeds (76411, 76423, 76439), two private learners per
seed/architecture, 32 training episodes of 512 ticks, and eight unseen
development evaluation maps per seed. Decisions occur every four world ticks:
4,096 training decisions per resident and architecture. The full budget is
172,032 world ticks with a 600-second run cap and 512 MiB evidence cap. An
independent audit has its own 600-second cap. No retries or extra training are
authorized by a budget stop.

The 8,192-tick infrastructure check completed in 15.30 seconds; its audit
replayed every transition and 3,072 frozen decisions in 12.82 seconds. That
throughput, rather than its behavioral scores, determined the pilot budget.
Evidence: `runs/category-smoke-20261005/`.

Each resident begins in a separate 7x7 room with empty inventory, fullness 4,
health/other needs 100, and four reachable berries. Half the cases have an
opaque divider between the initial position and food. Rotation, translation
and food location vary across lives. Rooms isolate individual food acquisition
from theft or social assistance. This assay does not test competition,
cooperation or sustained survival. Ecology, hazards and automatic wild-food
refill are absent; ordinary planting and seed conversion remain possible.

The primary outcome is **gathering ordinary food and eating it before that
resident ever reaches zero fullness in the life**. Report both layouts and
each training seed; lives sharing a learned brain are clustered observations.
Meals after the deadline remain in the meal counts but do not count as success.

Six frozen-policy conditions use identical starting worlds and matching action
RNG seeds: trained flat, trained category, each one's own initialization,
uniform feasible primitive actions, and uniform feasible categories followed
by uniform feasible arguments. Those last controls separate learning from a
change in exploration probabilities. All model families retain private brains,
optimizers and recurrent state. Core, value, argument and predictor weights
match at initialization. The new category head uses a separate seeded stream.

The existing sensory encoder, recurrent core, individual reward, update
schedule and terminal context reset are unchanged. Curiosity remains zero.
Joint-action entropy uses the same coefficient; its maximum still favors
uniform primitive actions, so entropy pressure is not neutral between verbs
with different numbers of arguments. This tradeoff is recorded, not changed
alongside the head.

The development advancement gate requires category-trained to outperform
flat-trained in at least two of three seeds and in the pooled rate, and to
exceed its own initialization and both random controls by at least ten
percentage points in **every** seed. This is a conservative pilot screen,
not a significance test and not the GT-01 confirmation gate. The latter still
requires the five-seed, 100-life-per-condition design and absolute reliability
criteria in [MILESTONES.md](MILESTONES.md).

## Evidence and verification

Before simulation, the runner writes the protocol, its SHA-256 and an archive
of the exact source. It retains initial/trained checkpoints, every private
decision input, primitive action, per-tick body and individual reward, and
episode results. The separate auditor reconstructs the worlds and frozen
policy decisions, recounts food deadlines and behavior, checks matching initial
tensors, verifies training-weight continuity and checks the declared gate.

Focused tests cover masks, joint probabilities and entropy, gradients,
unchanged flat probabilities, matched initialization, private optimizers,
exact continuation across a PPO update, action-schema rejection, reachable
food, hidden/visible starts and controls that cannot confuse random exploration
with learning. The live game is only checked through `/health`; no browser
heartbeat, restart, resident reset, weight import or Laya substitution occurs.
