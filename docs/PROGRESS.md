# Blissful Ignorance: progress so far

October 6, 2026. Previously developed as **Godhood Trials**.

Blissful Ignorance is a playable artificial-life research world. Its inhabitants
have individual observations, memory and learning state. The player can walk
among them using the same physical actions; their observations do not contain
a flag identifying the player as the creator.

**The software foundation works. The latest experiment demonstrates a narrow
feeding improvement. Reliable survival and an emergent society remain research
goals.**

## The latest result

Six retained experimental category-PPO brains were tested after different
amounts of additional practice. All conditions used matched fresh test maps
and frozen policies: they did not learn during evaluation.

| Additional practice | Prompt nearby-food acquisition | Prompt eating of carried fruit |
|---|---:|---:|
| Starting checkpoint | 75/96 (78.13%) | 58/96 (60.42%) |
| 16 more practice lives | 82/96 (85.42%) | 65/96 (67.71%) |
| 64 more practice lives | **95/96 (98.96%)** | **95/96 (98.96%)** |

“Prompt acquisition” means gathering ordinary food and eating it **before tick
64**, before reaching zero fullness. The carried-fruit deadline is **before
tick 16**. The starting checkpoint already had 6,144 training decisions per
brain. The comparison has no random-policy arm and is not an architecture
ranking. The six brains form three historical training groups; the 96 lives
are not 96 independent model initializations.

All six predeclared development checks passed. Previously weak responders
improved while retaining their brains. At +64 practice, all 96 nearby-food
lives eventually acquired and ate food before zero fullness; one missed the
prompt deadline. This is improvement on new instances of the same simple
fixture, **not demonstrated navigation or sustained survival**.

[Open the interactive snapshot](progress/index.html) for per-brain scores,
activity totals and actual recorded playback. GitHub displays HTML source;
download the repository and open that file locally to use it.
[Full experiment and limitations](PRACTICE-AMOUNT.md).

## What is built

| Area | Working mechanics | Behavior still to demonstrate |
|---|---|---|
| World | Local browser game; player avatar; pause, step, fast-forward; persistent saves | Useful adaptation across changing environments |
| Bodies and senses | Local vision with occlusion; directional sound; touch; bodily needs; approximate visual memories | Reliable learned use of memory and useful curiosity |
| Resources | Finite food, shared harvests, seed conversion, crops, irrigation, baskets | Sustained feeding from agents' own farming |
| Construction | Walls, doors, floors, shelters, dismantling | Building that measurably improves survival |
| Social actions | Attention taps, ten discrete tones, gifts and revival | Learned meaning, voluntary cooperation and leadership |
| Learning | Independent recurrent PPO, complete checkpoints, controlled experiments and replay audits | Robust survival, long-term retention and useful transfer |

Mortality is off. Passing out and recovery are implemented. Reproduction is a
later research milestone. The simulation defines possible actions and physical
consequences; the current PPO controllers select their own actions rather than
following hunger-triggered routes. Historical scripted baselines remain
explicitly labeled in their reports.

## Keep model families separate

- **Live PPO population:** eight independent recurrent PPO brains, with private
  learning state and histories.
- **Laya:** currently a distinct model using existing local NPU assets, frozen
  pretrained weights and private bounded consequence context. The user has
  proposed retirement; the replacement selection is pending.
- **Category-PPO candidates:** six separate experimental learners used in the
  practice result above. Their weights have not been imported into live bodies.
- **Clef and earlier candidates:** comparison experiments, not deployed replacements.

## What did not work

Negative results guide the next experiment and remain in the public record.
These comparisons have different fixtures and budgets; do not rank them by
combining their scores.

| Experiment | Observed result | Decision |
|---|---|---|
| [Voluntary coordination](VOLUNTARY-COORDINATION.md) | Initially uninformed residents ate in 2/24 intact and 2/24 shuffled-tone cases | Communication gate failed |
| [Nearby-food diagnostic](NEAR-FOOD-DIAGNOSTIC.md) | Category PPO: 23/48 prompt successes; its initialization: 20/48; random: 19/48 | Both architectures failed their gates |
| [Selective restart](CULLING-TRIAL.md) | Continuing: 32/48; selective restart: 29/48; random restart: 31/48 | Keep the existing brains |
| [Value-gradient intervention](VALUE-INTERFERENCE.md) | Reinforced more meal choices, but mean fixed-probe improvement was slightly worse | Keep the existing learning rule |
| [Additional practice](PRACTICE-AMOUNT.md) | 75 → 82 → 95 prompt feeding lives out of 96 | Pilot passed; test navigation next |

## Evidence and the next gate

The practice experiment's recorded suite passed **217 tests**. The current
public source passed **231 tests**, including PettingZoo compatibility, after
three preservation fixes; seven [implementation findings](AUDIT-2026-10-06.md)
remain open. The separate learning audit reproduced **181,248 world transitions, 90,624 decisions,
384 updates and 24 full individual checkpoint states**. It shares the native
simulator and brain; the replay and outcome recount are separately implemented.
Software correctness and learned competence are assessed separately.

**GT-00 is complete; GT-01 remains open.** The next proposed comparison tests
frozen +64 policies with food farther away and behind an occluder, alongside
starting-policy and random controls. Confirming GT-01 requires five fresh
training groups, held-out layout families, at least 90% success and the
specified advantage over controls. It has not run yet.

The portable snapshot contains scores, source hashes and one recorded example.
Raw traces, full checkpoints and live saves remain local; a reader cannot
reproduce the full training audit from the snapshot alone.

[Milestone criteria](MILESTONES.md) · [Implementation history](STATUS.md) ·
[Public snapshot provenance](progress/README.md) · [Publication status](PUBLISHING.md)
