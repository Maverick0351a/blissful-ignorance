# GT-01: feeding reward and update diagnostic

October 6, 2026 (America/Los_Angeles). **Diagnostic complete and audited.**

The nearby-food and culling pilots did not establish reliable feeding or a
memory defect. This bounded follow-up inspects how recorded feeding experience
changes the experimental category-PPO policy. It preserves every live resident,
Laya and every source checkpoint. It introduces no new training trajectories.

## Result and interpretation

All **214 ordinary meals** received positive normalized learning credit.
The ordinary-eating probability increased after its update for **184/214
meal decisions (85.98%)** and decreased for the other 30. Eating occupied only
1.74% of the 12,288 recorded decisions. Food credit reached the eating actions;
the probability changes were uneven across brains.

| Experimental brain | Ordinary meals | Meal choices reinforced | Carried-food eating probability, before → after | Adjacent-food gathering probability, before → after |
|---|---:|---:|---:|---:|
| c0 (77501/r0) | 29 | 27 | 16.45% → 17.33% | 27.63% → 28.44% |
| c1 (77501/r1) | 36 | 35 | 19.55% → 22.23% | 27.00% → 29.45% |
| c2 (77513/r0) | 49 | 43 | 24.19% → 35.24% | 27.58% → 33.84% |
| c3 (77513/r1) | 30 | 18 | 20.35% → 20.67% | 26.72% → 27.78% |
| c4 (77527/r0) | 38 | 38 | 20.05% → 27.28% | 24.95% → 30.89% |
| c5 (77527/r1) | 32 | 23 | 18.32% → 18.98% | 28.12% → 29.18% |

These probabilities average four identical zero-context probes per fixture,
before and after sixteen recorded updates. They are action preferences, not
feeding-success percentages. The replay reproduces the prior continuation
branch's states and cannot turn its 32/48 prompt-feeding result into a pass.
In particular, c4 improved while retaining its original experience, despite
being eligible for replacement in the earlier selection screen.

The first meal occurred in the first continuation life for all six brains.
Five had an immediate eating-probability gain on that fixed observation and
retained more than half the gain after **15 later updates**, in both zero and
held-fixed original context. c3's first update slightly decreased its eating
probability, so it was not eligible for a retention claim; its final preference
was nevertheless above the pre-update level. This is retention during further
training on the same food task, not a task-switch forgetting test or proof of
episodic recall. A memory-capacity defect remains unestablished.

Reward attribution also reached preceding non-meal actions, as expected from
GAE. Gathering had positive normalized credit for 347/408 decisions, with a
mean of 1.60; eating's mean was 2.27. Mean normalized advantages for moving,
resting and tones were negative. Positive raw food-credit components on earlier
tones therefore must not be described as a tone reward or evidence that the
learner was generally encouraged to signal instead of eat.

The value-loss gradient norm averaged 2.97–5.18 times the actor-loss norm across
the six brains. Only 9/96 first-epoch combined gradients exceeded the clipping
threshold; all 96 updates ran all four epochs. The strongest improving brain,
c2, had the largest value/actor norm ratio. c3 and c5 had negative mean
actor/value alignment, but the full-vector alignments were small. These
measurements do **not** establish that critic interference or clipping caused
weak learning, and provide no reason to raise the learning rate blindly.

**Next bounded recommendation:** compare the ordinary update with an
experimental copy that blocks value-loss gradients into the shared policy
encoder, using the same recorded rollouts and optimizer starting states.
Measure eating-probability changes on recorded and fixed new observations,
especially c3/c5; preserve every result. This would test a specific interference
hypothesis before changing rewards, memory capacity or the live architecture.
A favorable update diagnostic would still require a fresh behavioral trial.
That follow-up was subsequently completed in the
[matched value-gradient test](VALUE-INTERFERENCE.md). It helped some brains
but failed its primary mean-benefit screen; no live learning change followed.

The recorded training had 214 ordinary meals, zero amber meals, 143 zero-food
resident-ticks out of 49,152 (0.29%), no unconscious ticks or injury, and a mean
17.52 unique tiles per life. It included 197 plantings, 102 seed conversions and
3,991 tones. Rooms separated residents; there was no social coordination test.
The short fixture and activity counts do not establish sustainable farming,
learned language or survival competence. These are old training observations
replayed exactly, not newly observed live behavior.

## Evidence and completed checks

Local evidence: `runs/feeding-credit-20261006/`.
Protocol SHA-256:
`5e55562f2232301c96c01bfedaaba74faa78fbe6af77c847fd39d379d3dec3c0`.

- Completion receipt (local evidence; not bundled): **117.58
  seconds**, 48 episodes, 24,576 replayed ticks, 12,288 decisions and 96 updates.
- Separate audit (local evidence; not bundled): **105.04 seconds**;
  all source hashes, exact trace subset, native actions/updates, private views,
  physics/rewards, scalar GAE, probabilities and complete final states passed.
- Interactive probability curves (local evidence; not bundled),
  summary (local evidence; not bundled) and
  full update records (local evidence; not bundled) preserve all
  six candidates and every update. The report has no network requests or live
  world controls.
- The full suite passed **207 tests in 105.01 seconds**, including the optional
  PettingZoo API test. The instrumented-versus-native test matched full learning
  states over consecutive updates; scalar tests cover delayed credit, terminal
  cuts and relative advantage. All six report selectors and four curves per
  brain passed the JavaScript DOM-stub check (local evidence; not bundled).
  This does not claim visual browser inspection.

The diagnostic's peak working set was about 768.62 MiB and peak private commit
1.84 GiB. Core evidence was 64.38 MiB before the derived report and verification
files, below the 512 MiB cap. The diagnostic ran alongside the regression suite;
these wall times are not standalone architecture throughput measurements.

No live state endpoint, browser heartbeat, restart or checkpoint write occurred.
The main server was already automatically paused (`manualPause=false`) during
the first read-only health check at tick 78,701, PID 69328, with all eight PPO
IDs and distinct Laya NPU ID. The final health check retained the same PID,
model IDs and automatic-pause state at tick 78,811; the world's own activity
advanced it between the checks. The health response is retained separately.
Health does not expose controller error details; no error-log inspection is
claimed. The source experiment and all historical failed gates remain intact.

## Protocol fixed before measurement

Replay the original culling pilot's **continuation branch only**, beginning
with its six archived private brains and ending at its existing final states.
Include all sixteen 512-tick lives per brain, not just successful episodes:
48 two-resident episodes, 24,576 world ticks, 12,288 recorded decisions and
96 updates. These are the same experiences already reported in that pilot,
not additional evidence from new lives or six independent population trials.

Each replayed action must be independently sampled by the unchanged native
learner and match the recorded action. Check every private observation, body
transition, reward and ending weight hash. Full final state equality includes
the model, predictor, optimizers, RNG, recurrent state and private history.
Instrumentation does not enter the policy input, reward or update rule.

Measure the following for every update:

- Immediate rewards, raw and normalized generalized advantage estimates (GAE),
  and their additive food, hunger, injury and value-estimate contributions.
  Normalization centers credit relative to other actions in the same rollout;
  a positive immediate reward need not imply a positive normalized advantage.
- Ordinary meal probability before and after the update, reevaluating the full
  recorded recurrent sequence from the same initial context. Count an increase
  only above 0.000001. Include unsuccessful choices and all action categories.
- Fixed hungry adjacent-food and carried-food probes, four orientations each,
  evaluated with zero recurrent context. These observations are never sampled
  for behavior, rewarded or added to training. They test changed preference
  on matching inputs, not actual feeding or calibrated confidence.
- The first ordinary meal encountered chronologically by each brain, tracked
  through all later updates on identical encoded input. Evaluate zero context
  and the held-fixed original decision context separately. A descriptive
  half-gain-retention flag applies only if the immediate gain exceeds 0.000001
  and at least four later updates exist. Old context is held fixed to separate
  parameter changes; this does not establish usable episodic memory.
- First-epoch actor, value and entropy gradient norms, their combined clipping
  scale and actor/value alignment. These are measurements before Adam and do
  not by themselves establish that any loss component caused poor learning.

Report all six brains individually. There is no behavioral pass gate, memory
diagnosis or live migration from this retrospective measurement. GT-01 remains
open regardless of technical success. Continued learning can change a probe
preference for reasons other than lost memory; a causal repair requires a
separate controlled intervention with fresh evaluation.

## Budgets and verification

The runner has a **600-second cap and 512 MiB evidence cap**, followed by a
separate audit capped at 600 seconds. Preserve partial evidence on failure;
do not extend the run or automatically retry. The five focused tests use
small disposable fixtures; there is no new behavior-selection smoke pilot.

Archive the source and machine-readable protocol before replay. Retain the
exact original trace subset, source receipt/audit, copied initial/final
checkpoints, update diagnostics, reproduced final states and report. The
separate auditor replays with the uninstrumented native learner, computes GAE
using an independent scalar recurrence, verifies matched probabilities and
checks complete final states. Gradient recomputation shares the measurement
helper; analytic tests and native-versus-observer equivalence test it separately.

No live browser heartbeat, checkpoint import, model download, reward change,
architecture change, mortality change, further culling or publication is part
of this diagnostic. Only read-only `/health` checks inspect the main server.
