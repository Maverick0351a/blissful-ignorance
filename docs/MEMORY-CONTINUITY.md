# Recurrent working-context diagnostic

October 5, 2026 (America/Los_Angeles). The user approved the isolated memory
diagnostic. No live resident, model, reward, save, or server setting is changed.
Laya remains a separate persistent local NPU model.

**Completed: the primary hypothesis failed.** Rebuilding context reduced
average zero-fullness time from 43.23% to 36.45%, but improved only two of
three seed groups, left the low-deprivation count at 4/12 lives, and increased
mean thorn contacts. Both trained arms fed less reliably on average than the
initial weights. No experimental controller or weights were promoted.

## Question and scope

The existing PPO controller deliberately clears recurrent state after each
128-decision update. At four physical ticks per decision, that is every 512
ticks during training. Frozen evaluation has no weight updates and therefore
no periodic reset. This is an implementation choice, not an established cause
of poor feeding. Stored visual sketches and learned weights are separate and
are not erased at this boundary.

The experiment compares the existing reset with recomputing recurrent state
from the last 128 private encoded observations using the newly updated weights.
It starts reconstruction from zero and does not propagate gradients through
that prefix. Information outside this bounded prefix can still be lost. This
does not implement unlimited memory, planning, or additional policy capacity.

The candidate lives in `agents/memory_ppo.py`; the live server does not select
or import it. The reset arm retains the original policy outputs, physical
trajectory, parameter updates and optimizer state. Both arms retain the same
visual recall channels, action masks, ten tones, body rewards, discounting and
zero curiosity coefficient. No action recipe or privileged location enters a
learner. Reconstruction adds computation, which is reported separately from
the matched physical experience and update counts.

## Preregistered comparison

- Three fresh training seeds: 73401, 73412, 73423. Two private residents per
  seed and arm, twelve trained brains overall.
- Reset versus 128-observation reconstruction after learning updates.
- 4,096 decisions per resident: eight 512-decision episodes with the established
  feeding/feeding/routes/routes/scarcity/scarcity/scarcity/scarcity curriculum.
  Each resident receives 32 PPO updates. Episode length is held fixed here;
  the longer-life comparison is a separate proposed experiment.
- Two fresh evaluation maps per seed, each lasting 8,192 physical ticks.
  Evaluate each trained checkpoint continuously and with a reset every 128
  decisions. Initial weights receive both evaluation treatments. Random legal
  actions and a privileged feasibility oracle are additional controls.
- All eight conditions share map seeds and starting bodies. Evaluation freezes
  weights, optimizers, consequence predictors and learned journals. Runtime
  hidden state, sensing, history, action RNG and counters can change.
- Forty-eight evaluation worlds, 96 individual evaluated lives. Repeated lives
  of one trained brain are not independent training replicates.
- One full run, capped at 1,800 wall seconds. No test-guided checkpoint choice,
  retries with favorable seeds, or automatic promotion to the live world.

The primary hypothesis requires reconstruction to reduce mean zero-fullness
time in **every** seed, compared with reset training when both are evaluated
continuously, and to increase overall mean fullness. This is a directional
three-seed diagnostic, not a statistical significance or deployment claim.
Periodic-reset evaluations separately measure sensitivity of the same frozen
brain to evaluation boundary handling.

## Food-location observations

An observer records ordinary food seen through the actual local observation.
When a previously visible, still-present food location leaves sight beyond
gathering reach, it opens one opportunity for that resident. A return means
coming within Manhattan distance one of that location within 128 decisions.
The observer records whether a learning/reset boundary occurred between the
cue and outcome. Expired, depleted and end-censored opportunities remain in
the denominator. Only one target is tracked at a time, chosen deterministically.

The observer's absolute coordinates and resource-availability checks are
audit data and never enter the policy. Visible-food records, physical movement,
boundary timing, event definitions and aggregate counts are independently
recounted from the paired traces.

Returning does not prove that a resident remembered a location: it may wander
back, reacquire sight, or use an existing visual sketch. The arms also encounter
different opportunities after their actions diverge. Counts therefore measure
activity, not recalled navigation or causal mediation. A memory-specific cue
ablation would be needed for that stronger claim. These isolated fork arenas
contain no amber fruit and do not measure communication or social competition.

The preregistered event counter includes a boundary marker at the observation
that closes an event. Review identified an important ordering distinction:
the position observed at decision d results from action d-1, whereas the update
inside decision d occurs afterward. A separately labeled post-hoc timing check
therefore reports boundaries from the opening decision **up to but excluding**
the closing decision. This prevents crediting a reset/update for movement that
already happened. Both original counters and stricter counts are preserved;
the primary feeding hypothesis is unchanged. Two focused timing tests pass.

## Verification and evidence

Six focused tests passed before the pilot. They check original-baseline
equivalence, reconstruction with updated weights, bounded detached private
inputs, exact resume across updates, frozen learning, physical observer parity,
food-return definitions and rejection of a falsified boundary marker.

Each new output directory receives its source archive, runtime versions, fixed
protocol and SHA-256 before training. Training and evaluation retain physical
and sensory traces, checkpoints, food accounting, initial/frozen digests and
completion receipts. Partial or failed runs remain preserved.

```powershell
$python = '<user-home>\Projects\laya-lab\.venv\Scripts\python.exe'
& $python experiments/memory_continuity.py --output runs/new-memory-diagnostic --wall-seconds 1800
& $python experiments/audit_memory_continuity.py runs/new-memory-diagnostic
& $python experiments/review_memory_timing.py runs/new-memory-diagnostic
```

The existing runtime supplies every dependency; no download is required.

## Run status

The pilot completed **12,288 ticks in 23.44 seconds**. Its independent audit
passed archived/current source hashes, all physical and sensory trace hashes,
food conservation, every context event, checkpoint provenance, equal budgets,
private finite parameters and frozen learning. Its 1,024-tick evaluation is
too short to establish sustained feeding. Pilot preregistration SHA-256:
`36af638cceaf36e758627dc68562c2a2ef8d945a1c610295f7a537a55de4adf2`.

**170 tests passed in 73.642 seconds**, including the optional PettingZoo API
and deterministic-seeding checks. The full log is retained at
`runs/memory-continuity-tests-20261005.txt`.

The full run completed at `runs/memory-continuity-20261005/official/`.
Its protocol was recorded before training with SHA-256:
`3d1748f13937a6a6f729aa573467aa7356f7e381da70e4ec4a9bb428e20bde24`.

## Completed results

The run completed **491,520 world ticks in 999.13 seconds (16.65 minutes)**:
98,304 training ticks and 393,216 evaluation ticks. Every private learner
received 4,096 decisions and 32 updates. Each policy has 89,173 parameters and
its separate consequence predictor has 63,623. Rebuild learners additionally
re-encoded 4,096 private past observations apiece without gradient updates.

The independent audit passed all archived/current source hashes, all 192
physical/sensory trace hashes, trace and event recounts, food conservation,
disjoint seed sets, equal update/experience budgets, private finite parameters,
checkpoint provenance and frozen evaluation state. The chronology review
excluded five boundary markers that occurred after the relevant movement:
two per training arm and one in initial-weight reset evaluation. Original
counts remain preserved, and the primary result is unaffected.

Every row below covers twelve individual evaluation lives over two maps for
each of three training seeds. The same learned brain is reused across its
evaluation treatments; these are not twelve independent training replicates.

| Training / evaluation | Zero-fullness time | Mean fullness | Mean ordinary meals | Lives below 1% zero fullness |
|---|---:|---:|---:|---:|
| Reset / continuous | 43.23% | 37.77 | 4.17 | 4/12 |
| Rebuild / continuous | 36.45% | 40.98 | 5.08 | 4/12 |
| Reset / periodic reset | 43.23% | 37.89 | 4.42 | 4/12 |
| Rebuild / periodic reset | 29.14% | 47.11 | 6.08 | 5/12 |
| Initial weights / continuous | 17.86% | 52.40 | 7.25 | 6/12 |
| Initial weights / periodic reset | 14.72% | 57.81 | 7.92 | 7/12 |
| Random feasible actions | 18.83% | 57.89 | 8.17 | 4/12 |
| Privileged feasibility oracle | 0.00% | 69.84 | 4.00 | 12/12 |

The primary paired comparison, with continuous frozen evaluation:

| Training seed | Reset zero-fullness time | Rebuild zero-fullness time | Direction |
|---|---:|---:|---|
| 73401 | 74.18% | 58.90% | Improved |
| 73412 | 21.95% | 36.74% | Worsened |
| 73423 | 33.57% | 13.73% | Improved |

The overall reduction was **6.78 percentage points**, but the preregistered
requirement was improvement in every seed. That failed. Periodically clearing
context during evaluation did not increase aggregate deprivation in either
trained arm; it reduced it in the rebuild arm. This does not support the
simple explanation that the training/evaluation reset mismatch alone caused
the feeding failures. It does not establish a universally better reset policy.

## Activity, memory limits and individual variation

In continuous evaluation, reset versus rebuild respectively produced mean
thorn contacts **10.17 versus 14.33**, unconscious time **10.83% versus 11.39%**,
unique pre-action tiles **15.75 versus 16.17**, and plantings **2.83 versus 3.83**.
Lower average deprivation did not coincide with fewer injuries or more lives
meeting the low-deprivation threshold. Neither arm demonstrates productive
farming plans or reliable survival.

During training, reset residents returned to previously seen, out-of-sight
food in **44/163 opportunities**, versus **39/159** for rebuild residents.
With strict timing, returns after an actual intervening update were **12/104**
versus **11/104**. Continuous frozen evaluations had **10/83** versus **6/66**
returns; initial weights achieved **54/167**, and random actions **61/165**.
Different trajectories create different opportunities, and visual recall
remains enabled. These counts do not establish learned recurrent recall or
prove the candidate lacks useful memory in other conditions.

The same rebuilt resident could have very different outcomes on two maps:
seed 73412/r1 spent **0%** and **87.81%** at zero fullness, with **13** and **0**
ordinary meals. Seed 73423/r0 spent **0%** and **8.02%**, eating four meals in
each. Aggregate scores should not hide this variation.

The 8,192-tick evaluation horizon, seeds and training budget differ from the
earlier 12,000-tick learning curve. Cross-report percentages are not a matched
architecture ranking. No amber food was present; harmful-food avoidance is
unmeasured. The oracle remains a privileged feasibility control, never a teacher.

## Decision and remaining work

Keep this controller experimental and retain the existing live baseline.
Reconstruction operates correctly but does not show a consistent survival or
food-return advantage under this protocol. The proposed longer-life experiment
can now keep a declared memory-boundary policy fixed; it has not run here.
Curiosity comparisons and additional brain architectures remain separate work.

A read-only health check before the pilot showed the main process at tick
58,499; after the experiment it was the same PID 69,328 at tick 58,757, with
all eight `recurrent-ppo` residents and distinct `laya-npu` resident present.
It was locally available and automatically paused (`manualPause=false`).
No browser heartbeat, stop/restart, live-save write, reward change, model
download or live controller change was performed by this diagnostic.

Evidence includes `preregistration.json`, `source.zip`, `training.json`,
`results.json`, `complete.json`, `audit.json`, `timing-review.json`, the archived
timing-review source, six full population checkpoints and all raw traces.
