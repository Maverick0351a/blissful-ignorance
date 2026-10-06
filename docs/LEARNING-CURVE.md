# First standardized PPO learning curve

October 5, 2026. This is the first benchmark through the approved
[parallel environment adapter](PARALLEL-ENVIRONMENT.md). It measures learning
at predetermined budgets, preserving the current world physics, senses,
primitive actions and private recurrent PPO baseline.

## Preregistered design

- Fresh training seeds **71201, 71212**, with two independent residents each.
- Measure checkpoints at **2,048, 8,192 and 16,384 decisions per resident**.
- One decision advances four world ticks. Each training episode lasts 512
  decisions, with an eight-episode cycle: feeding twice, routes twice,
  scarcity four times. Each episode has its own predetermined world seed.
- Two separate validation world seeds per training seed, evaluated at each
  checkpoint. Two additional final-test world seeds per training seed are
  evaluated only at the maximum budget, regardless of validation results.
- Each evaluation lasts 12,000 ticks. Initial weights, random feasible actions
  and a privileged feasibility oracle are recorded separately; they never
  teach the learners. Each policy's weights, optimizer and learned evidence
  remain frozen during evaluation.
- Four private recurrent PPO learners; 40 evaluation runs / 80 individual
  evaluated lives. Repeated lives of the same learned brain are not independent
  training replicates. This is a two-seed diagnostic, not a conclusive ranking.
- Maximum 30 minutes of benchmark wall time; partial runs are retained and
  cannot be reported as complete. No trial weights are promoted automatically.

The source snapshot, fixed seeds, schedule, evaluation protocol and runtime
versions were recorded before training. Preregistration SHA-256:
`9226b20715f81cbd0175fd83a3234b2f2b6567208fe7a9add76932177ddc29e7`.

PPO keeps its existing body reward, local consequence predictor and zero
curiosity coefficient for this baseline. No teacher supplies actions. The
curriculum moves the same finite food stock closer in early episodes; it
does not change food totals or supply rewards for intermediate recipes.
The 2,048-decision checkpoint ends after the first feeding/routes episodes;
the larger budgets also include scarcity episodes. The curve measures prefixes
of this fixed curriculum, not training length isolated from task exposure.
Time-limit cutoffs now bootstrap the value function and reset recurrent
carry. Earlier experiments used terminal cuts, so cross-report differences
cannot be attributed solely to training length or PettingZoo.

The reliable-feeding gate requires less zero-food time and higher fullness
than initial weights in each seed; at least 80% of lives below 1% zero fullness;
and safe, active crossing evidence in at least 80% of lives, alongside reduced
thorn-contact rate. These are behavioral measurements, not judgments inferred
from reward or a model's explanation.

## Pilot and audit

The first native pilot completed but its independent audit rejected a missing
`thorn_opportunities` report field. The runner was corrected to count proximity
from the conscious resident's pre-action position. The original failed pilot
is retained; no results were overwritten.

A fresh output folder using the actual PettingZoo wrapper completed 5,632 ticks
in 9.75 seconds. Its independent audit passed source/archive hashes, all trace
totals, food conservation, disjoint seed sets, exact frozen evaluation state,
checkpoint provenance and private finite parameters. The 512-tick pilot horizon
is shorter than initial-food depletion and cannot establish reliable feeding.

Evidence folders (local and ignored):

- `runs/learning-curve-20261005/pilot-native/` — failed audit, preserved.
- `runs/learning-curve-20261005/pilot-pettingzoo/` — corrected pilot and audit.
- `runs/learning-curve-20261005/official/` — full run, status below.

The auditor independently recounts every evaluation trace, verifies source and
checkpoint hashes, compares checkpoint weights/optimizer state with frozen
evaluation digests, checks each private parameter allocation, recomputes gates,
and reconciles training/evaluation tick counts. Per-tick fullness aggregates
come from the recorded simulator trace; the audit checks aggregation, not a
second implementation of world physics.

```powershell
$env:PYTHONPATH = '<user-home>\Applications\Godhood-RL-Tools-20261005'
$python = '<user-home>\Projects\laya-lab\.venv\Scripts\python.exe'
& $python experiments/learning_curve.py --interface pettingzoo --output runs/new-learning-curve --wall-seconds 1800
& $python experiments/audit_learning_curve.py runs/new-learning-curve
& $python experiments/learning_curve_behaviors.py runs/new-learning-curve
```

The native interface is also supported via `--interface native`. It runs the
same physical core without optional PettingZoo imports. Both routes use CPU
inference/training and reuse the existing PyTorch installation.
Actions and physical steps use the selected interface; existing PPO sensing
uses the core's private raw-observation compatibility bridge. The declared
packed-array interface has separate parity/space tests, not a new external
trainer in this benchmark.

The optional behavior report is explicitly post-hoc and requires a passing
audit. It records per-life action frequencies, consecutive repeated choices,
visited tiles, first-meal observation intervals and farming actions. It cannot
change the preregistered gate or select a checkpoint. Repetition and tone use
are descriptions, not proof of a failure mechanism or communication.

## Completed results

The run completed **611,072 world ticks in 1,010.80 seconds (16.85 minutes)**.
The independent audit passed all source/archive and checkpoint hashes, all
40 trace recounts, conservation, frozen learning, distinct seed sets, private
finite parameters and recomputed gates. Each of the four learners finished
with 128 PPO updates. These are behavioral results from the actual adapter,
not synthetic optimizer data.

| Validation checkpoint | Zero-fullness time | Mean fullness | Mean safe meals | Lives below 1% zero fullness |
|---|---:|---:|---:|---:|
| 2,048 decisions | 39.55% | 40.36 | 7.75 | 2/8 |
| 8,192 decisions | 42.35% | 28.15 | 3.50 | 3/8 |
| 16,384 decisions | 68.76% | 18.62 | 3.13 | 2/8 |
| Initial weights | 16.62% | 61.36 | 11.38 | 1/8 |
| Random feasible actions | 15.17% | 60.29 | 10.50 | 3/8 |
| Feasibility oracle | 0.00% | 56.33 | 4.00 | 8/8 |

**Every validation gate failed.** Average feeding worsened with more training,
although individual residents sometimes improved on particular maps. The
8,192-decision checkpoint had three low-deprivation lives but only two active,
safe-route lives; this is insufficient for the 80% requirements. Lower thorn
contact at later checkpoints did not coincide with sustained feeding.

| Final-test condition | Zero-fullness time | Mean fullness | Mean safe meals | Lives below 1% zero fullness |
|---|---:|---:|---:|---:|
| Trained, 16,384 decisions | 45.88% | 28.41 | 3.63 | 2/8 |
| Initial weights | 15.13% | 60.42 | 9.88 | 1/8 |
| Random feasible actions | 13.26% | 69.55 | 13.13 | 1/8 |
| Feasibility oracle | 0.00% | 56.32 | 4.00 | 8/8 |

**The final gate failed.** Trained policies had two low-deprivation lives but
zero lives passing active-safe-route criteria, and worse mean deprivation/fullness
than initial weights in both seed groups. Initial/random controls can have
better averages yet fewer lives below a stringent 1% cutoff; report both.
The oracle demonstrates feasibility and has privileged information, so it is
not evidence that a comparable learned policy was supplied or discovered.

The four private trained residents varied sharply. The two percentages in
each row correspond to its two final-test maps, both starting from the same
frozen checkpoint:

| Training seed / resident | Zero-fullness time by map | Safe meals by map | Plantings by map |
|---|---|---|---|
| 71201 / r0 | 39.60%, 0.00% | 2, 4 | 0, 0 |
| 71201 / r1 | 0.00%, 91.68% | 4, 0 | 0, 0 |
| 71212 / r0 | 25.01%, 27.41% | 4, 15 | 0, 12 |
| 71212 / r1 | 91.68%, 91.68% | 0, 0 | 0, 0 |

One resident planted 12 times and ate 15 meals in a final life, but still spent
27.41% of that life at zero fullness. Executing farming actions is insufficient
evidence of timely, reliable survival or an understood farming strategy.

## Descriptive behavior findings and next experiment

The post-hoc report (`behavior-summary.json`) records all 80 evaluated lives
and hashes every trace. Final trained lives visited a mean **13.63 distinct
tiles**, versus **23.00** for random actions, and planted **1.50** times versus
**27.13**. Consecutive identical choices were **13.08%** versus **6.50%** of
conscious decisions. These observations are consistent with reduced exploration,
but do not establish its cause.

Tones occupied **71.32%** of trained and **69.59%** of random conscious choices.
The ten separate tone variants make up a large share of the feasible action
set. High tone frequency is present in both conditions and cannot by itself
explain the survival gap; the separated arenas also cannot establish useful
communication. Keep all tones available in future tests.

No checkpoint was promoted to the live population. Laya remains a separate
persistent model. This run supports the environment/benchmark implementation
and rejects the assumption that extending this exact training schedule reliably
improves feeding. It does not reject PPO in general or establish another core
as better.

Follow-up: the [memory-continuity diagnostic](MEMORY-CONTINUITY.md) compared
reconstruction after updates with the existing reset. It failed its consistent
improvement criterion, and periodic-reset evaluation did not reveal a general
feeding benefit from retaining hidden state. This informs the next experiment
without changing this curve's original protocol or results.

Recommended next diagnostic: keep the learner, reward and primitive actions
fixed while testing longer uninterrupted training lives at matched total
decision budgets. Explicitly measure exploration, delayed feeding and retention
of earlier competence. Then test a learned conditional verb/argument policy as
a separate factor, with every physical action and tone retained. CfC remains
a separately matched core experiment, not a demonstrated remedy for this result.

## Limits

These are isolated finite-stock fork arenas, not the shared live valley. They
test feeding/navigation and expose farming as an available primitive, but do
not establish competition, emergent language, general problem solving or
lifelong adaptation. Training worlds reset after 2,048 physical ticks, while
evaluation lasts 12,000 ticks continuously. This can leave long-life resource
planning undertrained even as total training experience increases; this run
does not isolate that possible mechanism. Only two training seeds are used. Repeated orientations
and route lengths can occur across distinct seeds: seed separation does not
prove an unseen structural task family. There is no amber fruit in these
arenas, so harmful-food avoidance is not measured. Longer training can worsen
behavior; the benchmark records that outcome rather than choosing a favorable
checkpoint from the final-test maps.
