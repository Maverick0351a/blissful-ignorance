# Short versus long training lives

October 5, 2026 (America/Los_Angeles). Authorized local experiment in separate
disposable worlds. The current eight PPO residents, their saved experience,
and distinct local NPU Laya remain untouched.

## Question and preregistered design

Does reducing fresh starts improve feeding on unfamiliar maps at the same
total training experience? The recent memory-continuity diagnostic did not
establish consistent benefit, so this comparison retains the existing PPO
memory handling, policy architecture, rewards and zero curiosity coefficient.

| Setting | Short lives | Long lives |
|---|---:|---:|
| Decisions per life | 512 | 4,096 |
| Physical ticks per life | 2,048 | 16,384 |
| Training lives per resident | 32 | 4 |
| Total decisions per resident | 16,384 | 16,384 |
| PPO updates per resident | 128 | 128 |

Three fresh training seeds: **75101, 75112, 75123**. Each arm contains two
private residents per seed, for twelve trained brains. No weights, optimizer,
experience or rewards are shared between residents. Arm execution order
alternates across seeds.

Four equal 4,096-decision blocks cover feeding, routes, scarcity and scarcity.
Within each block, both arms receive exactly the same initial map and bodies.
The short arm repeats that fresh map eight times; the long arm lives through
it once. Both use the same four map identities and stage exposure per seed.
Learned weights, optimizer state and outcome journal persist between lives.
Recurrent state clears after each 128-decision update and at life ends;
time limits bootstrap the value target and do not count as death.

**The reset treatment includes resource and body refresh.** Each start has
four ordinary food units and a body at eight fullness. Across training, this
introduces 128 food units per resident in the short arm and 16 in the long
arm. Inventories, sensory memories and physical development also restart.
These differences are part of the comparison; this cannot isolate longer
elapsed time or memory alone. Actual visited states and optimizer epoch counts
can differ despite matched decision and update counts. World time also
restarts in short lives. Decision counts include unconscious intervals;
conscious action opportunities can therefore differ between arms.

Evaluation freezes weights, optimizers, predictors and learned journals, while
allowing normal sensing and recurrent inference. Each seed has two new maps,
each run for **8,192 ticks** with five conditions: short-trained, long-trained,
initial weights, random feasible actions and a privileged feasibility oracle.
Controls never teach. Every condition shares starting maps and bodies. The
final checkpoint is always evaluated; no selection using test outcomes.

The primary directional hypothesis requires long lives to lower mean
zero-fullness time in **every** seed group and raise overall mean fullness
compared with short lives. This is a three-seed diagnostic, not a statistical
significance claim. Twelve evaluation lives per condition reuse six brains
across maps and are not twelve independent training replicates.

Separately, a competence gate requires each trained arm to improve deprivation
and fullness over both initial and random controls in every seed, have at least
80% of lives below 1% zero fullness, have at least 80% make a safe foodward
crossing with at least 80% of their completed crossings safe, and have no
higher thorn-contact rate than initial controls in any seed. No automatic
promotion into the live population follows a passing result.

## Evidence and limits

Record starting bodies, exact map-state digests, every four-tick action block,
local food opportunities, ordinary/amber meals, zero-food and unconscious
time, exploration, crop maturation, route crossings, finite private model
parameters, checkpoints, runtime and source hashes. A separate auditor
recounts traces and food conservation, checks equal budgets and frozen
evaluation, and verifies checkpoint provenance and matching fresh starts.

These isolated arenas contain thorns but no amber fruit or interacting
neighbors. Harmful-food avoidance, cooperation, language and societal behavior
are unmeasured. Crop maturation is physical output; it does not establish a
productive plan or who ultimately benefited. All primitive actions and ten
tones remain available under the existing physical feasibility mask.

The established fork factory leaves advanced resource ecology disabled.
These crops use the existing fixed 480-tick growth time and three-food yield;
the live world's soil, moisture and irrigation behavior is outside this test.
That setting is identical in both arms and their controls.

The main run is capped at 2,400 seconds and **638,976 world ticks**, with partial
evidence retained on failure. A separate short pilot checks the complete
recording/auditing path; its survival outcomes are not evidence of competence.
The existing Python/PyTorch runtime supplies every dependency. No downloads,
publications, live-save writes or server changes are needed.

## Execution record

Three focused tests passed, including exact observer parity across a PPO
update, equal full-rollout counts at different time limits, and rejection of
a falsified observed position. An initial test held a mutable world dictionary
instead of a snapshot; copying the test fixture corrected it. The experiment
already copies its starting resident records and hashes the world immediately.

The pilot completed **18,432 ticks in 40.76 seconds** at
`runs/life-length-20261005/pilot/`. Its separate audit passed source/trace hashes,
physical and local-opportunity recounts, food conservation, matching starts,
equal budgets, private finite parameters, frozen evaluation and checkpoint
provenance. Pilot preregistration SHA-256:
`325571a30a21bd6dd6be2e4953bbd6b28e1e81ee8c37541c0dec11dad576f88e`.
The pilot uses shorter schedules solely to exercise the full recording path.

The full suite passed **175 tests in 99.026 seconds**, including the optional
PettingZoo API and deterministic-seeding checks. Log:
`runs/life-length-tests-20261005.txt`.

The full run completed at `runs/life-length-20261005/official/` in
**1,583.43 seconds (26.39 minutes)**, within its 2,400-second cap. All
**638,976 world ticks** were accounted for. The protocol and exact source
snapshot were saved before training. Preregistration SHA-256:
`52599c25c2a4dd1911290d93923fec132b4ca9c1662013d4f709ff5a0ce032de`.

The separate auditor passed source/archive and trace hashes, physical and
local-opportunity recounts, food conservation, matched starts, disjoint map
seeds, equal decision/update budgets, private finite changed parameters,
checkpoint provenance and frozen evaluation. All twelve brains finished with
16,384 decisions, 128 updates and 512 optimizer steps; each policy has 89,173
parameters. Evidence includes `preregistration.json`, `source.zip`,
`training.json`, `results.json`, six checkpoints, physical/sensory traces,
`complete.json` and `audit.json`.

Reproduce the audit with the existing compatible Python environment:

```powershell
python experiments/audit_life_length.py runs/life-length-20261005/official
```

After later source changes, use `--archive-only` to verify against the retained
source archive. Raw evidence remains local and excluded from Git.

## Frozen evaluation results

**The longer-life hypothesis failed, as did both competence gates.** Long
training lives increased mean zero-fullness time in all three seed groups.
This is a result for the registered reset treatment and budget; it does not
establish that long lives are inherently worse for learning.

Each condition below covers twelve 8,192-tick lives on six new maps, reusing
six brains. Mean fullness is on the 0–100 body scale. Meal counts are totals;
percentages pool equal-length lives. The oracle is a privileged scripted
feasibility control, never a teacher or live resident.

| Condition | Zero fullness | Unconscious | Mean fullness | Ordinary meals | Lives below 1% zero fullness | Safe active lives |
|---|---:|---:|---:|---:|---:|---:|
| Short-trained PPO | 23.69% | 6.15% | 46.91 | 49 | 4/12 | 1/12 |
| Long-trained PPO | 39.90% | 9.68% | 37.94 | 62 | 4/12 | 2/12 |
| Initial weights | 16.95% | 8.68% | 54.66 | 97 | 6/12 | 0/12 |
| Random feasible actions | 21.65% | 10.13% | 54.04 | 95 | 3/12 | 1/12 |
| Feasibility oracle | 0% | 0% | 69.86 | 48 | 12/12 | 12/12 |

Long training raised aggregate deprivation by **16.22 percentage points**
despite more meals. Timing and distribution across individual lives matter;
total consumption does not establish dependable feeding. Both trained arms
reduced thorn-contact rates relative to initial weights in every seed, but
their low safe-route counts and substantial deprivation fail the wider gate.

| Training seed | Short zero fullness | Long zero fullness | Long improved? |
|---|---:|---:|---|
| 75101 | 16.19% | 25.32% | No |
| 75112 | 35.87% | 68.11% | No |
| 75123 | 18.99% | 26.28% | No |

Individual results retain substantial map dependence. Each cell below lists
zero-fullness percentages on the seed's first and second evaluation maps;
resident IDs identify matched initializations, not shared trained weights.

| Seed / resident | Short: map 1 / map 2 | Long: map 1 / map 2 |
|---|---:|---:|
| 75101 / r0 | 0 / 0 | 87.81 / 13.49 |
| 75101 / r1 | 19.74 / 45.03 | 0 / 0 |
| 75112 / r0 | 0 / 87.81 | 87.81 / 24.76 |
| 75112 / r1 | 4.21 / 51.48 | 72.08 / 87.81 |
| 75123 / r0 | 14.61 / 0 | 87.81 / 0 |
| 75123 / r1 | 9.53 / 51.82 | 0 / 17.30 |

## Activity and interpretation

**Exploration and tones.** Short-trained policies chose tones on 82.57% of
conscious decisions; long-trained policies on 82.26%, versus 69.39% for
initial weights and 69.01% for random feasible actions. These arenas contain
no interacting neighbors, so those sounds do not demonstrate communication.
Mean distinct visited tiles were 15.75 short, 11.17 long, 22.08 initial and
22.00 random. The physical trace and local-opportunity records agree on these
counts. Repetition comes from the evaluated policies; no NPC routine supplied
their action sequence.

**Farming.** Short-trained lives made 8 seed conversions, planted 11 crops
and matured 33 food units; long-trained lives made 46 conversions, planted
55 crops and matured 147 food units. Only one short evaluation life planted,
compared with five long lives. Initial policies planted 239 crops and matured
684 food units. Crop production alone therefore cannot establish learned
planning or an improvement over untrained behavior. Long-trained lives had
7.25 food units remaining on average, versus 2.00 short, alongside worse
deprivation; these population averages do not prove every hungry individual
had accessible food. All amber-meal counts were zero because the arenas
contain no amber fruit.

**Local opportunities.** Mean conscious decision counts with visible ordinary
food were 489.75 short and 978.00 long; with adjacent food, 192.25 and 579.33.
Food was carried on 74.67 versus 329.08 decisions per life, but it was carried
while fullness was below 60 on only 7.25 versus 14.50 decisions. These measure
available sensory/body conditions, not comprehension of food or successful
planning. They can help localize failures without granting global information
to a learner.

**Training exposure.** Both arms supplied 98,304 total resident decisions
across their six brains. Short had 92,497 conscious opportunities; long had
89,256. Training zero-fullness time was 17.78% short versus 28.85% long, and
unconscious time was 5.91% versus 9.20%. Short training recorded 434 meals and
804 matured food units; long recorded 206 and 813. Repeated fresh supplies
and bodies are part of the reset treatment, so equal decisions do not equate
nutrition availability or conscious experience. Training totals are descriptive
and do not change the preregistered evaluation verdict.

**Next candidate, not implemented:** compare the flat action head with a
learned choice of verb followed by learned arguments. The present vocabulary
contains ten separate tone choices; even random feasible actions frequently
select one. A learned verb/argument factorization could change that exploration
distribution while retaining all ten sounds and every physical action. The
current run does not isolate this mechanism or prove that it will help.
Any comparison needs fresh seeds, identical physics/rewards, independent
weights and the existing feeding/route gates. No feeding routine, assigned
job or hand-selected survival action should substitute for learning.

## Live population and documentation

No live save, weight, reward, architecture or Laya model was modified or
replaced. A read-only `/health` check after completion reported the original
server PID 69328, `independent-learning`, all eight `recurrent-ppo` IDs and
distinct `laya-npu`, at tick 61,333. It was automatically paused
(`manualPause=false`); no browser heartbeat or resume command was sent.
That endpoint does not expose controller errors, so this check establishes
the reported controller/roster and process continuity, not a fresh end-to-end
controller-error audit.

The user also identified stale present-tense descriptions of scripted agents
in older notes. Current-facing documentation was corrected, and historical
baseline descriptions were labeled as historical. This changes documentation,
not resident behavior. [Current controller boundaries](UNSCRIPTED-POPULATION.md)
distinguish model-selected actions from programmed physics, senses and rewards.
