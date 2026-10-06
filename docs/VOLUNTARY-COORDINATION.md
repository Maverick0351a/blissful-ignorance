# Voluntary food coordination

October 5, 2026 (America/Los_Angeles).

This experiment tests a prerequisite for emergent leadership: whether one
resident's voluntary tones help another obtain food. It does not appoint a
leader, define tone meanings, prescribe routes, or make a resident follow a
neighbor. The existing [three-tone diagnostic](TONE-TRIAL.md) learned a useful
code but used a forced message and scripted route skills. That historical
experiment remains reproducible; this is a separate, harder test.

## Completed result

The three-seed run completed **79,872 ticks in 183.04 seconds**: 60 training
episodes and 96 frozen evaluation episodes. **All three communication gates
failed.** No policy was promoted and no leadership behavior is established.

These results describe the resident that **initially could not see food**, over
24 matched evaluation cases per arm. The same six trained brains are reused
across cases; these are not 24 independent model initializations.

| Evaluation arm | Cases with an ordinary meal | Ordinary meals | Time at zero food | Time unconscious |
|---|---:|---:|---:|---:|
| Trained, intact tones | 2/24 (8.3%) | 3 | 47.83% | 4.30% |
| Trained, muted tones | 1/24 (4.2%) | 2 | 49.97% | 4.43% |
| Trained, shuffled symbols | 2/24 (8.3%) | 3 | 47.83% | 4.30% |
| Matching initial weights, intact tones | 0/24 | 0 | 51.37% | 4.69% |

For seeds 65113/65124/65135 respectively, intact fed-case counts were **1/8,
1/8, 0/8**; muted counts were **1/8, 0/8, 0/8**; shuffled counts matched intact.
Training improved this narrow feeding count over initialization in two seeds,
but never approached the communication threshold. The one extra fed case
relative to muting does not demonstrate a reliable communication benefit, and
there was no feeding advantage over relabeled symbols.

The information advantage alone was insufficient: even initially informed
residents ate in only **6/24** intact cases. Both residents ate in **1/24**.
Initially uninformed residents saw food at some point in **6/24** intact cases,
compared with **10/24** under matching initial weights. Basic exploration and
food acquisition remain weak in this task and training budget.

Across both residents in the intact arm:

- **4,168/5,944 conscious decisions (70.12%)** were tones. Uniform choice over
  each observation's physically allowed action slots would average **68.65%**
  tones. The high vocalization rate is consistent with having ten tone slots;
  it is not evidence of a meaningful language or intentional cooperation.
- Initially uninformed residents received tone memory while no ordinary food
  was visible in **1,635 decisions**. Repeated memories count repeatedly here;
  this is sensory exposure, not 1,635 distinct messages or successful requests.
- Residents made **1,307 successful moves**, visited **14.15 unique tiles per
  life on average**, tapped **eight** times, gave items **zero** times and
  planted **22** times. Planting counts do not establish a productive farming
  plan. No harmful meals occurred because the arena contained no harmful food.

The useful next experiment is the already proposed **learned action hierarchy**:
let the model choose an action category, then its direction, item or tone. All
ten symbols remain available. Compare that policy with the flat action head on
matched ordinary foraging before returning to coordination. The current result
motivates that comparison; it does not prove action-slot imbalance caused the
failure. No action hierarchy or reward change was deployed by this work.

## Protocol fixed before the behavioral run

- Three fresh initialization seeds: **65113, 65124, 65135**. Two independent
  recurrent PPO brains per seed, using the existing network, action mask,
  optimizer and individual physiological reward. No pretrained model, shared
  weights, centralized critic, team reward, signaling bonus or behavior script.
- Four warmup episodes with food visible to both residents, then sixteen
  episodes where only one initially sees it. All episodes last 512 world ticks
  (128 decisions per agent); each agent receives 2,560 training decisions.
- Both begin with fullness 2/100, empty packs and otherwise healthy bodies.
  Four berries are shared in a small room. An opaque divider has an open route
  around each end. Food appears on either side; room orientation and which
  resident initially sees food vary. There are no hazard resources, water
  sources or ecological replenishment in this narrow task. Ordinary seed-making
  and planting actions remain possible after gathering food.
- Eight evaluation cases per seed combine four orientations with two food
  locations. The divider is longer and food farther sideways than in training.
  Each resident starts with the informational advantage in four cases. Both can
  move, gather, eat, tap, give, plant or select any of the ten tone symbols.
- Four paired arms: trained/intact, trained/muted, trained/shuffled and matching
  untrained initialization/intact. Evaluation weights are frozen, recurrent
  state starts empty and policy RNGs match across arms of each case. Every arm
  starts with the identical world, bodies, inventories and world RNG.
- **79,872 world ticks** total, a **600-second run budget**, a **256 MiB evidence
  cap**, and an optional `STOP` file. A timeout or error retains partial evidence
  and produces no completion receipt. No automatic extra training or retuning
  follows a failed gate.

The advancement gate requires **every seed** to feed the initially uninformed
resident in at least 75% of intact evaluations, at least 25 percentage points
more than each tone intervention, and in more cases than its matching initial
weights. Eight cases per seed are a small pilot, not a significance test. The
gate concerns ordinary meals; high signaling counts or nearby movement alone
cannot pass it.

## What the controls change

Muted removes tones from both instantaneous hearing and private auditory
memory. Shuffled applies a new per-episode, per-recipient permutation of the ten
symbols, with no symbol left unchanged. The mapping stays consistent within an
episode, including repeated memories of the same event. Timing, direction,
strength and repeated patterns remain available. This is a **symbol-label
intervention**, not a temporal shuffle; per-label frequencies are relabeled,
not preserved under their original labels. It can disrupt a learned code while
leaving a positional beacon useful.

Other sensory channels stay intact, including visual movement, footsteps and
taps. Therefore, lack of an intact-versus-shuffled advantage would not rule out
every form of useful interaction. Conversely, a positive result would still
require larger replication and tests of stable voluntary influence before
calling it leadership.

Role labels, food coordinates and evaluation conditions exist only in the
recorder. A brain receives the ordinary private `World.observe()` data after
the specified tone intervention. Its actions are the existing primitives; there
is no food-destination selector or forced signal/listen phase. Rewards are the
unchanged individual's food relief, hunger distress and injury consequences.
That preserves the question of whether useful signaling develops, but may offer
weak credit for helping another resident within these short episodes.

## Evidence and reproduction

Use the existing local PyTorch Python environment; no installation is required:

```powershell
& '<user-home>/Projects/laya-lab/.venv/Scripts/python.exe' experiments/coordination_trial.py --output runs/coordination-20261005 --wall-seconds 600
& '<user-home>/Projects/laya-lab/.venv/Scripts/python.exe' experiments/audit_coordination.py runs/coordination-20261005 --wall-seconds 300
```

The output directory must be new. The runner writes its protocol and source
hashes before training, archives the sources, saves initial/final checkpoints,
and records every action, actual private input, tick's bodies and episode's
complete starting/ending world in a compressed trace. The separate auditor
replays all world transitions, reconstructs private inputs, reruns every frozen
evaluation decision from the saved model, recounts outcomes and verifies hashes
and the advancement gate.

The 320-tick infrastructure smoke check and its independent replay passed.
Its behavioral scores are not evidence for or against communication. Six new
unit tests cover private asymmetric information, reachability, role balance,
both tone channels, stable label intervention, independent brains, unchanged
individual rewards, frozen determinism and rejection of activity without a
channel benefit.

These are disposable experimental residents. The main population, including
its eight PPO learners and distinct NPU Laya, is neither loaded nor modified.
Clef and Laya are not participants in this pilot; it does not rank either model
as a potential leader. No experiment weights are promoted to the live game.

Local evidence is in `runs/coordination-20261005/`, including
`preregistration.json`, its SHA-256 receipt, `source.zip`, `trace.jsonl.gz`, six
initial/final checkpoint files, `results.json`, `analysis.json` and
`complete.json`. The initial infrastructure run is retained separately at
`runs/coordination-smoke-20261005/`. Evidence occupied about 40 MiB before the
final audit receipt, below the 256 MiB cap.

The separate audit **passed**, reproducing all **156 episodes / 79,872 world
transitions** and all **24,576 frozen evaluation decisions** in **110.08
seconds**. It matched private inputs, exact ending worlds, actual food relief,
deprivation, behavior counts, frozen parameter hashes and the failed gates.
The full repository suite passed **181 tests**, including the optional
PettingZoo compatibility tests. Preregistration SHA-256:
`9824d933bb7e839e87f846f78c3093cadc3c2c54b1b2f226c7d4ad12748d1fd6`.

Read-only main-world health checks retained server **PID 69328** and the same
eight `recurrent-ppo` IDs plus `laya-npu`. The world progressed independently
from tick 64,585 to 64,695 during this work and was automatically paused at the
final check (`manualPause=false`, `publicDemo=false`). No browser heartbeat,
main-world experiment, restart or checkpoint migration was performed. The final
health response is preserved as `live-health-after.json`.
