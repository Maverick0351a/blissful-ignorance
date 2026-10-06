# GT-01: archived culling and restart pilot

October 6, 2026 (America/Los_Angeles). **Pilot complete and audited;
development gate failed.** The user selected a separate experimental
population. Every live PPO resident and Laya remains preserved.

## Result

The comparison completed **135,168 world ticks in 343.95 seconds**. Selective
restart scored **29/48 prompt feeding lives**, below both continued learning
(32/48) and random restart (31/48). It missed the 80% absolute threshold and
the required advantage over controls. No milestone or live deployment follows.

| Branch | Gather then eat within 64 ticks / 48 | Rate | Feed before zero fullness over 512 ticks / 48 | Ordinary meals | Zero-food ticks / 24,576 |
|---|---:|---:|---:|---:|---:|
| Continue existing learning | 32 | 66.67% | 43 | 119 | 65 (0.26%) |
| Restart selected candidates | 29 | 60.42% | 41 | 110 | 91 (0.37%) |
| Restart random candidates | 31 | 64.58% | 42 | 104 | 78 (0.32%) |

The two evaluation halves scored 14/24 and 18/24 for continuation, 13/24 and
16/24 for selective restart, and 13/24 and 18/24 for random restart. All three
branches had zero amber meals and unconscious ticks. The short, hazard-free
fixture starts with enough fullness for roughly 500 ticks, so small zero-food
percentages do not establish sustained survival.

The two screens selected **c0 and c4**, each with 7/16 prompt successes. Random
restart selected **c1 and c3**. All six source brains were archived before
these decisions. Candidate labels refer only to this experimental pool.

| Slot and source | Screen 1 / 8 | Screen 2 / 8 | Final continue / 8 | Final selective restart / 8 | Final random restart / 8 |
|---|---:|---:|---:|---:|---:|
| c0: 77501/r0 | 5 | 2 | 4 | 4 (new brain) | 4 |
| c1: 77501/r1 | 5 | 5 | 6 | 6 | 5 (new brain) |
| c2: 77513/r0 | 6 | 8 | 6 | 6 | 6 |
| c3: 77513/r1 | 6 | 7 | 5 | 5 | 5 (new brain) |
| c4: 77527/r0 | 4 | 3 | 4 | 1 (new brain) | 4 |
| c5: 77527/r1 | 6 | 4 | 7 | 7 | 7 |

Replacing c0 tied its continuing counterpart; replacing c4 lost three prompt
successes. The c0 replacement is `selected-restart-new-0`, and c4 is
`selected-restart-new-1`. Random replacements have their own corresponding
branch-prefixed identities. Retained candidates' complete final learning
states matched their continuation counterparts exactly in a supplementary
checkpoint comparison. This supports isolation of the branches' surviving
brains; it is not evidence that the newborns inherited anything.

This one-pool result does **not** establish that culling is generally harmful,
that the original candidates cannot improve, or that their memory is defective.
It does fail to support replacing these candidates with fresh random brains
under this future-training budget. Selection scores also varied between the
two screens, reinforcing the limit of treating a short run as an agent's
permanent ability.

**Next bounded recommendation:** inspect the reward attribution and policy
update from a recorded successful feeding experience using experimental
copies. Compare the same hungry, food-carrying observation immediately before
and after that update, then examine retention separately. Preserve trained
brains while diagnosing why learning remains weak. Do not automatically start
another culling generation, add mortality, widen memory or change rewards on
the strength of this failed pilot.

That recommendation was subsequently completed in the
[feeding-credit diagnostic](FEEDING-CREDIT.md): the existing continuation
trajectory and complete final learning states reproduced exactly. All 214
meals received positive credit, with uneven policy gains. See that report for
the current next recommendation; no further culling generation was run.

## Evidence and verification

Evidence directory: `runs/culling-pilot-20261006/`. Preregistration SHA-256:
`7f79d287ceb21c365229cfc3a122d7f5b0de42554a3600893b880cf2df08831e`.
The completion record (local evidence; not bundled),
retirement decisions (local evidence; not bundled),
analysis (local evidence; not bundled) and
audit (local evidence; not bundled) remain local. The audit passed
in **169.80 seconds**, reproducing all **264 episodes, 135,168 transitions and
30,720 frozen selection/evaluation decisions**. It checked source/archive
integrity, survivor and replacement states, identities, private views, masks,
rewards, training-weight continuity, equal future budgets, retirement decisions
and the failed development gate.

The recorded comparison (local evidence; not bundled)
shows the first evaluation map and first pair in all three branches. This is
selection by index, not a chosen success. Its JavaScript DOM/canvas stub check
passed at the first and last frame of every branch; this is not a visual
browser inspection. Playback does not contact the live game. Original source
archives and the nearby-food evidence were preserved unchanged.

Each branch received **12,288 new decisions and 96 PPO updates**. Recorded
training time was 61.10 seconds for continuation, 71.47 for selective restart
and 79.98 for random restart, including 16.82, 19.30 and 21.23 seconds of learning
respectively. Branch order was fixed. These are equal-interaction comparisons,
not evidence that one branch is intrinsically cheaper in wall time. Peak
working set was **692.71 MiB** and peak private commit **1.78 GiB**. Evidence
including derived replay/analysis occupied about **117.41 MiB**, below 512 MiB.

During final evaluation, continuation/selected/random branches visited a mean
17.25/17.23/16.83 unique tiles and made 76/86/88 successful planting actions.
They emitted 2,022/1,929/2,109 tones among 6,144 decisions each. Those counts
do not establish productive farming, useful exploration or learned
communication; no neighboring resident was in reach. All recorded injuries
and invalid actions were zero. Reward components remain in the analysis.

Read-only health checks retained server PID **69328**, all eight live
`recurrent-ppo` IDs and the distinct `laya-npu` ID. Its own activity advanced
from tick 76,816 to 77,237; the final check was automatically paused
(`manualPause=false`, public sharing off). No live state endpoint, browser
heartbeat, restart, controller replacement, reward change or mortality change
was used. Health does not expose controller error details, so this is not a
fresh error-log inspection. No additional culling generation was launched.

## Protocol fixed before execution

Use copies of the six category-PPO brains from the audited
[nearby-food diagnostic](NEAR-FOOD-DIAGNOSTIC.md). Each already received 4,096
training decisions and 32 PPO updates. Keep the complete source checkpoints
byte-for-byte in a new run directory before selection; record hashes and new
replacement identities. This is one six-candidate pool drawn from three
training seeds, not three independent culling experiments.

Candidates c0/c1 originate from seed 77501, c2/c3 from 77513 and c4/c5 from
77527; each pair maps to the original r0/r1 private brains. Evaluate each frozen
candidate on **two separate eight-map screens** with food within reach and an
empty pack. A candidate is eligible for retirement only if it fails to reach
80% gather-then-eat success within 64 ticks in **both** screens. Retire at most
two eligible candidates, ordered by total prompt successes, then full-life
feeding, ordinary meals and stable ID. No candidate is retired merely to fill
a quota. Eight maps per screen remain a small development sample; eligibility
is an experimental management rule, not proof of inherent incapability.

Compare three branches cloned from the same original pool:

1. **Continue:** retain all six brains and their learning state.
2. **Selected restart:** archive and replace the selected candidates with new
   private random initializations. Retain the other brains unchanged at the
   branch boundary.
3. **Random restart:** retire the same number of candidates chosen with fixed
   RNG seed 79107. Use the same fresh initializations by replacement ordinal
   as the selected branch (seeds 79201 and 79213, assigned in stable slot-ID order).

Each branch then receives **16 further 512-tick training lives per candidate**
(2,048 new decisions), followed by eight fresh frozen evaluation maps per
candidate: 48 final lives per branch. All branches receive equal future
interactions. Survivors have 6,144 lifetime training decisions afterward;
newborns have 2,048. The loss of prior experience is part of the restart
treatment and must remain explicit. No weights are inherited by newborns;
there is no mutation, architecture growth or genetic-evolution claim.

Use the existing category architecture, configuration, private observations,
physical masks, actions, rewards, terminal handling and near-food fixture.
All ten tones remain available. There is no food-priority script, teacher
action, mortality change or reward bonus for selection. This isolates candidate
replacement, not social competition. Room orientation is balanced; starting
worlds match across pairs and branches. Candidate-specific action RNG seeds
use `stage_base * 10 + case * 6 + slot`, avoiding the previous cross-map seed
overlap. The stage bases are 79100000 (selection), 79200000 (continuation) and
79300000 (held-out evaluation). Selection never sees final evaluation outcomes.

The development gate requires selected restart to reach **80% prompt feeding**
and outperform both controls by **ten percentage points overall and in each
four-map evaluation half**. Even a pass would require independent new
population replications and the GT-01 confirmation design. All failed outcomes,
including no eligible candidates or identical random/selected retirements,
are retained without changing the protocol.

The smoke check uses prefixes of the same declared map sequences and the same
fresh initialization seeds, with reduced episode counts. Its weights and
retirement decisions are never imported into the main run, and the protocol
source is unchanged between runs. Thus final maps are held out from the main
learners' training, but are not an experimenter-blind confirmation set. Any
confirmation must use new population and map seeds.

## Budget and verification

The main comparison totals **135,168 world ticks**: 24,576 selection, 73,728
continuation-training and 36,864 final evaluation ticks. Its wall cap is 600
seconds and evidence cap 512 MiB; the separate replay audit has a 600-second
cap. A distinct smoke run uses two maps per screen, two continuation lives and
two evaluation maps, totaling 24,576 ticks. Smoke outcomes are infrastructure
checks, not the pilot result. A cap or failure preserves evidence and does not
authorize a retry or more training.

The separate auditor verifies archived source hashes, full survivor learning
state, fresh replacement state/identities, private observations, physics,
actions and rewards, frozen decisions, matched future learning budgets,
training-weight continuity, retirement ranking and final gates. Focused tests
cover two-screen eligibility, incomplete data, no mandatory culling, stable
ties, independent copies, matched newborn initialization, collision-free RNG
coordinates and controls that prevent a spurious improvement claim.

This runner cannot operate the live world and has no server endpoint. It reads
only the named audited experimental source and writes a new directory under
`runs/`. Replacing experimental candidates does not complete GT-07, reproduce
organisms in the game or resolve whether the current PPO's learning limitation
comes from credit assignment, retention or another cause.

The full suite passed **202 tests** in 96.24 seconds. The 24,576-tick smoke
run completed in 63.16 seconds alongside regression tests. Its separate audit
passed in 41.48 seconds, reproducing all 48 episodes and 7,680 frozen decisions.
These are verification results only.
