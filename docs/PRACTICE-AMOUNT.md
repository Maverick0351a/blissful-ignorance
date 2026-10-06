# GT-01: retained learners and additional practice

October 6, 2026 (America/Los_Angeles). **Pilot complete and audited;
development screen passed.** GT-01 remains open.

Evidence directory: `runs/practice-amount-20261006/`. The protocol and source
archive were saved before training. Preregistration SHA-256:
`3c2b2291a22389bbfe6b2693c2e4434d8d672bbe17c9be91d95b46326ead96e5`.

The user authorized a comparison of 16 versus 64 additional practice lives
after the [value-gradient diagnostic](VALUE-INTERFERENCE.md) failed its screen.
This tests practice amount with the existing learning rule. It preserves the
six experimental brains, all live residents and Laya's separate model.

## Result and interpretation

Additional practice substantially improved feeding in this retained pool on
fresh seeded instances of the **same adjacent-food task**. It did not require
replacing the weak responders, changing memory, or removing value gradients.
The primary comparison passed every predeclared development-screen check.

| Frozen policy | Gather then eat before tick 64 | Acquisition before zero fullness over 512 ticks | Eat carried fruit before tick 16 | Eat carried fruit over 64 ticks |
|---|---:|---:|---:|---:|
| Retained starting policy | 75/96 (78.13%) | 88/96 | 58/96 (60.42%) | 69/96 |
| After 16 additional lives | 82/96 (85.42%) | 88/96 | 65/96 (67.71%) | 75/96 |
| After 64 additional lives | **95/96 (98.96%)** | **96/96** | **95/96 (98.96%)** | **96/96** |

The long condition improved prompt acquisition by **13.54 percentage points**
over +16 and **20.83 points** over starting. Carried-food prompt success improved
by **31.25** and **38.54 points**, respectively. Starting is an already trained
checkpoint, not an untrained baseline. Its scores differ from earlier reports
because this test uses different evaluation worlds and action-sampling seeds;
only comparisons within this matched experiment support the dose result.

| Experimental brain | Starting prompt acquisition / 16 | +16 / 16 | +64 / 16 | Starting carried-food / 16 | +16 / 16 | +64 / 16 |
|---|---:|---:|---:|---:|---:|---:|
| c0, previously weak policy-gain responder | 10 | 12 | **16** | 6 | 6 | **16** |
| c1 | 14 | 16 | **16** | 9 | 13 | **16** |
| c2 | 14 | 14 | **16** | 11 | 11 | **16** |
| c3, previously weak policy-gain responder | 13 | 14 | **15** | 10 | 10 | **16** |
| c4 | 15 | 15 | **16** | 12 | 15 | **15** |
| c5, previously weak policy-gain responder | 9 | 11 | **16** | 10 | 10 | **16** |

All three historically paired seed groups improved over +16: **28→32/32**,
**28→31/32**, and **26→32/32**. Five brains improved over +16 and c1 remained
at its ceiling. The predesignated weak responders c0, c3 and c5 all improved.
c3's sole long-condition acquisition miss was case 10: gather at tick 108,
meal at 116, without reaching zero fullness. c4's carried-food deadline miss
ate at tick 16, which correctly fails the strict **before 16** criterion.
Neither late meal is relabeled as a prompt success.

**Decision:** retain the existing learning rule and the archived +64 candidates.
More practice helped this simple feeding task. This does not prove why previous
learning was slow, rule out every memory limitation, or establish sustained
foraging. Fresh world seeds mostly vary translation and cardinal direction in
the existing small-room fixture; they do not introduce a new layout family.
There are six retained brains rather than independent new populations for each
practice amount. The comparison also changes update count alongside experience.

**Next bounded recommendation:** test the frozen +64 policies on ordinary food
that requires movement, first in open rooms and then around an occluder. Keep
an adjacent-food sanity condition, the retained starting-policy control and a
matched category-random control; preserve all source checkpoints. This asks
whether the acquired feeding sequence transfers to navigation before choosing
additional navigation training. Preregister its maps and budget separately;
that follow-up has not run and no live promotion is implied.

## Measured activity

Every adjacent-food condition contains **49,152 resident-ticks and 12,288
decisions**. All recorded zero amber meals, injuries, unconscious ticks and
invalid actions; hazards were absent. Initial fullness lasts roughly 500 of
the 512 ticks, so zero-food occupancy is a weak survival test here. The +64
policies did actually eat in all 96 lives, but longer scarcity, hazards and
reliable navigation remain separate tests.

| Adjacent-food policy | Ordinary meals | Zero-food ticks | Mean unique tiles | Tone actions | Seed conversions | Plantings |
|---|---:|---:|---:|---:|---:|---:|
| Starting | 243 | 104 (0.21%) | 18.10 | 3,909 | 71 | 142 |
| +16 | 283 | 104 (0.21%) | 18.33 | 3,620 | 54 | 107 |
| +64 | 378 | **0** | 16.45 | 4,872 | 5 | 10 |

Fewer seed conversions/plantings accompanied more immediate eating in this
short fixture. Ordinary timed crops still exist with resource ecology disabled,
but the recorder does not attribute consumed food to crop provenance. These
counts establish neither better nor worse productive farming. No buildings,
taps, item gifts or revivals occurred. Separated rooms and tone counts cannot
demonstrate learned communication. Reduced unique-tile count does not by itself
measure useful curiosity; the unchanged curiosity coefficient here is zero.

Each carried-food condition covers 6,144 resident-ticks and 1,536 decisions;
its initial fullness cannot run out within 64 ticks. All had zero zero-food,
unconscious, injury and amber measurements. Dropped starting fruit can be
gathered again; those actions do not create new-food acquisition.

During the four successive 16-life training blocks, ordinary meals totaled
**258, 307, 348 and 377**, with **104, 65, 52 and 0** zero-food resident-ticks.
Each block contains 96 resident lives and 49,152 resident-ticks. These are
changing-policy training measurements, distinct from the frozen test scores.
All training blocks had zero amber meals, unconscious ticks and injury.

## Evidence and verification

The run completed **181,248 world ticks in 483.57 seconds (8.06 minutes)**,
with 192 paired training episodes and 288 paired evaluation episodes. It made
49,152 training decisions, 41,472 frozen decisions and 384 updates. All full
checkpoint files were retained. Peak working set was about **665.13 MiB** and
peak private commit **1.74 GiB**. Each category policy/value network has
**90,538 parameters**, plus its separate **63,623-parameter consequence
predictor**. Architecture and reward definitions were unchanged.

- Completion receipt (local evidence; not bundled)
- Separate audit (local evidence; not bundled)
- All frozen scores and activity (local evidence; not bundled)
- Exact protocol (local evidence; not bundled)
- Interactive report (local evidence; not bundled)
- Recorded first matched map (local evidence; not bundled)
- Per-brain training blocks (local evidence; not bundled)

**217 tests passed in 101.97 seconds**, including optional PettingZoo checks.
Five new tests cover source-history checks, world/sampling-seed isolation,
exact training continuation with a populated optimizer, frozen-copy isolation,
missing/duplicated evaluations and every advancement-screen constraint. The
full suite finished before training. The separate audit passed in **468.30
seconds (7.80 minutes)**: every one of 181,248 world transitions, 49,152
training and 41,472 frozen decisions, 384 updates, and all **24 full individual
checkpoint states** reproduced. It independently recounted the outcome and all
six passed screen checks. Shared native simulator/brain and hashing code are
disclosed; this was a separate implementation run inline, not a second human
or subagent review.

The offline report's JavaScript check passed all 14 scope/task views and three
replay conditions, with nine sampled replay frames. It confirmed the 95/96
headline and passed-screen label, with no external requests. This is a DOM-stub
check, not a visual browser inspection. Derived report source is retained
alongside the evidence.

Read-only main-server health checks retained PID **69328**, eight
`recurrent-ppo` model IDs and distinct `laya-npu`, with public sharing off.
Its own activity advanced from tick **81,877** to **82,041**; both checks
reported automatic pause, `manualPause=false`. No main browser heartbeat,
reset, policy import, live save write, restart, model download or publication
was performed. Health does not expose controller errors, so no fresh
controller-error inspection is claimed. The live roster has not been replaced
with these experimental category-PPO candidates.

## Protocol before execution

Start from the audited culling pilot's `continue-trained.pt`: category-PPO
candidates c0–c5, each with **6,144 prior decisions and 48 updates**. Retain
weights, both optimizers, private learning journals and model identity. Run a
single 64-life continuation per brain and save full states after 16, 32, 48 and
64 lives. The 16-life condition is exactly the prefix of the 64-life condition;
no alternative restart initialization is involved.

The existing adjacent-food fixture and 512-tick life length remain unchanged:
empty packs, fullness 4, four ordinary berries within gathering reach, all
other body needs and health 100. Rooms are separate. Food direction and room
position vary; ecology, automatic refill and hazards are absent. All ordinary
physical actions and ten tones remain. World/body resets between independent
practice lives use the existing native terminal semantics; they do not erase
trained weights or the learner's private journal. The live world is untouched.

Each brain receives 8,192 new training decisions and 64 updates in the longest
condition. Starting / +16 / +64 lifetime totals are **6,144 / 8,192 / 14,336
decisions**, with **48 / 64 / 112 updates**. This is deliberately an unequal
practice-budget comparison, not an equal-experience architecture ranking.

After all training is finished, evaluate independent frozen copies of starting,
+16 and +64 policies on the same 16 fresh maps per historical seed group:
**96 resident lives per condition**. The primary measure is actual ordinary
gather-then-eat before tick 64 and before first zero fullness. A separate
carried-food probe gives one fruit and measures eating before tick 16; its life
horizon is 64 ticks, and it never supplies training experience. Do not pool its
scores with gathering scores. Evaluation state is discarded, not fed back into
practice. Body, maps and per-life action sampling seeds are matched across
conditions; sampling coordinates are unique across residents and fixtures.

The development screen requires every condition below:

- +64 prompt acquisition reaches at least **80%**.
- It exceeds **both +16 and starting by at least 10 percentage points**.
- It exceeds +16 in at least **two of three historical seed groups**.
- Carried-food prompt success falls by no more than **five points** relative
  to either shorter condition.
- Adjacent-food zero-fullness occupancy rises by no more than **0.5 percentage
  point** relative to either shorter condition.

Report every candidate and group, including previously weak policy-gain
responders c0, c3 and c5 specified in advance. Their subgroup does not replace
the primary result. Track ordinary/amber meals, zero-food/unconscious time,
injuries, exploration, farming actions, social actions and reward components.
Farming activity cannot establish productive crop use; separated rooms cannot
demonstrate communication. No random-policy arm is included in this dose test.

One **600-second run**, **600-second separate audit** and **512 MiB evidence
cap** are fixed in advance. The planned workload is **181,248 world ticks**,
**49,152 training decisions**, **41,472 frozen decisions** and **384 updates**.
It matches the prior near-food diagnostic's workload, which took about 500
seconds. Budget/correctness failures preserve partial evidence without an
automatic retry or extension. The independent audit replays all native
decisions, physics, rewards and all 24 individual full checkpoint states, then
recounts the result. It shares the native simulator/brain and hashing helpers,
but not the experiment's rollout or summary functions.

This is one retained developmental pool, with three historical seed groups
and correlated lives/maps. It cannot close GT-01, establish navigation,
long-term survival, cross-task retention or a memory defect, or authorize live
weight imports. No architecture, reward, mortality or Laya change is included.
