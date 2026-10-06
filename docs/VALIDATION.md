# Validation record

## Short versus long training lives — October 5, 2026

**175 tests passed in 99.026 seconds**, including the optional PettingZoo API
and seeding checks. Three new tests verify equal experience and map exposure,
exact observer/world/brain parity across a PPO update, equal full-rollout
counts at different time limits, and rejection of a false observed position.
Python compilation and diff checks passed. Only experiment tooling and
documentation changed; no browser behavior changed.

The 18,432-tick pilot and the **638,976-tick full comparison** both passed
separate audits. The full run took **1,583.43 seconds**. The auditor verified
source/archive/trace hashes, physical and local-opportunity recounts, food
conservation, matched starts, disjoint map seeds, all twelve finite private
changed networks, 16,384 decisions and 128 updates per brain, checkpoint
provenance and unchanged learned state during evaluation.

Long training increased zero-fullness time in all three seeds: aggregate
**39.90% versus 23.69%** for short training. Both competence gates failed.
The main server retained PID 69328 and its eight PPO plus distinct NPU Laya
IDs on a read-only health check at tick 61,333. No live checkpoint or
controller was changed; the health endpoint does not establish absence of
controller errors. Stale current-tense descriptions of earlier scripted
baselines were corrected without changing those historical results.
[Protocol, individual results and limits](LIFE-LENGTH.md).

## Recurrent context reconstruction — October 5, 2026

The full **170-test suite passed in 73.642 seconds**. Six new checks cover exact
reset-baseline equivalence, reconstruction under updated weights, bounded
private detached inputs, exact resume, frozen learning, unchanged recorded
trajectories, food-return accounting and rejection of a false boundary marker.
Two subsequent chronology checks also passed. Python compile and diff checks
passed; no browser UI change was made.

The 12,288-tick pilot passed its audit. The full three-seed comparison completed
491,520 ticks in 999.13 seconds, with twelve private trained brains, 48 frozen
evaluation worlds and 96 evaluated lives. Separate audits verified source and
trace hashes, checkpoint provenance, every trace/event total, food conservation,
private finite parameters, equal budgets and unchanged learned evaluation state.
A labeled timing review excludes five markers that occurred after movement
already achieved the measured outcome; the primary feeding result is unaffected.

The primary hypothesis failed. Rebuilding improved mean deprivation in two
seeds but worsened the third; low-deprivation lives stayed at 4/12 under
continuous evaluation. The original live server PID and all nine model IDs
remained present. No live checkpoint, controller or Laya model was changed.
[Protocol, results and limitations](MEMORY-CONTINUITY.md).

## Parallel environment and first learning curve — October 5, 2026

**164 tests passed in 67.898 seconds**, including nine new environment tests.
The optional dependencies were downloaded only after exact-artifact approval,
hash/archive checked, inspected, and installed offline into an isolated target.
The existing Torch/NumPy environment was reused.

PettingZoo 1.27.0's official Parallel API and deterministic-seeding tests passed.
Additional checks establish declared observation-space membership, exact native
and wrapped trajectories, exact existing-world/PPO-state parity through actual
updates, invalid-batch atomicity, deterministic resets, snapshot continuation,
time-limit truncation, private sensory inputs and all ten tones. A corrected
5,632-tick wrapper pilot passed its independent trace/checkpoint audit. Its
short horizon is a mechanics check, not survival evidence.

The full 611,072-tick run completed in 1,010.80 seconds. All 40 evaluation traces
were independently recounted, and source/checkpoint hashes, food conservation,
private finite parameters, frozen learned state, seed separation and gates
passed the audit. Every behavioral gate failed. Final trained PPO spent 45.88%
of time at zero fullness, versus 13.26% for random actions. The post-hoc behavior
report shows per-resident variation and reduced exploration; it changes no gate
or model selection. [Full results](LEARNING-CURVE.md).

No live world or Laya model was loaded, reset or promoted by these experiments.
Details and dependency review: [PARALLEL-ENVIRONMENT.md](PARALLEL-ENVIRONMENT.md).

## Private retention and exploration — October 5, 2026

The 2x2 screen completed 602,304 ticks in 639.13 seconds, with three training
seeds, 24 private trained brains, 42 evaluations and 84 evaluated lives. Its
source/checkpoint hashes, trace recount, conservation, frozen evaluation,
parameter/update budgets, private state and bounded replay audit passed.
All four conditions failed reliable feeding; both directional main-effect
hypotheses failed. More retained/replayed nutrition did not establish reliable
survival. No weights were promoted. Five new retention tests passed as part
of the suite. [Full report](RETENTION-EXPLORATION.md).

## Matched visual replay comparison — October 5, 2026

150 tests passed in 24.509 seconds. Seven new tests cover masked Double-DQN targets and terminal cuts, gradient-free recurrent burn-in with temporal credit afterward, feedforward absence of temporal gradients, replay sequence coverage/episode boundaries, independent resident updates, frozen learned state, all ten available tones and local-only features, and exact resumed behavior through updates with pending rewards and both private RNGs.

The separate pilot completed 6,400 ticks in 8.36 seconds and passed trace/hash/accounting checks. The fresh three-seed confirmation completed 797,184 ticks in 977.94 seconds: 48 frozen evaluations, six held-out maps, 12 lives per condition. Source/preregistration/archive hashes, identical arenas, food conservation, trace recounts, frozen learning and independently calculated gates passed. A further checkpoint audit verified all 18 learned networks changed from initial weights, were finite and independently stored, preserved replay episode boundaries, and matched the evaluation's learned-state digests.

**All feeding gates failed.** Trained zero-food time was 26.69% PPO, 81.97% feedforward replay and 68.76% recurrent replay. A post-hoc retention audit found no eating transitions retained in 10/12 replay learners despite meals earlier in training. No live controller or Laya state was changed. [Protocol, results and limits](REPLAY-COMPARISON.md).

The user separately approved ALIEN's 61,489,415-byte official portable ZIP. Archive CRC/path checks passed; it was extracted into its own new directory and launched as v5.0.0-alpha.36. A responsive process and program log confirmed the RTX 5080, simulation initialization and CUDA/OpenGL interop, using 3,180 MB GPU memory at startup. Its first-run default simulation was downloaded automatically by the application. No native screenshot, evolutionary benchmark or agent-connection setup was performed. [ALIEN installation record](ALIEN-EVALUATION.md).

## Continuing lives and distinct Laya resident — October 5, 2026

Full suite: 142 tests passed in 23.531 seconds. After the installed Laya checkpoint exposed a 256-token question head, the conservative 192-token budget compatibility check was corrected; all seven Laya tests (including the new configuration regression) and all three mixed-population tests passed. JavaScript syntax and both modified PowerShell script parsers passed.

New checks cover deterministic recorded versus unrecorded continuation across neural updates, absolute decision timing from a non-aligned tick, untouched pending experience on stop, independent trace deprivation recounts, exclusive checkpoint ownership, both model types round-tripping together, original PPO preservation on Laya arrival, and inference failure pausing before advancing other brains or world time.

Actual work: two eight-PPO observation windows continued 9,216 ticks, with a separate subagent audit of archived source/checkpoint hashes, every tick, counters, and cross-window state equality. The real existing NPU model then controlled Laya for 16 choices over 64 ticks. Main launcher restart restored her history and the eight PPO residents. Browser verification displayed nine residents, explicit PPO/Laya labels, her 16 decisions, frozen weights, 15 measured outcomes, and paused tick 47,392. Screenshot: `runs/laya-population-verified.png`.

This establishes instrumentation, persistence and actual inference. Reliable survival, intentional farming and meaningful communication remain unproven. See [POPULATION-WATCH.md](POPULATION-WATCH.md).

## October 5: independent primary population

129 tests passed in 20.941 seconds after switching the playable entry point to independent learning for all eight residents. New tests verify real separate weight updates, exact continuation of eight brains, preservation of both older pair-checkpoint formats, unchanged world migration and backup bytes, rejection of missing controllers without a scripted fallback, and read-only learned hosting. JavaScript, PowerShell parsing and diff checks passed. Live migration at tick 37,342 preserved the complete world, with eight fresh brains and zero scripted residents. Browser stepping, the previously scripted Reed's neural inspector, manual saving and a real graceful stop/restart passed. The resumed tick 37,349 checkpoint retained two decisions and one completed transition for every brain. Document and viewport width both measured 1,503 pixels. Screenshot: `runs/unscripted-population.png`. Historical checks below describe prior versions. [Controller transition](UNSCRIPTED-POPULATION.md).

October 4, 2026, America/Los_Angeles. Local v0.2 foundation; no public deployment.

- **44 Python tests passed** in 14.6 seconds using the existing PyTorch-capable runtime. The cases cover deterministic save continuation, local/occluded observations, material accounting, construction legality, layered floors, doors/occupancy, roofed rooms, local fading/attenuated sounds, independent lossy memories, old saves, neural input/state isolation, and HTTP lifecycle/control behavior.
- `node --check web/app.js` passed.
- The updated CPU/CUDA quick benchmark and synthetic recurrent optimizer checks passed. [Hardware observations](HARDWARE.md) give the sample sizes, timing boundaries, and limitations. No gameplay policy was trained.
- Browser controls built a wood wall and rejected movement into it. Visible tiles dropped from 80 to 73. Dismantling recovered its two wood. A door was built and opened; the visible count returned to 80. A floor was placed under the player. Temporary test structures were dismantled and the materials returned.
- Switching to Pip's perspective showed a distinct 69-tile visible area and three personal recalled sketches with different similarity scores and ages. The global minimap was hidden. Ordinary world time was resumed after the checks.
- Browser console had no warnings/errors at final inspection; document width equaled the normal viewport width (1503 CSS pixels). A separate narrow-viewport test was not performed in this update.
- The pre-update world was backed up locally. The updated autosave was read back: original structures matched, player position matched, all test materials were restored, and nine separate memory owners were saved. The existing manual checkpoint was not overwritten by the UI tests.
- An independent read-only review of the simulation, visual-memory module and neural encoder found no additional confirmed correctness bug or cross-resident observation leak. It identified chronological sound truncation and old-sketch retention during approximate merging as limitations, now documented. This review is additional evidence, not a formal proof of isolation.

Those checks established the mechanics and local interface, not learned survival, useful memory-driven planning, emergent culture, or general intelligence. At that historical milestone the visible policies were the labeled scripted baseline; neural learning integration followed later.


## v0.3 hazards and fainting

53 Python tests passed in 15.0 seconds after adding hazards and unconscious recovery. JavaScript syntax passed. Disposable browser fixtures verified amber consumption (100 to 80 health), thorn contact (80 to 72), and fainting at 20 health with a sideways sprite and rejected movement. Automated checks cover blocked actions, save restoration and waking on thorns with time to leave. No live player injuries were used for testing. Browser console reported no warnings/errors.

The running server was stopped gracefully and its final autosave copied to `runs/before-hazards-20261004.json`. Migration preserved both structures, all existing resources, positions and inventories, and added 45 hazard plants on available grass. The restarted browser showed v0.3, 100 player health and the existing eight carried items. The manual checkpoint was not overwritten. Screenshots of the disposable fixtures are `runs/hazards-preview.jpg` and `runs/fainting-preview.jpg` (ignored).

Those checks established consequences and recovery, not learned avoidance. Visible residents were still scripted at that historical milestone.

Revival follow-up: 54 tests cover local posture visibility, adjacency, energy gating and partial recovery. A disposable browser test revived Moss to 30 health with pain remaining and reduced helper energy from 100 to 90; no console errors. Screenshot: `runs/revive-preview.jpg`.


## v0.4 finite resources and farming

- 59 tests passed in 14.6 seconds; JavaScript syntax and diff checks passed.
- A disposable browser fixture verified Make seeds (2 food -> 1 food + 2 seeds), Plant (one seed consumed), visible seedling and shared crop totals. The 1280px viewport had no horizontal overflow. Screenshot: `runs/farming-preview.jpg`.
- Crop presence encoding and all 51 neural actions were checked using the existing PyTorch runtime. This did not train a policy.
- Graceful shutdown saved the live world at tick 9244. `runs/before-farming-20261004.json` is the backup. An exact JSON roundtrip preserved all 9 characters, 3 structures and remaining state; the server restarted and the browser showed v0.4 and the shared harvest counter.
- Laya remains a permanent supported experiment track; its local assets and adapters were retained. No new download, live Laya deployment, weight training or NPU benchmark occurred in this update.
- A focused subagent review caught a Remove-preview mismatch for seedlings. The UI now excludes growing crops, matching the authoritative server rule. No other actionable findings were reported.


## Farming learner experiment

- 62 tests passed in 17.0 seconds, including new experiment accounting, positive-control, terminal-return and no-evaluation-learning checks.
- 150 episodes completed: 72 Q training, 72 small-policy evaluation, 6 local NPU Laya evaluation. Total 1,440,000 simulation ticks in disposable worlds; the live world was not loaded or saved by the experiment.
- Preregistered source hashes still matched at completion. Food/seed accounting passed for every recorded episode; rewards were restricted to successful eating. Independent subagent audit confirmed reported means and trace totals, and identified the lack of a demonstrated starvation-prevention benefit.
- Results and limitations: [FARMING-TRIAL.md](FARMING-TRIAL.md). Raw evidence is in ignored `runs/farming-trial-20261004/`. No gameplay controller was switched and no Laya weights were trained.


## Mobile farming and explicit-recipe Laya

65 tests passed in 16.7 seconds. The mobile run completed 42 worlds (806,400 ticks) with independently learned agents; its source hashes and evaluation resource accounting passed. The overall hypothesis failed in one of three seed groups. One of six evaluation layouts duplicates a training layout; the report retains this limitation. The Laya diagnostic completed four worlds (76,800 ticks), passed hash/accounting checks and failed its farming hypothesis. Both experiment outputs are local, disposable, and separate from the live saved world. Reports: [mobile](MOBILE-FARMING-TRIAL.md), [Laya](LAYA-RECIPE-TRIAL.md).


## v0.5 hunger and curiosity

77 tests passed. Hunger physiology, save compatibility, private curiosity, closed-door exploration and integrated reward accounting verified. Browser fixture confirmed eating relief and food-assisted revival; preserved live save migrated at tick 10,598 with nine characters and three structures. Details: [MOTIVATION.md](MOTIVATION.md). Earlier experimental results use prior physiology/reward versions and are not new-policy performance evidence.

## v0.6 neural trials, network growth, taps and tones

93 tests passed in 16.7 seconds; JavaScript syntax and diff checks passed. Tests cover independent replay/weights, target masking, frozen evaluation, capacity-preserving offspring, social range/occlusion/cooldown/privacy, save migration and both tap action orders. Disposable browser fixture verified Moss received “tapped from west” and “Hears tone high west”, with zero damage. No horizontal overflow at the current desktop viewport. Screenshot: ignored `runs/social-tones.jpg`.

The official 189-episode trial source and preregistration hashes were independently audited before social edits; original sources are archived in the run. Parent recount confirmed both conditions achieved 14/18 reliable lives and failed deployment. A trained 64-to-80-unit offspring retained outputs in a numerical smoke check; this is capacity preservation, not a fitness improvement.

Live save was gracefully stopped, backed up to `runs/before-social-20261004.json`, verified field-by-field through migration at tick 12,677 (four structures, nine characters), then restarted on port 8788. Existing structures, inventories, residents and RNG were preserved. Tone and tap controls verified in the live page. No new model download, cloud call, neural deployment or learned-language claim.

## Learned signaling trial

96 tests passed in 16.98 seconds. The 6,660-episode run completed in 21.82 seconds; source and preregistration hashes, channel interventions, equal evaluation worlds, shuffled message frequencies and food outcomes were independently audited. All three seeds passed the narrow gate. Live game and saves were untouched. See [report](TONE-TRIAL.md).

## v0.7 ten-tone sequences

100 tests passed in 17.03 seconds. Ten distinct actions, nonoverlap, repeated symbols, ordering, gaps, private acquisition, expiry, save/load and neural sequence distinctions verified. Browser fixture showed ten choices and recipient memory `2 > 7 > 2` after sounds ended, without horizontal overflow. Screenshot: `runs/ten-tone-sequence.jpg`. Live save preserved field-by-field at tick 14257, with 4 structures and 9 characters; backup `runs/before-ten-tones-20261004.json`. No ten-tone learning run or live neural promotion was performed.

## v0.8

105 tests passed; exact brain/world continuation, ecological conservation and browser interactions verified. See [full validation and negative results](LIVING-POPULATION.md).

## Physical action filters

108 tests passed; exact checkpoint continuation and legacy pending transitions remain valid. Python compile, JavaScript syntax and diff checks passed. The browser shows the enabled filter. See [experimental audit](PHYSICAL-ACTIONS.md).

The 79,200-tick comparison completed in 910 seconds. Independent audit verified source/preregistration hashes, nine evaluation traces and gate calculations. Trained filtered agents achieved 4/6 low-starvation lives versus 5/6 for initial-weight filtered agents; all three seed improvement gates failed. Conscious invalid choices fell to 0.145% versus 74.437% unfiltered. This validates filtering utility, not learned survival superiority. Trial checkpoints were not promoted.

## Sequence learning

117 tests passed in 18.929 seconds after the accounting correction. New checks verify delayed credit and terminal/cutoff handling, gradients reaching earlier observations through recurrence, policy constancy during collection, independent policy/predictor storage, frozen evaluation, exact continuation through an update after a mid-rollout save, stale-policy rejection, no healing reward, no false nutrition at zero fullness in both populations, local inputs, legacy checkpoint loading and runtime restore. Python compilation and JavaScript syntax checks passed.

Browser testing verified a real sequence update, the private prediction/outcome inspector, 20x controls, pause, manual save and restoration of tick 874 with policy version 1 and 90/128 buffered transitions. Desktop document width equals scroll width. Screenshot: `runs/sequence-inspector.jpg`. Development traveler placement was backed up and verified to preserve learner weights. Main and previous experimental saves were not replaced. See [experiment report](SEQUENCE-LEARNING.md).

The 289,728-tick confirmation completed in 862.616 seconds. Separate recount verified source/preregistration hashes before the accounting patch, archived sources afterward, three training budgets, eighteen frozen evaluation traces and final gate. Twelve trained lives and all controls stayed fed; zero trained lives met the specified thorn-avoidance threshold. The gate failed despite lower harmful eating and unconscious time. Later worlds are available as layouts, without learned farming/cooperation claims.

Both interactive learner populations restarted with the corrected accounting and remain paused: previous population tick 2469 / 617 updates, development tick 874 / 1 update / 90 buffered transitions. Pre-restart complete checkpoints are `before-accounting-fix.pt` in each population's inner runs directory. Brain weight hashes and ticks were recorded in ignored `runs/preserved-populations-20261005.json`. The development launcher starts paused; the older launcher supports `-Paused` without changing its default.

## Scarcity and safe routes

124 tests passed in 19.311 seconds, including new route geometry, finite stocks, private observations, physical crossing counts, inactive-agent gate rejection, positive-control feeding, actual crop-yield conservation and playable traveler/checkpoint continuation. Python compilation, JavaScript syntax and diff checks passed.

The 435,456-tick confirmation completed in 509.128 seconds. The separate auditor verified current/archived source and preregistration hashes, all training budgets, 24 matched/frozen evaluations, decision traces, nutrition/injury/deprivation and route totals, food conservation and the original failed gate. Trained lives suffered more deprivation than matching initial policies. A labeled post hoc action diagnostic independently checks action-count totals against the audited conscious-decision counts; it does not alter acceptance. See [full results and limits](SCARCITY-ROUTES.md).

The fresh interactive arena runs paused at http://127.0.0.1:8791/ . Browser checks verified 02 residents, accessible traveler movement at tick 1, manual restoration to tick 0 with both private brains, expanded Learning record, no console errors and no horizontal overflow at width 1,265. Screenshot: `runs/scarcity-arena.png`. Only this new population was restarted for the roster update; the main world and earlier learner populations were not reset or replaced.
