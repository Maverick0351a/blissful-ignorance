# Blissful Ignorance milestones

Established October 5, 2026 (America/Los_Angeles), at the user's request.

Public title updated October 6, 2026; the existing `GT-` identifiers remain
stable. The [progress showcase](progress/index.html) and [public summary](PROGRESS.md)
present working mechanics, pilot results and remaining gates separately.

**Next milestone: GT-01, reliable food acquisition.** The
[nearby-food diagnostic](NEAR-FOOD-DIAGNOSTIC.md) is complete and audited:
category PPO scored 23/48 prompt feeding lives versus 20/48 initially and 19/48
for category-random. Both architectures failed both behavioral gates.
The subsequent [archived culling/restart pilot](CULLING-TRIAL.md) is also
complete and audited: continuation 32/48, selective restart 29/48, random
restart 31/48. Its gate failed. All live residents and Laya remain preserved.
The [feeding-update diagnostic](FEEDING-CREDIT.md) reproduced all 96 recorded
continuation updates and complete final states. All 214 meals received positive
credit and 184 became more likely; gains were uneven and did not establish a
memory defect. The [matched value-gradient test](VALUE-INTERFERENCE.md) then
failed its primary screen: 200/214 meals reinforced, but a small mean decrease
in carried-food probe gains. Keeping the existing rule, the subsequent
[practice-amount comparison](PRACTICE-AMOUNT.md) **passed its development
screen**: prompt feeding 75→82→95/96 at starting / +16 / +64 lives, with a full
decision/checkpoint replay audit. The weak responders improved while retaining
their brains. Next propose frozen evaluation with food requiring navigation.
GT-01 stays open; no live weights were imported.

This is the forward roadmap. The `GT-` identifiers distinguish it from the
original M0–M6 build-plan numbering. Historical experiment gates and negative
results remain unchanged. [STATUS.md](STATUS.md) records implementations and
results; this file defines what earns the next capability claim.

## Starting point

**GT-00 — playable world and auditable learning infrastructure: complete.**
The documented foundation includes the browser world, player controls,
construction and farming mechanics, private senses and memories, ten tones,
pause/fast-forward, saved independent controllers and reproducible experiments.
The latest verified suite passed 217 tests. This completion concerns the
software foundation; the learned behavior milestones below remain open.

The subsequent [October 6 implementation audit](AUDIT-2026-10-06.md) passed
218 current software tests and identified ten additional defects/risks outside
that coverage. Saved-identity and controller-failure hardening (AUD-01–03)
is now installed: 231 passing tests, an unchanged 528-tick synthetic trajectory,
and 86,727 identical live checkpoint values after a paused restart. Seven audit
findings remain; next fix the UI controls/layout and tap mask. No behavioral
gate was advanced.

October 6 presentation follow-up puts the working map and characters on the
front page, with character following, an all-residents camera, expandable map,
touch movement and linked research. This remains GT-00 work; see
[UI validation](PROGRESS-VALIDATION.md), not evidence of a new learned skill.

October 6 publication follow-up makes the source public on GitHub and the
[recorded showcase](https://huggingface.co/spaces/Maverick03511/blissful-ignorance)
available on a free static Hugging Face Space. This extends GT-00 delivery;
the hosted page runs recorded playback and charts, without a live learning
population. [Hosting validation](HUGGING-FACE.md) keeps these boundaries explicit.

The main population has eight independent recurrent PPO residents and a
distinct Laya NPU resident. Laya's pretrained weights are frozen; her private
context is persistent. Clef is an installed experimental comparison, with no
live resident integration. These model families remain explicitly identified.
The user has since indicated Laya may be retired; replacement selection is
pending. Preserve her checkpoint before an intentional migration.

The latest [voluntary coordination pilot](VOLUNTARY-COORDINATION.md) passed its
technical audit but failed every communication gate: the initially uninformed
resident ate in 2/24 intact, 1/24 muted and 2/24 relabeled-tone cases. Even the
initially informed resident ate in only 6/24 intact cases. Reliable foraging is
the immediate development priority.

## Milestone sequence

These are **prospective engineering targets**, not capabilities already
demonstrated or promised dates. Exact fixtures, budgets, metrics and evaluation
seeds must be frozen before a confirming run. Later milestones may need
protocol refinements; revisions must be dated and cannot turn an old failed run
into a pass.

| ID | Ability we want to see | Completion target | Dependency / status |
|---|---|---|---|
| **GT-01** | Find, gather and eat ordinary food independently | At least 90% success on held-out foraging lives; at least 10 percentage points above both matching initialization and random feasible actions; no training seed below 80%. | **Nearby-food practice pilot passed; confirmation still open** |
| **GT-02** | Stay fed, avoid learned dangers and retain useful behavior | At least 90% of longer evaluation lives spend under 1% of ticks at zero food and under 1% unconscious; pass hazard and retention checks below. | After GT-01; planned |
| **GT-03** | Explore usefully and adapt when the world changes | Discover at least 20% more usable resource sites than a matched no-curiosity control without reducing GT-02 reliability; pass adaptation and noisy-distraction checks. | After GT-02; planned |
| **GT-04** | Produce food and improve living conditions | Sustain feeding after starting wild food runs out, consume genuinely grown harvests over three crop cycles, and demonstrate a measurable benefit from self-built shelter. | After GT-02; planned, with separate farming and building gates |
| **GT-05** | Develop useful communication | At least 75% task success with intact signals and at least a 25-point advantage over silence and the intervention that removes the claimed information, in every confirmation seed. | After GT-02; planned; current pilot failed |
| **GT-06** | Form useful, voluntary leadership and cooperation | Stable, learned influence across new tasks and sessions; a matched influence-removal test reduces group success by at least 20 points in at least 4/5 seed groups without merely removing a worker. | After GT-04 and GT-05; planned |
| **GT-07** | Reproduce and improve across generations | Viable unforced reproduction and care over three generations; claim evolution only after descendants outperform matched non-selected controls under equal resources and learning budgets. | After GT-02 and GT-04; research milestone |
| **GT-08** | Transfer a learned ability to a useful outside task | At least 90% success on 100 held-out local test tasks across five seeds, plus a measurable advantage over a fresh task-trained control at matched compute. | Separate research branch after GT-02/03; not dependent on reproduction |

GT-03, GT-04 and GT-05 can become separate branches once their prerequisites
pass. Generations depend on stable feeding and food production; useful transfer
has its own branch. Only one substantive learning change is the active
implementation target at a time. Preliminary fixtures may be built sooner;
their availability does not complete the corresponding behavior milestone.

## How completion is decided

Keep three records separate: **implementation works**, **behavior confirmed**,
and **live integration verified**. A passing software test or a completed
experiment does not by itself satisfy a behavior gate.

For the next confirmation, use at least **five independent training seeds**,
**ten unseen world seeds per training seed**, and **two independently trained
residents per world**: at least 100 evaluated lives per condition. Balance two
layout families, including open and occluded food access. Resident lives and
maps within a training seed are clustered observations; report each seed's
results and uncertainty rather than treating every tick as an independent
sample. Later multi-resident protocols must declare their corresponding
independent training groups and held-out worlds before running.

The common completion requirements are:

1. A preregistered hypothesis, primary metric, controls, thresholds, interaction
   budget and wall-time limit. Development maps cannot become confirmation maps.
2. All actors use private observations, independent learned state and ordinary
   game actions. World rules define physical possibilities and consequences.
   Scripted or privileged feasibility controls, if needed, remain clearly
   labeled benchmarks and supply no actions, experience or rewards to learners.
3. Matched bodies, opportunities and resource budgets across conditions. Record
   architecture size, updates, inference/training time, memory and each reward
   component. If two architectures cannot match both compute and interactions,
   report those as separate comparisons.
4. Frozen-policy evaluation for retained skills and a separately labeled online
   phase for adaptation. Report ordinary and amber meals, deprivation,
   unconscious time, injuries, useful exploration, produced-and-consumed food,
   and social outcomes when relevant. Mortality remains off; staying fed and
   conscious are the functional survival measures.
5. A separate trace/checkpoint audit, source hashes and regression checks. A
   short watchable replay should make the claimed behavior inspectable. Policy
   action probabilities are not automatically calibrated success probabilities.
6. No unresolved correctness failure and no material regression in previously
   confirmed abilities. Update this roadmap with the evidence link and outcome.

A pilot may justify more work but cannot close a milestone. A budget stop is
incomplete evidence. A failed gate keeps the milestone open and the result in
the record. After two bounded unsuccessful pilots of the same candidate,
review the mechanism and choose a specific diagnostic before spending more
compute or adding another component.

## GT-01: next proposed increment

The retained +64 category-PPO candidates now reliably acquired adjacent food
in the narrow development fixture. Test transfer with **frozen policies** on
food requiring movement, first in open rooms and then around an occluder.
Include the adjacent-food sanity condition, the retained starting policy and
matched category-random control. Preregister fresh maps, deadlines, interaction
and time budgets before execution. Preserve all candidate checkpoints and live
residents. This comparison is proposed, not started by the practice run.

Its purpose is to locate the next missing skill before choosing navigation
training. The existing GT-01 confirmation still requires five fresh training
groups, held-out open/occluded layouts and both absolute and control-margin
gates. A six-brain nearby-food dose comparison cannot close it.

## GT-01: original action-category increment

**Hypothesis:** a learned choice of action category, followed by its argument,
will make food acquisition easier to learn while retaining all ten tone symbols
and the existing physical actions. The coordination pilot's 70.12% tone share
was near the 68.65% expected from uniform feasible action slots. That motivates
the experiment; it does not establish the cause of poor feeding.

The increment is:

1. Implement the experimental policy head: the model selects an activity, then
   its direction, item or tone as applicable. Use the joint action probability
   in PPO and verify probability normalization, masking, gradients and exact
   save/resume. No hunger-triggered action rules or assigned routes are added.
2. Compare learned flat and learned hierarchical policies. Include their own
   frozen initializations and feasible random controls. A category-uniform
   random control helps separate changed exploration probabilities from learned
   competence. Keep observations, rewards, recurrent core and update schedule
   matched; report the changed head's parameter count.
3. Define food success as actual ordinary-food acquisition and consumption
   before initial fullness is exhausted. Starting packs are empty; food is
   reachable but locations vary. Repeated trials change patch location so an
   initial lucky meal cannot stand in for navigation. This is a foraging assay,
   not yet evidence of a self-sustaining resource economy.
4. Run a three-seed development pilot. Use measured throughput to choose a
   fixed interaction budget that fits a **ten-minute wall-time cap**. Archive
   results whether or not the candidate helps. Do not tune on confirmation maps.
5. If the mechanism looks promising, freeze the protocol and use the larger
   confirmation design above. GT-01 also requires improvement over both
   controls in at least 4/5 training seeds. Its absolute and comparative gates
   must pass together; a small gain from a poor baseline is insufficient.

The deliverable is a tested candidate policy, a matched result, an audited trace
and an explicit pass/fail decision. Laya and Clef comparisons are a separate
follow-up using matched observable information and action opportunities. Their
pretraining, prompt compression, frozen weights and hardware costs must remain
visible in the comparison. Laya's proposed retirement requires an explicit
candidate selection and archived checkpoint; neither comparison automatically
promotes a model or makes it a teacher.

## What the later gates mean in practice

**GT-02 — sustained competence.** Evaluate at least five in-world days per life
with finite, adequate food budgets and resource relocation. The deprivation
threshold applies to each qualifying life, not just a favorable population
average. Hazard trials require real, recorded opportunities to observe harmful
food and thorns, successful ordinary feeding and at least 50% fewer injuries per
comparable opportunity than initial/random controls. No exposure cannot pass
as learned avoidance. Revisit old held-out tasks after learning another task:
ordinary-food success may fall by no more than five percentage points and must
remain above the GT-01 absolute threshold. Checkpoint/restart must preserve the
next actions and all private state. These are separate tests within GT-02.

**GT-03 — useful curiosity and adaptation.** Define discovery as first observing
a previously unseen, reachable, usable resource site, counted once per site.
Novelty loops, repeatedly inspecting one plant and cosmetic noise do not count.
Compare the same learner with and without the candidate curiosity mechanism;
change one mechanism per experiment. After resource relocation, target recovery
of the prior feeding-success level within 500 decisions in at least 4/5 seed
groups. In a separate noisy-distractor condition, require the same GT-02
reliability and no more than a five-point loss of foraging success. Report
information seeking as behavior without inferring a subjective experience.

**GT-04 — productive habitats.** First test farming with adequate land, soil and
water but no automatic wild-food refill. After the starting supply is exhausted,
at least 80% of ordinary nutrition must come from traceable planted harvests
over three complete crop cycles, while the GT-02 feeding criterion holds in
at least 90% of lives. Track seeds, mature yield and actual later consumption;
planting counts alone are insufficient. Then test construction separately:
collect materials, build and use shelter on held-out layouts. Require at least
25% less cold-exposure time than a matched shelter-effect-disabled control,
without a feeding regression. Define the cold threshold before running. Keep
these two results separate until both gates pass.

**GT-05 — useful communication.** Let agents decide when to tap or emit one of
ten tones; meanings and responses remain learned. Start with a reliable food
task where information is divided between residents. Test silence, symbol
relabeling and a control matched to the claimed communication mechanism: for
example, changing direction if claiming directional guidance, or message timing
and order if claiming a sequence code. Preserve appropriate non-semantic
properties and verify that controls affect both immediate and remembered cues.
Require the table's gate with the same frozen policies and held-out layouts.
A useful directional beacon may survive symbol relabeling; use the directional
intervention to test that claim and do not call it a learned symbolic code.
Choose the primary mechanism test before confirmation, not after inspecting
its results. One useful signal convention is enough for this milestone;
sequences, large vocabularies and language need additional tests. This new
prospective protocol does not change the failed gate of the earlier pilot.

**GT-06 — voluntary leadership.** Define influence from observable behavior:
neighbors can choose to respond to an individual's demonstrated knowledge or
signals, ignore them, leave, or learn a different source of guidance. Provide
no leader flag, obedience action or reward simply for following. Measure who
influences whose useful decisions across multiple tasks and sessions. Compare
against equally capable random advisers and signal interventions that retain
the same workers and physical labor. Remove or change the informative
individual's influence while preserving the labor/resource budget. A useful
group-performance drop supports influence; crowding, loudness or a missing
worker's lost production does not. Test recovery or selection of another useful
source when circumstances change. Freeze operational influence measures before
examining the trace.

**GT-07 — generations and evolution.** Confirm the lifecycle mechanics before
claiming learned parenting: resource costs, independently chosen pairing,
juvenile care, saved lineage and configurable population/compute budgets.
Start offspring with fresh private brains as previously planned; inheritance
of trained weights would be a separately labeled design and experiment.
Declare which body or learning-rule traits can be inherited and mutated before
the evolutionary comparison, and record each lineage's actual changes.
Target at least 80% of juveniles reaching defined maturity while at least 90%
of all lives meet the feeding criterion, across three generations and five
world seeds. Evolution is a second gate: descendants must improve a fixed
held-out capability score by at least 10% relative to matched non-selected
lineages in at least 4/5 seeds, with equal lifetime experience, no survival
regression and no parent memories silently copied. Larger brains alone do not
complete that gate.

**GT-08 — useful transfer.** Select one concrete task before implementing an
outside-world interface, such as organizing synthetic local files or solving a
resource-allocation task. Start in a disposable, bounded local workspace.
Measure task correctness, sample efficiency, retention and boundary adherence
against fresh task-trained and simple conventional baselines. Require the
table's accuracy target, zero out-of-scope actions and at least a 20% reduction
in new-task training interactions to reach that accuracy under equal compute
caps relative to the fresh learner, in at least 4/5 seeds. Then validate whether
the result helps the user on that task. Game competence does not guarantee such
transfer; this branch may require new observations, actions and learning
mechanisms. Real documents, applications and external actions are a separate
integration decision.

## Live residents, compute and delivery

Continue preserving the named residents, their own learning histories and
Laya's separate local model. Experiments use fresh residents or explicitly
identified copies. Successful trial weights are never silently imported into
the continuing world. A live rollout needs a concrete migration/rollback plan,
preserved checkpoints and verification of every resident's identity and state.
Passing a behavior gate does not itself authorize that rollout.

Use existing local runtimes for the next increment. Research success has no
credible fixed calendar date. Budget development pilots at up to ten minutes;
size later resumable confirmation batches from measured throughput, normally
up to thirty minutes each. Record total planned interactions, batches, storage
and audit time before starting. Exceeding a budget ends that run as incomplete;
it does not automatically authorize more batches. Resource caps are adjustable
operational limits, not a permanent ceiling on model capacity.

Work inline under the current cost preference. New downloads, paid services,
public GitHub/Hugging Face publication and extra machines remain subject to the
existing user instructions. The [publication plan](PUBLISHING.md) is a separate
delivery track; an honest preview can ship without pretending research gates
have passed.

## Progress record and update format

Every substantive implementation or experiment update should identify the
active milestone, the bounded work completed, its evidence, the gate outcome
and the single next action. Use **planned**, **building**, **pilot complete**,
**confirmation pending**, **confirmed**, or **blocked**, with the actual reason.
Keep live integration status separate. A milestone's `blocked` label here
describes project evidence; it does not control any app goal or automation.

| Date | Milestone | Evidence / decision | Next action |
|---|---|---|---|
| 2026-10-06 | GT-00 public source delivery confirmed | [GitHub repository](https://github.com/Maverick0351a/blissful-ignorance) published under Apache-2.0; clean copy passed 231 tests; private lives, weights and raw traces excluded. GT-01 and seven audit findings remain open. | Fix AUD-05–08; navigation transfer remains the next proposed behavioral experiment. |
| 2026-10-06 | GT-00 preservation repair confirmed | [AUD-01–03](AUDIT-2026-10-06.md) fixed; 231 passing tests; successful trajectory unchanged; complete paused live restart preserved all 86,727 checkpoint values. No behavior gate advanced. | Fix AUD-05–08; Laya replacement selection is separately pending. |
| 2026-10-05 | GT-00 confirmed | Playable independent population; 181 passing tests; coordination replay audit (local evidence; not bundled) validates the latest experiment infrastructure. | Preserve the foundation and residents. |
| 2026-10-05 | GT-01 planned | [Coordination result](VOLUNTARY-COORDINATION.md) leaves basic feeding unresolved. No behavior milestone is advanced by the failed pilot. | Implement and test the experimental learned action-category head. |
| 2026-10-05 | Roadmap established | User requested milestones for work from here onward. GT-01–08 gates are prospective; no new training or live change was performed while writing this plan. | Track subsequent work against these IDs and link its evidence. |
| 2026-10-05 | GT-01 pilot complete; gate failed | [Action-category report](ACTION-CATEGORY-PILOT.md): 172,032 ticks, three seeds, 191 passing tests and full replay audit. Timely feeding: category PPO 15/48, flat PPO 6/48, category-random 18/48. Live population preserved; no confirmation or promotion. | Propose a bounded nearby-food gather-then-eat diagnostic before longer navigation training. |
| 2026-10-06 | GT-01 diagnostic complete; gates failed | [Nearby-food report](NEAR-FOOD-DIAGNOSTIC.md): 181,248 ticks, 196 passing tests, 480-episode audit. Category prompt feeding 23/48; initial 20/48; category-random 19/48. Carried-food gate also failed. | User selected a separate culling/restart comparison; archive experimental candidates and preserve every live resident and Laya. |
| 2026-10-06 | GT-01 culling pilot complete; gate failed | [Archived restart report](CULLING-TRIAL.md): 135,168 ticks, 202 tests, 264-episode audit. Continued learning 32/48, selective restart 29/48, random restart 31/48. Sources archived; every live resident and Laya preserved. | Inspect learning credit and a recorded feeding update on experimental copies; no automatic new generation. |
| 2026-10-06 | GT-01 update diagnostic complete; behavior unconfirmed | [Feeding-credit report](FEEDING-CREDIT.md): exact replay of 24,576 existing ticks and 96 updates; all 214 meals had positive credit, 184 reinforced; separate native audit and 207 tests passed. Same-task retention did not establish a memory defect. | Propose a matched experimental test of value-loss interference in the shared encoder; preserve live residents and rewards. |
| 2026-10-06 | GT-01 interference diagnostic complete; screen failed | [Value-gradient report](VALUE-INTERFERENCE.md): 96 independent paired updates, full-state audit, 212 tests. Meal reinforcement improved 184→200/214, but mean carried-food gain fell 0.00504 pp/update. No live change. | Propose a bounded comparison of practice amounts with retained trained brains and fresh evaluation; keep the existing learning rule. |
| 2026-10-06 | GT-01 practice pilot complete; screen passed | [Practice-amount result](PRACTICE-AMOUNT.md): 181,248 ticks; 75→82→95/96 prompt feeding lives; all seed groups and weak responders improved. Separate replay reproduced 90,624 decisions, 384 updates and 24 full states; 217 tests passed. GT-01 remains open; no live imports. | Propose a frozen navigation-transfer test with distant/occluded food, adjacent sanity condition, retained starting policy and category-random control. |
