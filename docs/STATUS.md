# Implementation status

Updated October 6, 2026 (America/Los_Angeles).

## Free Hugging Face showcase — October 6

The [recorded showcase](https://huggingface.co/spaces/Maverick03511/blissful-ignorance)
is public and running as a free static Space. It shows the playable map and
characters in a screenshot, interactive score filters and recorded feeding
playback. No live models run there. The local population stayed private and
manually paused at tick 85,447 with the same eight PPO learners and Laya.

The package has 11 reviewed files from the published GitHub snapshot. All 14
chart views matched the aggregate data locally; three replay conditions and
hosted filtering, scrubbing, playback and restart were checked. The hosted
desktop page had no document overflow or browser warnings/errors. The Space's
floating header has dedicated clearance above navigation. See
[reproduction and evidence limits](HUGGING-FACE.md).

This is GT-00 delivery work. GT-01 and the seven implementation audit findings
remain open; publication does not establish another learned capability.

## Public GitHub source — October 6

The first public source snapshot is at
[Maverick0351a/blissful-ignorance](https://github.com/Maverick0351a/blissful-ignorance),
under Apache-2.0. It contains the playable map, current preservation fixes,
tests, roadmap, aggregate results and selected experimental replay. Its clean
copy passed **231 tests in 92.466 seconds**, including PettingZoo. The final
publication documentation and notice do not change the tested executable code.
The file inventory, hashes, local links and JavaScript syntax were checked.

This is GT-00 delivery work; GT-01 remains open and seven implementation audit
findings remain. Raw traces, checkpoints, live histories, operational watch
logs and private Git history stay local. The source publication itself did not
deploy hosting; the subsequent static Space is recorded above. No shared game
server is deployed. See [publishing status](PUBLISHING.md).

## Preservation fixes installed — October 6

GT-00 hardening: [AUD-01–03](AUDIT-2026-10-06.md) are fixed and installed.
Incomplete live saves now fail without recreating controllers; maximum speed
stops immediately on controller failure; failed decisions restore complete
learning state and keep the player's queued action for an explicit retry.
Failed rollback or partial physics/reward advancement blocks saving and further
ticks until a complete reload/restart. Historical experimental bystanders and
scripted legacy migrations remain compatible.

**231 tests passed in 95.756 seconds**, including thirteen new preservation
tests and PettingZoo. A 528-tick synthetic comparison matched every action and
all 38,275 final saved values against the old source. Snapshot overhead reduced
that PPO-only sample to 45.1 ticks/second, about twice its former elapsed time.
The real service restarted private and manually paused at tick **84,935** with
no controller error; **86,727 checkpoint values matched the retained backup**.
All eight PPO learners and Laya remain unchanged by this repair. Seven audit
findings remain open; the next engineering batch is UI fixes AUD-05–07 and tap
mask AUD-08. This does not advance GT-01 or publish the project.

The user has indicated Laya can be retired in favor of an alternative. The
earlier permanent-retention preference is no longer assumed; the candidate
selection is pending between category-PPO, Clef Flash, or retirement without a
replacement. Archive her checkpoint before an intentional migration. No such
migration or deletion has occurred. Shared local model assets remain intact.

## Expert audit — October 6

Three specialized subagent reviews and a parent evidence pass produced a
[prioritized implementation audit](AUDIT-2026-10-06.md): two P1, seven P2 and
one P3 actionable findings. The complete suite passed **218 tests in 98.123
seconds**, including PettingZoo. Isolated probes exposed additional gaps:
incomplete-checkpoint identity replacement, post-error max-speed advancement,
partial learning state after a failed decision, experimental checkpoint
ownership, three UI control/layout defects, tap cooldown masking, paused
read-only demos and initial action-credit attribution. AUD-01–03 were fixed
in the follow-up above; the remaining seven are open.
During the audit, the main world stayed private and manually paused at tick 84,935 with the same
eight PPO residents, distinct Laya backend and server PID. No live experiment
or policy change was part of the audit. Its recommended GT-00
checkpoint/failure hardening (AUD-01–03) is now complete;
GT-01 remains the next behavioral milestone.

## Map-first playable front page — October 6

GT-00 presentation follow-up: the actual game map leads the app, with character
portraits beside it and the research report one click away. Portrait selection
now follows that resident; Everyone fits the roster in view, Expand hides the
inspector, and narrow screens have directional movement buttons. Names label
visible characters at normal zoom, including only seen characters in individual
perspective. The pause notice no longer covers the center of the map.
Research assets are served from an explicit allowlist; private saves and watch
logs stay unavailable. The offline research page remains usable by itself.
Validation uses a separate real PPO world; this is not a new behavioral result.
See [validation](PROGRESS-VALIDATION.md) for checks and live preservation.

## Public progress preview — October 6

The user chose **Blissful Ignorance** as the public/game title, formerly
Godhood Trials. The README, browser title and Windows launchers now use it;
existing saved-world paths, service IDs and historical evidence stay compatible.
The portable [progress showcase](progress/index.html) presents the verified
practice results, per-brain filters, selected recorded playback, failed
experiments and open milestones. [PROGRESS.md](PROGRESS.md) is the GitHub-readable
summary; [PLAYING.md](PLAYING.md) retains the full controls and mechanics.
The exporter verifies four recorded source hashes and does not rerun training
or read live saves. This is GT-00 documentation/presentation work, not a newly
confirmed learning milestone. At that preview stage, no repository or Space
had been published; the later GitHub publication is recorded above.
Browser filters, recorded playback, desktop/narrow layouts and ten existing
server tests passed; see [preview validation](PROGRESS-VALIDATION.md).

## Latest learning result

The authorized [practice-amount comparison](PRACTICE-AMOUNT.md) is complete
and audited; **its development screen passed**. On the same fresh adjacent-food
test instances, retained category-PPO candidates scored **75/96** prompt
feeding lives at start, **82/96** after 16 further lives, and **95/96** after
64. All three historical seed groups improved; c0/c3/c5 improved without being
replaced. The +64 policies ate in all 96 lives before reaching zero fullness.
Carried-food prompt success rose from 58→65→95/96. The audit reproduced every
one of 90,624 decisions, all 384 updates and 24 full individual checkpoint
states. The run took 483.57 seconds and audit 468.30 seconds; 217 tests passed.
This confirms improvement in the existing small-room task, not general
navigation or GT-01 completion. Next propose a frozen navigation-transfer test
with distant/occluded food. The existing learning rule and live roster remain
intact; no experimental weights were imported.

Forward work follows [MILESTONES.md](MILESTONES.md). **GT-00, the playable and
auditable foundation, is confirmed; GT-01, reliable food acquisition, remains
open.** The [nearby-food diagnostic](NEAR-FOOD-DIAGNOSTIC.md) completed 181,248
ticks in 499.97 seconds and passed its separate 209.22-second replay audit.
Prompt gather-then-eat success was 23/48 for category PPO, 20/48 for its
initialization and 19/48 for category-random; flat PPO scored 17/48. Both
architectures failed the nearby-food and carried-food gates. The policy
probabilities changed modestly; a memory defect was not established.

The authorized [archived culling/restart pilot](CULLING-TRIAL.md) is complete
and audited. Continued learning scored 32/48 prompt feeding lives, selective
restart 29/48 and random restart 31/48; the development gate failed. Copies of
experimental candidates c0 and c4 were replaced only in that branch, with all
source checkpoints preserved. Its audit replayed 135,168 transitions and
30,720 frozen decisions. All live residents and Laya retained their model IDs
and server PID. No additional culling generation or live mortality was enabled.

The [feeding-update diagnostic](FEEDING-CREDIT.md) then replayed the existing
continuation branch exactly: **24,576 ticks, 12,288 decisions and 96 updates**.
All 214 ordinary meals had positive normalized learning credit; 184 became
more likely after their update. Carried-food eating preference rose in all six
experimental category-PPO brains, from small gains in c0/c3/c5 to 24.19%→35.24%
in c2. Five first-meal anchors gained immediately and retained that gain through
15 later same-task updates; c3 improved later. This does not establish reliable
feeding, episodic recall or a memory defect. The separate audit reproduced full
final models, optimizers, RNGs and private histories.

The [value-gradient intervention](VALUE-INTERFERENCE.md) completed and passed
its separate audit: 96 matched updates, each experimental copy starting from
the corresponding original full state. Blocking value gradients reinforced
200/214 meal choices versus 184/214 normally, but mean fixed-probe eating gain
was 0.00504 percentage point worse per update. Four brains benefited slightly;
c2 lost larger gains. The predeclared diagnostic screen failed, so the existing
learning rule remains. Its proposed practice-amount follow-up is now complete
as recorded above; longer practice helped without changing that rule. No new
live controller or memory change was introduced.

[Tiny Theory of Mind](TINY-THEORY-OF-MIND.md) was reviewed as a later text
benchmark and embodied social-evaluation inspiration. No dataset artifact was
downloaded and no model was trained on it.

The [action-category pilot](ACTION-CATEGORY-PILOT.md) completed **172,032 ticks in 328.13 seconds**. Trained category PPO fed before zero fullness in **15/48 lives**, compared with **6/48** for trained flat PPO, **8/48** for its own initialization and **18/48** for category-random. All three seeds failed the comparative learning gate. The separate audit reproduced every transition and **36,864 frozen decisions** in 183.59 seconds; the full suite passed **191 tests**. The recorded replay (local evidence; not bundled) is local evidence, not a live game change. No experimental weights were promoted; the server retained its PID and all nine model IDs.

Current controller: **eight independent recurrent PPO residents plus a distinct persistent Laya resident on the local NPU**. Laya uses frozen pretrained weights with her own bounded context history; she is not controlled by PPO or a script. All prior inhabitants and their progress were preserved. The latest recorded verification ended and saved at tick 47,392; subsequent live play has resumed beyond that checkpoint. See [POPULATION-WATCH.md](POPULATION-WATCH.md) and the [model comparison](MODEL-SURVIVAL-COMPARISON.md).

The approved local [Clef Flash comparison](CLEF-FLASH-EVALUATION.md) completed 48 supplied-rule scenarios in three option orders per model. Clef Q4 on the GPU scored 128/144 versus typed-decisions Laya on the NPU at 70/144; median decision times were 80.5 versus 220.1 ms. Both failed half the farming-deadline cases. Clef made twelve clear high-confidence deadline errors plus three in a case with conflicting wording; the report preserves primary scores and discloses that limitation. This is a frozen decision diagnostic, not learned survival or a replacement for Laya. Downloads were verified, Gemma was restored, and the separate test processes stopped. Main-world health retained the same server and roster.

The [voluntary coordination pilot](VOLUNTARY-COORDINATION.md) completed **79,872 ticks in 183.04 seconds** with fresh independent PPO pairs, full primitive actions, private senses and individual rewards. Initially uninformed residents ate in 2/24 intact, 1/24 muted and 2/24 shuffled-symbol cases; all three seeds failed the communication gate. The separate audit reproduced every world transition and all 24,576 frozen evaluation decisions. No leader role, scripted route or live policy change was introduced. It motivated the current [action-category comparison](ACTION-CATEGORY-PILOT.md). The live server retained the same PID and all nine model IDs, ending the final health check automatically paused at tick 64,695.

Latest automated verification: **217 tests passed** across the suite, including the optional PettingZoo API and deterministic-seeding tests. The authorized [short/long-life comparison](LIFE-LENGTH.md) completed **638,976 ticks in 26.39 minutes** and passed its separate source/trace/checkpoint audit. Long-trained policies spent **39.90%** of evaluation time at zero fullness versus **23.69%** short-trained, worsening all three seeds. Both competence gates failed; no trial weights were promoted. Its read-only health check retained the original server PID and all nine model IDs at tick 61,333. Older present-tense scripted-controller descriptions have been corrected or explicitly marked historical.

The [memory-continuity diagnostic](MEMORY-CONTINUITY.md) completed 491,520 ticks and passed its separate trace/checkpoint audit. Rebuilding recurrent context reduced aggregate deprivation from 43.23% to 36.45%, but worsened one of three seed groups, retained only 4/12 low-deprivation lives, and increased thorn contacts. The primary hypothesis failed; the candidate remains experimental.

The [parallel environment](PARALLEL-ENVIRONMENT.md) preserves exact existing simulator/learner behavior and private observations. The earlier full [learning curve](LEARNING-CURVE.md) completed 611,072 ticks and passed its independent audit. All behavioral gates failed: final trained PPO spent 45.88% of time at zero fullness versus 13.26% for random actions. These experiments use different budgets/horizons and are not a matched cross-report ranking. The earlier real NPU and mixed-model restart verification remains recorded in [VALIDATION.md](VALIDATION.md).

The 602,304-tick [retention/exploration screen](RETENTION-EXPLORATION.md) completed and passed its audit. Retention kept more nutrition experiences, but all four conditions failed reliable feeding; no weights were promoted. The earlier 797,184-tick [matched replay comparison](REPLAY-COMPARISON.md) also failed every family's reliability gate.

The [small-agent research review](SMALL-AGENT-SWARMS-RESEARCH.md) compares CfC/NCP, continual backpropagation, e-prop, evolutionary populations and independent discrete communication. Reset-frequency and working-context comparisons have now failed their directional gates. Learned action factorization and curiosity remain separate candidate tests; CfC remains a proposed matched core comparison. No candidate brain package was installed or substituted for a live resident. The four approved PettingZoo/Gymnasium support packages were inspected and installed in a separate local folder. ALIEN was separately installed and launched successfully; see its [local trial record](ALIEN-EVALUATION.md).

The milestones below retain their historical controller descriptions; the current live controller is described above.

## First playable milestone

Implemented the deterministic Python world, browser renderer and controls, local server, save/load, pause/step/fast-forward, original procedural graphics, and Windows launch/stop scripts. At this first milestone, the UI labeled scripted resident behavior. Needs changed while mortality stayed disabled. Reproduction and learned communication were not enabled.

Browser interaction verified player movement, gathering food, pause/step, and saving. The live world subsequently progressed beyond these checks, and its state is preserved. Automated validation covers server controls and save restoration in disposable test worlds.

Independent untrained neural controllers are present separately in `agents/network.py`. The initial hidden size is 128 and is configurable at construction. Architecture manifests describe the backend, capacity, and observation/action versions. This does not implement automatic network growth or migration.

The first quick trial measured eight serial CPU forward passes at about 1.06 ms median, excluding observation encoding. GPU was slower for this small inference workload. NPU device availability was confirmed, but the resident model has not been run there. See [HARDWARE.md](HARDWARE.md).

## Historical construction and learning milestones

### Construction, senses, and visual memory update

Added modular wood/stone walls, floors, doors, placement previews, and full-material dismantling. Floors accept another piece above them; enclosed floor regions up to 64 tiles provide a roof. A door can open/close, and cannot close on an occupant. Existing shelter saves still load unchanged.

Each resident has an independent local observation with wall/tree/stone occlusion, nearby directional sounds, touch, exposure, and a private visual-memory collection. The browser's Through their eyes view uses that resident's visibility mask. Visual impressions use a 41-byte palette sketch and an 8-byte difference hash; similarity lookup is approximate and discards absolute position, identities, and precise quantities. Capacity is a saved world setting, initially 64 impressions per resident. See [details and limits](SENSES-AND-MEMORY.md).

At this milestone, the optional controller accepted these senses and recalled sketches, with a versioned, configurable parameter count. It was then untrained and disconnected from the visible scripted behavior. Memory acquisition/retrieval worked; learned use of those memories had not been demonstrated.

Browser checks verified building a wall, blocked movement, reduced vision, material recovery, opening a door, floor placement, and switching to another resident's view with different memories. Temporary test pieces were removed with full refunds; pre-existing structures and player position were preserved. The prior world was backed up before the update.

### Actual policy learning

The selected approach was [independent recurrent PPO with staged curiosity](ARCHITECTURE-DECISION.md). That decision recorded per-resident isolation, training requirements, acceptance experiments and when to consider a world-model alternative. Selection alone did not implement the learner; the later primary-population transition did.

The [first six-candidate architecture screen](ARCHITECTURE-TRIAL.md) ran random, tabular, feedforward, recurrent, fly-inspired and local NPU Laya variants in disposable food/reversal probes. Small online learners adapted; the fly variant matched tabular behavior, and the tested frozen Laya configuration preferred one label. These experiments are not full-game policy integration or PPO validation.

The [navigation and foraging follow-up](FORAGING-TRIAL.md) completed 30 disposable lives, including paired frozen-learning controls. The small MLP consistently acquired safe food but still suffered thorn injuries; GRU/fly results varied by cue mapping. Frozen Laya navigated and ate food but still ate harmful fruit and remained substantially slower. At that milestone, the proposed next confirmation prioritized the MLP and longer horizons; that experiment did not replace live residents.

The subsequent implementation connected per-resident observations, actions, rewards and recurrent rollouts to PPO, with resumable private checkpoints and held-out food-learning comparisons. The [primary-population transition](UNSCRIPTED-POPULATION.md) records that completed work. Reliable learned survival remains an open objective.

The implementation should follow the [expandable learning proposal](LEARNING-ARCHITECTURE.md): brain capacity is not part of permanent resident identity. Candidate additions include continual-plasticity methods, external experience memory, learned dynamics/curiosity, and skills. These are proposals awaiting experiments, not capabilities currently running in the game.

## Release boundary

The [hazard and revival experiment report](HAZARD-EXPERIMENTS.md) records a bounded three-seed offline pass: repeated food choices showed no learned avoidance, and repeated rescue on thorns caused an injury/revival loop unless the resident moved away. These are scripted-baseline observations, not trained-policy results. Reproduce with `python experiments/hazards.py`.

Apache-2.0 was selected. A local Git repository exists. No public GitHub repository or Hugging Face Space has been created for this project. The [publication plan](PUBLISHING.md) records remaining deployment work.

## Hazards and passing out

Version 0.3 adds harmful amber fruit, contact thorns, health/pain, injury counters, passing out below the health threshold and automatic recovery. Mortality remains off. See [mechanics and learning boundaries](HAZARDS.md).

## Shared farming update

Implemented finite stocks, food-to-seed conversion, timed public crops, scripted farming, player controls and crop sensory channels. See [resource economy](RESOURCE-ECONOMY.md) for that milestone's rules, limits and validation. Laya was retained as a permanent supported experimental architecture; the live residents were still scripted at that stage.

## Delayed farming learning pilot

[The farming experiment](FARMING-TRIAL.md) completed 150 disposable episodes. Tabular Q(lambda), rewarded only for useful food consumption, improved mean nutrition over an untrained control in all three seeds both solo and with competition. Aggregate gains were 22.4% and 59.3%. Frozen NPU Laya consumed starting food without farming in six evaluations; permanent support was the preference at that time. This is a fixed-plot pilot, not navigation generalization or demonstrated starvation prevention. Live residents were still scripted at that milestone. Source hashes, resource accounting and independent subagent review passed; 62 tests passed.

## Longer lives with movement and independent competitors

[Mobile farming](MOBILE-FARMING-TRIAL.md) completed 42 worlds/84 individual lives with separate learner tables, traces and RNGs. Two of three seed groups improved, but the overall preregistered gate failed; substantial zero-food time remains. Scripted controls stayed fed. Six learner checkpoints and raw evaluation traces are retained. A state-coverage audit found 65.6% of evaluated trained states unseen during training. This motivates testing generalization, not deploying these policies.

[Explicit-recipe Laya](LAYA-RECIPE-TRIAL.md) completed four long fixed-plot lives (Laya and scripted references, solo and competitive). Laya did not plant despite supplied recipe instructions, and spent 41.4% of ticks at zero fullness. At that stage it remained a supported frozen experimental model without live control. 65 tests passed; that experiment left live saves and resident controllers unchanged. Laya's later arrival is recorded in [the population record](POPULATION-WATCH.md).

## v0.5 hunger and curiosity

Implemented escalating internal hunger signals, reduced energy recovery, weak movement, nonfatal starvation fainting and food-assisted revival. The original v0.5 live residents used private saved scripted exploration counts. Mobile learning protocol 2 added integrated hunger cost and private fading novelty rewards. 77 tests passed; browser food/relief/revival checks and existing-world migration passed. No full protocol-2 learning run is claimed. See [motivation](MOTIVATION.md).

## Neural learning and social signals (v0.6)

The replay neural trial improved feeding but failed its deployment gate; the live residents remained scripted at that milestone. Opt-in growth of a trained model preserved outputs, but evolutionary improvement was unproven. That update introduced gentle taps and three meaningless tones, saved and locally perceived. It did not establish learned communication. See [trial](NEURAL-FARMING-TRIAL.md), [evolution](EVOLUTION.md), and [signals](SOCIAL-SIGNALS.md).

## Learned signaling diagnostic

Three independent sender/receiver pairs learned useful tone conventions in a scaffolded food-route task. All passed preregistered muted/shuffled controls. Learned signaling remains experimental and separate from the live population. [Report](TONE-TRIAL.md).

## Ten-tone channel (v0.7)

Agents have ten discrete tone actions with a brief silent gap, private ordered auditory memory, and sequence-aware general neural inputs. The browser verified 2 > 7 > 2 in the recipient memory after all sounds expired. These mechanics do not demonstrate learned multi-symbol language. [Details](SOCIAL-SIGNALS.md).

## v0.8 persistent population and resource ecology

Soil, irrigation and movable shared food baskets are live. Two independent online neural learners run in a separate local world with saved brains. Persistence passed; feeding competence did not. See [implementation and evidence](LIVING-POPULATION.md).

## Physical action filtering

Experimental residents can exclude locally known impossible actions while keeping harmful-but-possible choices. Filter settings and pending-transition masks persist in checkpoints. The existing population retains its weights and world. See [protocol and results](PHYSICAL-ACTIONS.md).

## Sequence learning and development worlds

Independent masked recurrent PPO now runs in a separate development world at http://127.0.0.1:8790/ . Each learner has its own contiguous rollout, optimizer, consequence predictor and private outcome journal. The browser inspector shows measured predictions, outcomes, preferences and policy versions. Old checkpoints retain their backend; no trial-trained weights are imported automatically. Near-food, depletion, hazard, farming and cooperative experimental layouts are available. Curiosity rewards default to off. See [protocol, implementation and evidence](SEQUENCE-LEARNING.md).

The 289,728-tick confirmation reduced harmful eating and unconscious time relative to initial weights, but failed its gate: feeding improved on only two of six maps and no life met the thorn-avoidance criterion. All controls also stayed fed. The zero-fullness nutrition accounting defect is now corrected in both backends; the confirmation had no zero-food ticks and its recorded results are unaffected.

## Finite-stock route experiment

[Scarcity and safe routes](SCARCITY-ROUTES.md) completed 435,456 ticks in 509.128 seconds with three training seeds and 24 frozen evaluations. The separate trace/hash/food-conservation audit passed. The learning gate failed: trained agents spent 51.28% of time at zero food versus 12.90% for matching initial policies, despite fewer thorn contacts. Only 3/12 lives met low-starvation and 2/12 met active-safe-route criteria. Post hoc traces show tones taking 81.24% of conscious decisions, motivating an action-category exploration comparison rather than a competence claim.

A separate fresh playable arena is available at http://127.0.0.1:8791/ and starts paused. Save/load, traveler access and the learning inspector passed browser checks. No trial-trained policy was imported into the named inhabitants. The roster now reports the actual resident count. Laya/fly options and existing saved worlds remain supported.
