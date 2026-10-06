# Continuing-population observation

October 5, 2026. We observed the actual saved inhabitants, preserved their learning state, and continued the playable checkpoint. This is one shared world's life history, not a controlled architecture comparison.

## Completed windows

| Window | World ticks | Advanced | Recorded runtime |
|---|---|---:|---:|
| Eight-PPO pilot | 38,111 → 39,135 | 1,024 | 13.78 s |
| Eight-PPO follow-up | 39,135 → 47,327 | 8,192 | 108.83 s |
| Laya integration check, after arrival and one UI step | 47,328 → 47,392 | 64 | 14.67 s |
| October 6 continuing mixed population | 84,935 → 85,447 | 512 | 85.32 s |

The latest dated observation log (local evidence; not bundled) records four
ordinary PPO meals, no PPO zero-food time, and Laya at zero food throughout
the window. Starting bodies and resources differed. The full roster and
complete ending checkpoint were verified after a paused restart; no behavior
gate or replacement decision follows from this uncontrolled observation.

The first two windows advanced 9,216 ticks, equivalent to 38 minutes 24 seconds at normal four-tick-per-second playback, or 3.84 in-world days. The latter window includes Laya, so its results are kept separate. All runs continued online learning for the existing PPO brains. No trained trial weights, scripts, recipes, resource injections or player actions were applied during observation.

## Which residents stayed fed?

Results below cover the **8,192-tick follow-up**, all using the same recurrent PPO architecture with independent weights and experience. Ordinary meals and harmful amber meals are separate.

| Resident | Time at zero food | Time unconscious | Ordinary meals | Amber meals | Successful plantings | Unique tiles visited |
|---|---:|---:|---:|---:|---:|---:|
| Moss | 0% | 0% | 5 | 1 | 18 | 148 |
| Pip | 0% | 0% | 3 | 4 | 18 | 132 |
| Ember | 18.0% | 4.9% | 1 | 6 | 10 | 112 |
| Fern | 14.5% | 2.9% | 9 | 0 | 40 | 77 |
| Cove | 53.4% | 12.3% | 0 | 2 | 0 | 26 |
| Ochre | 24.8% | 6.2% | 4 | 0 | 10 | 103 |
| Luma | 0% | 0% | 5 | 2 | 10 | 137 |
| Reed | 0% | 0% | 7 | 1 | 54 | 114 |

Zero-food time is post-step fullness <=0, and unconscious time is post-step occupancy of that state. They are not counts of deaths: mortality is off. Starting bodies and locations differed, so these are individual outcomes, not evidence that one initialization has learned a better general strategy. Pip had previously starved during 51.3% of the shorter pilot; this later improvement is observational.

Each brain made exactly 2,048 decisions and completed 16 weight updates in the follow-up. Four residents avoided starvation, but several still ate harmful amber fruit. Sustained safe feeding is a stronger criterion than simply having food in the body.

## Behavior and environment

The group completed **160 plantings, 91 food-to-seed conversions, 466 construction actions and 410 dismantles**. It ate 34 ordinary meals and 16 amber fruits, revived neighbors five times, tapped 42 times, and successfully gave items 32 times. Construction actions included 339 floor placements, 84 wood walls, 19 stone walls and 24 doors. Repeated construction/dismantling can inflate action counts; structures present increased from 54 to 106.

Harvestable ordinary food on the ground increased from **617 to 895**. Berry-resource locations increased from 166 to 301, with growing crops moving from five to seven. Farming mechanics are producing food in this continuing world. The recorder does not reconstruct exact maturation yields or who later harvested whose crop, so it cannot prove productive farming plans, ownership, intentional cooperation, or a self-sustaining society.

**Cove's difficulty was local access and behavior, despite global abundance.** Cove never had ordinary food in a policy observation or saw it within gathering reach. It did see ordinary food at distances two through eight in 401 of 1,794 conscious decisions. Its 51 successful gathers yielded 41 wood, six stone and four amber fruit; its only meals were two amber fruits. It moved 270 times yet covered only 26 tiles, and then stayed at zero food for the final 4,371 ticks. This supports investigating local navigation and competing material actions. It does not prove permanent entrapment or isolate a cause.

Tone actions occupied roughly 54–63% of conscious decisions. The expected tone share from uniformly selecting physically allowed action slots was also roughly 55–63%. Ten tone slots versus one gather slot explains why even broad random exploration can look highly vocal; this run does not establish learned communication. Instantaneous sounds had expired at policy decision times, but private auditory memories retained them. Cove received auditory memory in 1,103 conscious observations.

## Laya remains a separate model

Laya arrived as a ninth autonomous resident at tick 47,327, using the existing local typed-decisions model on the Intel NPU. The original eight bodies, brains, optimizers, memories and progress were retained. Laya's own saved RNG, chosen-action history and measured consequences persist through restarts. Her weights are currently frozen; she is not a renamed PPO resident and does not yet train her weights from world experience.

Her first 64 observed ticks produced **16 model decisions: seven moves and nine tones**, all physically successful. She saw no ordinary food in these decisions, ate nothing, and began fully fed with an empty pack. No starvation during this short arrival check is **not survival success**.

Laya uses compressed local body, touch, nearby-resource, neighbor and sound information, including the last five tone memories. Up to three recent real consequences are supplied when they fit; older context may be omitted. Full visual-memory sketches are not yet encoded into her prompt. Every feasible primitive action can enter a shuffled model tournament, including all ten tones. This accommodates her 512-token, 16-option NPU interface, but it differs from PPO's full action distribution. The conservative question budget is checked against the installed checkpoint's actual 256-token head configuration.

The mixed-model check ran much slower than the PPO-only windows: 64 ticks in 14.67 seconds including first model loading and report processing. This is an integration measurement, not a controlled hardware benchmark. Fast-forward remains bounded by actual inference speed. If model inference fails, the world pauses and displays the error; no script replaces Laya.

## Repeatable monitoring

Run `scripts/Watch-Population.ps1 -Ticks 512 -WallSeconds 120` from the project. It gracefully saves/stops the owned local server, obtains exclusive checkpoint ownership, backs up the start, observes a bounded continuation, independently recounts key trace metrics, atomically publishes the ending checkpoint, and restarts paused. A STOP file in the active output directory requests an early finish. Failures preserve the original live save and evidence. A crashed process may leave `runs/population-owner.json`; verify its recorded process has stopped before removing that specific lock.

Each dated `runs/population-watch-*` directory contains start/end full checkpoints, exact per-decision local observations and action masks, every tick's bodies, events, periodic world summaries, source archive and hashes, summary and completion receipt. World-wide crop/structure/cache changes are sampled rather than traced as a full resource ledger. Live save/load remains full-population persistence; manually loading an older pre-Laya save intentionally restores that earlier roster.

Observation checkpoints accumulate disk space. The wrapper stops creating further runs at 4 GiB of recorded observation evidence and asks for reviewed archiving; it never automatically deletes history.

Raw evidence:

- `runs/population-watch-20261005T092440361Z/` — pilot.
- `runs/population-watch-20261005T092552823Z/` — follow-up and independent subagent audit.
- `runs/population-watch-20261005T093428648Z/` — real Laya NPU integration continuation.
- `runs/laya-arrival.json` — prior checkpoint hash and backup reference.

The independent audit matched trace counts, identities, decisions, updates, configurations, checkpoint continuity and archived-source hashes. Consecutive checkpoint files had different serialized hashes but every field and tensor matched. A deterministic test also confirmed that adding the recorder produces exactly the same world, weights, RNGs and recurrent state as an unrecorded continuation across PPO updates.

## Next informative experiments

Preserve these residents and compare architectures in copies with matched starting bodies, local cues, actions and resource budgets. Prioritize recurrent PPO versus the promising feedforward replay learner, while keeping Laya and the fly-inspired model as separate comparisons. Compare flat action exploration with a model-selected category hierarchy that retains all tone symbols; do not insert survival routes. Evaluate ordinary feeding, harmful consumption, deprivation, useful crop yield, navigation and generalization across several new maps. Use muted/shuffled-tone controls before claiming communication.

The historical [model comparison](MODEL-SURVIVAL-COMPARISON.md) shows why no architecture should yet be declared the winner.
