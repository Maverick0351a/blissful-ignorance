# Mobile farming and independent learning competitors

October 4, 2026. Exploratory follow-up to the fixed-plot farming result. Implemented and run by a subagent; simulation interface, accounting and conclusions independently reviewed by the parent.

## Design

A 9×9 walkable sand arena contains six fertile grass plots sampled from nine separated sites. Two residents start apart, each with 40 fullness and two food. Movement, rest, eating, seed conversion, planting and harvesting are available through nine actions. The plant action selects the nearest locally visible reachable empty grass; target selection and physical legality masks are supplied scaffolding, not learned abilities.

Each resident owns a separate Q table, eligibility traces and action RNG. Only World.observe inputs are used for policy state: fullness/inventory bins, direction and distance bins to visible ripe fruit and free plots, nearby crop stage, and visible-neighbor presence. Observer trace positions and crop ownership are excluded from policy inputs. Neither agent can access the other's parameters or experience.

Three base seeds 7413/8524/9635 each train an independent pair for eight episodes, then evaluate that pair with updates off on two fresh seeds. Compare untrained pairs and recipe-informed scripted pairs on the same evaluation seeds. Training changes both competitors, so this is a population comparison, not a controlled causal estimate for one actor against a fixed rival. Reward is only useful fullness from eating divided by 25. Q hyperparameters match the prior pilot: alpha .15, gamma .995, lambda .9, epsilon .2 during training and 0 during evaluation with random ties.

Every decision advances 120 actual simulation ticks, explicitly resting all residents between decisions. Each 160-decision life spans 19, 200 ticks: 153.6 fullness depletion exceeds 90 available from initial fullness and food. Crop maturation remains 480 ticks. Mortality is off and zero fullness does not cause injury; zero-food time measures deprivation, not deaths or complete survival.

The arena layout changes across seeds. Inspection found five of six evaluation layouts absent from that pair's training layouts; one evaluation in seedgroup 8524 repeats a training layout. Thus evaluation uses fresh seeds, but is not a strictly unseen-layout benchmark. No reroll was performed after observing this overlap.

Source hashes, seeds, hyperparameters and hypothesis were recorded before the official run. Hypothesis: trained pairs achieve both higher mean nutrition and fewer zero-food ticks than untrained pairs in each of the three seed groups. A separate positive-control seed 123456 verified navigation and renewable feeding before evaluation.

## Results

42 world episodes (84 individual lives) completed in 54.6 seconds: 24 training episodes and 18 evaluation episodes. Six independent trained Q checkpoints and eighteen raw evaluation traces were saved.

| Seed group | Nutrition trained / untrained | Zero-food ticks trained / untrained | Gate |
| --- | ---: | ---: | --- |
| 7413 | 50 / 50 | 8, 072 / 9, 033.25 | Fail: nutrition tie |
| 8524 | 106.25 / 75 | 2, 684.25 / 5, 182 | Pass |
| 9635 | 62.5 / 50 | 6, 388 / 7, 950.5 | Pass |

Each value averages two actors over two evaluated worlds. **Overall preregistered hypothesis failed.** Two seed groups improved; reliable sustained farming was not established.

| Policy pair | Mean nutrition per actor | Mean zero-food ticks | Mean fullness | Mean plantings | Mean harvests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained | 72.92 | 5, 714.75 | 25.79 | 2.08 | 3.42 |
| Untrained | 58.33 | 7, 388.58 | 20.20 | 2.50 | 4.00 |
| Scripted reference | 200 | 0 | 73.51 | 8.83 | 21.83 |

Trained actors visited 39.58 distinct positions on average; untrained 41.25, scripted 6.5. Movement alone is not productive foraging. Aggregate trained nutrition is 25% higher, but plantings/harvests are lower than the untrained average. Do not interpret the nutrition gain as universally better farming behavior. The scripted reference establishes that navigation and the resource cycle can sustain feeding in all evaluated worlds.

A post-run diagnostic found 1, 259/1, 920 trained evaluation decisions (65.6%) had compact states absent from that actor's saved training table. This supports investigating state fragmentation and limited coverage; it does not isolate the cause of failure or prove that another architecture will succeed. New states use zero values and random tie-breaking.

The observer-only competitor_harvest metric is **approximate and excluded from conclusions**: resource selection was recorded before shuffled resolution, so a concurrent gather can deplete that resource and cause a different successful harvest. Shared stock itself remains authoritative; this ambiguity affects attribution only.

## Validation and next decision

65 tests pass, including independent Q/trace state, starting resources/legal masks, and actual step counts. Preregistered source hashes match. Parent audit reconstructed food/seed accounting and nutrition sums from every evaluation trace. The source uses local observations only; absolute positions are retained solely in diagnostic traces. No live save was loaded or changed.

Next recommended experiment: a small trainable function approximator with stored experience and delayed credit assignment, compared with this tabular baseline under the same longer-life protocol. Freeze the candidate, budgets, truly disjoint layouts and metrics before testing. Do not switch the live population until feeding remains reliable under movement and competition.

Laya remains permanently supported. The separate [explicit-recipe diagnostic](LAYA-RECIPE-TRIAL.md) failed to farm with frozen weights; it is not part of this independently learning-pair score.

## Artifacts and reproduction

- `runs/mobile-farming-trial-20261004/official/`: preregistration, 42 episode records, six Q checkpoints, 18 evaluation traces, completion receipt, subagent audit and parent accounting audit.
- `runs/mobile-farming-trial-20261004/pilot/`: separate positive-control run.
- `experiments/mobile_farming_trial.py`; run with `python experiments/mobile_farming_trial.py --output runs/mobile-farming-new` using the existing local Python runtime. Output must be a new directory; no downloads are needed.

Checkpoints are per-agent state/Q-value JSON. They are retained experimental artifacts; the live game has no loader for this restricted nine-action policy.
