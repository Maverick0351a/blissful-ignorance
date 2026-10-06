# Physical action filtering

The optional neural population can filter actions using only each resident's observation. This changes which actions can be sampled, not which goals it should pursue.

Examples: an empty pack excludes eating; missing materials exclude construction; visible occupancy blocks moving into a neighbor; a full pack excludes gathering. Building costs and action rules are built-in mechanics, so this is supplied knowledge rather than learned affordance discovery.

Hazardous choices remain available. A creature may eat amber fruit or approach thorns. The filter does not use private inventories of other residents, the global irrigation reserve, unseen resources, absolute coordinates or player identity. Unknown conditions can still lead to failed attempts, and another agent can change the world after an observation. Tap cooldown and some legacy-crop distinctions are not represented, so the filter is intentionally not a perfect action oracle.

The actor samples and trains against the same stored action mask. Previous unfiltered pending transitions use an all-actions mask when resumed. The filter setting is saved in each brain checkpoint; older checkpoints load unfiltered by default. Model weights, optimizer and recurrent states are retained when enabling filtering. Historical manual saves retain their original settings.

Frozen evaluation disables learning but still updates recurrent state and chooses actions stochastically. Independent training seeds use matched network initialization, world seeds and budgets. The initial-weight filtered control distinguishes benefits from filtering alone from benefits of weight updates.

## Long-life test protocol

Training seeds: 31041, 31052, 31063. Each trains filtered and unfiltered populations for 1,200 ticks. Evaluation uses fresh world seeds offset by 100,000, 8,000 ticks per condition, and three conditions: trained filtered, trained unfiltered, initial-weight filtered. Each world has two learners and six scripted residents. Learners begin with 20 fullness and one food; these reserves cannot alone cover the horizon. Evaluation sampling RNGs match across conditions; weights are frozen and checked.

Primary gate, set before running: in every seed, trained filtered must improve mean nutrition and reduce zero-food ticks against both controls; at least five of six filtered lives must spend under 1% of ticks at zero food. Conscious invalid-action rate is a secondary measure, not a survival result. There is only one evaluation world per training seed; this remains a small experiment. Nutrition includes all bodily food relief, including help from other residents, and is not proof of independent farming.

Raw checkpoints, periodic traces, preregistration receipt and results are stored locally under `runs/affordance-trial-20261004/`. Historical main-world and experimental saves are not used by the comparison.

## Results: survival gate failed

The run completed in 910 seconds over 79,200 total ticks. An independent local audit verified the preregistration and source hashes, all nine evaluation traces and the recomputed gate. The frozen evaluation covered six lives per condition:

| Condition | Mean nutrition restored | Mean zero-food ticks / 8,000 | Conscious invalid choices | Lives below 1% zero-food |
| --- | ---: | ---: | ---: | ---: |
| Trained, filtered | 106.172 | 426.5 | 0.145% | 4/6 |
| Trained, unfiltered | 102.921 | 786.0 | 74.437% | 3/6 |
| Initial weights, filtered | 110.328 | 242.5 | 0.458% | 5/6 |

None of the three seeds met the required improvement against both controls. Filtering sharply reduces physically impossible conscious choices, but this run does not establish learned survival improvement: the initial-weight filtered control had better aggregate food outcomes than the trained filtered population. Injury-related unconsciousness remains possible, including with adequate food. Nutrition also includes amber fruit and assisted revival, so it cannot establish farming skill or independence.

Accounting correction found October 5: this historical implementation also counted up to .008 false food relief per tick at zero fullness because decay clips at the floor. The nutrition values above retain the original recorded metric and overstate bodily relief in starved lives. Zero-food, conscious invalid-action and reliability counts are unaffected; the deployment verdict remains failed. Both current learner backends now correct this floor case, covered by regression tests. The later sequence comparison had no zero-food ticks and is unaffected. No historical rerun or exact corrected nutrition totals are claimed.

The physical filter is enabled in the separate experimental population. No trial-trained checkpoint was promoted into either live world. The next learning investigation should address delayed consequences and multi-step food acquisition, with the initial-weight filtered control retained.

Local evidence: `results.json`, `complete.json`, `audit.json`, and `audited-source.zip` in the run directory. Preregistration SHA-256: `cea1d52a7a769b99e9e92b1b185d905d6bbfecd59ccac78b2e0d3c353634f7f3`.

## Integration and checks

108 tests passed. New tests cover locally known feasibility, preservation of harmful choices, rejection of privileged input fields, basket weight, masked sampling and compatibility with old pending transitions. Existing exact-checkpoint continuation tests still pass. Python compilation, JavaScript syntax and diff checks passed.

The two existing experimental residents were backed up to `runs/learning-population/runs/before-physical-filter.pt`. Enabling the filter preserved their world at tick 2422 and all model weights. The browser confirmed the saved filter state and the population remains paused. Trial-trained checkpoints do not replace those residents. The main-world simulation and residents are unchanged. Historical manual saves restore their own prior filter settings; cumulative inspector failure counts include choices made before this update.

Screenshot: `runs/physical-filter.jpg`. A separate post-run diagnostic samples one starting observation from seed 31041; its trained and initial action probabilities remain fairly flat. That single observation is not a general policy analysis or confidence-calibration result.
