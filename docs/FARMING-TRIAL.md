# Farming learning pilot

October 4, 2026. Restricted delayed-reward task using the real resource economy. The live saved world is not read or modified.

## Question and preregistration

Does a small tabular Q(lambda) learner, rewarded only for useful fullness gained by eating, learn to sacrifice food and wait for a harvest? Compare solo farming with a neighbor competing for the same fruit. The preregistered success condition requires trained nutrition to exceed the untrained baseline in both conditions on all three base seeds, with more than two meals and at least one harvest.

Before the run, `runs/farming-trial-20261004/preregistration.json` recorded seeds, hypotheses, hyperparameters, episode lengths and SHA-256 hashes of the experiment and simulation. Base seeds: 4813, 5824, 6835. Twelve training episodes per seed/condition (960 decisions), then three evaluation episodes with weight updates off. Laya gets one evaluation episode per seed/condition because local NPU inference is substantially slower. No tuning on these evaluation results.

## Setup

A resident stays beside one empty plot with two food, 40 fullness, no seeds and no wild food. It can rest, eat, make seeds, plant or gather. One decision is followed by 120 real simulation ticks; every resident explicitly waits between decisions. Crop maturation remains 480 ticks. Eighty decisions span 9,600 ticks, equivalent to 40 minutes at normal game speed. No within-life resets, food replenishment or artificial crop acceleration occur.

The competition condition adds an adjacent scripted resident who gathers visible ripe fruit or eats its own collected food when not full. It does not plant. Both characters' commands are resolved by the world's shuffled order, so the last fruit cannot be duplicated. The rival can consume its supply and continue competing rather than becoming permanently pack-full.

Reward is actual useful fullness /25 from successful eating. There are no planting, seed-making, harvest or exploration bonuses. We separately measure mean and final fullness, ticks below20 and at0, meals, conversions, harvests, rival harvests and final inventory. Mortality remains off; zero fullness does not cause injury.

The learner uses current local observation: fullness in20-point bins, exact inventory, visible crop stage and amount, and visible neighbor presence. Q-values start at0. Alpha .15, gamma .995, lambda .9; training epsilon .2, evaluation0 with random ties. Replacing eligibility traces clear after non-greedy choices and reset per episode. Terminal transitions do not bootstrap. Each seed/condition owns a separate learner. Legal-action masks encode physical prerequisites only.

Comparisons: untrained frozen zero-Q, masked random actions, a recipe-informed scripted farmer, and frozen pretrained local NPU Laya. Laya sees the same compact sensory variables plus three recent action/reward records. Its legal option order rotates; no cloud service is used. The adapter and checkpoint remain unchanged.

## Validation and boundaries

A separate fixture seed9991 confirmed that the scripted farmer completes the full cycle in both conditions: five meals,125 nutrition and no zero-fullness ticks. Three experiment tests cover this positive control, food/seed accounting, terminal credit assignment and disabled learning during evaluation. A subagent reviewed the method and implementation without modifying the code.

This tests an observation-limited controller in a fixed plot, not navigation or fully observed optimal planning. Solo evaluation worlds are mechanically identical; fresh seeds mainly change policy randomness, while competition also varies action resolution. They are not new maps or evidence of environmental generalization. Crop stages alias different ages.

Frozen zero-Q with random ties is effectively another masked-random baseline, not an exposure-matched training control. Laya differs in pretrained knowledge, language encoding and recent memory; this is not a controlled architecture ranking. Its prompt does not state the exact recipe, growth duration or yield. A failure here cannot establish failure after explicitly teaching those rules. Rotating options reduces positional bias but does not prove balanced frequencies as legal actions change.

Total-nutrition ceiling is approximately136.8: initial hunger60 plus76.8 fullness lost over the run. Eating two initial food provides50; more than two meals with harvests establishes use of cultivated food. Nutrition alone is not stable survival, hence the separate fullness measurements. This is a small pilot, not a statistical claim about open-ended intelligence. MLP, recurrent and fly-inspired farming extensions are not implemented in this experiment; all remain project candidates alongside permanent Laya support.

## Results

Completed **150 episodes**: 72 training, 72 small-policy evaluation and 6 frozen Laya evaluation. Each episode spans 9,600 simulation ticks. All 62 tests passed. Source hashes still match preregistration; food/seed accounting and eating-only rewards were checked.

| Condition | Policy | Evaluation lives | Mean nutrition | Mean fullness | Mean harvested |
| --- | --- | ---: | ---: | ---: | ---: |
| solo | q_lambda | 9 | 123.15 | 86.75 | 17.44 |
| solo | frozen | 9 | 100.62 | 70.95 | 10.33 |
| solo | random | 9 | 97.48 | 67.67 | 9.56 |
| solo | scripted | 9 | 125.00 | 78.16 | 18.00 |
| solo | laya | 3 | 50.00 | 51.28 | 0.00 |
| competitive | q_lambda | 9 | 126.89 | 78.97 | 11.89 |
| competitive | frozen | 9 | 79.66 | 57.22 | 5.00 |
| competitive | random | 9 | 67.05 | 50.46 | 3.00 |
| competitive | scripted | 9 | 125.00 | 79.41 | 20.00 |
| competitive | laya | 3 | 50.00 | 51.28 | 0.00 |

**Positive narrow result:** trained Q(lambda) exceeds untrained mean nutrition in all six seed-condition pairs, with more than two meals and at least one harvest in every Q evaluation. This meets the preregistered gate when evaluated by per-seed means. It does not win every paired episode: three losses and one tie remain. Mean nutrition improves about 22.4% solo and 59.3% with competition.

An independent subagent audit confirmed means, trace reward sums and food/seed conservation. Importantly, both trained and frozen agents have zero zero-fullness ticks. Initial 40 fullness + 50 from the two starting foods exceeds 76.8 depletion. Therefore this is **not evidence of learned starvation prevention or sustained survival**. The learner also eats partial meals near fullness, so meal counts alone overstate nutrition; use actual fullness gained.

Laya consumed its two starting foods and did not convert, plant or harvest in any of its six lives. Median decision latency averaged about 214-215 ms on the local NPU. Laya remains supported regardless of this outcome. Its frozen parameters and short history did not change during these runs. A follow-up should compare explicit recipe teaching and longer consequence memory, with the same observation budget across policies, before judging its suitability.

Next gate: longer lives that outlast initial supplies, changing plot locations and an independently learning competitor. Freeze the evaluation plan before extending training. This run does not replace any live resident controller. Raw observations, action/reward traces, preregistration, complete marker and audit are in `runs/farming-trial-20261004/`.


## Reproduce

Use the existing local Python environment and a new output directory:

```powershell
python experiments/farming_trial.py --output runs/farming-trial-new --laya-adapter PATH/TO/laya_lite.py
```

Omit the adapter for the four small-policy conditions only. Nothing is downloaded. The output directory must not already exist. Policy tables are retrained from fixed seeds in this pilot; persistent learned checkpoints and live controller integration are not implemented here.
