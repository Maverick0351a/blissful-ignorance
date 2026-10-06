# Explicit-recipe Laya diagnostic

October 4, 2026. A local frozen-model check accompanying the mobile farming pilot. It does not change the live game or the existing Laya checkpoint.

## Design

The prior Laya prompt did not explain the farming conversion, yield or wait. This prompt explicitly states: one food becomes two seeds, one seed becomes three fruit after four decisions, eating gives up to 25 fullness, each decision loses0.96 fullness, and neighbors may harvest. It advises preserving a way to grow food. This is supplied knowledge, not discovery or weight learning.

Use the previous fixed-plot arena for 160 decisions (19,200 actual simulation ticks), starting at 40 fullness with 2 food. Potential hunger loss 153.6 now exceeds the 90 available from starting fullness and food. Compare frozen Laya with the recipe-informed scripted positive control, solo and with the previous scripted harvesting neighbor. Fresh seed 79021; one life per policy/condition. This is an illustrative diagnostic, not a statistically established comparison. It is separate from the independently learning mobile residents.

Observation, reward, legal masks, simulation stepping and metrics come from `farming_trial.py`. The only overrides are episode length and the Laya question. We preserve each agent's compact observations and last three action/reward entries. Choice order rotates. Before inference, token length is checked against the 512-token limit rather than silently truncating. Existing local OpenVINO NPU adapter is loaded offline; no cloud or download is involved.

Preregistration in `runs/laya-recipe-20261004/preregistration.json` fixes source hashes and the success condition: Laya converts, plants, harvests and eats more than two meals in both conditions. The prior 80-turn trial differs in horizon, so this diagnostic cannot estimate a causal effect of recipe teaching against that historical result. Hydration, injury and navigation are not evaluated; mortality is off.

## Results

**Hypothesis failed.** In both conditions, Laya consumed its two initial foods, made no seeds, planted nothing and harvested nothing. It gained 50 fullness from food, ended at 0 and spent 7,950/19,200 ticks (41.4%) at zero fullness. Mean fullness was 26.21. It committed no invalid actions. Median decision latency was214 ms solo and222 ms with a neighbor.

The scripted positive control gained 200 fullness, ate eight meals and spent zero ticks at zero fullness in both conditions. Its mean fullness was 80.07 solo and 80.70 with the neighbor. Resource accounting and all preregistered source hashes passed the post-run audit. The diagnostic completed; no worker remains running.

This checkpoint/prompt/short-memory combination did not execute a productive farming sequence despite an explicit recipe. That is not proof that the Laya architecture cannot learn it. The weights were frozen and no task-specific training was attempted. Keep the supported Laya track; require a measured improvement from calibration, experience representation or task training before assigning it survival-critical control. Raw traces, results and audit are in `runs/laya-recipe-20261004/`.

Reproduce with `python experiments/laya_recipe_trial.py --output runs/laya-recipe-new --laya-adapter PATH/TO/laya_lite.py` using the existing local runtime. The output directory must be new.
