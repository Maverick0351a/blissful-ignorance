# Which models stay fed?

October 5, 2026. These are measured project results, not a universal ranking of architectures. Zero-food time means the fraction of simulated ticks at zero fullness. Mortality is disabled; repeated fainting is still failure to sustain oneself.

## Latest PPO training-life comparison

The [matched short/long-life experiment](LIFE-LENGTH.md) gave each private
brain 16,384 decisions and 128 PPO updates. Long-trained policies spent
**39.90%** of held-out time at zero fullness versus **23.69%** short-trained,
worsening all three seed groups. Both achieved only **4/12** low-deprivation
lives. Initial weights were at **16.95%**, random feasible actions at **21.65%**,
and the privileged feasibility oracle at **0%**. Both competence gates failed.

The short arm received eight times as many fresh bodies and resource starts;
that is part of the reset treatment, not an isolated memory comparison.
Long-trained lives produced more crop food but explored fewer distinct tiles
on average. Both trained arms chose tones on about 82% of conscious decisions
in isolated arenas without listeners. This is measured policy behavior,
not established planning or communication. The 638,976-tick run passed its
independent evidence audit; no live controller or Laya state was changed.

## PPO working-context comparison and learning curve

A subsequent [memory-boundary comparison](MEMORY-CONTINUITY.md) tested the same
PPO with reset versus reconstructed working context at matched 4,096-decision
budgets. In continuous frozen evaluations, deprivation was **43.23% versus
36.45%**, with **4/12** low-deprivation lives in both arms. Reconstruction helped
two seeds and worsened one; its preregistered consistency hypothesis failed.
Initial weights were at 17.86% and random actions at 18.83%. Different horizons
and seeds prevent comparing these percentages directly with the curve below.
This was a boundary-handling test, not a new model-family winner or live change.

The [standardized curve](LEARNING-CURVE.md) tested four private PPO learners at
2,048, 8,192 and 16,384 decisions. Validation zero-fullness time was **39.55%,
42.35%, and 68.76%**, respectively. On separate final maps, the maximum-budget
policies spent **45.88%** of time at zero fullness, versus **15.13%** for initial
weights, **13.26%** for random actions and **0%** for the privileged feasibility
control. Only **2/8** trained final lives were below 1% zero fullness, and no
trained life passed the active-safe-route criterion. Every gate failed.

This is a two-seed diagnostic using a different curriculum and time-limit
bootstrap treatment from the earlier comparisons, so it cannot rank those
experiments against one another. It shows that more experience under this
particular schedule did not reliably improve feeding. All 611,072 ticks were
accounted for; no policy was promoted. Laya remains her own model and was not
part of this PPO-only curve.

## New matched full-senses comparison

The [matched replay comparison](REPLAY-COMPARISON.md) used identical local sensory encoding, action vocabulary, physical masks, rewards and experience budgets, with three training seeds and 12 held-out lives per condition. All reliability gates failed.

| Family | Trained zero-food time | Its initial weights | Trained lives under 1% zero food |
|---|---:|---:|---:|
| Recurrent PPO | 26.69% | 10.13% | 6/12 |
| Full-senses feedforward replay | 81.97% | 87.36% | 0/12 |
| Recurrent replay | 68.76% | 91.68% | 0/12 |

Uniform random legal actions spent 8.58% of time at zero food; a privileged feasibility control stayed fed. Recurrent replay improved over its own initial weights in all three seeds but remained unreliable. No trial weights were promoted, and Laya stays a distinct supported/live model.

This is closer to a fair interface comparison than the older studies below, but parameter counts, evaluation exploration and learning algorithms still differ. In 10/12 replay learners, recent-only FIFO memory had evicted all eating transitions by the end despite earlier meals. The follow-up below tests retention and exploration separately. Original studies and their limitations remain below.

## Completed retention/exploration follow-up

The [2x2 screen](RETENTION-EXPLORATION.md) held the recurrent replay model and
2,048-decision budget constant. FIFO/flat exploration spent **81.38%** of time
at zero fullness; retention alone **85.16%**; balanced exploration alone **82.65%**;
both **79.55%**. Each had **0/12 reliable lives**. Initial weights were at 91.68%,
random at 11.07%, and the feasibility oracle at 0%. Retention kept more nutrition
experience, but both directional main-effect hypotheses failed across seeds.
No live model was replaced. The subsequent [baseline learning curve](LEARNING-CURVE.md)
tested larger predetermined training budgets; its results and further
diagnostics are recorded above.

## Comparable results within each experiment

| Model and experiment | Trained/model zero-food time | Same-study control | What was established |
|---|---:|---:|---|
| Feedforward DoubleDQN with replay, mobile farming, curiosity off | 4.41% | Initial weights 63.11%; random 32.20%; script 0% | Large feeding improvement; only 14/18 trained lives passed the reliability criterion, so deployment gate failed. |
| Same DoubleDQN, curiosity on | 8.93% | Same controls above | More nutrition but more deprivation than curiosity off; neither configuration passed. |
| Tabular Q(lambda), long mobile farming | 29.76% | Untrained 38.48%; script 0% | Partial improvement; one of three seed groups regressed and the overall gate failed. |
| Frozen local NPU Laya, explicit-recipe farming diagnostic | 41.4% | Script 0% | Consumed its two starting meals; no seed conversion, planting or harvesting. One seed, two conditions. |
| Recurrent PPO, replenished development rooms | 0% | Initial weights and script also 0% | Injury and unconscious time improved, but abundant food made zero starvation uninformative. |
| Recurrent PPO, finite-stock route arenas | 51.28% | Initial weights 12.90%; script 0% | Training worsened feeding. Only 3/12 lives met low-starvation and 2/12 met active-safe-route criteria. |

Evidence: [neural replay](NEURAL-FARMING-TRIAL.md), [mobile tabular](MOBILE-FARMING-TRIAL.md), [Laya diagnostic](LAYA-RECIPE-TRIAL.md), [sequence learning](SEQUENCE-LEARNING.md), [scarcity routes](SCARCITY-ROUTES.md). Neural replay percentages divide the published mean tick counts by 19,200. Tabular percentages use the same denominator in its own experiment.

The feedforward trial uses a compact 23-input, two-layer 64-unit DoubleDQN with a nine-action vocabulary. The live PPO agents use local visual/sensory encoding, recurrence and a much larger construction/social/ecology action vocabulary. The different observations, actions, tasks, horizons and training budgets prevent reading this table as a head-to-head architecture leaderboard.

## Fly-inspired and earlier small learners

In the 384-tick [foraging screen](FORAGING-TRIAL.md), the MLP averaged three ordinary meals and no harmful meals. GRU and fly-inspired learners varied by seed and failed the swapped-food case; fly averaged 1.67 ordinary and two harmful meals. Frozen Laya averaged 1.67 ordinary and one harmful meal. Starting fullness and the short horizon prevented a meaningful sustained-starvation comparison.

The food-choice [architecture screen](ARCHITECTURE-TRIAL.md) reset bodies between choices. Its high late safe-choice rates for tabular, fly, MLP and GRU models show adaptation in that probe, not survival competence. Frozen Laya had an option-label bias in that screen. The short fixed-plot [farming trial](FARMING-TRIAL.md) also ended before starting supplies could produce a strong starvation test.

## Direction supported by this evidence

No learned family has passed the longer reliable-feeding gate. Keep the feedforward replay learner as a serious comparison, retain Laya as its own model track, and test them against recurrent PPO on matched local observations, actions, resource budgets and held-out worlds. Preserve the live inhabitants; do controlled architecture comparisons in copies.

The scarcity PPO trace allocated 81.24% of conscious choices to tones, compared with 69.36% for initial weights. Ten tone actions versus a single gather action can distort a flat action distribution. A matched comparison of flat versus category-balanced exploration is justified, but this observational correlation does not establish that tones caused starvation. Keep every tone available and let models choose; do not substitute a survival script.
