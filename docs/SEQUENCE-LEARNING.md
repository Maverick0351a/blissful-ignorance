# Sequence learning and development worlds

October 5, 2026. A new independent recurrent PPO backend uses the existing local senses and full primitive-action set. Each resident owns a 64-unit LSTM, policy/value weights, optimizer, action RNG, rollout, consequence predictor and bounded private outcome journal. Width and budgets are configurable. Historical one-step populations retain their original backend and saves.

## Training mechanics

The collection policy remains fixed over 128 decisions. Four optimization epochs use contiguous sequences, stored behavior log probabilities and physical masks, bootstrapped generalized advantages, PPO clipping and KL early stopping. Gradients cross the whole sequence. Gamma .997 and lambda .98 operate per four-tick decision. Learning rate .0003, clip .2, entropy .01, gradient cap 1. This is our implementation in existing PyTorch, informed by [recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html); no package was downloaded or installed.

After an update the recurrent state deliberately resets. This makes rollout boundaries explicit; it limits uninterrupted recurrent memory to a rollout. Pausing and fainting do not create terminal events or reset identity. Experimental episode boundaries are explicitly terminal and reset body/location; they are not deaths in the live population. Checkpoints include unfinished sequences, pending observations, all optimizers, hidden state, RNG, policy version and diagnostics.

Feedback uses observable food relief, hunger distress and injury. Injury costs use health loss or increasing pain, including at the health floor. Automatic recovery earns no positive reward. No reward is assigned to construction, planting, rescue or talking. Hazardous-but-possible choices remain available.

The compact predictor learns seven short-term changes: fullness, hydration, health, pain and carried food/wood/seeds. It uses only the current resident's encoded senses and chosen action. It has its own optimizer; it does not modify the policy encoder or plan actions. This is a consequence-prediction scaffold, not a Dreamer implementation or a model of crop growth. The inspector compares predicted and observed deltas in bodily/inventory units. Its MAE uses normalized outcomes and is not confidence calibration. Most transitions have small changes, which can make aggregate MAE look good without predicting rare injuries accurately.

A capped optional curiosity proxy compares fast and slow prediction-error averages. Raw surprise earns nothing. The coefficient defaults to zero and is zero in the confirmation. Comparing these moving averages can confound predictor improvement with changed experience; it is not yet evidence of effective learning-progress curiosity.

## Experimental environments

`experiments/development.py` supplies near-food, depletion, hazards, farming and cooperation layouts through ordinary world mechanics. The first three form the current training curriculum. No stage label, global position, hidden resource list or scripted goal enters a brain. Separate initial rooms give residents independent experience; offline comparisons remove all scripted neighbors and player intervention.

Farming retains real soil, moisture, crop timing, finite food and irrigation. The cooperative layout connects the rooms, puts food and water on different sides, and provides shared storage and asymmetric materials. Tones retain no assigned meanings and social actions earn no bonus. These two later environments are implemented opportunities; their competence and learned cooperation have not been evaluated here.

## Confirmation protocol

Seeds 42071, 42082, 42093. Four episodes of 512 decisions at each of three stages give 6,144 decisions per agent. Each seed has two held-out worlds, each evaluated for 12,000 ticks. Conditions: trained, matching initial weights, scripted reference. Evaluation weights and predictors are frozen; resources relocate left-right-left after each third of a life. This tests response to spatial changes with fixed weights, not online adaptation or a cue-meaning reversal.

Acceptance requires improved nutrition over initialization on every map, at least 80% of scripted nutrition, at least 90% low-starvation lives, and at least 80% of lives halving thorn contacts per adjacent opportunity against initialization with nonzero exposure. The small rooms and shared cue semantics limit generalization claims. The confirmation has a 1,800-second wall cap and a source/preregistration hash receipt; failed and partial results are retained.

Local evidence is under `runs/sequence-confirmation-20261005/`. Run the separate historical recount with `python experiments/audit_sequence.py runs/sequence-confirmation-20261005 --archive-only`. The eight-second pilot checks throughput and plumbing only; its short survival horizon cannot establish competence.

## Confirmation results

The 289,728-tick comparison completed in 862.616 seconds. Three independent training seeds produced six held-out worlds and twelve lives per condition. The independent recount verified archived source/preregistration hashes, training budgets, all eighteen frozen evaluations and trace totals. The original source also matched the workspace when initially audited. Preregistration SHA-256: `a05ddacb71809f3f5d9044670d77a527b8385c942d6e11b412070e1a577b4253`.

| Mean per held-out life | Trained | Matching initial weights | Scripted reference |
| --- | ---: | ---: | ---: |
| Nutrition restored | 172.643 | 173.629 | 163.917 |
| Zero-food ticks / 12,000 | 0 | 0 | 0 |
| Unconscious ticks | 405.667 | 1,576.333 | 1,430.583 |
| Observed injury | 407.667 | 584.313 | 594.333 |
| Ordinary-food eating actions | 13.750 | 9.250 | 4.917 |
| Harmful-fruit eating actions | 4.083 | 8.417 | 3.417 |
| Thorn contacts | 40.750 | 52.000 | 65.750 |
| Normalized short-term prediction MAE | .03164 | .08583 | n/a |

Training reduced harmful eating by 51.5%, observed injury by 30.2% and time unconscious by 74.3% relative to initialization in this comparison. These are descriptive paired outcomes from a small curriculum experiment, not statistical proof of architecture superiority. No matched one-step learner was trained at this larger budget. The predictor is diagnostic and did not select actions, so these gains do not establish planning through a world model.

**The preregistered gate failed.** All twelve trained lives stayed fed, as did both controls. Only two of six maps improved nutrition over initialization, although all exceeded 80% of scripted nutrition. Zero of twelve lives halved thorn contacts per adjacent opportunity. Reduced raw injury cannot be substituted for that exposure criterion. The available food and replenishment during spatial changes make starvation a weak discriminator here. Fullness saturation also places nutrition near a ceiling; future criteria should assess safe feeding and scarcity explicitly. The recorded verdict is retained.

Resources were restocked during the two relocation interventions; the run is not a demonstration of self-sustaining agriculture. Six maps share room geometry and cue semantics. Neither crop learning, real-time adaptation after a meaning reversal, language nor cooperative society was confirmed. Trial-trained copies were not deployed to the named interactive residents.

After the audit, a nutrition-accounting defect was corrected in both learner backends: decay clipped at zero fullness must not earn food relief. All recorded training and evaluation transitions in this run stayed above zero food, so its numbers and learned weights are unaffected. The original source remains in `source.zip`; historical archive recounts correctly report that the current source differs. New runs hash the shared corrected accounting helper too. Earlier trials with starvation retain a documented metric caveat rather than silently replacing their results.

## Interactive inspector

Launch `scripts/Start-Development.ps1` for a separate local development world at http://127.0.0.1:8790/ . Stop with `scripts/Stop-Development.ps1`; both brains and world save together. The stage argument applies to a new directory only; existing saves resume their original environment. Select Moss or Pip, then open **Learning record** to inspect rollout fill, policy version, preferences and recent measured consequences. Preferences are not calibrated success probabilities, and the UI does not invent inner dialogue.

The development launcher starts paused. Resume to continue actual online experience, or step to advance one tick. After the accounting fix, this world resumed at tick 874 with one sequence update and 90 buffered transitions; the previous one-step population resumed paused at tick 2469 with 617 updates. Both have pre-upgrade complete backups and preserved weights.

For a new later-stage world, use for example `scripts/Start-Development.ps1 -RunName farming-development -Stage farming -Port 8791`; stop it with `scripts/Stop-Development.ps1 -RunName farming-development`. This creates fresh independent learners. It does not transfer a trained curriculum checkpoint or reset an existing directory. These extra worlds have not been started by default.

The original main world and historical experimental population keep their existing saves. Trial-trained copies are not automatically imported into named interactive residents. Laya and fly-inspired experiment options remain supported.

## Engineering verification

117 tests passed in 18.929 seconds after the accounting correction. Delayed credit, sequence gradients, collection-policy stability, physical masks, independence, frozen parameters, exact mid-rollout continuation, both checkpoint formats, zero-fullness nutrition and runtime restore passed. Python compile, JavaScript syntax and diff checks passed. The browser displayed a completed update, measured learning records and successful save/load at tick 874. No desktop horizontal overflow. Screenshot: `runs/sequence-inspector.jpg`.
