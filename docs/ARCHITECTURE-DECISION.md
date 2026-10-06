# Selected learning architecture

Decision: October 4, 2026 (America/Los_Angeles). Research and implementation selection; no trained-policy result is claimed. This selects the next implementation from the earlier architecture proposals.

Implementation update, October 5: the independent recurrent PPO backend, private consequence predictor and development-world interface are now implemented separately. The older sizing and implementation-state discussion below records the decision snapshot. Current settings and evidence are in [SEQUENCE-LEARNING.md](SEQUENCE-LEARNING.md).

## Decision

Use **independent recurrent PPO**, built around the existing local observation encoder and configurable LSTM. Each resident owns its policy, local value function, optimizer, recurrent state, experience stream, normalization statistics and checkpoint. Start without pretrained weights. Keep the existing private visual impressions as inputs; add curiosity as a separately measured RND condition after basic consequence learning works.

PPO is the learning algorithm; the convolutional encoder and LSTM are the neural architecture. Independent here means separate learned parameters and experiences, including the critic. A shared process or GPU is acceptable; a shared population policy or omniscient critic is not this experiment.

The current network has 148,148 parameters at width 128, 32 patch channels, 81 other features and 51 actions (observation schema 6, action schema 5). Width is configurable. These are initial resource choices, not promises of automatic growth. The current view uses engineered local tile/resource channels plus lossy sketches, not end-to-end learning from raw screenshots. That sensory representation is an explicit prior.

## Permanent Laya track

User decision, October 4: retain **Laya permanently as a supported architecture option**, alongside the small learners and fly-inspired track. The existing local NPU adapter, checkpoint and experiment entry points remain in place. A poor result in one screen is evidence for improvement, not a reason to remove this track. Both `experiments/architecture_trial.py` and `experiments/foraging_trial.py` retain their Laya conditions.

Current Laya is a frozen pretrained decision model with private per-agent history in experiments. It is not a blank-slate learner, has no online weight updates here, and is not yet connected to the live population. A future live adapter should use the same local observation/action contract with separately stored history for every resident and an explicit decision budget. Sharing a frozen inference engine can save memory; it must not share private histories. Keep the supported track without starting an always-on service or assigning Laya to all residents by default.

## Why this choice

Our [hazard experiments](HAZARD-EXPERIMENTS.md) exercised a scripted policy. Its failure to change food preference says nothing about trained PPO, Dreamer or neural capacity: the running population has no weight-update path. The existing controller chooses an argmax from untrained logits; the benchmark only tests synthetic updates. The first missing mechanism is real experience collection and credit assignment.

| Candidate | Decision and tradeoff |
| --- | --- |
| Recurrent PPO | Selected. Fits the existing PyTorch model and local partial observations. Lowest integration burden among the reviewed general learners. On-policy training can need substantial experience; independent populations add nonstationarity. Neither limitation disappears through fast-forward alone. |
| DreamerV3 | Strongest later challenger for learning from fewer real interactions and anticipating consequences. It learns latent dynamics and an actor-critic through imagined trajectories. Separate world models/replay per resident add engineering and compute costs that have not been measured here. Author code uses JAX and lists Linux/Mac testing; do not assume native Windows compatibility. |
| MinGRU / GRU | Alternative recurrent cores, not replacements for a learning algorithm. Compare only if memory behavior or measured runtime warrants it. A faster untrained recurrence would still not learn. |
| Fly-inspired associative plasticity | Useful narrow cue/outcome challenger for rapid adaptation. Whole-connectome simulation is not selected for the first population. The prior research did not establish a ready lifelong learner from anatomical wiring. |
| Vendor and Chinese frameworks | Prior reviews remain background references. They provide training, simulation or planning components, not evidence that a different framework alone meets these objectives. No stack migration is needed for this first learner. |

The maintained [Craftax baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) supply PPO-RNN and exploration references for a related survival domain. They demonstrate relevant precedents, not success in Godhood Trials. [DreamerV3's author implementation](https://github.com/danijar/dreamerv3) documents its world-model learning and platform requirements. These pages were rechecked for this decision; nothing was downloaded or installed.

## Training contract

1. Collect real transitions from `World.observe(rid)`. Store previous action, observed consequence, reward components, policy version and recurrent boundary state. No `World.state()` input to policy or critic. Extra diagnostic world truth stays in evaluation only.
2. Sample categorical actions during training; use explicitly defined stochastic or deterministic evaluation. Record actual behavior log probabilities. Action masks enforce visible physical legality only; do not mask harmful food or thorns to manufacture avoidance.
3. Use contiguous recurrent rollouts, bootstrapped advantages, PPO clipping, entropy and value losses, gradient clipping and KL diagnostics. Do not shuffle individual timesteps or reuse old arbitrary experience as fresh on-policy data. Freeze collection policy during each rollout and update at a synchronization boundary.
4. Treat rollout cutoffs, fainting, world pause and terminal events distinctly. Fainting is not death or a new identity. Recompute or deliberately reset recurrent state after weight updates with a recorded policy; do not silently reuse stale hidden states. Resume checkpoints must include world, optimizer, RNG, normalizers, buffers and policy/recurrent versions.
5. Physiology supplies survival feedback: useful food/water intake, observed injury and prolonged unmet needs. Exclude automatic health recovery as a positive reward, so injury/recovery loops cannot farm healing rewards. Keep injury events observable at the health floor rather than relying only on clipped health deltas. Fix numerical reward weights and training budgets in the executable experiment manifest before running.
6. No unconditional revival reward. In the later social curriculum, evaluate sustained awake/safe outcomes and the helper's costs together. Any designed social reward is an explicit incentive, not evidence of spontaneous altruism. No automatic imports of one resident's experience or weights into another.

The live game must label each resident as scripted, untrained or learning. Learned mode must bypass scripted food seeking and automatic helping; the scripted policy remains a separate control. Preserve live saves before enabling it.

## Curiosity and memory

Add **Random Network Distillation (RND)** as a controlled optional module: each resident has a fixed random target encoder and a predictor trained only on its own observations. Prediction error supplies a bounded novelty bonus. This is an engineered curiosity signal, not a claim of subjective curiosity. The [original paper](https://arxiv.org/abs/1810.12894) supports the mechanism; it does not establish lifelong society formation.

First establish survival learning without intrinsic reward. Then compare identical budgets with/without RND, including a changing visual distractor. Novelty inputs should exclude observer journal, timestamps and global position. Tune and freeze its weighting on development seeds, then test on new seeds. Reject a variant that increases exploration by sacrificing food acquisition or repeatedly injuring residents.

Keep private sketches and recurrent state; test whether recall improves delayed decisions against a no-recall control. A sketch store alone is not learned episodic reasoning. Associating remembered actions/consequences, a learned dynamics model and skills are later additions, evaluated separately.

## First acceptance experiment

Use disposable learners, fresh development seeds and five held-out world seeds declared before training. Their experiences/checkpoints are not silently imported into named live residents. Pilot throughput first, then freeze both an interaction budget and wall-time cap; no learning-time estimate can be inferred from the 22-second scripted experiment.

| Stage | Proposed acceptance rule, fixed before the confirming run |
| --- | --- |
| Food consequences | With two edible choices equally available, at least 80% safe choices over the final 100 opportunities in at least 4/5 held-out runs; at least 25 percentage points above the frozen initialization. Confirm actual consumption so abstention cannot pass. |
| Mobile survival | On new resource layouts, at least 50% less injury per opportunity to enter a visible thorn tile than frozen/random controls, while obtaining at least 80% of the scripted control's useful nutrition. Report raw injuries, exposure counts, hunger and time unconscious; zero exposure is not an avoidance success. |
| Retention and adaptation | Change cue/outcome relationships in a separate labeled experimental world, then revisit the original task. Measure recovery speed and old-task retention. Passing fixed-color food selection alone establishes neither one-shot learning nor generalization to new objects. |
| Curiosity | At equal experience and compute budgets, improve discovery of useful resources on held-out layouts without worsening survival metrics. Report noisy-distractor failures. |
| Social behavior | Only after individual learning: learn local helping and subsequent safe behavior with the scripted helper disabled. Score outcomes over time, not raw revive counts. Physical carrying/dragging is currently absent and cannot be learned until added as an action. |

The first two rules are engineering gates, not statistical proof or results already obtained. Show each run and uncertainty, retain failures, and use separate new seeds after tuning. Construction, communication and reproduction need additional tasks and mechanics; basic survival cannot certify them.

## Growth and fallback

Keep the versioned brain/identity separation in [LEARNING-ARCHITECTURE.md](LEARNING-ARCHITECTURE.md). Continuing gradient updates does not guarantee continuing adaptability: [continual-backpropagation research](https://github.com/shibhansh/loss-of-plasticity) motivates measuring loss of plasticity. Trial its intervention in feedforward encoders only if the task-switch test exposes a problem, rather than resetting recurrent units speculatively.

If consequence learning fails, first inspect gradients, action/reward alignment, observations, recurrent boundaries and comparable data budgets. If a correct PPO learner is interaction-limited, compare a compact world-model learner against it on the same task and budgets. Do not call a small custom world model DreamerV3 or assume the published large-system result transfers.

CPU inference is the starting point; profile training on the existing CPU/CUDA installation. The NPU is not on the critical path. No paid service or new runtime is required for this first implementation. We can avoid fixed architectural ceilings, but no reviewed architecture guarantees emergent society, unlimited improvement or real-world assistant capabilities from this tile-world experience alone.
