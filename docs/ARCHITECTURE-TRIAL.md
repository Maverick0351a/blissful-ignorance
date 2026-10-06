# First agent architecture screen

Executed October 4, 2026 (America/Los_Angeles). Six candidates ran locally in isolated worker processes. No live-world saves, residents, server settings or policies were changed. No packages, weights or other files were downloaded.

## Candidates actually run

| Candidate | Mechanism | What adapts |
| --- | --- | --- |
| Random | Uniform two-action control | Nothing |
| Tabular | Exponentially updated expected reward for each food cue | Two values; no neural network |
| Feedforward | 32-unit hidden-layer MLP over the last eight outcomes | 1,122 neural parameters, chosen-action reward prediction |
| Recurrent | 24-unit GRU over the same history | 2,210 neural parameters, chosen-action reward prediction |
| Fly-inspired | 256 sparse sensory units, 16 active per cue, reward-error-modulated output synapses | 256 output weights; fixed sensory expansion |
| Laya | Existing local typed-decisions OpenVINO model on the Intel NPU | Frozen weights; recent outcomes supplied as text context |

The fly candidate is an engineered mushroom-body-inspired sparse associative learner. It is not FlyWire anatomy, a whole-brain simulation, a spiking model or a validated biological learning circuit. This two-cue task does not use delayed eligibility traces.

The small neural candidates use online regression to observed reward with epsilon-greedy action selection. They are not PPO, not the existing full 147K-parameter resident model, and not trained general-world controllers. The GRU recomputes state over its eight-event window; it does not carry an unlimited lifetime hidden state. Every seed starts fresh weights/state. Laya shares immutable frozen weights across sequential trials but receives a separate history per trial.

## Protocol and provenance

Run `python experiments/architecture_trial.py --laya-adapter PATH/TO/EXISTING/laya_lite.py --output runs/NEW-TRIAL`. Without an explicit local adapter the runner records Laya as blocked. Existing result directories are not reused by the orchestrator.

Seeds: 1031, 2053, 4099. Each candidate makes 80 choices per seed: 40 with one neutral food label safe, then 40 after reversing the label-to-food mapping. All three selected seeds are odd, so B is initially safe and A is safe after reversal. This is an initial-label confound, explicitly diagnosed below; it must be counterbalanced in a confirming experiment.

The environment executes the real `World.apply_action(... eat ...)` effect. At every choice, inventory is replenished and health/fullness are reset to 100/50. Only the learning state/history persists. This deliberate bandit probe removes navigation, resource scarcity, waiting, fainting and natural feeding schedules. It is not a continuing life simulation.

Feedback is useful fullness gain divided by 25 minus health loss divided by 10: safe food yields +1; harmful food yields -1.52. The learner never receives the safe label. The neural candidates receive action/outcome history with padding indicators; Laya receives the same bounded history serialized as choice/outcome records plus an explicit objective. Tabular/fly candidates retain their own outcome estimates. Every nonrandom candidate has 15% uniform exploration; ties are randomized. Thus even a correct learner still makes exploratory mistakes.

No hyperparameter search occurred. Neural learning rate is 0.01 with Adam and gradient clipping; tabular/fly step size is 0.3. Capacities, algorithms and pretrained knowledge differ, so this screens complete configured candidates rather than isolating architectural superiority. Preregistration and source SHA-256 values were written before execution to `runs/architecture-trial/preregistration.json`. Raw per-choice outcomes, timing, logs and status are preserved beside it. All six workers completed.

## Observed food-choice results

Percent choosing safe food, pooled across three seeds. Each late column covers 60 choices; early reversal covers 30. These are exploratory small-sample measurements, not confirmation of the architecture-decision acceptance gates.

| Candidate | Initial phase, last 20 choices | First 10 after reversal | Reversed phase, last 20 choices | Median decision latency |
| --- | ---: | ---: | ---: | ---: |
| Random | 50.0% | 46.7% | 51.7% | <0.001 ms |
| Tabular | 93.3% | 46.7% | 95.0% | 0.001 ms |
| Feedforward | 93.3% | 23.3% | 93.3% | 0.058 ms |
| Recurrent GRU | 93.3% | 10.0% | 95.0% | 0.213 ms |
| Fly-inspired | 93.3% | 46.7% | 95.0% | 0.007 ms |
| Frozen Laya + context | 8.3% | 76.7% | 93.3% | 213.107 ms |

Latency is the median of three seed medians for action selection only. It excludes neural updates, environment stepping and rendering; Laya includes tokenization and its NPU request. The maximum of seed p95 decision latencies was 219.072 ms for Laya, 0.265 ms for GRU and 0.086 ms for MLP. These are different hardware paths and model sizes, not a CPU/NPU hardware comparison.

The fly and tabular candidates produced identical recorded action trajectories across these seeds. Sparse expansion supplies no demonstrated advantage here. Both changed their estimates faster immediately after reversal than the neural regressors. The GRU eventually recovered, but its poor first ten reversal choices are an important negative result; recurrence did not improve this simple task.

Laya's apparent reversal improvement does not demonstrate adaptation. It largely preferred A throughout, which became safe at reversal. Its initial late-phase seed rates were 10%, 10%, 5%; reversed late-phase rates were 100%, 95%, 85%. The prescribed exploration explains many departures from its preferred action.

## Laya follow-up: label preference

After seeing that pattern, a separate eight-case diagnostic was run using `experiments/laya_choice_diagnostic.py --adapter PATH/TO/EXISTING/laya_lite.py --output runs/architecture-trial/laya-diagnostic.json`.

The histories alternated A/B observations and independently varied which label had positive outcomes, candidate order (AB/BA), and whether outcomes were numbers or words (benefit/harm). **Laya selected A in all eight cases**, including the four where B had better outcomes. Reversing candidate order did not fix it. Scores changed with evidence, but never enough to change the selected action. This is a post-hoc prompt diagnostic, not a predeclared confirmation or a tuned replacement result.

This establishes a problem with the tested local checkpoint, adapter and prompt combination. It does not establish that all Laya checkpoints or a fine-tuned Laya policy cannot perform the task. No current PyTorch-versus-NPU numerical parity comparison or alternative-checkpoint evaluation was run, so checkpoint quality and adapter effects cannot be separated here. The context-fit check rejected truncation, and returned scores were finite.

## Local memory and speed implications

Peak worker working sets reported by Windows were 22.2 MiB random, 22.3 MiB tabular, 583.7 MiB MLP, 584.4 MiB GRU, 488.0 MiB fly, and 2,541.5 MiB Laya. These include Python/library/driver overhead and initialization; they are not incremental per-resident model memory, private committed bytes, or total NPU/GPU memory. Small models shared within one process would not each incur the entire runtime overhead.

Laya adapter initialization/compilation took 2.3 seconds in this run. It was used sequentially and did not train weights. Its measured median corresponds to approximately 4.7 serial decisions/second. Eight inhabitants each deciding four times/second need 32 decisions/second, before fast-forward. That arithmetic indicates this measured serial configuration cannot sustain that target; batching, fewer decisions and other backends remain unmeasured possibilities. It does not predict the speed or memory of eight independently trained Laya copies.

## Decision after the screen

Keep the general recurrent learner as the planned full-world architecture, but do not claim this tiny GRU screen validates PPO. Carry the fly-inspired associative learner and tabular control into a navigation/consequence test; rapid local cue learning may be useful as a separate memory mechanism. Preserve feedforward as an inexpensive control because this experiment shows no recurrent advantage.

Keep Laya available as an experimental comparator, not an automatic replacement for resident brains. Its tested configuration showed a label preference and substantially higher latency. A next Laya experiment should diagnose checkpoint/runtime parity and use counterbalanced prompts before investing in independent online training.

Next meaningful test: independent residents choose movement, gathering and food actions from local senses, with resources at unfamiliar positions and no physiological reset. Counterbalance cue mappings, measure useful intake and injury together, include frozen-learning controls, and test retention after reversal. No candidate here has demonstrated learned construction, curiosity, social rescue, communication or open-ended intelligence. The live population remains scripted.

Validation: Python compilation succeeded; all 1,440 recorded outcomes matched actual safe/harmful food rewards; source hashes matched the executed files; six status records report completion. Raw results remain separate from live world files.
