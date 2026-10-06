# GT-01: matched value-gradient interference diagnostic

October 6, 2026 (America/Los_Angeles). **Comparison complete and audited;
diagnostic screen failed.**

The [feeding-credit diagnostic](FEEDING-CREDIT.md) found positive credit for
every ordinary meal, but uneven policy improvement. This follow-up tests one
specific cause: whether value-estimation updates interfere with action learning
through their shared sensory encoder and recurrent network. A larger value
gradient alone was not evidence of interference.

## Result and interpretation

The intervention **failed the predeclared diagnostic screen**. Its mean extra
eating-probability gain was **−0.00504 percentage point per matched update**,
below the required +0.1 point. Four brains had positive mean effects, but c2's
larger negative effect outweighed them. No alternative trained population was
produced, and no live model change follows.

The table reports gains from each update's own matching pre-update state,
averaged across sixteen boundaries. Every detached copy starts again at the
original native state; the columns must not be added as a projected learning
curve or mistaken for a sequentially trained detached policy.

| Brain | Standard eating gain (pp/update) | Detached eating gain (pp/update) | Extra gain (pp/update) | Meal choices reinforced: standard → detached |
|---|---:|---:|---:|---:|
| c0 | +0.05453 | +0.05711 | +0.00259 | 27/29 → 28/29 |
| c1 | +0.16759 | +0.16234 | −0.00525 | 35/36 → 36/36 |
| c2 | +0.69055 | +0.62561 | −0.06494 | 43/49 → 39/49 |
| c3 | +0.01950 | +0.04235 | +0.02285 | 18/30 → 27/30 |
| c4 | +0.45230 | +0.45908 | +0.00678 | 38/38 → 38/38 |
| c5 | +0.04130 | +0.04900 | +0.00770 | 23/32 → 32/32 |

The detached update reinforced **200/214 recorded meal decisions (93.46%)**,
versus **184/214 (85.98%)** under standard PPO. That higher count did not imply
stronger overall improvement: the magnitude of the gains also matters. Fixed
carried-food probes and the actual recorded meal states are different inputs,
so their effects are kept separate.

c3 and c5 benefited from this immediate intervention. c2, previously the
strongest improver, lost eating and gathering gains. Mean paired gathering
change was **−0.00349 pp**, within the screen's allowed −0.1 pp margin.
The other three checks passed; the primary absolute benefit check failed.

The value predictor's mean error against the same recorded targets rose from
**0.41430 to 0.43351 (4.64% worse)**. Those targets are training estimates, not
ground-truth long-term success. Both conditions ran all four epochs at every
boundary. Blocking current value gradients left historical optimizer momentum
intact and could change global clipping, so this experiment does not isolate
each of those contributors individually or settle a long-term architecture
comparison.

**Decision:** retain the existing learning rule. The effect is heterogeneous
and does not support a blanket removal of value gradients. A memory defect or
missing food-reward signal is still unestablished; repeated culling would not
address the mechanism measured here.

**Next bounded recommendation:** measure a longer, checkpointed practice curve
using copies of the preserved trained category-PPO brains, with unchanged
rewards, memory, actions and life length. Compare frozen feeding performance
after 16 versus 64 additional practice lives on fresh evaluation maps, with a
frozen starting-policy control. This directly tests whether the weak responders
need more useful experience before attempting another architecture change.
It is a training-budget comparison, not an equal-experience model ranking.
Freeze the exact interaction/time budgets and advancement criteria before
execution. That follow-up was subsequently completed in the
[practice-amount comparison](PRACTICE-AMOUNT.md): longer practice improved
actual frozen feeding performance and passed its development screen, with the
existing learning rule retained. No live weights were imported.

## Evidence and completed checks

Evidence is local in `runs/value-interference-20261006/`. Protocol SHA-256:
`dda0f274c81156768917a5351575f17c1c60fe161770f0f1a5e70a3fae9b551e`.

- Completion receipt (local evidence; not bundled):
  **137.54 seconds**, 24,576 original ticks, 12,288 sampled native decisions,
  96 reproduced native updates and 96 independent experimental updates.
- Separate audit (local evidence; not bundled): **114.96
  seconds**. An independently written detached recurrent graph reproduced every
  experimental full state; all native states matched the archived continuation.
  Probabilities, value errors, isolation and the failed screen were recounted.
- Interactive paired effects (local evidence; not bundled),
  summary (local evidence; not bundled) and
  every comparison (local evidence; not bundled) preserve
  beneficial and harmful cases. The last six independent shadows are retained
  only as evidence, not as a sequentially trained alternative population.
- **212 regression tests passed in 102.32 seconds**, including the optional
  PettingZoo API test. The native update source remained unchanged.

Peak working set was about **746.81 MiB**, and peak private commit **1.82 GiB**.
Evidence before the derived report and verification files was **71.84 MiB**,
below 512 MiB. The comparison ran alongside regression tests; the timings are
not standalone model-performance measurements. No budget extension or retry
was needed.

Read-only main-server health checks retained PID **69328**, eight
`recurrent-ppo` IDs and distinct `laya-npu`, with public sharing off. Its own
activity advanced from tick **80,996** (running) to **81,087** (automatically
paused, `manualPause=false`). No live browser heartbeat, reset, policy import,
save modification or restart was performed. Health does not expose controller
errors; no new error-log inspection or behavior observation is claimed.

## Protocol fixed before measurement

Use the same six experimental category-PPO brains and all **96 original update
boundaries** from the audited continuation replay. The native policy must again
reproduce 24,576 recorded world ticks, 12,288 sampled decisions, every original
update and its complete final state. No newly generated training lives or live
resident data enter this experiment.

At each boundary, clone the complete original model, predictor, optimizers,
RNGs, recurrent state, pending experience and private history. Compare:

1. **No update:** the unchanged pre-update probabilities.
2. **Standard:** the existing native PPO update, required to reproduce the
   original recorded result exactly.
3. **Detached value input:** the same native PPO update, with the value head's
   input detached from gradient propagation. It produces the same values at
   the start, and the value head still trains, but new value-loss gradients
   cannot change the shared CNN, feature encoder or recurrent network.

The detached copy receives **one update only**, then is retired. Every next
comparison starts again from that boundary's original native checkpoint. Never
chain detached copies while feeding them observations from a diverged baseline.
Consequently this is an immediate update intervention, not a newly trained
population, retention test, architecture competition or survival result.

Rewards, value targets, action masks, all ten tones, observations, optimizer
momentum, learning rate, entropy, global gradient clipping and KL/epoch rules
remain unchanged. Prior momentum still contains the native learner's history.
Changing gradient flow can alter the common clipping scale and later optimizer
steps; these are part of the intervention, not separately isolated causes.

## Measurements and development screen

The primary outcome is the paired difference in ordinary-eating probability
after the two updates, on the same four carried-food observations used by the
previous diagnostic. Average equally over sixteen updates per brain and then
over six brains. These probes are never training examples, but they have already
been inspected in the earlier diagnostic and are not a blind confirmation set.

Report each brain, each update, all recorded meal-choice probabilities,
adjacent-food gathering probes, value-target mean squared error, epochs and KL
diagnostics. c3 and c5 were previously weak responders; report them without
excluding the other four. Probability changes are not feeding-success rates.

The intervention earns a **promising diagnostic** label only if all hold:

- Mean paired carried-food eating improvement is at least **0.1 percentage
  point** per update (0.001 probability).
- The mean paired improvement is positive by more than 0.000001 probability
  in at least **four of six brains**.
- At least as many recorded ordinary meal decisions gain probability by more
  than 0.000001 as under standard PPO.
- Mean adjacent-food gathering probability is not reduced by more than
  **0.1 percentage point** relative to standard PPO.

Preserve all failed checks. This screen cannot close GT-01, establish a memory
defect or authorize a live deployment. A favorable result would require a
separate online learning comparison with fresh behavioral evaluation. This is
one historical six-brain pool, with correlated updates and no significance claim.

## Verification and resource budget

Before the run, archive the protocol, sources, parent evidence, starting/final
native checkpoints and trace. The runner has a **600-second wall cap and
512 MiB evidence cap**; the separate replay audit has a 600-second cap.
Budget or correctness failure preserves partial files without an automatic
retry or increased budget. No new behavior-selection smoke run is needed.

The runner uses a temporary value-head input hook, removed even on exception.
The separate auditor uses an explicit recurrent unroll with detached value
inputs, never the hook. It must reproduce every detached full state, measured
probability and value error, and independently recount the screening decision.
Trace replay and hashing infrastructure are shared and disclosed.

Five focused tests verify unchanged forward values and actor gradients,
blocked shared value gradients, still-trainable value-head weights, exact
hook-versus-explicit update equality with populated optimizer state, copy
isolation, exception cleanup, the native identity control and screening rules.
The full regression suite follows. Live residents and Laya remain preserved;
no model download, public release, reward change or mortality change is included.
