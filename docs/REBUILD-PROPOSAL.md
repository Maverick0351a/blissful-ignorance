# Rebuild recommendation

October 5, 2026. The user approved starting with the environment adapter and
learning-curve benchmark. Preserve the live residents, Laya, saved world and current experiment
records. Existing worlds already run headlessly; this proposal standardizes
and improves that infrastructure rather than starting the game over.

Implementation progress: the [parallel adapter](PARALLEL-ENVIRONMENT.md) now
passes official PettingZoo compatibility and deterministic-seeding checks,
with exact existing-world/learner parity. The [learning-curve report](LEARNING-CURVE.md)
records its protocol, pilot audit and full-run outcome. Later core/action-head,
world-model and communication changes remain proposals.

The approved [memory-continuity diagnostic](MEMORY-CONTINUITY.md) is also
complete. Reconstructing private working context improved average deprivation
but failed the three-seed consistency criterion and did not establish learned
recall or dependable feeding. The live baseline is unchanged. The authorized
[short/long-life comparison](LIFE-LENGTH.md) is also complete: at matched
16,384-decision budgets, long training increased deprivation in all three
seeds (39.90% versus 23.69% overall), with only 4/12 low-deprivation lives
in either arm. Both competence gates failed. Fresh body and resource supply
differ as part of that reset treatment; this does not establish an intrinsic
disadvantage of longer lives.

The latest traces make learned action factorization a concrete next candidate:
both trained arms spent about 82% of conscious decisions emitting tones in
isolated arenas, and random feasible actions spent 69%. A learned verb then
argument choice would retain all ten sounds and primitive physics while
changing the action distribution. This is a hypothesis to test, not a
demonstrated remedy or an instruction to script feeding. Curiosity remains a
separate proposed comparison; neither change has been applied to live agents.

## First objective

Four independent learners reliably feed themselves on unfamiliar layouts,
adapt when resources move, and retain earlier competence. Measure each life,
not only the population mean. Compare trained policies with their starting
weights, random behavior, and a feasibility oracle kept out of training.

The current 2,048- and 6,144-decision screens are small comparisons. Their
failures do not establish an architecture's ultimate capacity. A rebuild
needs learning curves across predetermined larger budgets and validation
maps separate from final test maps, before selecting a new brain family.

## What to preserve and replace

| Component | Recommendation |
|---|---|
| Pixel world and browser | Preserve rendering, controls, construction, farming, hazards and bodily consequences. |
| Individual perception | Preserve local occluded vision, hearing, touch and private compressed memories. No global state enters a resident policy. |
| Environment boundary | Add a tested PettingZoo Parallel API adapter; retain the simulator as the source of physical outcomes. |
| Learner | Begin with one tested recurrent actor-critic baseline; compare a compact CfC core with the LSTM under matched budgets. |
| Actions | Trial a learned conditional policy over verb then arguments. Preserve primitive physics and all ten tones. |
| Experience | Separate recent memory, durable consequence memory and replay sampling. Validate the chosen retention scheme against survival, not just memory counts. |
| Training | Batch simulation/encoding where measured useful, keep parameters and experience streams independent, and measure throughput before increasing population size. |
| Evaluation | Keep reproducible checkpoints, frozen evaluations, source snapshots, conservation audits, negative controls and per-agent results. |

PettingZoo supports simultaneous actions and separate per-agent spaces. Its
spaces and possible-agent list are fixed within a run. Reproduction therefore
needs configured identity slots and a versioned migration path for larger
populations; the adapter does not make capacity unlimited. Its optional global
state interface must stay outside private learner observations.
[Primary API](https://pettingzoo.farama.org/api/parallel/).

CfC is a candidate, not a demonstrated upgrade. The minimal core reviewed for
80 inputs and 64 state units has a derived count of 37,120 trainable parameters,
close to the existing LSTM core's 37,376. Preserve the sensory encoder and
readouts in that comparison. See [source review and constraints](SMALL-AGENT-SWARMS-RESEARCH.md).

## Development sequence

1. Establish a stronger baseline with predetermined learning budgets, a fixed
   validation protocol and a separate final test set. Accessible food precedes
   distant food, hazards and scarcity. Curriculum changes environments, not
   supplied actions. Evaluate retention when switching tasks.
2. Test a private predictive model of action consequences and curiosity based
   on measurable learning progress. Include unlearnable/noisy distractors and
   stable familiar resources as controls. Prediction error alone is not enough.
3. After prediction is useful, test short imagined trajectories for decision
   learning. DreamerV3 is a scientific/software reference for this approach,
   not a drop-in tiny model or proven Windows solution. The official MIT code
   uses JAX and documents Linux/Mac testing. Porting or adopting it would be a
   separately reviewed effort. [Author implementation](https://github.com/danijar/dreamerv3).
4. Introduce interactions where another agent's information can help. Test
   assistance and tone usefulness against shuffled and silent hearing. Private
   data and gradients remain private; experienced social consequences can be
   observed through the world.
5. Separate lifetime learning from evolutionary inheritance. Specify which
   traits or weights children inherit, and test local reproduction against
   evaluation-only population optimization. Laya keeps her distinct model and
   history throughout.

Larger networks and new abilities should be versioned, measurable extensions.
Hardware always imposes a budget; extensible capacity is different from a
guarantee of open-ended intelligence. Usefulness on real computer tasks needs
its own isolated transfer tests and permissions, rather than being inferred
from successful simulated survival.

The first approved changes, the environment adapter and learning-curve
benchmark, are complete. Their evidence and the follow-up diagnostics support
testing the next candidate in separate worlds while preserving the live
population throughout development.
