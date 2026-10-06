# Proposed expandable learning architecture

The [architecture decision](ARCHITECTURE-DECISION.md) selects independent recurrent PPO for the next implementation, with RND curiosity introduced only after consequence learning passes. The other mechanisms below remain evaluated extensions, not a simultaneous implementation requirement.

Design proposal, 2026-10-04. No implementation or training claim. The goal is independent lifetime learning with replaceable capacity and learning mechanisms. The research evidence is in [ARCHITECTURE-RESEARCH.md](ARCHITECTURE-RESEARCH.md).

Current implementation is tracked in [STATUS.md](STATUS.md). The first independent visual-memory mechanism and sensory interface are now implemented; the full learner, growth/migration system and additional modules below remain proposals.

## Resident identity and brain state

A resident has a stable identity, body, lineage, and lifetime event history. Its brain has a separate architecture identifier, state-schema version, policy version, parameter manifest, and migration history. Changing architecture never silently creates a different resident or erases its history.

Each resident owns its weights, optimizer, short-term recurrent/plastic state, observation memory, training buffer, random state, and optional skill/world-model modules. Batch execution may share computation infrastructure and immutable definitions; learned resident parameters remain separate. Readout and memory must use only actually observed information.

The initial encoder/LSTM/PPO controller is one backend. GRU, MinGRU, plastic associative memory, world-model controllers, and connectome-based controllers can implement the same interaction contract later. Features listed below are optional future modules, not present capabilities.

## Proposed versioned BrainProtocol

| Operation | Contract |
| --- | --- |
| `act(observation, legal_actions, state)` | Produce an action, updated internal state, and diagnostic metadata tagged with the policy version. Diagnostics describe measured quantities; they do not invent thoughts. Action legality comes from the resident-visible physical contract. |
| `observe(transition)` | Record the resident's observation, action, consequence, reward components, terminal/lifecycle flags, and versions. Training data records provenance and policy age. |
| `learn(budget)` | Perform bounded updates compatible with that backend. For PPO, preserve on-policy ordering and recurrent sequence masks; older memory is not silently treated as fresh PPO data. |
| `snapshot()` / `restore(manifest)` | Save or restore all learned and dynamic states, RNG, observation/action schema, module versions, and pending rollout boundary with the world. Reject unsupported schemas explicitly. |
| `propose_growth(evidence, budget)` | Produce a declarative candidate architecture/memory migration and its expected resource cost. This is a proposal; improvement must be measured. |
| `migrate(candidate, checkpoint)` | Construct a candidate from this resident's own state, validate preservation where claimed, and record parameter/optimizer/internal-state migration rules. Preserve the previous complete checkpoint. |

An architecture manifest should identify the encoder, recurrent core, heads, optional modules, tensor shapes, state schemas, and allowed migrations. It should not serialize a backend by assuming every resident always has exactly 128 recurrent units.

## Modules that can grow

- **Own-experience memory:** retain observed landmarks, resources, action consequences, and encounters with provenance. A finite working cache can query a larger disk-backed store. Storage growth is configurable and measured; more stored events do not automatically produce useful recall.
- **Fast plastic memory:** optional sparse associative or learned plastic module for quick cue–outcome learning. Its traces and plastic weights belong to the resident and persist where the task design requires it.
- **Dynamics model and curiosity:** predict consequences from own experience; use capped intrinsic signals tied to useful information or progress. Test learnable change against random/noisy distractors. Report prediction calibration and survival tradeoffs.
- **Skills:** learned temporally extended action policies with initiation/termination conditions, selected by a trainable controller. A skill is accepted because behavior is reproducible on held-out tasks, not because it received a descriptive name.
- **Capacity:** grow an encoder or add a gated module before attempting recurrent-width changes. Validate any claimed function preservation. A recurrent migration must explicitly handle hidden state, optimizer tensors, and unfinished rollouts.

## Growth and continuation

First measure why learning stalled: insufficient experience, optimization faults, observation ambiguity, reward shortcuts, forgetting, or representational capacity. Add capacity only when the evidence supports that intervention. Introduce a candidate at a complete update/checkpoint boundary, then compare preservation and new-task learning against a fixed-capacity control. Record versions and retain rollback.

Named residents obtain new experience in their actual world. Disposable evaluation worlds can assess copied checkpoints with learning disabled or isolated experimental state; their observations and adapted parameters never flow back automatically. If evaluated adaptations are proposed for import, label the origin and obtain an explicit design decision, since this changes the lifetime-experience experiment.

Curiosity can choose destinations or interactions within the current world. A separate experimental curriculum generator can vary resource layout, delayed dependencies, climate, or social information. It should target diverse achievable challenges rather than merely maximizing difficulty. Transferring a solution between residents is a separate, explicit social-learning mechanism, not a hidden training optimization.

Reproduction initially keeps the existing fresh-child learning rule. Future evolution can mutate declared architecture or physiological genes; inherited learned weights require a different experiment and label. Do not equate this generational selection with growth of an individual's competence.

Memory and compute budgets are operational settings that can increase after measured milestones. Brain size, memory capacity, and available modules should be configurable rather than built into identity. Adding new physical actions or senses requires a versioned simulator/interface change. These provisions create an expandable research platform; successful growth still requires evidence.
