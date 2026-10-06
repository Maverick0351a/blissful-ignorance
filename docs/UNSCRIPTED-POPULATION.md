# Independent playable population

Latest update: Laya has been added as a ninth, distinct local NPU resident. The eight PPO lives below remain preserved. Her pretrained weights are frozen; her action/consequence history persists separately. See [continuing observations](POPULATION-WATCH.md). The remainder documents the original all-eight migration.

October 5, 2026. The user specified that the population must not be scripted. The primary `server.py` entry point and Windows game launcher now run one independent recurrent PPO brain for each of the eight autonomous residents. No survival script, assigned job, route planner, farming recipe controller or emergency scripted rescue chooses their actions. Each brain starts from its own random initialization and learns from its local observations and experienced food, hunger and injury consequences.

The shared action vocabulary, bodily rules, food rewards and private physical-feasibility masks are still designed mechanics. They define what an action means and which locally known actions are physically possible; they do not choose a survival strategy. Harmful-but-possible choices remain available. Curiosity rewards remain off pending a controlled exploration comparison. This change does not establish reliable survival, learned farming, cooperation or language.

## World and identity preservation

At first primary launch, an existing `runs/autosave.json` (or older manual `world.json`) is read without resetting the world. Its resident bodies, inventories, positions, resources, structures, sounds, memories, world tick and RNG are retained. Original JSON saves are copied into a timestamped `runs/legacy-scripted-*` directory. `controller-transition.json` records the source hash, original tick, backup path and fresh brain identities. Scripted histories are not converted into fabricated learned skills or training examples.

The actual saved valley migrated at tick **37,342**. Its full serialized world matched the original before any neural action. Original autosave SHA-256: `fc3857ef9b9dddb67de6b27cefb9b5d1201c2915543c436d8ccd35c98f69f23f`. All eight new brains had zero training updates at handover. The previous mixed learning worlds were gracefully stopped with their checkpoints preserved.

Existing partial neural checkpoints can add a brain for every remaining resident while retaining the original learners' weights, optimizers, recurrent state, rollout, RNG and counters. The original checkpoint is backed up before promotion. New experimental rooms contain only their two learners and traveler, rather than six passive placeholder residents.

## Persistence and enforcement

Primary launch resumes `runs/population.pt` preferentially and starts paused. Autosaves, manual saves, pre-load and pre-fast-forward snapshots contain the entire world and every brain together. A shutdown receipt records the completed checkpoint size, world tick and brain identities. Older manual JSON saves remain loadable by attaching fresh independent brains; loading them cannot re-enable scripted survival. Public read-only demos also reject direct load calls.

The live runtime rejects a tick when any autonomous resident is missing a brain. At the physics boundary, unscripted stepping requires an explicit action for every autonomous resident and rejects missing actions before invoking any fallback. Between neural decisions, wait actions represent the fixed decision cadence; the traveler rests when the human queues no action. Decision cadence and body recovery are simulation mechanics, not scripted NPC strategies.

The old scripted policy remains available for offline tests and matched reference experiments. It is not selected by the primary playable entry point. Missing PyTorch or unreadable saved state produces an error rather than silently substituting scripted agents or resetting the valley.

## Verification

**129 tests passed in 20.941 seconds.** New checks verify all eight networks independently update from experience, exact multi-brain continuation, preservation of prior pair checkpoints in both supported backends, unchanged migrated world and original save bytes, zero calls to the scripted policy, rejection of missing brains, and read-only learned hosting. JavaScript syntax, PowerShell launcher parsing and diff checks passed.

Live migration verification confirmed eight recurrent PPO brains, zero scripted residents, and equality with the original saved world. These engineering checks establish controller isolation and persistence; they do not establish better survival. Previous scarcity failures remain valid historical results, with their source archives preserved. Historical source-hash audits should use `--archive-only` after this source change.

Browser verification stepped the migrated world seven ticks, inspected Reed's independent action preferences and actual consequence record, and saved all brains. The real Windows stop/start path resumed tick 37,349 with two decisions and one completed transition per brain, matching the manual checkpoint. The world was left paused after that verification; later play and Laya's arrival are recorded separately. The two-resident scarcity world could resume separately without changing its saved brains. The two older mixed worlds were stopped with their original checkpoints available for analysis. Document and viewport width both measured 1,503 pixels; screenshot: `runs/unscripted-population.png`.

Open http://127.0.0.1:8788/ and select any resident's **Learning record**. The footer reports the actual controller population. Resume or step to let them experience the world.
