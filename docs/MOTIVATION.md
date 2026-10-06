# Hunger and curiosity

Implemented October 4, 2026, local v 0.5. These are engineered internal signals and behaviors, not evidence of subjective feelings.

**Current controller status:** eight independent recurrent PPO learners and
distinct local NPU Laya choose the main valley's resident actions. The
physiology below still supplies bodily feedback. The scripted route cache
and Q-learning curiosity experiment are historical v0.5 implementations,
not the live policy. See [the population record](POPULATION-WATCH.md) and
[current sequence-learning design](SEQUENCE-LEARNING.md).

## Hunger in the running world

| Fullness | Internal warning | Consequence |
| --- | --- | --- |
| 60–100 | Satiated | Normal rest recovery and movement |
| Below 60 | Hungry | Increasing discomfort; reduced rest recovery |
| Below 20 | Weak | Increased movement cost; at most one move every 2 ticks |
| Zero | Starving | No energy from rest; one move every 4 ticks; starvation strain rises |

Discomfort is clamp((60-fullness)/60, 0, 1)×100. Rest restores .25×(1-discomfort/100) energy. Movement costs .035×(1+2×discomfort/100) energy. Normal movement accepts one move per tick. Fullness still declines .008 per tick. Warnings are available through the agent's own body observation, not only the observer UI; stage changes are journaled without an every-tick log flood.

After 240 conscious ticks at zero fullness (one minute at 1×), the creature faints. Hunger fainting lasts at least 80 ticks (20 seconds at 1×), and waking requires health≥30. Hunger wake gives a minimum 10 energy and resets the starvation counter, allowing another 240 conscious ticks to find food. It gives no food or immediate health. Passive health regeneration continues at the ordinary awake rate during hunger fainting. Injury-only faint recovery is retained. Mortality stays off; hunger does not directly subtract health or add injury pain.

Ordinary food restores up to 25 fullness and immediately reduces discomfort/reset starvation strain. An adjacent helper can still revive a creature for 10 energy. If the recipient has fullness below 20 and the helper has ordinary food, help-up also consumes one food and restores 25 fullness; the button describes this cost. Without food, revival still works but hunger remains.

Body observations expose hunger_discomfort, hunger_stage and starvation_strain in addition to existing needs, health, pain and unconscious state. The UI shows Hunger distress and a stage warning. Observer state derives these fields from the same physiology. Saves add starvation_ticks, hunger_faint_until and next_move_tick with defaults for old saves.

## Historical scripted curiosity baseline

The original v0.5 live residents were scripted. When immediate scripted needs permitted exploring, each resident preferred its least-tried available direction in its current lossy visual scene. Counts were private to that resident and persisted in saves. Closed doors remained usable exploration routes. Unconscious residents did not explore. The baseline also stopped resting indefinitely at critically low fullness simply because energy was low.

The per-resident route cache retains up to 2, 048 scene fingerprints, evicting the oldest entry when full. This is a bounded implementation memory store, not demonstrated unlimited growth or a learned curiosity policy. Other residents' route counts are not sensory inputs.

## Historical Q-learning curiosity and hunger experiment

`agents/rewards.py` provides a separate private novelty memory to each experimental learner. It hashes only its visible local terrain, resource kinds and structure state. Absolute coordinates, identities, body fluctuations, exact quantities and hidden-tile data are excluded. Repeating the identical view earns zero. A changed view earns at most 0.02, decreasing as 0.02/sqrt(visits) when revisited. Failed actions and unconscious transitions cannot earn the bonus. Remembering the initial scene earns no transition reward.

`mobile_farming_trial.py` protocol 2 now trains on:

    useful nutrition /25 - summed hunger discomfort /48000 + scene novelty

A 120-tick decision at maximum discomfort costs 0.25; a new scene earns at most 0.02. Hunger cost includes unconscious ticks, so fainting cannot escape the penalty. Eating earns nutrition and reduces subsequent hunger cost. No bonus directly instructs farming. The policy also receives its own energy, discomfort, starvation strain and unconscious state. Curiosity state is independent and saved separately beside each experimental Q checkpoint.

Curiosity counts continue updating during evaluation, while Q updates are disabled. Trained agents start familiar with their training scenes and untrained controls do not. Therefore curiosity/total-reward comparisons include different familiarity histories; use separate nutrition, deprivation and exploration outcomes when evaluating learning.

That update verified the feedback path and private memories. It did **not** establish that learned curiosity improved survival or deploy neural controllers into the then-scripted live population. A full protocol-2 training comparison was not part of that update. Later neural deployment is recorded separately in [the controller transition](UNSCRIPTED-POPULATION.md). Previous experiment reports describe their original rules; source hashes and commits preserve that distinction.

The optional neural scaffold has 148, 148 parameters at width 128, 32 patch channels, 81 other features and 51 actions. Overall/observation schema 6, action schema 5. Older neural weights need explicit migration/retraining. World saves remain compatible.

## Verification

77 tests pass, including hunger boundaries, rest/movement effects, faint/recovery/revival, eating relief, old saves and deterministic continuation, independent exploration memory, opening closed exits, curiosity decay, no idle/failed/unconscious curiosity reward, hidden/body novelty exclusion, and integrated experiment reward accounting.

Browser fixture: fullness 0/distress 100/strain 92%; eating restored fullness 25, reduced distress to 58 and cleared the starvation warning. Helping an unconscious hungry neighbor used one food and 10 energy. Live browser shows v 0.5 and the new body warning; no horizontal overflow at the observed 1518px viewport. Screenshot: `runs/hunger-curiosity.jpg`.

Before restart, `runs/before-hunger-curiosity-20261004.json` backed up tick 10, 598 with nine characters and three structures. Every prior saved field was checked for preservation through migration. A subagent implemented/tested core physiology, then reviewed the motivation integration; the parent checked behavior, reward accounting, UI and the full suite.
