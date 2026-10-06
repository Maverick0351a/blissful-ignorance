# Navigation and foraging architecture screen

October 4, 2026, America/Los_Angeles. This extends the [food-only screen](ARCHITECTURE-TRIAL.md) into real movement, gathering, eating, drinking, injury and fainting in disposable worlds. The live valley remains untouched.

## Protocol

Script: `python experiments/foraging_trial.py --laya-adapter PATH/TO/EXISTING/laya_lite.py --output runs/NEW-FORAGING-TRIAL`. It uses only the existing Python/PyTorch installation and optional local NPU adapter. No download or external service is used. Raw rows, logs, source hashes and preregistration are under `runs/foraging-trial/` (ignored).

Three fixed seeds: 1701, 2702, 3703. Each run starts a fresh independent learner in a bounded 11-by-11 clearing, with eight ordinary berry plants, eight amber plants, ten thorn plants and three water tiles. Placement differs by seed. The resident begins with fullness 25, hydration 40 and health 100. Every run lasts 384 world ticks (96 simulated seconds). Physiology and inventory persist: there are no resets after choices or injuries.

The 41 input features derive only from the resident's local observation: needs, pain/health, carried food, orientation/contact, adjacent visible categories, nearest visible food offsets and the current tile. They are engineered sensory features, not raw pixels. A/B food cue meanings are swapped for the even seed; C denotes a distinct unfamiliar plant appearance. No safe/danger label or absolute location is supplied. Nine actions are available: four moves, rest, gather, eat A, eat B and drink. Masks enforce observed movement/inventory/fullness constraints but do not remove harmful choices. An unconscious resident has only a waiting action; the simulator still rejects actions until recovery.

Candidates: random, tabular Q values, 64-unit feedforward network, 32-unit GRU, fly-inspired sparse expansion (256 units with 16 active), and frozen Laya with three recent action/reward records. Neural candidates receive up to four recent sensory vectors. Laya uses a textual rendering of the same available sensory information, an explicit survival objective, and natural-language priors. The smaller candidates learn from reward and have no language pretraining.

The four experience-learning candidates also have paired frozen-learning controls with the same initial seed, network/expansion, legal-action rules and exploration. All nonrandom candidates use 15% exploration; frozen does not mean the world or input history is frozen. Tabular and fly controls start with zero values, so their tied choices match each other. Different capacities, representations and learning rules mean this is a configured-system comparison, not an isolated test of architecture.

Online updates are one-step Q-learning with discount 0.95. MLP/GRU use chosen-action Huber loss, Adam at 0.001, gradient clipping and bounded targets; tabular/fly use local reward-error updates. There is no target network, replay buffer or PPO update here. GRU state is recomputed from a four-observation window. These deliberately small pilots do not validate the selected full-world recurrent PPO design.

Reward combines fullness/water gained, a small gathering bonus, observed damage, failed actions, low food and unconscious time. Automatic healing receives no positive reward. Gathering reward is an explicit shaping choice; a learner may optimize it by hoarding, so meals and health are reported separately. The raw field `useful_nutrition` means fullness gained from either food type, including harmful fruit; it must not be read as safe nutrition.

## Observed results

Means per 384-tick life across three seeds. Safe/harmful meals are successful consumption events; damage is cumulative before healing. Frozen rows disable learning updates, not exploration.

| Candidate | Safe meals | Harmful meals | Fullness gained | Damage | Unconscious ticks | Unique tiles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1.67 | 3.00 | 73.89 | 110.67 | 213.67 | 42.00 |
| tabular | 3.00 | 0.33 | 75.10 | 89.33 | 81.00 | 65.67 |
| tabular-frozen | 2.33 | 0.33 | 62.33 | 94.67 | 137.33 | 46.00 |
| mlp | 3.00 | 0.00 | 75.00 | 58.67 | 0.00 | 56.67 |
| mlp-frozen | 0.00 | 1.67 | 20.00 | 62.67 | 64.33 | 20.67 |
| gru | 2.00 | 1.33 | 66.00 | 50.67 | 68.33 | 38.00 |
| gru-frozen | 2.00 | 0.00 | 50.00 | 24.00 | 0.00 | 16.33 |
| fly | 1.67 | 2.00 | 65.67 | 64.00 | 45.33 | 27.67 |
| fly-frozen | 2.33 | 0.33 | 62.33 | 94.67 | 137.33 | 46.00 |
| laya | 1.67 | 1.00 | 53.67 | 46.67 | 0.00 | 29.33 |

The learning MLP consumed three safe meals and no harmful meals in every seed, versus zero safe meals for its frozen controls. It did not faint, but cumulative damage was 48, 72 and 56: food acquisition improved without establishing dependable thorn avoidance.

The learning GRU consumed three safe meals in seeds 1701 and 3703, but zero safe and four harmful meals in seed 2702, where it fainted twice. Its frozen control also found safe food in the first and third seeds and incurred less average damage. A recurrent advantage is not established.

The fly-inspired candidate reduced mean damage and unconscious time against its frozen control, but safe meals decreased and harmful meals increased. It also failed the even-seed food mapping (four harmful meals, two faints). The earlier food-only result does not transfer into consistently safe foraging.

Laya consumed five safe and three harmful meals total, drank in one run and never fainted. It took 247.3 seconds for the three runs, with roughly 212.7 ms median per decision. This is more capable than a fixed label selection in the prior bandit probe, but does not establish weight learning, effective use of memory, or calibrated choices. There is no no-history Laya control in this pass. Different prompt, legal choices and task make the two experiments unsuitable for a direct numerical comparison.

All ten conditions completed: 30 lives, 11,520 world ticks, 265.4 seconds summed timed trial execution. Process startup and Laya initialization are excluded. This is three seeds per condition, not thirty independent training seeds.

### Next decision

Carry the feedforward learner into a longer, independently seeded foraging confirmation because it improved safe intake most consistently in these runs. Retain GRU and fly-inspired alternatives and paired frozen controls; this result does not overturn recurrent PPO as an untested broader learning candidate. Keep Laya as a pretrained comparison, not a claim of an independently growing brain.

Before selecting a live population controller, require repeated feeding/watering cycles, saved learned checkpoints, fixed-policy evaluation on unseen layouts, balanced cue mappings, and a hazard-encounter denominator. The current live population remains scripted.

## Interpretation boundaries

The seeds are development runs with learning enabled in the evaluation world, not held-out frozen-policy transfer. Paired controls can show effects of updates under this protocol, but three trajectories do not establish reliable superiority. Identical layouts do not imply identical experience once policies diverge.

Fullness falls only 3.072 points during a run. Starting at 25 means an agent can avoid the below-20-food threshold without eating. Thus low damage, staying awake or merely reaching the end cannot establish survival. Sustained foraging needs longer runs with repeated feeding opportunities. Resource regeneration starts at tick 600 and does not occur in these runs. Mortality is disabled, and automatic recovery remains active.

Damage is reported raw alongside exploration and food outcomes. It is not normalized to matched hazard encounters; increased exploration can increase injury exposure. The `invalid` counter includes forced waiting while unconscious, so it is not purely a bad-action statistic. Laya receives recent action indices rather than fully described past commands; that serialization is another prompt limitation to diagnose before generalizing its result.

No social partners, construction, reproduction, learned communication or curiosity mechanism are tested. Learned weights from these disposable trials are not imported into live residents. The experiment can be reproduced but does not currently export neural checkpoints.

## Validation

Source hashes match the preregistered files. Arena counts and 41-feature/nine-action shapes were checked. Recorded fullness matches initial fullness plus actual intake minus all elapsed metabolism, confirming that needs were not reset during a life. Health remains within the mortality-off range. Python compilation passed. No simulation/game source or live save was changed.
