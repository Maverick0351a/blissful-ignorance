# Hazard and revival experiments

Run October 4, 2026, 21:49:47–21:50:09 America/Los_Angeles. The current build produces repeatable injury, fainting and rescue consequences. Its scripted residents do not learn food avoidance. Helping someone up on thorns can create a repeated injury/rescue loop unless they leave the plant.

## Reproduction and scope

Simulation source: Git commit `30e10c95211e040bfa3a232269bde5758c391bf5`. Experiment script was an added, uncommitted file at execution. Python command, run from the repository root:

```powershell
& '<user-home>/Projects/laya-lab/.venv/Scripts/python.exe' experiments/hazards.py
```

The script constructs disposable worlds in memory. It does not load saves, send server commands, restart services or modify the player's world. It writes only `runs/hazard-experiments/preregistration.json` and `runs/hazard-experiments/results.json`. These ignored artifacts contain the declared hypotheses, exact source hashes, every food choice and all scenario rows. Re-running replaces those two experiment artifacts. No dependencies or downloads were needed.

The hypotheses and source hashes were written before executing the scenarios. Seeds were **713, 1729 and 20261004**, fixed before running. Total experiment compute time was **22.572 seconds**. Source SHA-256:

```text
sim/world.py
b1b588a55bd17807ac7c272208ec2ce65e179ce9fadfcbbc5455e1abca77bbd4
experiments/hazards.py
ac8986db37acf18b4819724fac288274fbdc85b4ea8971b4d5535f118bac8365
```

## 1. Does repeated injury change food selection?

Hypothesis: with both foods available every time, the scripted policy will not systematically reduce amber consumption after injury.

Each seed starts one resident on empty grass. Before each of 80 choices, both inventory food types are replenished to one and fullness is set to 50. The experiment invokes the existing scripted policy and executes its selected food action. Health, pain, counters and visual memories persist; after a faint, ordinary simulation steps continue until natural recovery. There is no health reset or trained policy. Resetting fullness is a deliberate way to create equal food-choice opportunities, not a natural feeding schedule.

| Seed | Amber in first 40 choices | Amber in last 40 choices | Amber / all choices | Faints | Actual cumulative damage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 713 | 16/40 | 19/40 | 35/80 | 32 | 700 |
| 1729 | 15/40 | 16/40 | 31/80 | 28 | 620 |
| 20261004 | 17/40 | 19/40 | 36/80 | 33 | 720 |
| Total | **48/120 (40%)** | **54/120 (45%)** | **102/240 (42.5%)** | **93** | **2,040** |

Observed: harmful consumption did not decline in any of these three seeded sequences. This agrees with the implementation, which randomly chooses between available foods without updating preferences. It is not evidence of a learner failing: no learner is connected. Nor does the five percentage-point increase establish a preference for harmful food. Cumulative damage exceeds starting health because natural recovery restores health between later injuries. Frequent later faints reflect waking near 30 health and consuming another harmful fruit.

## 2. Does rescue solve the hazard?

Hypotheses: adjacent help reduces unconscious time; standing on the same plant after rescue allows repeated injury, while moving off prevents further contact.

Controlled resident starts at health 1 after a 99-point injury on a thorn tile. Each scenario lasts 400 ticks (100 simulated seconds). Compare no helper, a helper one tile west, and a helper two tiles west. Helpers stay in place: they use the existing baseline's revive action when selected and otherwise rest. The injured resident either stays still or is commanded east on its first awake turn. These explicit interventions isolate contact and assistance; they are not autonomous rescue demonstrations.

All three seeds returned identical measurements below. Faint totals include the controlled initial faint. Unconscious ticks count state at the start of each tick.

| Helper | Injured resident's behavior | First awake tick | Unconscious ticks | Thorn contacts | Faints | Successful revives | Final health |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | Stay on plant | 185 | 336 | 4 | 3 | 0 | 14.8 |
| None | Leave when awake | 185 | 185 | 0 | 1 | 0 | 38.8 |
| Adjacent | Stay on plant | 1 | 13 | 25 | 13 | 13 | 22 |
| Adjacent | Leave when awake | 1 | 1 | 0 | 1 | 1 | 44.44 |
| Two tiles away | Stay on plant | 185 | 336 | 4 | 3 | 0 | 14.8 |
| Two tiles away | Leave when awake | 185 | 185 | 0 | 1 | 0 | 38.8 |

Observed: adjacent rescue wakes the resident within the first tick, versus 185 ticks (46.25 simulated seconds) for natural recovery from health 1. The two-tile helper cannot revive remotely. Its position is fixed by the experiment, so this comparison does not measure whether wandering residents might eventually encounter someone who needs help.

The repeat loop is real: helping an idle body back onto its feet on thorns increases awake exposure, causing 25 contacts and 13 faints/revivals. Remaining unconscious suppresses further thorn contact, explaining the lower contact count without help. A lower raw injury count therefore need not indicate better behavior. The adjacent helper finishes the repeated-rescue condition at 66.75 energy; resting between rescues partly replenishes its 10-energy rescue costs. This is consistent with current mechanics, not a test failure or demonstrated learned strategy.

## 3. Short unmodified population runs

Hypothesis: short fresh-world runs may have too little relevant exposure to evaluate behavior. Each seed ran eight ordinary scripted residents plus an idle player for 600 ticks (150 simulated seconds); no objects, needs or policies were changed. Metrics below exclude the player.

Visible-thorn resident-ticks count one awake resident seeing at least one thorn during one pre-action observation. This is a rough exposure denominator, not a count of independent encounters or a probability of touching each visible plant.

| Seed | Visible-thorn resident-ticks | Thorn contacts | Contacts / 1,000 visible-thorn resident-ticks | Visible-amber resident-ticks | Food choices | Faints / revives | Mean final health | Items gathered |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 713 | 1,572 | 14 | 8.91 | 1,694 | 0 | 0 / 0 | 95.275 | 94 |
| 1729 | 2,880 | 17 | 5.90 | 2,716 | 0 | 0 / 0 | 95.965 | 94 |
| 20261004 | 1,956 | 19 | 9.71 | 2,289 | 0 | 0 / 0 | 87.215 | 89 |

Observed: thorns were encountered and caused injury; no resident fainted in these short runs. There were **zero food-choice opportunities**, including zero opportunities with both foods carried. Fullness starts at 100 and drops only 4.8 over 600 ticks, ending at 95.2, above the scripted eating threshold of 75. Consequently, zero amber consumption tells us nothing about avoidance. The controlled food-choice experiment above supplies the missing decision opportunities. These three short populations do not establish long-term health stability or rescue frequency.

## Implications and next experiment

The mechanics provide consequences worth learning from, but current memory acquisition does not change decisions. A later learned policy should be tested against frozen/random/scripted controls using the same opportunities and held-out seeds. Measure harmful choices per available-food decision, injury conditional on exposure, time unconscious, food obtained and health together. Report trajectories per resident; aggregate improvement could otherwise hide some residents repeatedly suffering while others remain safe.

Rescue rewards alone would be a poor training target: repeated faint/revive cycles could score highly without moving anyone to safety. A future experiment should measure sustained safe, awake time after assistance and compare leaving a hazard with staying in place. This report recommends that evaluation; it does not introduce rewards, teach avoidance or alter rescue mechanics.

Mortality remains disabled. There is no neural training, learned cooperation, preference update or long-term transfer result here. Three seeds are a bounded smoke experiment, not a statistical generalization claim. Existing hazard regression tests were also run with `python -m unittest discover -s tests -p test_hazards.py -q`: **10 passed**. No simulation or game files were changed.
