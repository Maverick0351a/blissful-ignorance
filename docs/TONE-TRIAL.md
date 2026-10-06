# Learned tone signaling — October 4, 2026

## Result

The preregistered diagnostic passed in all three training seeds. Two independent small neural policies learned a useful three-symbol signaling convention using food outcomes. This is a narrow cooperative communication result, not general language or a live population deployment.

| Frozen evaluation | Food eaten | Mean nutrition restored |
|---|---:|---:|
| Intact tones | 720 / 720 (100%) | 25.00 |
| Muted | 240 / 720 (33.33%) | 8.33 |
| Shuffled tones | 239 / 720 (33.19%) | 8.30 |

Per-seed shuffled success: 28.75%, 35.42%, 35.42%. Each seed exceeded the preregistered intact >=80% and >=20 percentage point advantage over both controls. Evaluation trials within a trained pair are correlated; there are only three independent training seeds, not 720 independent learners.

## What learned

A sender sees one berry location through its own World observation and receives a three-category left/center/right feature. A receiver starts out of food-vision range and receives only a four-way tone/silence input. Each policy has its own 16-unit neural hidden layer, weights, optimizer and action RNG. They learn simultaneously with independent squared-error action-value updates and 20% exploration. Both receive the same team reward: nutrition actually restored to the receiver, divided by 25. There is no pitch label, route label or gradient sharing.

The sender chooses low/mid/high through the actual World tone action. The receiver observes the normal hearing channel, selects one of three route skills, then walks, gathers and eats through World.step. Tones carry no sender identity or coordinates. No reward is given just for vocalizing.

Learned sender mappings for left/center/right:

- Seed 24051: low / mid / high.
- Seed 24062: low / mid / high.
- Seed 24073: high / low / mid.

A post-run diagnostic reconstructed initial policy choices: each initial pair matched only one of three food locations. This is an additional policy-lookup check, not a preregistered physical baseline run.

## Protocol and limits

1,500 training episodes per seed, with four relative layouts. Evaluation used four disjoint relative layouts, 240 balanced food-location episodes per condition per seed. All conditions used identical evaluation worlds and frozen weights. Shuffling permuted the intact tone list, preserving exact pitch frequency while breaking its relationship to food. Muting uses an unseen silence input, so the shuffled control is the stronger causal comparison.

6,660 total world episodes completed in 21.82 seconds. Training seeds: 24051, 24062, 24073. Three fixture/privacy/outcome/independence tests pass; the full repository suite passes 96 tests.

Route execution is scripted and knows all candidate positions, but never which contains food. Food perception is deliberately compressed to three categories. Layout transfer therefore tests the scaffolded task and physical channel; it does not establish learned navigation, raw-vision transfer or broad generalization. The world is empty, food is static, there is one message and one choice, and rewards are cooperative. Agents do not need to learn when to communicate, listen, tap, remember a sequence or resolve conflicting interests. No hunger endurance or competitive social benefit was measured. Action values are not calibrated probabilities; confidence calibration was not measured.

## Evidence and reproduction

Run with the existing PyTorch environment:

```powershell
& <user-home>/Projects/laya-lab/.venv/Scripts/python.exe experiments/tone_trial.py --output runs/tone-trial-fresh
```

The output directory must not exist. Local ignored evidence in `runs/tone-trial-20261004/`: training decisions, all evaluation rows, model weights, preregistration, SHA receipt, completion/gates and independent audit. The audit verified source hashes, matched evaluation worlds, preserved shuffled frequencies and actual nutrition accounting. No live save was read or changed.

Preregistration SHA256: `434de8e363e1bd447f4f512998671deb06099358908077e91113c4945ef34824`.

Next milestone: a harder experiment in which agents choose whether and when to signal, with food changing over time and communication competing with movement or gathering. Repeat muted/shuffled controls before connecting learned social behavior to live residents.

Historical reproduction note: use commit `fa5bcfd` for the exact three-tone run and source hashes. v0.7 changes sound duration and adds ten symbols; the runner now explicitly restricts its protocol to 0/1/2. Results above are not a ten-tone sequence-learning result.

October 5 follow-up: the [voluntary coordination pilot](VOLUNTARY-COORDINATION.md)
uses private raw sensory inputs, all ten tones, primitive movement and individual
rewards, with no forced signaling or route execution. Its three seeds failed
the communication gate. That result keeps this earlier scaffolded success
separate from evidence of voluntary useful communication or leadership.
