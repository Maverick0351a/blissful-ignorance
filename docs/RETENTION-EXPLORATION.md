# Private memory and exploration screen

October 5, 2026. The user approved the next experiment after the small-agent
research review. The live valley and Laya are outside this experiment.

## Preregistered comparison

Four recurrent replay conditions form a 2x2 design:

| Condition | Memory | Training epsilon exploration |
|---|---|---|
| `fifo_flat` | Recent FIFO | Uniform feasible action |
| `retained_flat` | Consequence reservation + recent FIFO | Uniform feasible action |
| `fifo_balanced` | Recent FIFO | Uniform available verb, then variant |
| `retained_balanced` | Consequence reservation + recent FIFO | Uniform available verb, then variant |

Each resident owns its weights, optimizer, replay and three RNGs (action,
priority sampling, retention). Every condition retains at most 64 chunks of
16 learning transitions with up to eight preceding burn-in observations.
Up to 16 chunks can be protected by uniform reservoir sampling over chunks
containing that resident's observed nutrition or injury. The oldest unprotected
chunk is evicted when full. Chunks have one copy; unused protected slots can
hold recent experience. Learning samples all chunks through the same priority
distribution and beta=1 correction to a uniform retained-chunk objective.

This is a declared memory-selection bias, not an action script. Nutrition and
injury are the same local body changes already used by reward. Common injury
experiences may dominate the reservoir; the audit reports positive and negative
retention separately. No external meal, route, behavior or reward is supplied.

Balanced exploration groups by the existing verb strings. Directions, items
and the ten tones remain selectable within each category. This changes only
the 25% epsilon draws during training. All trained and initial policies use
the same flat 5% epsilon exploration during evaluation. Changing an exploration
prior does not demonstrate intentional communication or freely chosen goals.

Fresh seeds: **70131, 70142, 70153**. Each arm trains two independent residents
for 2,048 decisions each: feeding 512, routes 512, scarcity 1,024. The same
initial weights, observations, architecture, memory capacity, reward, worlds,
update count and evaluation exploration are used across arms. Parameter count
and update count are verified. Actual buffer context lengths and elapsed time
can differ. Two held-out world seeds per training seed give 12 evaluated lives
per condition at 12,000 ticks each. Shared initial-weight, random-action and
privileged feasibility controls add three evaluation conditions. No control
provides experience to a learner.

Main retention/exploration effects and their interaction are reported within
each seed. Directional support requires less zero-food time and higher fullness
in every seed. Three seeds and this short training budget constitute a mechanism
screen, not significance or a deployment decision. The original reliable-feeding
gate is also reported. Useful factors require longer confirmation and fresh seeds.

The worlds use separated fork arenas with finite ordinary food and normal crop
physics. They do not test competition or communication. There is no amber fruit,
so zero harmful meals cannot establish food-hazard avoidance. Mortality is off.

## Verification and evidence

All **155 tests passed** in 29.785 seconds. Five new tests cover exact baseline equivalence, bounded retention and priority
alignment through eviction, availability of all action variants, identical flat
evaluation exploration, private learning, frozen retention state, and exact
save/resume across updates and evictions in all four conditions.

The distinct pilot seed 91231 completed 6,656 ticks in 12.70 seconds. Its source,
trace recount, conservation, frozen checkpoint and equal-budget audit passed.
The 512-tick pilot horizon is too short to measure sustained feeding; it only
checks experiment mechanics.

Run folders are local and ignored:

- `runs/retention-screen-20261005/pilot/`
- `runs/retention-screen-20261005/official/` (confirmation status recorded below)

Each includes preregistration and source hashes, an archived source snapshot,
full evaluation traces, learning curves and complete private checkpoints.

```powershell
python experiments/retention_trial.py --output runs/new-retention-screen --wall-seconds 1800
python experiments/audit_retention.py runs/new-retention-screen
```

The new extension hooks preserve the earlier replay baseline's behavior, as
verified by the baseline equivalence test. They do change source bytes: use
`--archive-only` with the older `audit_replay_comparison.py` when auditing the
previous October 5 comparison against its saved source archive.

## Completed results

The official screen completed **602,304 ticks in 639.13 seconds**. All four
conditions failed reliable feeding, with zero of 12 lives under 1% zero fullness.
Preregistration SHA-256, recorded before results:
`9e742405bd4d969c8bafba884a642000b09cb01902599289d70398609ba7f932`.

| Condition | Zero-fullness time | Mean safe meals | Mean thorn contacts |
|---|---:|---:|---:|
| FIFO, flat | 81.38% | 0.50 | 8.67 |
| Retention, flat | 85.16% | 0.25 | 11.67 |
| FIFO, balanced | 82.65% | 0.42 | 4.42 |
| Retention, balanced | 79.55% | 0.83 | 2.17 |
| Initial weights | 91.68% | 0.00 | 3.00 |
| Random legal actions | 11.07% | 11.17 | 41.67 |
| Privileged feasibility oracle | 0.00% | 4.00 | 0.00 |

Retention preserved more nutrition transitions: 23 versus 4 total across the
six flat-exploration learners, and 30 versus 11 with balanced exploration.
They also replayed more nutrition transitions (385 versus 273, and 416 versus
301). This establishes that the memory intervention operated; it does not
establish useful sustained survival. Both main-effect directional hypotheses
failed across seeds. Retention's deprivation effect was beneficial, unchanged,
and harmful in the three seeds, respectively. Exploration's effect was harmful,
harmful, then beneficial. The combined arm's slightly better aggregate does
not overcome this inconsistency or its zero reliable lives.

The audit verified all 42 evaluation traces, conservation, source/archive and
checkpoint hashes, identical layouts and initial weights, equal parameter and
update budgets, private finite parameters, episode boundaries, memory caps and
the exact frozen checkpoints used in evaluation. Each learner had 89,108 policy
parameters, 125 updates and 1,024 retained learning transitions at the end.

No trial weights were promoted. These results support improving the training
benchmark and examining learning curves before treating another architecture
or a larger swarm as a solution. See the [approved rebuild direction](REBUILD-PROPOSAL.md).
