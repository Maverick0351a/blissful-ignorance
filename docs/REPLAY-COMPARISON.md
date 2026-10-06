# Matched replay comparison

October 5, 2026. **All three families failed the reliable-feeding gate.** The
confirmation completed 797,184 ticks in 977.94 seconds. No trial model was
promoted into the playable valley, and Laya remains her own supported model.

## Results

Twelve held-out lives per condition, 12,000 ticks each. Zero food means zero
fullness, not death; mortality is disabled. Each trained family is paired with
its own initial weights on the same six held-out layouts.

| Controller | Zero-food time | Lives under 1% zero food | Safe meals per life | Conscious tone choices |
|---|---:|---:|---:|---:|
| PPO, trained | 26.69% | 6/12 | 7.08 | 73.54% |
| PPO, initial | 10.13% | 2/12 | 10.08 | 69.69% |
| Feedforward replay, trained | 81.97% | 0/12 | 0.50 | 98.10% |
| Feedforward replay, initial | 87.36% | 0/12 | 0.92 | 92.62% |
| Recurrent replay, trained | 68.76% | 0/12 | 1.33 | 81.99% |
| Recurrent replay, initial | 91.68% | 0/12 | 0.00 | 76.15% |
| Uniform random legal actions | 8.58% | 6/12 | 10.92 | 69.87% |
| Privileged safe-route feasibility control | 0% | 12/12 | 4.00 | 0% |

Recurrent replay improved deprivation and mean fullness over its own initial
weights in all three training-seed groups. It did not produce a single life
below 1% zero-food time. Feedforward replay improved deprivation in only one
seed group. PPO reduced thorn contacts in all groups but increased mean
deprivation in all three. Its mean and low-starvation count move differently
because failures are concentrated in some individuals; neither metric alone
describes the distribution adequately.

The trained replay models repeated the exact previous conscious action on
89.30% (feedforward) and 87.78% (recurrent) of decisions, versus 8.47% for PPO
and 6.30% for random. This is a diagnostic, not proof of a particular internal
cause. Repetition is not always failure: the safe oracle mostly rests after
securing enough food, and repeats 99.50% of choices.

Random actions produced more planting and matured crop yield than trained PPO
in these permissive small arenas. Planting activity by itself therefore does
not establish planning or farming intent. Neither trained replay variant
planted during evaluation.

All three implementations remain experimental. These results do **not** mean
that replay, recurrence or PPO is generally inferior. Exploration, network
size, optimizer/update scheme and limited retention differ. In particular,
replay evaluation uses 5% epsilon while PPO remains stochastic. The older
small nine-action replay success cannot be transferred to this much wider
action interface.

## Post-hoc retention diagnosis and next direction

A separate checkpoint audit found that **10 of 12 replay residents** had eaten
during training but had **zero eating choices left in final replay**. Each
retains 1,024 learning transitions plus burn-in context. Sampling priorities
only affect retained data: FIFO eviction can still discard rare useful older
experience. This does not establish that eviction caused starvation, or that
the weights forgot those meals. It gives us a concrete intervention to test.

Before adding more complexity, compare FIFO with bounded episodic/reservoir
retention at equal memory and update budgets, and separately compare flat
versus category-balanced exploration. Keep all primitive choices and avoid
survival scripts. Use fresh seeds and retain this failed confirmation. A
world-model candidate remains a research direction, not an implemented or
validated capability.

## Runtime and completed audit

CPU training per independent two-resident pair took 41.97–52.48 seconds for
PPO, 37.18–41.41 seconds for feedforward replay and 48.62–52.94 seconds for
recurrent replay. Policy parameter counts are 89,173, 61,076 and 89,108,
respectively; target networks, optimizer state and PPO's separate predictor
are additional. ALIEN was launched during the run, so timings are observations
under concurrent use, not an isolated hardware benchmark.

The trace auditor passed all 48 evaluations: source/preregistration hashes,
archived source, held-out identities, matching layouts, every reported trace
total, food conservation, frozen learning and independent gate calculations.
The checkpoint auditor verified all 18 networks changed from initial weights,
remained finite/distinct, preserved replay episode boundaries, and matched the
exact learned-state digests used in evaluation. Evidence files are `audit.json`
and `checkpoint-audit.json` in the official run directory.

## Question and implementations

Does replay improve feeding when it receives the same sensory interface and
primitive actions as the recurrent PPO residents? Does recurrent sequence
replay add useful behavior beyond feedforward replay?

`agents/sequence_replay.py` implements independent visual Double-DQN variants:

- **Feedforward replay:** the existing PPO sensory encoder followed by two
  64-unit feedforward layers. This adapts the earlier replay approach to full
  senses and actions; it is not the old 23-input, nine-action model/checkpoint.
- **Recurrent replay:** the same encoder followed by a 64-unit LSTM. Contiguous
  16-transition learning chunks retain up to eight prior observations for
  burn-in. Online and target networks reconstruct context separately.

Both use private prioritized replay (64 chunks), batches of four chunks,
one-step masked Double-DQN targets, full importance correction (beta=1), and a
target update every 32 optimization steps. Each resident owns its weights,
optimizer, replay, priorities, action RNG and replay RNG. Complete checkpoints
preserve pending rewards and recurrent carry. Replay does not blend residents.

This is **R2D2-inspired**, not a reproduction of distributed R2D2. The stored
behavior carry is an approximation under newer weights. Burn-in reduces that
mismatch but does not eliminate it. No multi-step off-policy return is used.

All three candidates see the same local vision, body, sound/tone history,
contact and lossy visual recall. They can choose the full action vocabulary,
including ten tones and physically possible harmful actions. Only impossible
actions are masked. Reward comes from useful nutrition, hunger and injury;
curiosity is zero in this comparison. There are no route or farming bonuses.

## Preregistered experiment

Evidence folder: `runs/replay-comparison-20261005/official/` (local, ignored).

Preregistration SHA-256, recorded before confirmation results:

`47c978eb443f14bf31ca563116315e41c1601e5636b43b7ec0650843e7beeea7`

Three fresh seeds: 69071, 69082 and 69093. Each family receives 6,144 decisions
per resident: two feeding episodes of 512 decisions, two route episodes of
1,024, and two scarcity episodes of 1,536. Each world has two independent
residents in separate rotated/transposed finite-food fork arenas.

Evaluation freezes learning for 12,000 ticks on two held-out layouts per seed,
giving 12 evaluated lives per family/condition. Each family is compared with
its own initial weights. A random policy is an additional baseline. A
privileged safe-route oracle is feasibility evidence only and supplies no
training data. All start with identical body state, zero carried food and four
ordinary food units per arena. Crops may generate food through normal physics;
the recorder audits conservation each tick.

The inherited scarcity gate requires every seed group to improve deprivation
and fullness and halve thorn-contact frequency versus its initial policy;
at least 80% of lives must spend under 1% of time at zero food and at least 80%
must make a safe foodward crossing with at least 80% safe completed crossings.

Experience budgets are matched; wall-compute budgets and parameter counts are
not. PPO samples its policy at evaluation; replay uses 5% epsilon exploration.
Algorithm, memory and exploration therefore cannot be isolated as separate
causal effects. Small separated arenas do not test competition, society,
language or intentional farming. There is no amber fruit in these arenas, so
zero harmful meals is not avoidance evidence. Three training seeds give only
an initial comparison, not a universal architecture ranking.

## Validation before confirmation

All 150 tests passed in 24.509 seconds, including seven new tests covering
masked Double-DQN and terminal targets, burn-in gradients, feedforward absence
of temporal credit, complete sequence coverage and episode boundaries,
independent learning, frozen evaluation, and exact continuation through
optimizer updates after save/load with a pending reward.

A separate pilot seed (90211) completed 6,400 ticks in 8.36 seconds. The
independent trace recount, food conservation, source archive and preregistration
hashes, matched layouts, held-out world identities, and frozen learning checks
passed. Its short horizon is not a survival result.

Reproduce with the existing PyTorch Python environment:

```powershell
python experiments/replay_comparison.py --output runs/new-replay-confirmation --wall-seconds 2400
python experiments/audit_replay_comparison.py runs/new-replay-confirmation
python experiments/audit_replay_checkpoints.py runs/new-replay-confirmation
```

The runner creates a new evidence directory and refuses an existing one. It
does not contact a server or read, reset or save any live world.

## References

- [Recurrent DQN with PyTorch](https://docs.pytorch.org/rl/stable/tutorials/dqn_with_rnn.html)
- [R2D2 paper](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf)
- [Earlier project comparison](MODEL-SURVIVAL-COMPARISON.md)
