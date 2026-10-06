# Neural replay trial — October 4, 2026

189 episodes completed in 339.47 seconds. Three training seeds, 24 training worlds per seed/condition, three held-out worlds per seed, two independent learners. All nine evaluation layouts excluded all training layouts. Curiosity on/off shared starting weights and budgets. Frozen evaluation checked weights, target network, optimizer, replay, novelty counts and replay RNG.

| Evaluation mean per life | Curiosity on | Curiosity off | Initial weights | Random | Scripted |
|---|---:|---:|---:|---:|---:|
| Nutrition restored | 169.26 | 162.62 | 16.67 | 70.83 | 200 |
| Hunger cost | 8.345 | 5.907 | 32.624 | 24.925 | 2.869 |
| Zero-food ticks / 19,200 | 1715.50 | 846.78 | 12117.44 | 6182.06 | 0 |
| Unconscious ticks | 417.78 | 206.17 | 2997.11 | 1505.50 | 0 |
| Food harvested | 12.56 | 15.06 | 0 | 3.11 | 21.72 |

Both conditions improved nutrition and hunger cost over initial weights in every seed group. Both failed deployment: 14/18 lives met the feeding criterion, below 90%. No live resident was replaced. Curiosity increased nutrition slightly but worsened deprivation in this small study; this is not evidence against curiosity generally.

Limitations: compact observations alias crop timing/capacity; historical novelty rewards remain in replay; three-step off-policy returns lack importance correction; three seeds and nine layouts; historical tabular trials used different budgets/physiology. This does not establish neural superiority or general intelligence. Calibration of action confidence was not measured.

Local ignored evidence: `runs/neural-farming-20261004/official/` contains raw results, all evaluation traces, checkpoints, gate verdict, summary, preregistration, parent hash audit and `audited-source.zip`. Preregistration SHA256: `930b367256874af12098293de9be5291edb6b56578b0c0a5b6827549830e15b9`. The source archive preserves the pre-social world actually used by this trial; later live changes do not reproduce that exact experiment automatically.
