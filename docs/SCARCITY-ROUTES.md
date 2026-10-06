# Scarcity and safe routes

October 5, 2026. The sequence learner now has a harder, finite-stock route experiment. The previous confirmation kept all controls fed and could not discriminate reliable feeding. This protocol tests whether injury avoidance accompanies useful food acquisition.

## Environment

Two independently trained residents occupy isolated fork arenas in one authoritative World. Each starts at 8 fullness with no carried food, seeds or materials. Each arena contains four ordinary-food units, a thorn in the two-move shortcut between the fork junctions, and a safe bypass taking six or eight moves between those junctions. Orientations, translations, bypass direction and height vary with the world seed. Water is available beside the home room. Room-to-room route accounting also accepts entering through room corners; those shortest safe crossings take four or six moves.

The experiment never refills or relocates stocks. The normal primitive action set remains available, including ten tones, seed conversion, planting and construction when materials permit. A resident may generate additional food through real crop maturation. Food conservation is checked every tick: initial stock plus matured yield, minus ordinary eating and food-to-seed conversions, equals all remaining ground, carried and cached food. Such yields are recorded separately; observing planting alone is not proof of learned farming.

Each brain receives its existing private local observations and physically possible action mask. Route labels, global positions, goals and oracle actions are absent from neural inputs. Shared parameter definitions do not share learned weights, optimizers, rollouts or action RNGs. No player, scripted neighbor, gift or rescue participates in the formal experiment.

## Preregistered comparison

The unchanged 64-unit recurrent PPO backend has zero curiosity coefficient. Feedback remains food relief/25, hunger distress at up to -0.005 per tick, and observed injury divided by -20. There is no route, movement, seed, harvest or tone bonus.

Three training seeds: 53081, 53092 and 53103. Each resident completes four feeding episodes of 512 decisions, four route episodes of 1,024 decisions and four scarcity episodes of 1,536 decisions: 12,288 decisions per resident. Easier stages place two or one of the same four food units beside the starting room. Weights and optimizer persist across explicitly terminal training episodes; bodies and worlds reset. These are disposable experimental copies, not the named interactive inhabitants' lived histories.

Each training seed has two held-out world seeds (+100,000 and +200,000), evaluated for 12,000 ticks under four conditions:

- Trained, frozen policy and predictor weights.
- Matching training initialization, also frozen, with a matched action-sampling RNG.
- The existing local-observation scripted reference, which can choose harmful routes.
- A privileged safe-route script with global resource/terrain knowledge, used only to prove physical feasibility. It never supplies training targets or observations to a learner.

Every condition starts with the same map, body and stock. Recurrent state and visual memories may change through experience during evaluation, while weight digests must remain unchanged. There are six map-seed comparisons and twelve lives per condition; lives share small layout families and are not twelve independent architecture experiments. New seeds do not imply every normalized geometry is novel.

Before the run, source files and protocol were SHA-256 hashed and archived in `runs/scarcity-confirmation-20261005/`. Preregistration SHA-256: `69858c041c569f192414e1a4114f54f8cecddc357759953ccb5db39c18266c10`.

Acceptance requires every training-seed group to reduce mean zero-food ticks, increase mean fullness, and halve thorn contacts per conscious decision against matching initial policies. At least 80% of trained lives must spend less than 1% of time at zero fullness. At least 80% must complete a foodward crossing, a safe foodward crossing, and use safe paths for at least 80% of all completed crossings. Standing still cannot satisfy the route criterion. Aborted attempts, adjacent hazard opportunities and raw injury remain separate diagnostics.

The formal run is capped at 1,800 seconds. Failed, partial and pilot runs are retained. The 3,584-tick pilot completed in 5.224 seconds and passed its separate accounting/hash/route recount; its 512-tick evaluations do not outlast initial food and cannot establish survival competence. A subsequent diagnostic correction subtracts each persistent predictor's pre-episode counters; the pilot's original source remains archived.

## Confirmation results

The 435,456-tick confirmation completed in **509.128 seconds (8.5 minutes)**: 147,456 training ticks and 288,000 evaluation ticks. All 24 evaluations completed. A separate recount verified current and archived source/protocol hashes, matching arenas, all training budgets, frozen policy/predictor digests, physical route traces, food conservation and the unchanged acceptance verdict.

| Mean per held-out life | Trained | Matching initial policy | Local script | Privileged safe script |
| --- | ---: | ---: | ---: | ---: |
| Useful nutrition | 57.93 | 162.77 | 100.00 | 100.00 |
| Time at zero fullness | 51.28% | 12.90% | 0% | 0% |
| Mean fullness | 27.15 | 61.03 | 57.75 | 56.32 |
| Unconscious ticks | 1,525.92 | 683.42 | 1,269.83 | 0 |
| Thorn contacts | 3.92 | 41.83 | 72.25 | 0 |
| Contacts per 1,000 conscious decisions | 1.50 | 14.79 | 26.90 | 0 |
| Completed crossings | 0.75 | 10.58 | 50.83 | 1.00 |
| Safe share of all completed crossings | 22.22% | 26.77% | 27.54% | 100% |
| Crops planted | 3.17 | 26.25 | 0 | 0 |
| Food generated by mature crops | 6.92 | 65.42 | 0 | 0 |
| Lives with <1% zero-food time | 3/12 | 4/12 | 12/12 | 12/12 |

**The preregistered gate failed.** All three training-seed groups reduced thorn contacts per conscious decision by at least half, but none improved mean fullness or deprivation. Only 3/12 trained lives met the low-starvation criterion, and 2/12 met the active-safe-route criterion, versus the required >=80% for both. Lower raw injury was accompanied by less useful activity and nearly four times as much time at zero food. It is not evidence of competent safe foraging. The safe script proves food can be acquired without injury on every evaluation map; its global knowledge does not establish a learner result.

The food-producing action combinations were more common in initial policies than trained policies. This demonstrates that real crop production can happen through exploratory behavior without establishing learned farming or investment. No intervention isolated the value of planting, and no social interaction was possible in these arenas.

### Post hoc action diagnostic

`experiments/describe_scarcity.py` counts the actual conscious decisions in each evaluation trace, with totals checked against the audited metrics. Trained agents chose tones on **81.24%** of those decisions, versus **69.36%** for matching initial policies. Movement fell from 18.84% to 10.75%; gathering fell from 1.77% to 0.37%. Six trained lives received no nutrition at all, versus zero initial-policy lives. Four trained lives never completed a crossing, versus zero initial lives. These are descriptive checks after the run, not a newly passed hypothesis or proof of a causal explanation.

This suggests investigating exploration and action-category balance before adding more survival dependencies. Ten separate tone choices can occupy substantially more initial probability than a verb with one choice. A useful next ablation would preserve all ten symbols, balance exploration among feasible action types, and compare feeding, activity and hazards against the unchanged policy at the same budget. The present report does not claim that this proposed change will fix learning. No weights from this failed run were deployed into interactive residents.

## Play and reproduce

The interactive version adds a traveler in an accessible home-room tile. Human interventions can affect its inhabitants, so it is separate from the controlled comparison. It starts paused with freshly initialized learners and resumes existing complete checkpoints without resetting them. Trial-trained weights are not silently imported.

```powershell
.\scripts\Start-Development.ps1 -RunName scarcity-development -Stage scarcity -Port 8791
.\scripts\Stop-Development.ps1 -RunName scarcity-development
```

Open http://127.0.0.1:8791/ . Select Moss or Pip and expand **Learning record**. Move the traveler with WASD/arrows; step or resume for queued actions. The thorny shortcut and longer safe bypass are terrain, not instructions given to the residents.

With the existing compatible PyTorch environment:

```powershell
python experiments/scarcity_trial.py --output runs/scarcity-fresh --wall-seconds 1800
python experiments/audit_scarcity.py runs/scarcity-fresh
python experiments/describe_scarcity.py runs/scarcity-fresh
```

Output must be a new directory under `runs/`. The separate auditor checks archived/current source and protocol hashes, matched maps, completed training budgets, frozen evaluation digests, trace-derived nutrition/injury/deprivation/route counts, food conservation and the original acceptance verdict. Historical audits after a source change can use `--archive-only`.

## Limits

This is a small, fixed-physiology, shared-cue experiment. It does not test full-valley navigation, competition, communication, general planning, climate transfer or lifelong adaptation. The consequence predictor is diagnostic and does not choose actions. Crop production can occur through exploratory combinations; a matched control and a farming-specific intervention would be needed to establish learned investment in future harvests. Independent learner support remains separate from the permanently retained Laya and fly-inspired architecture options.

## Engineering verification

124 tests passed in 19.311 seconds. New tests cover the longer physically reachable bypass, finite/private initial resources, completed/aborted/unsafe route counts and teleport rejection, oracle feeding through authoritative physics, a gate that rejects inaction, real crop-yield accounting, and accessible traveler placement with checkpoint resumption. Python compilation, JavaScript syntax and diff checks passed.

Browser checks in the separate playable world verified the two-resident roster, paused startup, traveler movement on a single step, manual save and restoration of tick 0 with both brains, and the sequence-learning inspector. No browser errors or horizontal overflow; client and scroll width both 1,265 pixels. Screenshot: `runs/scarcity-arena.png`. This interactive population is fresh and paused, not an imported trial-trained pair. Existing experimental populations retain ticks 2,469 and 874; the original valley continues under its own controls.
