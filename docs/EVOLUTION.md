# Adaptation and experimental evolution

`agents/replay_policy.py` learns independent weights from experience replay. `agents/evolution.py` can load a locally generated checkpoint, fork an independent candidate, widen its hidden layers and vary learning rate or curiosity. Optimizer and replay start fresh; weights and private novelty memories copy by value. Checkpoints must be from trusted local runs, loaded with weights_only=True. Resume checkpoints are saved at episode boundaries; unfinished multi-step traces are not restored.

The actual trained 64-unit model was widened to 80 units (6,281 to 9,129 parameters). A 64-input smoke check measured exactly zero output difference. Tests also verify target preservation, independent updates and checkpoints. Evidence: `runs/evolution-smoke-20261004/lineage.json` and offspring checkpoint.

No offspring has demonstrated better fitness or been promoted. The proposed promotion gate requires matched fresh evaluations, at least three seeds, improved nutrition, no deprivation regression, and reliable feeding in >=90% of lives. Callers must enforce equal continuation-training budgets and fresh worlds. A per-run parameter budget is adjustable; this implementation grows within a two-hidden-layer family, not arbitrary architecture discovery. More capacity does not guarantee more intelligence. Laya remains a separate supported track.
