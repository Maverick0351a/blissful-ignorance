# Parallel learning environment

Implemented October 5, 2026 as the first approved rebuild milestone.

`agents/environment.py` exposes the existing simulator through a headless,
model-free parallel interface. `adapters/pettingzoo_env.py` adds the optional
PettingZoo `ParallelEnv` contract. The browser and live population do not import
the optional packages. This work creates disposable experiments and does not
load a live save, replace resident brains, or run Laya inference.

## Interface

```python
from adapters.pettingzoo_env import GodhoodParallelEnv

env = GodhoodParallelEnv(max_cycles=1000)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {
        agent: env.action_space(agent).sample(mask=observations[agent]["action_mask"])
        for agent in env.agents
    }
    observations, rewards, terminated, truncated, infos = env.step(actions)
env.close()
```

This example samples random actions to illustrate the interface; it is not
the learning policy. `LearningEnv` offers the same reset/step structure using
the existing PyTorch and NumPy installation, without PettingZoo or Gymnasium.

Each agent receives a dictionary containing its own encoded senses:

| Field | Space | Meaning |
|---|---|---|
| `patch` | float32, 36 x 9 x 9, [0,1] | Occluded local vision and private lossy recall channels |
| `features` | float32, `FEATURES`, [0,1] | Own body, inventory, relative hearing, touch and sensory history |
| `action_mask` | int8, `len(ACTIONS)`, 0/1 | Existing locally observable physical affordances |

The action vocabulary preserves every existing primitive, including the ten
individual tones, harmful food and physically accessible hazards. The adapter
does not choose actions, recommend recipes or award farming/communication bonuses.
Policies may submit any in-range action, even one masked as infeasible; the
world records its physical failure. Invalid batches fail before advancing the
world or consuming its RNG.

Pass an explicit seed to choose a scenario. An omitted seed currently selects
the deterministic default 1729 on every reset; this adapter does not generate
an automatic stream of new maps. The benchmark supplies every map seed itself.

All agents choose from the same pre-step instant. One decision advances four
physical ticks: the submitted commands on the first, then waiting commands on
three ticks. The original simulator resolves simultaneous conflicts with its
seeded action ordering. Returned rewards sum the existing nutrition relief,
hunger and injury terms across those ticks. `infos` contains only that agent's
body deltas, action result and reward accounting.

Mortality remains off. Reaching `max_cycles` produces truncation, not death,
and returns final observations for value bootstrapping. A fainted resident stays
in the active population. All declared IDs and spaces are fixed within a run;
births and larger populations need a future identity-slot/migration design.
Default experiments use four residents; the curve benchmark uses two per world.

The optional `state()` method deliberately raises `NotImplementedError`, so
standard learners are not handed a centralized observation. The trusted Python
harness can still inspect `env.core` for evidence and checkpoints: this is an
observation contract, not a process security sandbox. The core's separate
snapshot/restore API preserves world RNG and episode state. ANSI rendering is
an optional observer summary, not a policy input.

## Approved optional dependencies

The user approved these exact official PyPI/files.pythonhosted.org wheels before
download, inspection, isolated installation and testing:

| Wheel | Bytes |
|---|---:|
| pettingzoo-1.27.0-py3-none-any.whl | 668,492 |
| gymnasium-1.3.0-py3-none-any.whl | 953,904 |
| cloudpickle-3.1.2-py3-none-any.whl | 22,228 |
| farama_notifications-0.0.6-py3-none-any.whl | 2,897 |
| Total | 1,647,521 |

Exact URLs and SHA-256 values are pinned in
[`scripts/rl-dependencies.json`](../scripts/rl-dependencies.json).
[`prepare_rl_dependencies.py`](../scripts/prepare_rl_dependencies.py) verifies
size, hash, archive CRC, relative paths, duplicate names, symlinks, encrypted
members, executable extensions, pure-Python metadata and Python syntax. It
does not install or import package code; downloading requires its explicit
approval flag and the user's prior approval.

Focused review of the downloaded source covered package initializers, registry
startup, base environment classes, relevant spaces, official parallel/seed
test paths, Cloudpickle import-time setup, and the static Farama notification
dictionary. A whole-wheel text scan found no network client or shell-execution
calls in these Python files. PettingZoo's registry reads bundled environment
source to find parallel entry points; it does not instantiate those games.
Gymnasium registers built-in entry-point strings. This is a focused dependency
review, not a claim of exhaustive security analysis. No untrusted serialized
models are loaded through Cloudpickle.

Local artifacts are outside the repository:

- Downloads: `<user-home>/Downloads/godhood-rl-wheels-20261005/`, including `VERIFIED.json`.
- Optional packages: `<user-home>/Applications/Godhood-RL-Tools-20261005/`.
- Reused Python/Torch/NumPy: `<user-home>/Projects/laya-lab/.venv/Scripts/python.exe`.

The existing environment has no pip module. The already installed `uv` installed
the four local wheels with `--offline --no-index --no-deps --no-build
--no-python-downloads --no-cache --link-mode copy --target ...`. No existing
Torch/NumPy packages were replaced. Optional packages are selected with a
process-local `PYTHONPATH`, not a global environment change.

```powershell
$env:PYTHONPATH = '<user-home>\Applications\Godhood-RL-Tools-20261005'
$python = '<user-home>\Projects\laya-lab\.venv\Scripts\python.exe'
& $python -m unittest tests.test_environment tests.test_pettingzoo_adapter -v
```

## Verification

Nine focused tests passed in 43.735 seconds. These include PettingZoo 1.27.0's
official `parallel_api_test` (1,000 cycles, two resets) and `parallel_seed_test`
(500 cycles), actual observations contained in their declared spaces, exact
native-versus-wrapper trajectories, and deliberately absent global state.

The core tests additionally verify exact original-simulator and PPO learned-state
agreement through 145 decisions, including optimizer updates; deterministic
reset; complete snapshot continuation; local array bounds; action validation;
and truncation/terminal reward accounting. Optional tests skip explicitly when
their dependencies are unavailable.

The complete suite passed **164 tests in 67.898 seconds**. The local test log
and its source-hash record are in `runs/parallel-interface-20261005/`.

The [learning-curve benchmark](LEARNING-CURVE.md) exercises the actual wrapper
with private learners and separate validation/final maps. Passing the adapter
tests establishes a reusable experiment interface, not survival competence.
The existing PPO brain still consumes the core's copied `raw_observation`
compatibility bridge and applies its existing encoder. Standard packed arrays
are returned and tested, but an external trainer consuming them directly has
not yet been integrated. This preserves numerical parity at some redundant
encoding cost; it is not a throughput optimization.
