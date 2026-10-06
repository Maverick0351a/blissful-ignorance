# Chinese learning architectures for Godhood Trials

Research date: **2026-10-04, America/Los_Angeles**. Primary repositories only;
no packages, models, or checkpoints were downloaded or executed. Compatibility
and learning performance below remain untested unless explicitly marked.

**Recommendation:** keep the compact recurrent policy and add real experience
collection and independent PPO training first. Use HARL as a reference for
separate actors and recurrent rollout bookkeeping; use DI-engine as the more
comprehensive alternative if maintaining a training framework becomes costly.
LightZero is a later world-model/planning experiment. These are training systems,
not pretrained language-model towns or proof of lifelong intelligence.

## Three concrete candidates

| Candidate | Reusable capability | Independent brains and memory | Effort / platform limits |
|---|---|---|---|
| [PKU-MARL HARL](https://github.com/PKU-MARL/HARL), affiliated with Peking University and BIGAI | HAPPO/HAA2C and other multi-agent learners; discrete actions; recurrent rollout masks and checkpointing | HAPPO configuration defaults to separate actor weights and random initialization. Optional [GRU state](https://github.com/PKU-MARL/HARL/blob/main/harl/models/base/rnn.py) is per agent; recurrence defaults off. Stock runner uses one shared critic. | Medium adaptation: custom world adapter, local reward definition, observation encoder, recurrent configuration. Existing examples include older Gym dependencies and Linux-specific environments; native Windows integration is unverified. |
| [OpenDILab DI-engine](https://github.com/opendilab/DI-engine) | PPO/R2D2 training, advantage estimation, collectors, replay, exploration modules including RND/ICM | Small policies can train from scratch. Separate models, optimizers, buffers, checkpoints, and recurrent states must be configured explicitly; the framework does not imply independence. | Medium to high integration. Package metadata declares Windows support, but this laptop/runtime combination is untested. Its broad dependency set and older pins favor an isolated environment or selectively reusing utilities. |
| [OpenDILab LightZero](https://github.com/opendilab/LightZero) | MuZero/EfficientZero/UniZero families: learn latent dynamics, reward/value predictions, then plan with MCTS. A [vector-input MLP model](https://github.com/opendilab/LightZero/blob/main/lzero/model/muzero_model_mlp.py) exists. | Fresh model initialization is supported; preserving individual learning would require a separate model and replay for each resident. Latent dynamics and replay are learned memory, not automatic network growth. | High adaptation: world adapter, partial-observation history, stochastic transitions, bounded replay and search. README limits compilation support to Linux/macOS; Windows needs a port or a separately approved Linux environment. |

HARL's [HAPPO configuration](https://github.com/PKU-MARL/HARL/blob/main/harl/configs/algos_cfgs/happo.yaml)
sets `share_param: False` and `model_dir: null`; its
[runner](https://github.com/PKU-MARL/HARL/blob/main/harl/runners/on_policy_base_runner.py)
creates separate actors/buffers but a shared critic using `share_obs`. Therefore
stock HAPPO is a cooperation reference, not a direct replacement for fully local,
independent resident learning. Using local critics instead changes the algorithm;
its published guarantees cannot simply be carried over. Independent recurrent
PPO is the cleaner first baseline for personal survival objectives.

## Code and weights

DI-engine and LightZero declare **Apache-2.0** for code. HARL's
[package metadata](https://github.com/PKU-MARL/HARL/blob/main/setup.py)
declares **MIT**; a standalone license file was not found in the inspected root,
so confirm the distribution's license text before vendoring its code. None of
these options requires importing pretrained weights for this experiment.
External benchmark checkpoints were not assessed, and a code license should
not be assumed to cover arbitrary externally hosted weights.

DI-engine's README advertises DreamerV3, but its linked
`ding/world_model/dreamerv3.py` returned **404** during this check. That port is
an advertised lead, not verified reusable code here. Dreamer itself is not a
Chinese-origin algorithm merely because a Chinese framework advertises a port.

## Laptop fit and actual growth

The initial independent policy had 140,320 parameters. Eight CPU forward
passes measured about 1.06 ms median; CUDA measured about 2.33 ms, excluding
observation encoding. That supports retaining CPU inference while testing
small training workloads on the RTX 5080 Laptop GPU with 16 GB VRAM. It does
not prove a framework's training throughput or memory requirements. The later
construction/senses policy has 145,754 parameters; refreshed measurements are
in [HARDWARE.md](HARDWARE.md). Start with
a few residents and bounded rollouts; profile buffers and optimizer memory
before scaling. World-model search adds substantial work beyond one policy
forward pass. Hardware details and NPU boundaries are in [HARDWARE.md](HARDWARE.md).

The detected Intel AI Boost NPU is a possible later inference target. These
repositories do not provide a verified NPU training path for this game; conversion,
numerical agreement, compilation and warm latency still need separate tests.

All three candidates learn by updating finite networks. Recurrent state,
replay history and changing weights are distinct from permanent episodic memory,
lifelong retention or structural brain growth. None of the inspected sources
establishes automatic, indefinitely growing resident brains for this world.
Our next reusable milestone is a real rollout learner evaluated on unseen
seeds against the scripted and untrained baselines, with per-resident checkpoints,
local observations, and retention checks after the world changes.

## LLM society boundary

[Tsinghua FIB Lab AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety)
is explicitly LLM-native and its quick start requires an LLM provider. Its code
is Apache-2.0 except for the documented commercial folder; model weights and
provider terms are separate. Its social-environment and replay ideas may be
useful, but it does not supply tiny resident brains that learn from scratch
through our local experience loop. It is not the recommended controller path.
