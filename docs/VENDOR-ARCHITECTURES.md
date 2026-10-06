# NVIDIA and AMD architecture review

Reviewed 2026-10-05 UTC / 2026-10-04 America/Los_Angeles. Primary vendor documentation and repositories only. Research review; nothing downloaded, installed, or benchmarked. Hardware context supplied for this project: Windows, RTX 5080 Laptop GPU with 16 GB VRAM, Intel NPU.

**Recommendation:** keep Godhood Trials' small, independently trained resident policies and deterministic tile world. NVIDIA provides useful simulation and RL components; adopting a branded agent framework would not itself solve curiosity, lifelong learning, or capacity growth. The strongest vendor component for a later throughput bottleneck is **Warp**. **Isaac Lab** provides genuine policy training reference designs, with substantial adaptation overhead for this world. These are engineering recommendations inferred from the sources below.

## Four concrete candidates

| Candidate | License | Local compatibility | Learning and memory | Fit |
|---|---|---|---|---|
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | Apache-2.0; bundled libmathdx has separate NVIDIA terms | Official Windows x64 CPU/CUDA wheels; compatible driver still needs verification | Simulation kernels and differentiation; learner, resident memories, and growth must be implemented | Best optional acceleration component |
| [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Core BSD-3-Clause; Mimic Apache-2.0; dependencies/assets have separate terms | Windows/Linux documented; conventional Isaac Sim path lists 16 GB VRAM and 32 GB RAM minimum | RL/IL and multi-agent integrations; individual checkpoints possible through training design; persistent personal memory/growth remain custom | Useful training patterns; heavier than this tile world needs |
| [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | Apache-2.0; environment/dataset licenses vary | Windows through WSL2; library itself requires no GPU | Stateful environments, verifiers, rollouts; separate training framework updates models | Evaluation ideas useful; current generative-model stack is excessive |
| [AMD GAIA](https://github.com/amd/gaia) | MIT | Windows/Linux; advertised Ryzen AI acceleration does not establish Intel NPU support | LLM tools and retrieval; persistent memory exists, policy training/growth not supplied | Poor resident-brain fit; possible future researcher assistant |

**Warp:** GPU kernels could execute many replicas of the tile world and batch policy inputs while preserving separate policy parameters. This requires porting and testing the simulation; Warp is not a ready-made autonomous learner. Profile CPU stepping and learner updates first. Eight residents alone do not establish a GPU bottleneck. [Official repository and platform documentation](https://github.com/NVIDIA/warp).

**Isaac Lab:** its reinforcement-learning integrations and multi-agent support make it materially different from an LLM assistant. Randomly initialized small policies are a suitable use of conventional RL; a pretrained language model is unnecessary. However, the demonstrated applications are robotics. The current development branch also advertises Kit-less Newton workflows; it would be inaccurate to say every future path requires Isaac Sim. Treat that branch's compatibility as unverified here. [Framework capabilities and licenses](https://github.com/isaac-sim/IsaacLab), [documented conventional Windows requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Its **population-based training** replaces underperforming policies' weights with a leader's checkpoint and mutates hyperparameters. That is population optimization, not eight residents independently retaining their own life histories. Disable replacement for the individuality experiment, or explicitly describe descendants and inheritance in a separate experiment. [Official PBT behavior](https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html).

**NeMo Gym:** borrow its separation of environment, agent, verifier, and per-task state. Its documented ecosystem centers on generative-model training and tool-use benchmarks; online resident learning would require a custom adapter and learner. A small local evaluation runner gives most of the immediate benefit with much less infrastructure. [Official purpose, interfaces, and platform requirements](https://github.com/NVIDIA-NeMo/Gym).

**GAIA:** its opt-in memory uses SQLite and semantic/keyword recall. “Continuous learning” in that guide describes extracting knowledge from conversations. It does not establish gradient updates, increased network capacity, or blank-slate sensorimotor learning. A generic serving backend might broaden hardware compatibility, but that is untested here; do not confuse AMD Ryzen acceleration with this machine's Intel NPU. [Official memory guide](https://github.com/amd/gaia/blob/main/docs/guides/memory.mdx).

## Other vendor names checked

[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) is a pretrained vision-language-action robot model, currently Apache-2.0. The repository lists 16 GB+ for inference and recommends 40 GB+ for fine-tuning. It brings language/robot demonstration priors and a continuous-action robot interface, contrary to the proposed blank-slate creatures. Running inference is different from independently training eight such models.

[Cosmos 3](https://github.com/NVIDIA/cosmos) offers pretrained physical-world reasoning, generation, and policy models under OpenMDW-1.1. Its model family includes an Edge tier; large-model-only claims would be misleading. Even the smaller tier supplies learned priors and physical-AI interfaces, rather than a small resident learning its tile world from birth. It could be a future comparison condition, with its prior knowledge disclosed.

[AMD Quark](https://github.com/amd/Quark) is MIT model quantization/optimization tooling, not a resident architecture. Its repository includes Windows CUDA wheel options, so “AMD software cannot use NVIDIA” would be incorrect; backend/operation support still needs checking. It could compress a mature policy later. [ROCm's compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/compatibility.html) targets AMD Radeon/Ryzen hardware; CUDA is the relevant training route on the supplied NVIDIA GPU.

## Preserve room to grow

Define a replaceable resident-controller interface with versioned observation/action schemas. Save each resident's weights, optimizer, recurrent state, episodic/replay memory, and random generator separately. Add bounded memory or model modules only when measured transfer/retention improves, preserving the previous checkpoint as a control. This is a proposed experiment design, not a capability these products automatically deliver.

Evaluate fresh seeds, resource relocation, social partners, unfamiliar layouts, delayed building tasks, and return to old tasks. Compare against random actions, the scripted baseline, and a frozen policy; include compute and memory cost, survival needs, distinct resident outcomes, and forgetting. Accelerated training must produce experiences each resident actually receives. No reviewed vendor system demonstrates unlimited self-expansion or guarantees a route to AGI.
