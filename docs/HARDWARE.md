# Local hardware observations

Observed October 4, 2026 (America/Los_Angeles; artifact timestamps use UTC), using the existing Python environment. This is one
laptop trial, not a minimum specification or a performance guarantee.

| Component | Observed status |
|---|---|
| CPU | Intel Core Ultra 9 275HX, 24 cores / 24 logical processors |
| CUDA GPU | NVIDIA GeForce RTX 5080 Laptop GPU; PyTorch 2.11.0+cu128, CUDA 12.8 available |
| NPU | Intel AI Boost; the installed OpenVINO 2026.4.0 runtime lists `NPU` in `Core.available_devices` |
| NPU metadata | Architecture `3720`; reported capabilities `FP16`, `INT8`, `EXPORT_IMPORT`; driver property `1005540` |

The NPU check read device properties only. The resident network has not been
converted, compiled, or run on the NPU; its compatibility, latency, power use,
and benefit remain untested. No NPU backend is connected to the game.

## Hazard interface (schema 4)

The schema-4 controller had 147,054 parameters, 30 visual channels, 79 other features and 45 actions. Schema 5 adds farming: 148,116 parameters, 32 visual channels and 51 actions; no new timing benchmark was run. These dimensions include health, pain and unconscious state. Timing tables below are historical; this interface has not been rebenchmarked.

## Construction and senses update (schema 2)

The updated controller has **145,754 parameters**, 26 visual/recall channels,
66 other sensory features, and 41 actions at the default recurrent width.
The same bounded quick benchmark was rerun after adding these inputs:

| Independent resident models | CPU median / p95 | CUDA median / p95 |
|---|---:|---:|
| 1 | 0.130 / 0.386 ms | 0.348 / 0.546 ms |
| 8 | 1.169 / 1.253 ms | 2.991 / 3.741 ms |
| 24 | 4.574 / 5.139 ms | 9.445 / 10.298 ms |

This uses only twelve timing samples and excludes observation encoding.
The scripted simulation with senses and growing memory reached about 232
ticks/s over a short 120-tick run. This is not a filled-memory, long-run,
browser-rendering or online-training benchmark. Both CPU and CUDA synthetic
optimizer smoke checks again passed. Raw local measurements are in
`runs/benchmark-20261005T042946Z.json` (ignored by Git).

## Original schema 1 CPU and CUDA trial

`python -m experiments.benchmark --quick` measured independently initialized,
untrained 140,320-parameter models, each with a 128-unit LSTMCell. Each model
performed a separate batch-one forward pass. Three warmup passes preceded
twelve timing samples; CUDA timing synchronized completion. The measurements
exclude observation encoding and initialization.

| Independent resident models | CPU median / p95 | CUDA median / p95 |
|---|---:|---:|
| 1 | 0.126 / 0.149 ms | 0.412 / 0.591 ms |
| 8 | 1.055 / 1.228 ms | 2.335 / 2.503 ms |
| 24 | 3.530 / 3.782 ms | 6.743 / 8.075 ms |

CPU was faster for this small, serial inference workload. That result does not
predict performance for larger models, batched independent parameters, or
training. The scripted nine-resident simulation reached about 1,007 ticks/s
with rendering off. CUDA peak allocated tensors for the 24-model trial were
about 23.3 MB; that figure excludes the CUDA context and reserved memory.
The local result is `runs/benchmark-20261005T040205Z.json` (ignored by Git).

The bounded PPO smoke made two recurrent optimizer updates with invented
observations and rewards on CPU and CUDA. It verified finite gradients and
parameter changes; it did not collect game experience or establish learning.

## Sensible NPU next step

Keep CPU as the measured inference baseline for these tiny policies. An NPU
experiment could later export one frozen policy with fixed input shapes and
explicit hidden/cell state inputs and outputs, compare its outputs with
PyTorch, and measure compilation cost plus warm latency for 1/8/24 independent
policies. Per-resident weights and state must remain isolated. Device dispatch
and transfer costs may outweigh the small computation; that is an untested
hypothesis here. A useful NPU role may instead be reducing CPU/GPU contention
or power use, which also needs measurement.

OpenVINO documents [stateful NPU model support](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-devices.html)
and an [LSTMCell IR operation](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/lstm-cell-1.html).
Its [state API guide](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-request/stateful-models.html)
describes carrying recurrent state between inference calls. These general
features do not establish conversion or compilation support for this complete
policy on the installed NPU driver.

The existing training smoke uses PyTorch autograd on CPU/CUDA. This repository
has no NPU training backend. OpenVINO's documented [model preparation workflow](https://docs.openvino.ai/2026/openvino-workflow/model-preparation.html)
converts models for an inference application; an NPU deployment would be a
separate inference export refreshed from CPU/CUDA training. The
[NPU device guide](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
lists platform and shape constraints to check before that experiment.

Hunger observation schema 6 adds two body features: 81 scalar features and 148,148 parameters at width 128; action schema remains 5 with 51 actions. This update was functionally tested, not rebenchmarked.
