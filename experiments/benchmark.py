"""Bounded local inference benchmark and explicitly synthetic PPO update smoke.

Run: python -m experiments.benchmark --quick
No installation, downloads, game policy replacement, or learning claim occurs.
Raw measurements are saved under ignored runs/. Run --help for bounds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from sim.world import World


def summary(samples):
    ordered = sorted(samples)
    return {"samples": len(samples), "median_ms": statistics.median(samples) * 1000,
            "p95_ms": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] * 1000}


def simulation_benchmark(ticks):
    world = World(8128)
    started = time.perf_counter()
    for _ in range(ticks):
        world.step()
    elapsed = time.perf_counter() - started
    return {"ticks": ticks, "seconds": elapsed, "ticks_per_second": ticks / elapsed,
            "residents": len(world.residents), "rendering": False,
            "controller": "Scripted survival baseline", "learning": False}


def inference_benchmark(torch, device, count, warmups, samples, observations):
    from agents.network import Controller, encode_observation
    controllers = [Controller(seed=100 + index, device=device) for index in range(count)]
    inputs = [encode_observation(observations[index % len(observations)], device)
              for index in range(count)]
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def forwards():
        for controller, (patch, features) in zip(controllers, inputs):
            _, _, controller.state = controller.model(patch, features, controller.state)

    synchronize()
    warm_started = time.perf_counter()
    for _ in range(warmups):
        forwards()
    synchronize()
    warm_seconds = time.perf_counter() - warm_started
    measurements = []
    for _ in range(samples):
        synchronize()
        started = time.perf_counter()
        forwards()
        synchronize()
        measurements.append(time.perf_counter() - started)
    result = {"device": device, "independent_models": count, "batch_per_model": 1,
              "params_per_model": sum(p.numel() for p in controllers[0].model.parameters()),
              "warmup_passes": warmups, "warmup_seconds": warm_seconds,
              **summary(measurements),
              "resident_forwards_per_second": count / statistics.median(measurements),
              "includes_observation_encoding": False, "game_connected": False, "trained": False}
    if is_cuda:
        result["peak_tensor_memory_bytes"] = torch.cuda.max_memory_allocated()
    return result


def synthetic_ppo_smoke(torch, device):
    """Two updates on invented tensors/rewards; checks gradients, not learning.

    The recurrent state is recomputed through eight steps in each epoch. This
    small fixed-rollout clipped PPO objective proves that backward and optimizer
    paths function. It does not collect experience or measure survival reward.
    """
    from agents.network import ACTIONS, FEATURES, PATCH_CHANNELS, ResidentNetwork
    torch.manual_seed(20261004)
    model = ResidentNetwork().to(device)
    steps, batch = 8, 4
    patches = torch.rand((steps, batch, PATCH_CHANNELS, 9, 9), device=device)
    features = torch.rand((steps, batch, FEATURES), device=device)
    rewards = torch.randn((steps, batch), device=device) * .1
    actions, old_logs, old_values = [], [], []
    state = model.initial_state(batch)
    with torch.no_grad():
        for patch, feature in zip(patches, features):
            logits, value, state = model(patch, feature, state)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            actions.append(action)
            old_logs.append(distribution.log_prob(action))
            old_values.append(value)
    actions, old_logs, old_values = map(torch.stack, (actions, old_logs, old_values))
    returns = torch.zeros_like(rewards)
    future = torch.zeros(batch, device=device)
    for index in range(steps - 1, -1, -1):
        future = rewards[index] + .99 * future
        returns[index] = future
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    losses, gradient_norms = [], []
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(2):
        state = model.initial_state(batch)
        logs, values, entropies = [], [], []
        for index in range(steps):
            logits, value, state = model(patches[index], features[index], state)
            distribution = torch.distributions.Categorical(logits=logits)
            logs.append(distribution.log_prob(actions[index]))
            values.append(value)
            entropies.append(distribution.entropy())
        ratio = (torch.stack(logs) - old_logs).exp()
        surrogate = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
        loss = -surrogate.mean() + .5 * (torch.stack(values) - returns).square().mean() - .01 * torch.stack(entropies).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not gradients_finite or not bool(torch.isfinite(loss)):
            raise RuntimeError("Synthetic smoke produced non-finite loss or gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(norm.item()))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    delta = float((after - before).norm().item())
    if not delta > 0 or not math.isfinite(delta):
        raise RuntimeError("Synthetic optimizer did not make a finite parameter update")
    return {"device": device, "synthetic": True, "game_connected": False,
            "establishes_learning": False, "optimizer_updates": 2, "rollout_steps": steps,
            "batch": batch, "action_count": len(ACTIONS), "seconds": elapsed,
            "losses": losses, "gradient_norms": gradient_norms,
            "parameter_delta_l2": delta, "finite_gradients": True,
            "caveat": "Invented observations and rewards; no trained checkpoint, environment rollout, or learning evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True,
                        help="Default: 3 warmups, 12 samples/count/device, 120 simulation ticks.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto measures CPU plus CUDA when available; no dependencies are installed.")
    arguments = parser.parse_args(argv)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {"schema": 1, "utc": stamp, "quick": arguments.quick,
              "network_status": "Initialized from scratch; untrained and disconnected from game",
              "simulation": simulation_benchmark(120 if arguments.quick else 500),
              "inference": [], "synthetic_ppo": []}
    try:
        import torch
    except ImportError:
        result["optional_torch"] = "Unavailable; simulation benchmark completed; neural checks skipped."
    else:
        # Small independent models benefit from a single CPU thread; avoid large
        # thread pools overwhelming the laptop for these tiny tensor operations.
        torch.set_num_threads(1)
        devices = ["cpu"] if arguments.device == "auto" else [arguments.device]
        if arguments.device == "auto" and torch.cuda.is_available():
            devices.append("cuda")
        if "cuda" in devices and not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable in the existing PyTorch runtime")
        result["runtime"] = {"torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                             "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
        if torch.cuda.is_available():
            result["runtime"]["gpu"] = torch.cuda.get_device_name(0)
        world = World(20261004)
        observations = [world.observe(rid) for rid in world.residents if rid != "player"]
        for device in devices:
            for count in (1, 8, 24):
                measurement = inference_benchmark(torch, device, count,
                                                  3 if arguments.quick else 5,
                                                  12 if arguments.quick else 30, observations)
                result["inference"].append(measurement)
                print(f"{device}: {count} independent models, warmup {measurement['warmup_seconds']:.3f}s, "
                      f"median {measurement['median_ms']:.3f}ms, p95 {measurement['p95_ms']:.3f}ms", flush=True)
            result["synthetic_ppo"].append(synthetic_ppo_smoke(torch, device))
    result["total_seconds"] = time.perf_counter() - started
    directory = Path(__file__).resolve().parents[1] / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"benchmark-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Simulation: {result['simulation']['ticks_per_second']:.1f} ticks/s, rendering off", flush=True)
    print("Synthetic PPO smoke checks optimizer mechanics only; it does not establish learning.", flush=True)
    print(f"Result: {path}", flush=True)
    return result


if __name__ == "__main__":
    main()
