#!/usr/bin/env python3
"""Visualize a clipped Ornstein-Uhlenbeck noise process."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-std", type=float, default=0.05, help="Stationary OU standard deviation.")
    parser.add_argument("--ou-theta", type=float, default=7.0, help="Mean-reversion rate in s^-1.")
    parser.add_argument("--ou-mu", type=float, default=0.0, help="Long-term mean.")
    parser.add_argument("--noise-clip", type=float, default=0.1, help="Symmetric hard-clipping limit.")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=float, default=10.0, help="Visualization duration in seconds.")
    parser.add_argument("--seed", type=int, default=2025, help="NumPy random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/ou_noise_process.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def simulate_clipped_ou(
    *,
    noise_std: float,
    ou_theta: float,
    ou_mu: float,
    noise_clip: float,
    dt: float,
    duration: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same exact OU discretization as the benchmark."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if ou_theta < 0.0:
        raise ValueError("ou_theta must be non-negative")
    if noise_clip < 0.0:
        raise ValueError("noise_clip must be non-negative")
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("dt and duration must be positive")

    num_steps = int(round(duration / dt))
    time = np.arange(num_steps + 1, dtype=np.float64) * dt
    noise = np.empty(num_steps + 1, dtype=np.float64)
    raw_update = np.empty(num_steps + 1, dtype=np.float64)
    noise[0] = ou_mu
    raw_update[0] = ou_mu

    decay = math.exp(-ou_theta * dt)
    innovation_std = noise_std * math.sqrt(max(0.0, 1.0 - decay**2))
    normal_samples = np.random.default_rng(seed).standard_normal(num_steps)
    for step, normal_sample in enumerate(normal_samples, start=1):
        raw_update[step] = ou_mu + decay * (noise[step - 1] - ou_mu) + innovation_std * normal_sample
        if noise_clip > 0.0:
            noise[step] = np.clip(raw_update[step], -noise_clip, noise_clip)
        else:
            noise[step] = raw_update[step]

    return time, noise, raw_update


def main() -> None:
    args = parse_args()
    time, noise, raw_update = simulate_clipped_ou(
        noise_std=args.noise_std,
        ou_theta=args.ou_theta,
        ou_mu=args.ou_mu,
        noise_clip=args.noise_clip,
        dt=args.dt,
        duration=args.duration,
        seed=args.seed,
    )

    fig, ax = plt.subplots(figsize=(12.0, 5.4), constrained_layout=True)
    ax.axhspan(
        args.ou_mu - args.noise_std,
        args.ou_mu + args.noise_std,
        color="#4C78A8",
        alpha=0.12,
        label=r"stationary $\mu \pm \sigma$",
    )
    ax.axhline(args.ou_mu, color="#333333", linewidth=1.0, linestyle="--", label=r"mean $\mu$")
    if args.noise_clip > 0.0:
        ax.axhline(args.noise_clip, color="#D62728", linewidth=1.2, linestyle=":", label="clip limits")
        ax.axhline(-args.noise_clip, color="#D62728", linewidth=1.2, linestyle=":")

    ax.plot(time, noise, color="#1565C0", linewidth=1.25, label="clipped OU sample")
    clipped = np.flatnonzero(~np.isclose(noise, raw_update, rtol=0.0, atol=1e-15))
    if clipped.size:
        ax.scatter(time[clipped], noise[clipped], s=22, color="#D62728", zorder=3, label="clipped update")

    correlation_time = math.inf if args.ou_theta == 0.0 else 1.0 / args.ou_theta
    title = (
        "Clipped Ornstein-Uhlenbeck Noise\n"
        rf"$\sigma={args.noise_std:g}$, $\theta={args.ou_theta:g}\,\mathrm{{s}}^{{-1}}$, "
        rf"$\mu={args.ou_mu:g}$, clip=$\pm${args.noise_clip:g}, "
        rf"$\Delta t={args.dt:.5f}\,\mathrm{{s}}$"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("OU noise")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    ax.text(
        0.012,
        0.025,
        rf"seed={args.seed}   correlation time $1/\theta={correlation_time:.3f}$ s   "
        f"clipped updates={clipped.size}/{noise.size - 1}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved {args.output}")
    print(f"sample_mean={noise.mean():.6f}")
    print(f"sample_std={noise.std(ddof=0):.6f}")
    print(f"sample_min={noise.min():.6f}")
    print(f"sample_max={noise.max():.6f}")
    print(f"clipped_updates={clipped.size}")


if __name__ == "__main__":
    main()
