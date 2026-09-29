"""
sweep_fixed_policy.py - First experiment: the energy / QoS frontier.
====================================================================

Runs fixed (non-learned) sleep policies across the baseline paper's nine
scenarios - three traffic levels crossed with 2, 4 and 8 slices - and plots
the resulting trade-off.

Purpose. This is not a result about any algorithm. It is the Phase 1 evidence
that the simulator reproduces the qualitative behaviour the baseline reports,
and it establishes the frontier that a learned policy has to beat. Concretely
it should show:

  * energy saving rising monotonically with the sleep ratio;
  * QoS violations rising with traffic load;
  * QoS violations rising as the slice count grows, because fragmenting the
    PRB pool reduces statistical multiplexing gain - this is exactly the
    effect the paper describes for its Figures 5 and 7;
  * a knee, where further sleep buys little energy and costs a lot of QoS.

Run with:  python -m experiments.sweep_fixed_policy
Outputs to experiments/results/.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import OranSimEnv, ScenarioConfig, paper_scenarios

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_STEPS = 60
N_SEEDS = 3
SLEEP_FRACTIONS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]


# ---------------------------------------------------------------------------


def run_one(args) -> Dict:
    """Run one (scenario, sleep fraction, seed) combination."""
    scenario_name, traffic_level, n_slices, sleep_frac, seed = args

    cfg = ScenarioConfig(
        name=scenario_name,
        traffic_level=traffic_level,
        n_slices=n_slices,
        n_ue=8,
        n_ru=1,
        seed=seed,
    )
    env = OranSimEnv(cfg)
    env.reset()

    n_dl = env.n_dl_slots
    b = int(round(sleep_frac * n_dl))
    a = (n_dl - b) // 2
    c = n_dl - b - a
    action = {
        "sleep": np.array([[a, b, c]]),
        "slices": np.full((1, n_slices), 1.0 / n_slices),
    }

    acc = {"energy_saved": [], "violation": [], "reward": [], "delay": [], "throughput": []}
    for _ in range(N_STEPS):
        _, (r_e, r_q), _, info = env.step(action)
        acc["energy_saved"].append(info["energy_saved_fraction"])
        acc["violation"].append(info["qos_violation_ratio"])
        acc["reward"].append(r_e + r_q)
        acc["delay"].append(float(np.mean(info["delay_ms"])))
        acc["throughput"].append(float(np.mean(info["throughput_mbps"])))

    return {
        "scenario": scenario_name,
        "traffic_level": traffic_level,
        "n_slices": n_slices,
        "sleep_fraction": sleep_frac,
        "seed": seed,
        **{k: float(np.mean(v)) for k, v in acc.items()},
    }


def build_jobs() -> List:
    jobs = []
    for sc in paper_scenarios(n_ru=1, n_ue=8):
        for frac in SLEEP_FRACTIONS:
            for seed in range(N_SEEDS):
                jobs.append((sc.name, sc.traffic_level, sc.n_slices, frac, seed))
    return jobs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _aggregate(rows: List[Dict], key: str) -> Dict:
    """Group rows by (traffic_level, n_slices, sleep_fraction) and average."""
    out: Dict = {}
    for r in rows:
        k = (r["traffic_level"], r["n_slices"], r["sleep_fraction"])
        out.setdefault(k, []).append(r[key])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


LEVELS = ["light", "medium", "heavy"]
SLICES = [2, 4, 8]
COLORS = {2: "#2b6cb0", 4: "#d97706", 8: "#b91c1c"}


def plot_tradeoff(rows: List[Dict], path: str) -> None:
    """Energy saved against QoS violation - the frontier a policy must beat."""
    energy = _aggregate(rows, "energy_saved")
    violation = _aggregate(rows, "violation")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, level in zip(axes, LEVELS):
        for n_slices in SLICES:
            xs = [energy[(level, n_slices, f)][0] * 100 for f in SLEEP_FRACTIONS]
            ys = [violation[(level, n_slices, f)][0] for f in SLEEP_FRACTIONS]
            ax.plot(xs, ys, "o-", color=COLORS[n_slices], label=f"{n_slices} slices", lw=1.8, ms=5)
        ax.set_title(f"{level} traffic")
        ax.set_xlabel("Energy saved (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("QoS violation ratio")
    axes[0].legend(frameon=False)
    fig.suptitle("Energy / QoS trade-off frontier under fixed sleep policies", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_violation_bars(rows: List[Dict], path: str) -> None:
    """Figure 7 style: QoS violation per scenario, grouped by sleep level."""
    violation = _aggregate(rows, "violation")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    width = 0.8 / len(SLEEP_FRACTIONS)
    cmap = plt.get_cmap("viridis")

    for ax, level in zip(axes, LEVELS):
        x = np.arange(len(SLICES))
        for i, frac in enumerate(SLEEP_FRACTIONS):
            means = [violation[(level, s, frac)][0] for s in SLICES]
            errs = [violation[(level, s, frac)][1] for s in SLICES]
            ax.bar(
                x + i * width - 0.4 + width / 2,
                means,
                width,
                yerr=errs,
                capsize=2,
                color=cmap(i / max(len(SLEEP_FRACTIONS) - 1, 1)),
                label=f"sleep {frac:.0%}" if ax is axes[0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s} slices" for s in SLICES])
        ax.set_title(f"{level} traffic")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("QoS violation ratio")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("QoS violation by traffic level and slice count (baseline Figure 7 format)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_energy_curve(rows: List[Dict], path: str) -> None:
    """Energy saved against the commanded sleep ratio, with the proxy for comparison."""
    energy = _aggregate(rows, "energy_saved")

    fig, ax = plt.subplots(figsize=(6, 4))
    for level in LEVELS:
        ys = [np.mean([energy[(level, s, f)][0] for s in SLICES]) * 100 for f in SLEEP_FRACTIONS]
        ax.plot([f * 100 for f in SLEEP_FRACTIONS], ys, "o-", label=f"{level} traffic", lw=1.8)
    ax.plot(
        [f * 100 for f in SLEEP_FRACTIONS],
        [f * 100 for f in SLEEP_FRACTIONS],
        "k--",
        lw=1.2,
        label="baseline proxy  b/N",
    )
    ax.set_xlabel("Commanded sleep ratio (%)")
    ax.set_ylabel("Actual energy saved (%)")
    ax.set_title("Why the sleep-ratio proxy overstates energy saving")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = build_jobs()
    print(f"Running {len(jobs)} configurations on {os.cpu_count()} cores ...")

    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run_one, jobs))

    with open(os.path.join(RESULTS_DIR, "fixed_policy_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    plot_tradeoff(rows, os.path.join(RESULTS_DIR, "fig_tradeoff.png"))
    plot_violation_bars(rows, os.path.join(RESULTS_DIR, "fig_violation_bars.png"))
    plot_energy_curve(rows, os.path.join(RESULTS_DIR, "fig_energy_curve.png"))

    # Console summary so the run is auditable without opening the plots.
    print("\nMean over seeds and slice counts:")
    print(f"{'traffic':>8} {'sleep':>7} {'energy%':>9} {'violation':>10} {'delay ms':>9} {'thp Mbps':>9}")
    for level in LEVELS:
        for frac in SLEEP_FRACTIONS:
            sel = [r for r in rows if r["traffic_level"] == level and r["sleep_fraction"] == frac]
            print(
                f"{level:>8} {frac:>7.0%} "
                f"{np.mean([r['energy_saved'] for r in sel]) * 100:>9.1f} "
                f"{np.mean([r['violation'] for r in sel]):>10.3f} "
                f"{np.mean([r['delay'] for r in sel]):>9.1f} "
                f"{np.mean([r['throughput'] for r in sel]):>9.2f}"
            )
    print(f"\nWrote plots and raw data to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
