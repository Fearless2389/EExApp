"""
fixed_oracle.py - The best fixed (non-learned) policy for each scenario.
========================================================================

For every scenario, tries every sleep length b = 0 .. N_dl with an equal slice
split, and keeps the one with the highest mean reward. This is an *oracle*:
it is tuned per scenario with knowledge of the traffic level, which no
deployed policy has. It therefore serves two purposes in the comparison:

  * a reference line in Figure 5 - a learned policy that cannot reach it has
    not learned anything a lookup table would not give you;
  * the frontier for Figure 7 - the lowest violation a non-adaptive policy
    achieves at its best operating point.

Every b is evaluated on the same held-out seeds used for evaluating the
learned agents, so the numbers are directly comparable.

Run:  python -m experiments.fixed_oracle
Writes experiments/results/fixed_oracle.json
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from sim import OranSimEnv, ScenarioConfig, paper_scenarios

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Held-out evaluation seeds shared with experiments/train.py.
EVAL_SEEDS = (1000, 1001, 1002)
EVAL_STEPS = 150


def evaluate_fixed(args) -> Dict:
    """Mean metrics of one fixed sleep length on one scenario."""
    level, n_slices, b = args
    rewards, violations, energy, sleep = [], [], [], []
    for seed in EVAL_SEEDS:
        env = OranSimEnv(
            ScenarioConfig(name="oracle", traffic_level=level, n_slices=n_slices, n_ue=8, n_ru=1, seed=seed)
        )
        env.reset()
        n_dl = env.n_dl_slots
        # Sleep at the end of the frame (a = n_dl - b, c = 0). Traffic arrives at
        # the start of each frame, so late sleep lets the queue drain first.
        action = {"sleep": [[n_dl - b, b, 0]], "slices": [np.full(n_slices, 1.0 / n_slices)]}
        for _ in range(EVAL_STEPS):
            _, (r_e, r_q), _, info = env.step(action)
            rewards.append(r_e + r_q)
            violations.append(info["qos_violation_ratio"])
            energy.append(info["energy_saved_fraction"])
            sleep.append(info["mean_sleep_ratio"])
    return {
        "traffic_level": level,
        "n_slices": n_slices,
        "b": b,
        "reward": float(np.mean(rewards)),
        "reward_samples": [float(x) for x in rewards],
        "violation": float(np.mean(violations)),
        "energy_saved": float(np.mean(energy)),
        "sleep_ratio": float(np.mean(sleep)),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_dl = ScenarioConfig(name="probe").numerology.n_dl_slots
    jobs = [(sc.traffic_level, sc.n_slices, b) for sc in paper_scenarios() for b in range(n_dl + 1)]

    with ProcessPoolExecutor() as pool:
        rows: List[Dict] = list(pool.map(evaluate_fixed, jobs))

    best: Dict[str, Dict] = {}
    for sc in paper_scenarios():
        cands = [r for r in rows if r["traffic_level"] == sc.traffic_level and r["n_slices"] == sc.n_slices]
        best[sc.name] = max(cands, key=lambda r: r["reward"])

    with open(os.path.join(RESULTS_DIR, "fixed_oracle.json"), "w") as fh:
        json.dump({"best": best, "all": [{k: v for k, v in r.items() if k != "reward_samples"} for r in rows]}, fh, indent=1)

    print(f"{'scenario':>16} {'best b':>7} {'reward':>8} {'violation':>10} {'sleep':>7} {'energy saved':>13}")
    for name, r in best.items():
        print(
            f"{name:>16} {r['b']:>7d} {r['reward']:>8.3f} {r['violation']:>10.3f} "
            f"{r['sleep_ratio']:>7.2f} {r['energy_saved']:>13.3f}"
        )
    print("\nReward by sleep length (rows = scenario, columns = b):")
    for sc in paper_scenarios():
        cands = sorted(
            (r for r in rows if r["traffic_level"] == sc.traffic_level and r["n_slices"] == sc.n_slices),
            key=lambda r: r["b"],
        )
        print(f"{sc.name:>16} " + " ".join(f"{r['reward']:+.2f}" for r in cands))


if __name__ == "__main__":
    main()
