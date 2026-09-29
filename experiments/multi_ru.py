"""
multi_ru.py - Coordination (C2) and size generalisation (C3) with several RUs.
==============================================================================

The single-RU comparison tests the encoder swap. This experiment tests the
project's actual thesis: that RUs whose coverage overlaps must coordinate their
sleep, that independent agents cannot, and that message passing over the RU
graph lets them.

Setting
-------
4 RUs (one centre, three on a ring 120 m out) with overlapping coverage and 16
UEs, 3 slices, light and medium traffic. A UE in an overlap region can be
served by more than one RU; if every one of its candidate RUs sleeps in the
same slot while it has data queued, the coverage-continuity penalty applies.

Methods
-------
  indep       EExApp per RU. Shared weights, but each RU's actor sees only the
              UEs in its own cell; the critic sees the pooled state of all
              cells. Independent actors, centralised critic.
  gnn_noedge  Our GNN with the RU-RU relation removed. Identical network,
              minus the coordination channel.
  gnn         Our GNN, full.
  sync        Fixed: every RU sleeps the same b slots at the end of the frame.
              Best b chosen per scenario (an oracle-tuned policy).
  stagger     Fixed: every RU sleeps b slots, but windows are rotated so that
              neighbours sleep at different times. Best b per scenario. This
              is the obvious hand-designed coordination rule.

C3: agents trained on 4 RUs / 16 UEs are evaluated, unchanged, on 7 RUs / 28
UEs (a full hexagonal ring). ``gnn`` is also trained natively on 7 RUs as the
reference for what transfer should reach.

Run:
    python -m experiments.multi_ru            # train + evaluate everything
    python -m experiments.multi_ru --plot     # figures and tables only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "multi_ru")
FIG_DIR = os.path.join(HERE, "figures")

# Light and medium. At 4 UEs per cell both are fully provisioned (no
# violations with every RU awake), and they call for opposite coordination:
# under light load it is best for neighbours to sleep together, under medium
# load they must stagger. Heavy load already violates QoS for ~28% of UEs with
# every RU awake, so it is not a meaningful operating point for energy saving.
LEVELS = ["light", "medium"]
SEEDS = 3
TRAIN_STEPS = 3000
EVAL_SEEDS = (2000, 2001, 2002)
EVAL_STEPS = 100


# The paper's per-UE rates were set for 8 UEs on one isolated cell. With
# neighbours interfering, 6 UEs per cell at medium traffic already fails 25%
# of UEs with every RU awake; 4 per cell is provisioned at light and medium.
UE_PER_RU = 4


def scenario(level: str, n_ru: int = 4):
    """Multi-cell deployment. Propagation is 3GPP UMi with LOS probability,
    since links to a neighbouring site 120 m away are mostly non-line-of-sight;
    the single-RU study keeps the all-LOS model of the paper's indoor lab."""
    from sim import RADIO_DEFAULT, ScenarioConfig

    return ScenarioConfig(
        name=f"{level}_{n_ru}ru",
        traffic_level=level,
        n_slices=3,
        n_ue=UE_PER_RU * n_ru,
        n_ru=n_ru,
        inter_site_distance_m=120.0,
        cell_radius_m=100.0,
        radio=replace(RADIO_DEFAULT, propagation="umi_mixed"),
    )


def spec(method: str):
    from agents.policy import PolicySpec

    return {
        "indep": PolicySpec(encoder="transformer", dual=True, gat=True),
        "gnn_noedge": PolicySpec(encoder="gnn", dual=True, gat=True, d_model=28, ru_edges=False),
        "gnn": PolicySpec(encoder="gnn", dual=True, gat=True, d_model=28, ru_edges=True),
    }[method]


# ---------------------------------------------------------------------------
# Evaluation (returns the coverage metric too)
# ---------------------------------------------------------------------------


def evaluate_policy(policy, sc, seeds=EVAL_SEEDS, steps=EVAL_STEPS) -> dict:
    import torch

    from agents.obs import build_obs, collate
    from agents.ppo import to_env_action
    from sim import OranSimEnv

    acc = {k: [] for k in ("reward", "violation", "energy_saved", "sleep_ratio", "uncovered", "coverage_penalty")}
    policy.eval()
    with torch.no_grad():
        for seed in seeds:
            env = OranSimEnv(sc, seed=seed)
            obs = build_obs(env, env.reset())
            for _ in range(steps):
                out = policy(collate([obs]))
                a, b = out.sleep.mode()
                ue_obs, (r_e, r_q), _, info = env.step(to_env_action(a, b, out.slices.mode(), env.n_dl_slots))
                _record(acc, r_e, r_q, info)
                obs = build_obs(env, ue_obs)
    policy.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate_fixed(sc, b: int, stagger: bool, seeds=EVAL_SEEDS, steps=EVAL_STEPS) -> dict:
    from sim import OranSimEnv

    acc = {k: [] for k in ("reward", "violation", "energy_saved", "sleep_ratio", "uncovered", "coverage_penalty")}
    for seed in seeds:
        env = OranSimEnv(sc, seed=seed)
        env.reset()
        n, r = env.n_dl_slots, env.n_ru
        rows = []
        for i in range(r):
            a = ((i * (n - b)) // max(r - 1, 1)) if stagger else n - b
            a = min(a, n - b)
            rows.append([a, b, n - a - b])
        action = {"sleep": rows, "slices": np.full((r, env.n_slices), 1.0 / env.n_slices)}
        for _ in range(steps):
            _, (r_e, r_q), _, info = env.step(action)
            _record(acc, r_e, r_q, info)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _record(acc, r_e, r_q, info):
    acc["reward"].append(r_e + r_q)
    acc["violation"].append(info["qos_violation_ratio"])
    acc["energy_saved"].append(info["energy_saved_fraction"])
    acc["sleep_ratio"].append(info["mean_sleep_ratio"])
    acc["uncovered"].append(float(np.mean(info["uncovered_fraction"])))
    acc["coverage_penalty"].append(info["coverage_penalty"])


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _init_worker():
    import torch

    torch.set_num_threads(1)


def job_learned(method: str, level: str, seed: int, n_ru_train: int = 4) -> dict:
    path = os.path.join(OUT_DIR, f"{method}_{level}_{n_ru_train}ru_seed{seed}.json")
    if os.path.exists(path):
        return {"ok": True, "path": path, "skipped": True}
    try:
        from agents.ppo import PPOConfig, train

        t0 = time.perf_counter()
        sc = scenario(level, n_ru_train)
        policy, log = train(spec(method), sc, seed, PPOConfig(total_steps=TRAIN_STEPS))
        rec = {
            "method": method,
            "level": level,
            "seed": seed,
            "n_ru_train": n_ru_train,
            "curve": log["curve"],
            "n_params": log["n_params"],
            "eval": {str(n_ru_train): evaluate_policy(policy, sc)},
        }
        if n_ru_train == 4:
            rec["eval"]["7"] = evaluate_policy(policy, scenario(level, 7))  # zero-shot
        rec["seconds"] = time.perf_counter() - t0
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(rec, fh)
        return {"ok": True, "path": path, "seconds": rec["seconds"]}
    except Exception:
        return {"ok": False, "error": traceback.format_exc(), "method": method, "level": level, "seed": seed}


def job_fixed(level: str, n_ru: int) -> dict:
    path = os.path.join(OUT_DIR, f"fixed_{level}_{n_ru}ru.json")
    if os.path.exists(path):
        return {"ok": True, "path": path, "skipped": True}
    try:
        sc = scenario(level, n_ru)
        n = sc.numerology.n_dl_slots
        rows = []
        for b in range(0, n + 1):
            for stagger in (False, True):
                r = evaluate_fixed(sc, b, stagger)
                rows.append({"b": b, "stagger": stagger, **r})
        best = {
            "sync": max((r for r in rows if not r["stagger"]), key=lambda r: r["reward"]),
            "stagger": max((r for r in rows if r["stagger"]), key=lambda r: r["reward"]),
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"level": level, "n_ru": n_ru, "best": best, "all": rows}, fh)
        return {"ok": True, "path": path}
    except Exception:
        return {"ok": False, "error": traceback.format_exc(), "level": level, "n_ru": n_ru}


def run_all(workers: int) -> None:
    futures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        # Longest first: the natively trained 7-RU GNN.
        for level in LEVELS:
            for seed in range(SEEDS):
                futures.append(pool.submit(job_learned, "gnn", level, seed, 7))
        for method in ("gnn", "gnn_noedge", "indep"):
            for level in LEVELS:
                for seed in range(SEEDS):
                    futures.append(pool.submit(job_learned, method, level, seed, 4))
        for level in LEVELS:
            for n_ru in (4, 7):
                futures.append(pool.submit(job_fixed, level, n_ru))
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            status = "ok" if r["ok"] else "FAILED\n" + r["error"]
            print(f"[{i}/{len(futures)}] {r.get('path', '')} {status}", flush=True)


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

COLOR = {"indep": "#2a78d6", "gnn_noedge": "#4a3aa7", "gnn": "#eb6834", "sync": "#8a8984", "stagger": "#52514e"}
LABEL = {
    "indep": "Independent EExApp per RU",
    "gnn_noedge": "Ours, no RU-RU edges",
    "gnn": "Ours (GNN, full)",
    "sync": "Fixed, synchronised sleep (oracle b)",
    "stagger": "Fixed, staggered sleep (oracle b)",
}


def load():
    learned, fixed = {}, {}
    for p in glob.glob(os.path.join(OUT_DIR, "*.json")):
        with open(p) as fh:
            r = json.load(fh)
        if os.path.basename(p).startswith("fixed_"):
            fixed[(r["level"], r["n_ru"])] = r["best"]
        else:
            learned.setdefault((r["method"], r["level"], r["n_ru_train"]), []).append(r)
    return learned, fixed


def plot_and_tabulate() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.plot_comparison import INK_2, smooth  # shared style

    learned, fixed = load()
    os.makedirs(FIG_DIR, exist_ok=True)
    lines = ["## Table J - Multi-RU coordination (4 RUs, 16 UEs; mean ± s.d. over 3 seeds)", "",
             "| Traffic | Method | Reward | QoS violation (%) | Energy saved (%) | Sleep ratio | Coverage penalty |",
             "|---|---|---|---|---|---|---|"]
    summary = {}

    def row(level, name, vals):
        def ms(k, pct=False):
            a = np.array([v[k] for v in vals])
            f = 100 if pct else 1
            return f"{f * a.mean():.{1 if pct else 3}f}" + (f" ± {f * a.std():.{1 if pct else 3}f}" if len(a) > 1 else "")
        lines.append(f"| {level} | {LABEL[name]} | {ms('reward')} | {ms('violation', True)} | "
                     f"{ms('energy_saved', True)} | {ms('sleep_ratio')} | {ms('coverage_penalty')} |")
        summary.setdefault(f"{level}_4ru", {})[name] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}

    for level in LEVELS:
        for m in ("indep", "gnn_noedge", "gnn"):
            recs = learned.get((m, level, 4), [])
            if recs:
                row(level, m, [r["eval"]["4"] for r in recs])
        for f in ("sync", "stagger"):
            if (level, 4) in fixed:
                row(level, f, [fixed[(level, 4)][f]])
    lines.append("")

    # Zero-shot to 7 RUs
    lines += ["## Table K - Zero-shot transfer from 4 RUs to 7 RUs (28 UEs), no retraining", "",
              "| Traffic | Method | Reward on 7 RUs | QoS violation (%) | Energy saved (%) | Coverage penalty |",
              "|---|---|---|---|---|---|"]
    for level in LEVELS:
        for m in ("indep", "gnn_noedge", "gnn"):
            recs = learned.get((m, level, 4), [])
            if recs:
                v = [r["eval"]["7"] for r in recs]
                lines.append(f"| {level} | {LABEL[m]} (trained on 4) | {np.mean([x['reward'] for x in v]):.3f} | "
                             f"{100 * np.mean([x['violation'] for x in v]):.1f} | {100 * np.mean([x['energy_saved'] for x in v]):.1f} | "
                             f"{np.mean([x['coverage_penalty'] for x in v]):.3f} |")
                summary.setdefault(f"{level}_7ru_zeroshot", {})[m] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
        native = learned.get(("gnn", level, 7), [])
        if native:
            v = [r["eval"]["7"] for r in native]
            lines.append(f"| {level} | Ours (GNN) trained natively on 7 | {np.mean([x['reward'] for x in v]):.3f} | "
                         f"{100 * np.mean([x['violation'] for x in v]):.1f} | {100 * np.mean([x['energy_saved'] for x in v]):.1f} | "
                         f"{np.mean([x['coverage_penalty'] for x in v]):.3f} |")
        for f in ("sync", "stagger"):
            if (level, 7) in fixed:
                x = fixed[(level, 7)][f]
                lines.append(f"| {level} | {LABEL[f]} | {x['reward']:.3f} | {100 * x['violation']:.1f} | "
                             f"{100 * x['energy_saved']:.1f} | {x['coverage_penalty']:.3f} |")
    lines.append("")

    # Figure: learning curves + bar summary
    fig, axes = plt.subplots(1, 2 + 1, figsize=(14, 3.9))
    for ax, level in zip(axes[:2], LEVELS):
        for m in ("indep", "gnn_noedge", "gnn"):
            recs = learned.get((m, level, 4), [])
            if not recs:
                continue
            L = min(len(r["curve"]["reward"]) for r in recs)
            st = np.array(recs[0]["curve"]["step"][:L])
            ys = np.array([smooth(np.array(r["curve"]["reward"][:L]), 10) for r in recs])
            ax.plot(st, ys.mean(0), color=COLOR[m], lw=2, label=LABEL[m])
            ax.fill_between(st, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), color=COLOR[m], alpha=0.15, lw=0)
        for f, ls in (("sync", ":"), ("stagger", "--")):
            if (level, 4) in fixed:
                ax.axhline(fixed[(level, 4)][f]["reward"], color=COLOR[f], ls=ls, lw=1.4, label=LABEL[f])
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_title(f"4 RUs, {level} traffic")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
    axes[0].legend(fontsize=7.5, loc="lower right")

    ax = axes[2]
    methods = ["indep", "gnn_noedge", "gnn"]
    x = np.arange(len(LEVELS))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = []
        for level in LEVELS:
            recs = learned.get((m, level, 4), [])
            vals.append(np.mean([r["eval"]["7"]["reward"] for r in recs]) if recs else np.nan)
        ax.bar(x - 0.4 + (i + 0.5) * w, vals, w * 0.9, color=COLOR[m], label=LABEL[m])
    for j, level in enumerate(LEVELS):
        native = learned.get(("gnn", level, 7), [])
        if native:
            ax.plot([j - 0.42, j + 0.42], [np.mean([r["eval"]["7"]["reward"] for r in native])] * 2,
                    color=INK_2, ls="--", lw=1.2, label="GNN trained on 7 RUs" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([lv.capitalize() for lv in LEVELS])
    ax.set_ylabel("Reward on 7 RUs")
    ax.set_title("Zero-shot: trained on 4 RUs, run on 7")
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_multi_ru.png"))
    plt.close(fig)

    with open(os.path.join(FIG_DIR, "summary_multi_ru.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only produce figures and tables")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    if not args.plot:
        run_all(args.workers)
    text = plot_and_tabulate()
    with open(os.path.join(FIG_DIR, "summary_multi_ru.md"), "w", encoding="utf-8") as fh:
        fh.write("# Multi-RU results\n\nGenerated by experiments/multi_ru.py.\n\n" + text)
    print(text)


if __name__ == "__main__":
    main()
