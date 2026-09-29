"""
plot_comparison.py - Figures and tables for the paper-vs-ours comparison.
=========================================================================

Reads the per-run JSON written by experiments/comparison.py and produces:

  fig5_convergence.png      our version of the paper's Figure 5
  fig6_reward_cdf.png       our version of the paper's Figure 6
  fig7_qos_violation.png    our version of the paper's Figure 7
  fig_energy_saved.png      energy saved under the power model, per method
  fig_slice_transfer.png    zero-shot transfer across slice counts
  summary_tables.md         every number behind every figure, as tables
  summary.json              the same, machine-readable

Figures use one fixed colour per method across every figure (colour follows
the method, never its rank), validated for colour-vision deficiency. Three of
the hues sit below 3:1 contrast on white, so every figure has a numeric table
in summary_tables.md.

Run:  python -m experiments.plot_comparison
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "results", "runs")
ORACLE_PATH = os.path.join(HERE, "results", "fixed_oracle.json")
FIG_DIR = os.path.join(HERE, "figures")

LEVELS = ["light", "medium", "heavy"]
SLICES = [2, 4, 8]

# One colour per method, everywhere. Validated with the dataviz palette checker
# for the bar adjacency order used in Figure 7 and for the Figure 5 line set.
COLOR = {
    "eexapp": "#2a78d6",
    "wo_trans": "#eda100",
    "wo_gat": "#e87ba4",
    "wo_both": "#008300",
    "eexapp_plus": "#4a3aa7",
    "gnn": "#eb6834",
    "sasc": "#1baf7a",
    "released": "#8a8984",
    "oracle": "#52514e",
}
LABEL = {
    "released": "EExApp (released code)",
    "eexapp": "EExApp (paper, corrected)",
    "sasc": "SASC",
    "wo_trans": "w/o Trans",
    "wo_gat": "w/o GAT",
    "wo_both": "w/o Both",
    "eexapp_plus": "EExApp + slice features",
    "gnn": "Ours (hetero GNN)",
    "oracle": "Best fixed policy (oracle)",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs() -> Dict[str, Dict[str, List[dict]]]:
    """runs[method][scenario] -> list of run records (one per seed)."""
    runs: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(RUNS_DIR, "*", "*", "seed*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        runs[rec["method"]][rec["scenario"]].append(rec)
    return runs


def load_oracle() -> Dict[str, dict]:
    if not os.path.exists(ORACLE_PATH):
        return {}
    with open(ORACLE_PATH) as fh:
        return json.load(fh)["best"]


def scen(level: str, n: int) -> str:
    return f"{level}_{n}slice"


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Trailing moving average; the paper's curves are visibly smoothed."""
    if k <= 1 or len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y, dtype=float)
    out[k - 1 :] = (c[k:] - c[:-k]) / k
    out[: k - 1] = c[1:k] / np.arange(1, k)
    return out


def metric(runs, method, scenario, key) -> np.ndarray:
    return np.array([r["eval"][key] for r in runs.get(method, {}).get(scenario, [])], dtype=float)


# ---------------------------------------------------------------------------
# Figure 5 - convergence
# ---------------------------------------------------------------------------


def fig5(runs, oracle) -> None:
    methods = [("released", "--"), ("sasc", "-."), ("eexapp", "-"), ("gnn", "-")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharex=True, sharey=True)
    for ri, level in enumerate(LEVELS):
        for ci, n in enumerate(SLICES):
            ax = axes[ri, ci]
            s = scen(level, n)
            for m, ls in methods:
                recs = runs.get(m, {}).get(s, [])
                if not recs:
                    continue
                length = min(len(r["curve"]["reward"]) for r in recs)
                steps = np.array(recs[0]["curve"]["step"][:length])
                ys = np.array([smooth(np.array(r["curve"]["reward"][:length]), 10) for r in recs])
                mu, sd = ys.mean(0), ys.std(0)
                ax.plot(steps, mu, ls, color=COLOR[m], lw=2, label=LABEL[m])
                ax.fill_between(steps, mu - sd, mu + sd, color=COLOR[m], alpha=0.15, lw=0)
            if s in oracle:
                ax.axhline(oracle[s]["reward"], color=COLOR["oracle"], ls=":", lw=1.4, label=LABEL["oracle"])
            ax.axhline(0, color=INK_2, lw=0.6)
            if ri == 0:
                ax.set_title(f"{n} slices")
            if ci == 0:
                ax.set_ylabel(f"{level.capitalize()}\nReward")
            if ri == 2:
                ax.set_xlabel("Time step")
    axes[0, 0].set_ylim(-0.6, 1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Convergence under light, medium and heavy traffic (mean of 5 seeds, band = 1 s.d.)", y=1.055, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_convergence.png"))
    plt.close(fig)


def steps_to_converge(curve_steps, curve_reward, final_window=20, frac=0.9) -> float:
    """First step at which the smoothed reward reaches 90% of its final level.

    Measured relative to the curve's own start, so it is well defined for
    curves that begin negative. This is how "stabilises around 500 steps" in
    the paper is read quantitatively.
    """
    y = smooth(np.array(curve_reward, dtype=float), 10)
    final = float(np.mean(y[-final_window:]))
    start = float(y[0])
    if final <= start:
        return float("nan")
    thresh = start + frac * (final - start)
    hit = np.flatnonzero(y >= thresh)
    return float(curve_steps[hit[0]]) if hit.size else float("nan")


# ---------------------------------------------------------------------------
# Figure 6 - reward CDF
# ---------------------------------------------------------------------------


def fig6(runs) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ["released", "wo_both", "wo_gat", "wo_trans", "eexapp", "eexapp_plus", "gnn"]:
        samples = [x for s in runs.get(m, {}) for r in runs[m][s] for x in r["eval"]["reward_samples"]]
        if not samples:
            continue
        x = np.sort(np.clip(samples, -1.0, None))
        y = np.arange(1, len(x) + 1) / len(x)
        ls = "--" if m == "released" else "-"
        ax.plot(x, y, ls, color=COLOR[m], lw=2, label=LABEL[m])
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("Per-step reward (evaluation, all nine scenarios)")
    ax.set_ylabel("CDF")
    ax.set_title("Reward CDF - paper ablations and ours")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_reward_cdf.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 - QoS violations
# ---------------------------------------------------------------------------

FIG7_METHODS = ["eexapp", "wo_trans", "wo_gat", "wo_both", "eexapp_plus", "gnn"]


def fig7(runs) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    n_m = len(FIG7_METHODS)
    width = 0.8 / n_m
    top = 0.0
    for ax, level in zip(axes, LEVELS):
        x = np.arange(len(SLICES))
        for i, m in enumerate(FIG7_METHODS):
            means, sds = [], []
            for n in SLICES:
                v = metric(runs, m, scen(level, n), "violation") * 100
                means.append(v.mean() if v.size else np.nan)
                sds.append(v.std() if v.size else 0.0)
            top = max(top, np.nanmax(np.array(means) + np.array(sds)))
            ax.bar(
                x - 0.4 + (i + 0.5) * width,
                means,
                width * 0.9,  # a surface gap between adjacent bars
                yerr=sds,
                color=COLOR[m],
                ecolor=INK_2,
                error_kw={"lw": 0.8, "capsize": 1.5},
                label=LABEL[m],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in SLICES])
        ax.set_xlabel("Number of slices")
        ax.set_title(level.capitalize())
    axes[0].set_ylabel("QoS violations (%)")
    axes[0].set_ylim(0, max(5.0, top * 1.15))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_qos_violation.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extra figures
# ---------------------------------------------------------------------------


def fig_energy(runs, oracle) -> None:
    methods = ["released", "eexapp", "eexapp_plus", "gnn"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(LEVELS))
    width = 0.8 / (len(methods) + 1)
    for i, m in enumerate(methods + ["oracle"]):
        vals = []
        for level in LEVELS:
            if m == "oracle":
                v = [oracle[scen(level, n)]["energy_saved"] for n in SLICES if scen(level, n) in oracle]
            else:
                v = [x for n in SLICES for x in metric(runs, m, scen(level, n), "energy_saved")]
            vals.append(100 * np.mean(v) if v else np.nan)
        ax.bar(x - 0.4 + (i + 0.5) * width, vals, width * 0.9, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([lv.capitalize() for lv in LEVELS])
    ax.set_ylabel("RU energy saved (%)")
    ax.set_title("Energy saved under the RU power model (not the sleep-ratio proxy)")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_energy_saved.png"))
    plt.close(fig)


def slice_transfer_table(runs) -> Dict[str, Dict]:
    """reward[method][train_I][test_I], averaged over traffic levels and seeds."""
    out: Dict[str, Dict] = {}
    for m in ["eexapp", "eexapp_plus", "gnn"]:
        mat = {}
        for tr in SLICES:
            row = {}
            for te in SLICES:
                vals = []
                for level in LEVELS:
                    for r in runs.get(m, {}).get(scen(level, tr), []):
                        if te == tr:
                            vals.append(r["eval"]["reward"])
                        elif "cross_eval" in r and str(te) in r["cross_eval"]:
                            vals.append(r["cross_eval"][str(te)]["reward"])
                row[te] = float(np.mean(vals)) if vals else float("nan")
            mat[tr] = row
        out[m] = mat
    return out


def fig_transfer(table) -> None:
    methods = [m for m in ["eexapp", "eexapp_plus", "gnn"] if m in table]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        mat = np.array([[table[m][tr][te] for te in SLICES] for tr in SLICES])
        im = ax.imshow(mat, cmap="Blues", vmin=-0.2, vmax=1.0)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SLICES)
        ax.set_yticks(range(3))
        ax.set_yticklabels(SLICES)
        ax.set_xlabel("Evaluated on (slices)")
        ax.set_title(LABEL[m], fontsize=9.5)
        ax.grid(False)
    axes[0].set_ylabel("Trained on (slices)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean reward")
    fig.suptitle("Zero-shot transfer across slice counts (diagonal = trained and tested on the same count)",
                 fontsize=10.5, y=1.02)
    fig.savefig(os.path.join(FIG_DIR, "fig_slice_transfer.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def fmt(v, sd=None, pct=False, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if pct:
        return f"{100 * v:.1f}" + (f" ± {100 * sd:.1f}" if sd is not None else "")
    return f"{v:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "")


def build_tables(runs, oracle) -> str:
    methods = ["released", "eexapp", "sasc", "wo_trans", "wo_gat", "wo_both", "eexapp_plus", "gnn"]
    lines: List[str] = []
    summary: Dict = {"per_scenario": {}, "overall": {}, "convergence": {}, "tests": {}}

    # Overall means
    lines += ["## Table A - Overall (mean over 9 scenarios x 5 seeds)", "",
              "| Method | Params | Reward | QoS violation (%) | Sleep ratio | Energy saved (%) | Reward / oracle |",
              "|---|---|---|---|---|---|---|"]
    oracle_mean = np.mean([oracle[s]["reward"] for s in oracle]) if oracle else np.nan
    for m in methods:
        recs = [r for s in runs.get(m, {}) for r in runs[m][s]]
        if not recs:
            continue
        rew = np.array([r["eval"]["reward"] for r in recs])
        vio = np.array([r["eval"]["violation"] for r in recs])
        slp = np.array([r["eval"]["sleep_ratio"] for r in recs])
        eng = np.array([r["eval"]["energy_saved"] for r in recs])
        params = recs[0].get("extra", {}).get("n_params", "their code")
        summary["overall"][m] = {"reward": rew.mean(), "violation": vio.mean(), "sleep": slp.mean(),
                                 "energy_saved": eng.mean(), "n_runs": len(recs)}
        lines.append(
            f"| {LABEL[m]} | {params if isinstance(params, str) else f'{params:,}'} | {fmt(rew.mean(), rew.std())} | "
            f"{fmt(vio.mean(), vio.std(), pct=True)} | {fmt(slp.mean(), digits=2)} | "
            f"{fmt(eng.mean(), pct=True)} | {fmt(rew.mean() / oracle_mean, digits=2) if oracle else '-'} |"
        )
    if oracle:
        o = list(oracle.values())
        lines.append(
            f"| {LABEL['oracle']} | - | {fmt(np.mean([x['reward'] for x in o]))} | "
            f"{fmt(np.mean([x['violation'] for x in o]), pct=True)} | {fmt(np.mean([x['sleep_ratio'] for x in o]), digits=2)} | "
            f"{fmt(np.mean([x['energy_saved'] for x in o]), pct=True)} | 1.00 |"
        )
    lines.append("")

    # Per-scenario reward
    lines += ["## Table B - Evaluation reward per scenario (mean ± s.d. over 5 seeds)", "",
              "| Scenario | " + " | ".join(LABEL[m] for m in methods) + " | Oracle |",
              "|---|" + "---|" * (len(methods) + 1)]
    for level in LEVELS:
        for n in SLICES:
            s = scen(level, n)
            cells = []
            for m in methods:
                v = metric(runs, m, s, "reward")
                cells.append(fmt(v.mean(), v.std(), digits=2) if v.size else "-")
            lines.append(f"| {s} | " + " | ".join(cells) + f" | {fmt(oracle[s]['reward'], digits=2) if s in oracle else '-'} |")
    lines.append("")

    # Per-scenario violations (Figure 7 table)
    lines += ["## Table C - QoS violation (%) per scenario - the data behind Figure 7", "",
              "| Scenario | " + " | ".join(LABEL[m] for m in methods) + " |",
              "|---|" + "---|" * len(methods)]
    for level in LEVELS:
        for n in SLICES:
            s = scen(level, n)
            cells = []
            for m in methods:
                v = metric(runs, m, s, "violation")
                cells.append(fmt(v.mean(), v.std(), pct=True) if v.size else "-")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")

    # Convergence speed (Figure 5 table)
    lines += ["## Table D - Steps to reach 90% of final training reward - the data behind Figure 5", "",
              "| Scenario | " + " | ".join(LABEL[m] for m in ["sasc", "eexapp", "gnn"]) + " |", "|---|---|---|---|"]
    for level in LEVELS:
        for n in SLICES:
            s = scen(level, n)
            cells = []
            for m in ["sasc", "eexapp", "gnn"]:
                vals = [steps_to_converge(r["curve"]["step"], r["curve"]["reward"]) for r in runs.get(m, {}).get(s, [])]
                vals = [v for v in vals if not np.isnan(v)]
                summary["convergence"].setdefault(m, {})[s] = vals
                cells.append(f"{np.mean(vals):.0f} ± {np.std(vals):.0f}" if vals else "-")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")

    # Significance: ours vs each baseline
    lines += ["## Table E - Ours vs baselines: reward difference and Welch t-test (per scenario, 5 seeds each)", "",
              "| Scenario | vs EExApp: diff (p) | vs EExApp+slice: diff (p) | vs SASC: diff (p) |", "|---|---|---|---|"]
    for level in LEVELS:
        for n in SLICES:
            s = scen(level, n)
            g = metric(runs, "gnn", s, "reward")
            cells = []
            for base in ["eexapp", "eexapp_plus", "sasc"]:
                b = metric(runs, base, s, "reward")
                if g.size >= 2 and b.size >= 2:
                    t = stats.ttest_ind(g, b, equal_var=False)
                    p = float(t.pvalue) if np.isfinite(t.pvalue) else 1.0
                    summary["tests"].setdefault(base, {})[s] = {"diff": float(g.mean() - b.mean()), "p": p}
                    star = " *" if p < 0.05 else ""
                    cells.append(f"{g.mean() - b.mean():+.3f} ({p:.3f}){star}")
                else:
                    cells.append("-")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "`*` = p < 0.05. With 5 seeds per cell these tests have limited power; a non-significant "
              "difference is not evidence of equivalence.", ""]

    # Pooled test across all scenarios
    for base in ["eexapp", "eexapp_plus", "sasc"]:
        diffs = []
        for level in LEVELS:
            for n in SLICES:
                s = scen(level, n)
                g, b = metric(runs, "gnn", s, "reward"), metric(runs, base, s, "reward")
                if g.size and b.size:
                    diffs.append(g.mean() - b.mean())
        if len(diffs) >= 3:
            w = stats.wilcoxon(diffs) if np.any(np.array(diffs) != 0) else None
            summary["tests"].setdefault("pooled", {})[base] = {
                "mean_diff": float(np.mean(diffs)),
                "wins": int(np.sum(np.array(diffs) > 0)),
                "n": len(diffs),
                "wilcoxon_p": float(w.pvalue) if w is not None else 1.0,
            }
    if "pooled" in summary["tests"]:
        lines += ["## Table F - Pooled over the nine scenarios (Wilcoxon signed-rank on per-scenario means)", "",
                  "| Ours vs | Mean reward difference | Scenarios where ours is higher | Wilcoxon p |", "|---|---|---|---|"]
        for base, v in summary["tests"]["pooled"].items():
            lines.append(f"| {LABEL[base]} | {v['mean_diff']:+.3f} | {v['wins']} / {v['n']} | {v['wilcoxon_p']:.3f} |")
        lines.append("")

    # Slice transfer
    tt = slice_transfer_table(runs)
    summary["slice_transfer"] = tt
    lines += ["## Table G - Zero-shot transfer across slice counts (mean reward, averaged over traffic levels)", "",
              "| Method | Trained on | Tested on 2 | Tested on 4 | Tested on 8 |", "|---|---|---|---|---|"]
    for m, mat in tt.items():
        for tr in SLICES:
            lines.append(f"| {LABEL[m]} | {tr} | " + " | ".join(fmt(mat[tr][te], digits=3) for te in SLICES) + " |")
    lines.append("")

    # GAT attention learned
    lines += ["## Table H - Learned bipartite-GAT attention (EExApp, corrected), mean over all runs", "",
              "Rows are target actors, columns the critic each attends to.", "",
              "| Actor | attends to EE critic | attends to RS critic |", "|---|---|---|"]
    g_all = [r["eval"]["gat_gamma"] for s in runs.get("eexapp", {}) for r in runs["eexapp"][s] if "gat_gamma" in r["eval"]]
    if g_all:
        g = np.mean(np.array(g_all), axis=0)
        lines.append(f"| EE actor | {g[0][0]:.3f} | {g[0][1]:.3f} |")
        lines.append(f"| RS actor | {g[1][0]:.3f} | {g[1][1]:.3f} |")
        summary["gat_gamma"] = g.tolist()
    lines.append("")

    # Released code facts
    rel = [r for s in runs.get("released", {}) for r in runs["released"][s]]
    if rel:
        mb = max(r["eval"].get("max_b_chosen", 0) for r in rel)
        complete = all(r["extra"]["steps_completed"] == 2000 for r in rel)
        lines += ["## Table I - The released code on our simulator", "",
                  f"- Runs: {len(rel)}; all completed their 2000 steps: {complete}",
                  f"- Largest sleep length b ever chosen, in any run, training or evaluation: {mb} of 14 DL slots",
                  f"- Best mean evaluation reward in any scenario: "
                  f"{max(metric(runs, 'released', scen(l, n), 'reward').mean() for l in LEVELS for n in SLICES):.3f}",
                  "- Ceiling implied by b <= 1: 1/14 = 0.071. The paper's Figure 5 plateaus near 0.70.", ""]
        summary["released"] = {"max_b": mb, "all_complete": complete, "n": len(rel)}

    with open(os.path.join(FIG_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return "\n".join(lines)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    oracle = load_oracle()
    n = sum(len(v) for m in runs.values() for v in m.values())
    print(f"loaded {n} runs across {len(runs)} methods; oracle for {len(oracle)} scenarios")
    fig5(runs, oracle)
    fig6(runs)
    fig7(runs)
    if oracle:
        fig_energy(runs, oracle)
    fig_transfer(slice_transfer_table(runs))
    tables = build_tables(runs, oracle)
    with open(os.path.join(FIG_DIR, "summary_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("# Comparison results\n\nGenerated by experiments/plot_comparison.py.\n\n" + tables)
    print(tables[:3000])


if __name__ == "__main__":
    main()
