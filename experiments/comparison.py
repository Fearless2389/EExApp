"""
comparison.py - Train and evaluate every method on the paper's nine scenarios.
==============================================================================

This is the Tier 1 experiment: every method re-run inside one simulator, on
the same scenarios, seeds, reward and evaluation protocol. It produces the
data behind our versions of the paper's Figures 5, 6 and 7.

Methods
-------
  released     the authors' own released code, unmodified (agents/released.py)
  eexapp       EExApp as the paper describes it, defects corrected
  sasc         single-actor-single-critic (the paper's Fig. 5 baseline)
  wo_trans     EExApp with the Transformer replaced by an MLP   (Fig. 6/7)
  wo_gat       EExApp without the critic GAT                     (Fig. 6/7)
  wo_both      both removed                                      (Fig. 6/7)
  eexapp_plus  EExApp's Transformer given slice identity as features
  gnn          ours: heterogeneous GNN encoder, parameter-matched

Every learned method except ``released`` shares one implementation
(agents/policy.py, agents/ppo.py); they differ only in their PolicySpec.

Zero-shot slice generalisation
------------------------------
For ``eexapp``, ``eexapp_plus`` and ``gnn``, the policy trained on I slices is
also evaluated, without retraining, on the other two slice counts at the same
traffic level. The paper's architecture has no mechanism for this; whether a
policy survives it is claim C3 in the single-RU setting.

Usage
-----
    python -m experiments.comparison                     # everything, 5 seeds
    python -m experiments.comparison --methods gnn eexapp --seeds 3
    python -m experiments.comparison --workers 6

Results are written per run to experiments/results/runs/<method>/<scenario>/
seed<k>.json. Runs whose output already exists are skipped, so an interrupted
sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")

METHODS = ["released", "eexapp", "sasc", "wo_trans", "wo_gat", "wo_both", "eexapp_plus", "gnn"]
CROSS_EVAL_METHODS = {"eexapp", "eexapp_plus", "gnn"}


def method_spec(name: str):
    from agents.policy import PolicySpec

    return {
        "eexapp": PolicySpec(encoder="transformer", dual=True, gat=True),
        "sasc": PolicySpec(encoder="transformer", dual=False, gat=False),
        "wo_trans": PolicySpec(encoder="mlp", dual=True, gat=True),
        "wo_gat": PolicySpec(encoder="transformer", dual=True, gat=False),
        "wo_both": PolicySpec(encoder="mlp", dual=True, gat=False),
        "eexapp_plus": PolicySpec(encoder="transformer_plus", dual=True, gat=True),
        # d=28 gives 98k parameters against the Transformer's 105k.
        "gnn": PolicySpec(encoder="gnn", dual=True, gat=True, d_model=28),
    }[name]


def run_path(method: str, scenario: str, seed: int) -> str:
    return os.path.join(RUNS_DIR, method, scenario, f"seed{seed}.json")


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def run_job(job: dict) -> dict:
    """Train + evaluate one (method, scenario, seed). Writes its JSON and returns a summary."""
    method, level, n_slices, seed, steps = job["method"], job["level"], job["n_slices"], job["seed"], job["steps"]
    from sim import ScenarioConfig

    scenario = ScenarioConfig(name=f"{level}_{n_slices}slice", traffic_level=level, n_slices=n_slices, n_ue=8, n_ru=1)
    out_path = run_path(method, scenario.name, seed)
    t0 = time.perf_counter()
    try:
        if method == "released":
            from agents.released import run_released

            res = run_released(scenario, seed, total_steps=steps)
            record = {"curve": res["curve"], "eval": res["eval"], "extra": {
                "steps_completed": res["steps_completed"],
                "max_b_chosen_in_training": res["max_b_chosen_in_training"],
            }}
        else:
            from agents.ppo import PPOConfig, evaluate, train

            cfg = PPOConfig(total_steps=steps)
            policy, log = train(method_spec(method), scenario, seed, cfg)
            record = {"curve": log["curve"], "eval": evaluate(policy, scenario, cfg.eval_seeds, cfg.eval_steps),
                      "extra": {"n_params": log["n_params"], "train_seconds": log["train_seconds"]}}
            if method in CROSS_EVAL_METHODS:
                record["cross_eval"] = {}
                for other in (2, 4, 8):
                    if other == n_slices:
                        continue
                    target = replace(scenario, name=f"{level}_{other}slice", n_slices=other)
                    ev = evaluate(policy, target, cfg.eval_seeds, cfg.eval_steps)
                    ev.pop("reward_samples")
                    record["cross_eval"][str(other)] = ev
        record.update({"method": method, "scenario": scenario.name, "traffic_level": level,
                       "n_slices": n_slices, "seed": seed, "wall_seconds": time.perf_counter() - t0})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(record, fh)
        os.replace(tmp, out_path)
        return {"ok": True, "job": job, "reward": record["eval"]["reward"], "seconds": record["wall_seconds"]}
    except Exception:
        return {"ok": False, "job": job, "error": traceback.format_exc()}


def build_jobs(methods, seeds, steps):
    jobs = []
    for method in methods:
        for level in ("light", "medium", "heavy"):
            for n_slices in (2, 4, 8):
                for seed in range(seeds):
                    scenario = f"{level}_{n_slices}slice"
                    if os.path.exists(run_path(method, scenario, seed)):
                        continue
                    jobs.append({"method": method, "level": level, "n_slices": n_slices, "seed": seed, "steps": steps})
    # Slowest first so the tail of the sweep is not one long job.
    order = {"released": 0, "gnn": 1, "eexapp_plus": 2}
    return sorted(jobs, key=lambda j: order.get(j["method"], 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    jobs = build_jobs(args.methods, args.seeds, args.steps)
    print(f"{len(jobs)} runs to do on {args.workers} workers", flush=True)
    failures = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            j = r["job"]
            tag = f"{j['method']:>12} {j['level']:>6} {j['n_slices']}sl s{j['seed']}"
            if r["ok"]:
                print(f"[{i:>4}/{len(jobs)}] {tag}  reward={r['reward']:+.3f}  {r['seconds']:5.0f}s", flush=True)
            else:
                failures += 1
                print(f"[{i:>4}/{len(jobs)}] {tag}  FAILED\n{r['error']}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, {failures} failures", flush=True)


if __name__ == "__main__":
    main()
