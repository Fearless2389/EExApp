"""
released.py - Run the authors' released EExApp code against our simulator.
==========================================================================

This does not re-implement the baseline. It imports the authors' own modules
from ``algorithms/`` - their training loop (``ppo.ppo_mcma``), their GRU state
encoder, their bipartite GAT, their constrained actors - and runs them
unmodified. The only thing replaced is the environment: their
hardware-in-the-loop ``OranEnv`` is swapped for an adapter over our
simulator that speaks exactly the same interface.

So every behaviour of the released implementation is preserved, including the
defects documented in the analysis report:

  * the encoder and the GAT are never optimised;
  * the EE actor can only choose b in {0, 1};
  * the RS actor's log-probability ignores its softmax/clamp transform.

The reward is ours (the paper's Lagrangian form, correct units), because the
released environment's reward has a unit bug that zeroes the delay penalty; a
comparison on that reward would conflate two separate problems.

Unavoidable configuration changes, all confined to the authors' ``config``
object and all necessary for their code to address our scenarios:

  * ACTION_SPACE['ee']['sum'] = N_dl (their hardcoded 7 assumes mu=0);
  * ns_action_dim / num_slices = the scenario's slice count (theirs is fixed
    at 3, but the paper evaluates 2, 4 and 8);
  * for 8 slices, the per-slice minimum share is lowered from 20% to 6.25%,
    since 8 x 20% exceeds 100% and their clamp would force an equal split.

Their plotting module is replaced by a no-op stub: it needs pandas, seaborn
and plotly (none listed in their requirements.txt) and has no effect on
learning. Their per-step debug printing is silenced.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import types
from typing import Dict, List, Tuple

import numpy as np
import torch

from sim import OranSimEnv, ScenarioConfig

ALGO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithms")


# ---------------------------------------------------------------------------
# Environment adapter with the released OranEnv interface
# ---------------------------------------------------------------------------


class ReleasedEnvAdapter:
    """Presents our simulator through the released ``OranEnv`` interface.

    ``reset()`` returns a list of per-UE 17-dim arrays; ``step(action)`` takes
    ``[slice_1 .. slice_I (percent), a_t, b_t, c_t]`` and returns
    ``(state, (reward_ee, reward_ns), done, info)``.

    Their loop calls ``reset()`` every ``max_ep_len`` steps. A deployed xApp's
    network does not reset, and our other agents train on one continuing
    episode, so only the first ``reset()`` initialises the simulator; later
    calls return the current state.
    """

    def __init__(self, scenario: ScenarioConfig, seed: int):
        self.env = OranSimEnv(scenario, seed=seed)
        self.n_slices = scenario.n_slices
        self._obs = None
        self.log: Dict[str, List[float]] = {"reward": [], "violation": [], "sleep_ratio": [], "energy_saved": [], "b": []}

    def reset(self):
        if self._obs is None:
            self._obs = self.env.reset()
        return [row.copy() for row in self._obs]

    def step(self, action):
        action = np.asarray(action, dtype=float).ravel()
        shares = np.clip(action[: self.n_slices], 0.0, None)
        a, b, c = (int(round(x)) for x in action[self.n_slices : self.n_slices + 3])
        obs, (r_e, r_q), done, info = self.env.step({"sleep": [[a, b, c]], "slices": [shares / max(shares.sum(), 1e-9)]})
        self._obs = obs
        self.log["reward"].append(r_e + r_q)
        self.log["violation"].append(info["qos_violation_ratio"])
        self.log["sleep_ratio"].append(info["mean_sleep_ratio"])
        self.log["energy_saved"].append(info["energy_saved_fraction"])
        self.log["b"].append(b)
        return [row.copy() for row in obs], (r_e, r_q), False, info


# ---------------------------------------------------------------------------
# Importing the authors' modules
# ---------------------------------------------------------------------------


class _NoOpVisualizer:
    def __init__(self, *args, **kwargs):
        self.metrics = {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _import_released(n_dl: int, n_slices: int):
    """Import the released modules with their config patched for this scenario."""
    if ALGO_DIR not in sys.path:
        sys.path.insert(0, ALGO_DIR)

    stub = types.ModuleType("visualization")
    stub.ORANVisualizer = _NoOpVisualizer
    stub.calculate_network_metrics = lambda *a, **k: {}
    stub.calculate_action_metrics = lambda *a, **k: {}
    stub.calculate_multi_objective_metrics = lambda *a, **k: {}
    sys.modules["visualization"] = stub

    import warnings

    warnings.filterwarnings("ignore")
    with contextlib.redirect_stderr(io.StringIO()):
        import config as released_config  # noqa: E402  (the authors' config.py)
        import mcma_ppo  # noqa: E402
        import ppo as released_ppo  # noqa: E402

    cfg = released_config.config
    cfg.ACTION_SPACE["ee"]["sum"] = n_dl
    cfg.ACTION_SPACE["ee"]["max"] = n_dl
    cfg.ACTOR_CRITIC["ns_action_dim"] = n_slices
    cfg.ACTOR_CRITIC["num_slices"] = n_slices
    cfg.ENV["num_slices"] = n_slices
    cfg.ACTION_SPACE["ns"]["min"] = min(20.0, 50.0 / n_slices)
    released_ppo.logger.setLevel(logging.CRITICAL)
    return released_ppo, mcma_ppo


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------


def run_released(
    scenario: ScenarioConfig,
    seed: int,
    total_steps: int = 2000,
    eval_seeds=(1000, 1001, 1002),
    eval_steps: int = 150,
    log_every: int = 10,
) -> Dict:
    """Train the released EExApp on one scenario and evaluate it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = OranSimEnv(scenario, seed=seed)
    released_ppo, mcma_ppo = _import_released(probe.n_dl_slots, scenario.n_slices)

    adapter = ReleasedEnvAdapter(scenario, seed)
    captured = []

    def ac_factory(*args, **kwargs):
        ac = mcma_ppo.MCMA_ActorCritic(*args, **kwargs)
        captured.append(ac)
        return ac

    steps_per_epoch = 100
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        released_ppo.ppo_mcma(
            env_fn=lambda: adapter,
            actor_critic=ac_factory,
            ac_kwargs=dict(hidden_sizes=[64, 64]),
            steps_per_epoch=steps_per_epoch,
            epochs=max(1, total_steps // steps_per_epoch),
            max_ep_len=steps_per_epoch,
            save_freq=10_000,
            device=torch.device("cpu"),
            save_dir=tmp,
        )
    ac = captured[0]

    # Their training loop wraps each step in `except Exception: continue`, so a
    # failure would silently shorten training. Check every step happened.
    n_done = len(adapter.log["reward"])

    # Learning curve, block-averaged exactly like agents/ppo.py.
    curve = {"step": [], "reward": [], "violation": [], "sleep_ratio": [], "energy_saved": []}
    for start in range(0, n_done - log_every + 1, log_every):
        curve["step"].append(start + log_every)
        for k in ("reward", "violation", "sleep_ratio", "energy_saved"):
            curve[k].append(float(np.mean(adapter.log[k][start : start + log_every])))

    # Evaluation. The released actor has no deterministic mode, so it is
    # evaluated as it would run: sampling from its policy.
    rewards, violations, energy, sleep, per_slice, bs = [], [], [], [], [], []
    for es in eval_seeds:
        ev = ReleasedEnvAdapter(scenario, es)
        o = ev.reset()
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(eval_steps):
                out = ac.step(o)
                act = np.concatenate([out["ns_action"].numpy(), out["ee_action"].numpy()])
                o, _, _, info = ev.step(act)
                per_slice.append(info["per_slice_violation"])
        rewards += ev.log["reward"]
        violations += ev.log["violation"]
        energy += ev.log["energy_saved"]
        sleep += ev.log["sleep_ratio"]
        bs += ev.log["b"]

    return {
        "curve": curve,
        "steps_completed": n_done,
        "steps_requested": total_steps,
        "max_b_chosen_in_training": int(max(adapter.log["b"])) if adapter.log["b"] else None,
        "eval": {
            "reward": float(np.mean(rewards)),
            "reward_samples": [float(x) for x in rewards],
            "violation": float(np.mean(violations)),
            "energy_saved": float(np.mean(energy)),
            "sleep_ratio": float(np.mean(sleep)),
            "per_slice_violation": np.mean(per_slice, axis=0).tolist(),
            "max_b_chosen": int(max(bs)),
        },
    }
