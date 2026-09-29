"""
ppo.py - PPO training and evaluation for every method in the comparison.
========================================================================

Dual-actor-dual-critic PPO as the paper describes it, with the defects found in
the released implementation corrected:

  * the encoder and the critic GAT are inside the optimisation graph and are
    trained end to end (the released code gives them no optimiser and computes
    them under no_grad, so they stay at random initialisation);
  * the advantage baseline is a trained value function (the released code
    uses the untrained GAT's output);
  * action log-probabilities are exact (see policy.py).

Training is one continuing episode, as a deployed xApp experiences it: the
network is never reset, and the x-axis of every learning curve is simply the
number of decisions taken - the same axis as the paper's Figure 5.

Advantages
----------
The reward is r = r_energy + r_qos.

  dual, with GAT     actor j uses baseline V_hat_j from the GAT
  dual, without GAT  both actors use V_alpha + V_beta
  SASC               one actor, one critic V, one joint ratio

In the dual variants the two critics are also regressed onto the energy and
QoS returns separately, which is the paper's decomposition.

Multi-RU
--------
Every RU is an agent. Actions and log-probabilities are per RU; advantages are
shared (one team reward) and each agent's ratio is clipped separately, as in
MAPPO. With one RU this reduces exactly to single-agent PPO.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sim import OranSimEnv, ScenarioConfig

from .obs import build_obs, collate
from .policy import Policy, PolicySpec, count_parameters


@dataclass
class PPOConfig:
    total_steps: int = 2000  # the paper's Figure 5 x-axis
    rollout: int = 64
    epochs: int = 8
    minibatch: int = 32
    gamma: float = 0.9
    lam: float = 0.9
    clip: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_encoder: float = 3e-4
    ent_sleep: float = 0.01
    ent_slice: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    reward_scale: float = 0.1  # learning only; every reported reward is unscaled
    eval_steps: int = 150
    eval_seeds: Tuple[int, ...] = (1000, 1001, 1002)  # shared with fixed_oracle.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gae(rewards: np.ndarray, values: np.ndarray, last_value: float, gamma: float, lam: float):
    """Generalised advantage estimation for one continuing trajectory."""
    t_len = len(rewards)
    adv = np.zeros(t_len, dtype=np.float32)
    nxt_v, running = last_value, 0.0
    for t in reversed(range(t_len)):
        delta = rewards[t] + gamma * nxt_v - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        nxt_v = values[t]
    return adv, adv + values


def to_env_action(a: torch.Tensor, b: torch.Tensor, sl: torch.Tensor, n_dl: int) -> dict:
    """Policy outputs for one step (batch of 1) -> the simulator's action dict."""
    a = a[0].cpu().numpy()
    b = b[0].cpu().numpy()
    return {
        "sleep": np.stack([a, b, n_dl - a - b], axis=-1).astype(int),
        "slices": sl[0].cpu().numpy().astype(float),
    }


def _snapshot(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.reshape(-1)[0]) for k, v in values.items()}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def ppo_update(policy: Policy, opt, batch: Dict, cfg: PPOConfig) -> Dict[str, float]:
    """Run the PPO epochs over one rollout. ``batch`` holds tensors of length T."""
    spec = policy.spec
    t_len = batch["a"].shape[0]
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(t_len)
        epoch_kl = []
        for start in range(0, t_len, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            obs = {k: v[idx] for k, v in batch["obs"].items()}
            out = policy(obs)

            a, b, sl = batch["a"][idx], batch["b"][idx], batch["sl"][idx]
            lp_sleep = out.sleep.log_prob(a, b)  # [M, R]
            lp_slice = out.slices.log_prob(sl)  # [M, R]
            ent = cfg.ent_sleep * out.sleep.entropy(b).mean() + cfg.ent_slice * out.slices.entropy().mean()

            def clipped(ratio, adv):
                adv = adv[:, None]  # shared across agents
                return -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()

            if spec.dual:
                r_ee = torch.exp(lp_sleep - batch["lp_sleep"][idx])
                r_rs = torch.exp(lp_slice - batch["lp_slice"][idx])
                pi_loss = clipped(r_ee, batch["adv_ee"][idx]) + clipped(r_rs, batch["adv_rs"][idx])
                log_ratio = (lp_sleep - batch["lp_sleep"][idx]) + (lp_slice - batch["lp_slice"][idx])
                ratio_any = r_ee

                v = out.values
                v_loss = F.smooth_l1_loss(v["v_a"], batch["ret_a"][idx]) + F.smooth_l1_loss(
                    v["v_b"], batch["ret_b"][idx]
                )
                if spec.gat:
                    v_loss = v_loss + F.smooth_l1_loss(v["vhat_a"], batch["ret_hat_a"][idx])
                    v_loss = v_loss + F.smooth_l1_loss(v["vhat_b"], batch["ret_hat_b"][idx])
            else:
                lp = lp_sleep + lp_slice
                lp_old = batch["lp_sleep"][idx] + batch["lp_slice"][idx]
                ratio_any = torch.exp(lp - lp_old)
                pi_loss = clipped(ratio_any, batch["adv"][idx])
                log_ratio = lp - lp_old
                v_loss = F.smooth_l1_loss(out.values["v"], batch["ret"][idx])

            loss = pi_loss + cfg.vf_coef * v_loss - ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean())
                epoch_kl.append(approx_kl)
                stats["pi_loss"] += float(pi_loss)
                stats["v_loss"] += float(v_loss)
                stats["kl"] += approx_kl
                stats["clipfrac"] += float(((ratio_any - 1).abs() > cfg.clip).float().mean())
                stats["n"] += 1
        if np.mean(epoch_kl) > cfg.target_kl:
            break

    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}


def build_advantages(spec: PolicySpec, roll: Dict, last_vals: Dict[str, float], cfg: PPOConfig) -> Dict:
    """Turn a rollout's rewards and value snapshots into advantages and returns."""
    s = cfg.reward_scale
    r_e = np.array(roll["r_e"], dtype=np.float32) * s
    r_q = np.array(roll["r_q"], dtype=np.float32) * s
    r = r_e + r_q
    vals = {k: np.array([v[k] for v in roll["values"]], dtype=np.float32) for k in roll["values"][0]}
    out = {}

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    if spec.dual:
        _, out["ret_a"] = gae(r_e, vals["v_a"], last_vals["v_a"], cfg.gamma, cfg.lam)
        _, out["ret_b"] = gae(r_q, vals["v_b"], last_vals["v_b"], cfg.gamma, cfg.lam)
        adv_a, out["ret_hat_a"] = gae(r, vals["vhat_a"], last_vals["vhat_a"], cfg.gamma, cfg.lam)
        adv_b, out["ret_hat_b"] = gae(r, vals["vhat_b"], last_vals["vhat_b"], cfg.gamma, cfg.lam)
        out["adv_ee"], out["adv_rs"] = norm(adv_a), norm(adv_b)
    else:
        adv, out["ret"] = gae(r, vals["v"], last_vals["v"], cfg.gamma, cfg.lam)
        out["adv"] = norm(adv)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_optimizer(policy: Policy, cfg: PPOConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": policy.actor_parameters(), "lr": cfg.lr_actor},
            {"params": policy.critic_parameters(), "lr": cfg.lr_critic},
            {"params": policy.encoder_parameters(), "lr": cfg.lr_encoder},
        ]
    )


def train(
    spec: PolicySpec,
    scenario: ScenarioConfig,
    seed: int,
    cfg: Optional[PPOConfig] = None,
    log_every: int = 10,
) -> Tuple[Policy, Dict]:
    """Train one agent on one scenario. Returns the policy and a log dict.

    The log records the per-step reward of the *training* (stochastic) policy,
    averaged over blocks of ``log_every`` steps: this is what the paper's
    Figure 5 plots.
    """
    cfg = cfg or PPOConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = OranSimEnv(scenario, seed=seed)
    policy = Policy(spec, env.n_dl_slots)
    opt = make_optimizer(policy, cfg)

    ue_obs = env.reset()
    obs = build_obs(env, ue_obs)

    curve: Dict[str, List[float]] = {"step": [], "reward": [], "violation": [], "sleep_ratio": [], "energy_saved": []}
    block = {k: [] for k in ("reward", "violation", "sleep_ratio", "energy_saved")}
    update_stats: List[Dict] = []
    steps = 0
    t0 = time.perf_counter()

    n_updates = math.ceil(cfg.total_steps / cfg.rollout)
    for _u in range(n_updates):
        roll = {k: [] for k in ("obs", "a", "b", "sl", "lp_sleep", "lp_slice", "values", "r_e", "r_q")}
        for _t in range(cfg.rollout):
            if steps >= cfg.total_steps:
                break
            with torch.no_grad():
                out = policy(collate([obs]))
                a, b = out.sleep.sample()
                sl = out.slices.sample()
                lp_sleep = out.sleep.log_prob(a, b)
                lp_slice = out.slices.log_prob(sl)

            ue_obs, (r_e, r_q), _, info = env.step(to_env_action(a, b, sl, env.n_dl_slots))

            roll["obs"].append(obs)
            roll["a"].append(a[0])
            roll["b"].append(b[0])
            roll["sl"].append(sl[0])
            roll["lp_sleep"].append(lp_sleep[0])
            roll["lp_slice"].append(lp_slice[0])
            roll["values"].append(_snapshot(out.values))
            roll["r_e"].append(r_e)
            roll["r_q"].append(r_q)

            obs = build_obs(env, ue_obs)
            steps += 1

            block["reward"].append(r_e + r_q)
            block["violation"].append(info["qos_violation_ratio"])
            block["sleep_ratio"].append(info["mean_sleep_ratio"])
            block["energy_saved"].append(info["energy_saved_fraction"])
            if steps % log_every == 0:
                curve["step"].append(steps)
                for k in block:
                    curve[k].append(float(np.mean(block[k])))
                    block[k] = []

        if not roll["a"]:
            break

        with torch.no_grad():
            last_vals = _snapshot(policy(collate([obs])).values)
        adv = build_advantages(spec, roll, last_vals, cfg)
        batch = {
            "obs": collate(roll["obs"]),
            "a": torch.stack(roll["a"]),
            "b": torch.stack(roll["b"]),
            "sl": torch.stack(roll["sl"]),
            "lp_sleep": torch.stack(roll["lp_sleep"]),
            "lp_slice": torch.stack(roll["lp_slice"]),
            **adv,
        }
        update_stats.append(ppo_update(policy, opt, batch, cfg))

    log = {
        "curve": curve,
        "updates": update_stats,
        "train_seconds": time.perf_counter() - t0,
        "n_params": count_parameters(policy),
        "spec": asdict(spec),
        "ppo": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
    }
    return policy, log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    policy: Policy,
    scenario: ScenarioConfig,
    seeds=(1000, 1001, 1002),
    steps: int = 150,
    deterministic: bool = True,
) -> Dict:
    """Run a trained policy on held-out seeds and summarise it.

    Deterministic by default: the most likely sleep schedule and the mean
    slice split, which is how a trained policy would be deployed.
    """
    policy.eval()
    rewards, violations, energy, sleep, gammas = [], [], [], [], []
    per_slice = []
    for seed in seeds:
        env = OranSimEnv(scenario, seed=seed)
        obs = build_obs(env, env.reset())
        for _ in range(steps):
            out = policy(collate([obs]))
            if deterministic:
                a, b = out.sleep.mode()
                sl = out.slices.mode()
            else:
                a, b = out.sleep.sample()
                sl = out.slices.sample()
            ue_obs, (r_e, r_q), _, info = env.step(to_env_action(a, b, sl, env.n_dl_slots))
            rewards.append(r_e + r_q)
            violations.append(info["qos_violation_ratio"])
            energy.append(info["energy_saved_fraction"])
            sleep.append(info["mean_sleep_ratio"])
            per_slice.append(info["per_slice_violation"])
            if out.gamma is not None:
                gammas.append(out.gamma[0].numpy())
            obs = build_obs(env, ue_obs)
    policy.train()
    res = {
        "reward": float(np.mean(rewards)),
        "reward_samples": [float(x) for x in rewards],
        "violation": float(np.mean(violations)),
        "energy_saved": float(np.mean(energy)),
        "sleep_ratio": float(np.mean(sleep)),
        "per_slice_violation": np.mean(per_slice, axis=0).tolist(),
    }
    if gammas:
        res["gat_gamma"] = np.mean(gammas, axis=0).tolist()  # [target actor][source critic]
    return res
