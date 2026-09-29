"""
metrics.py - Reward and evaluation metrics.
===========================================

Two things live here: the reward the agent optimises, and the metrics the
report plots. They are kept apart on purpose - a metric used for evaluation
should not quietly become part of the training signal.

QoS violation ratio
-------------------
The baseline plots this in Figure 7 but never defines it. We define it
explicitly as the fraction of (UE, step) pairs that miss their throughput
requirement OR exceed their delay target.

The throughput requirement is ``min(slice target, offered load)``, not the
slice target alone. Measuring against the bare target makes a lightly-loaded
UE look permanently in violation - a UE offering 0.5 Mbps into a 5 Mbps eMBB
slice can never reach 5 Mbps no matter how well it is served - which inverts
the expected relationship between load and violations. QoS is satisfaction of
demand. Every violation number we report depends on this choice, so it is
stated in the report rather than buried here.

Reward
------
Follows the paper's Lagrangian relaxation, with three corrections and one
addition:

  * energy comes from the power model, not the ``b_t / N_sf`` proxy;
  * throughput and delay are compared in matching physical units - the
    released baseline compares normalised values against raw targets, which
    pins the throughput penalty near its maximum and makes the delay penalty
    identically zero;
  * the delay clip ``min(d, 2*D)`` from the paper is applied;
  * a coverage-continuity term, which is identically zero for a single RU and
    binding once cells overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .params import RewardWeights, SliceSpec
from .scheduler import StepTelemetry


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


def effective_targets_mbps(targets_mbps: np.ndarray, offered_mbps: np.ndarray) -> np.ndarray:
    """The throughput a UE actually needs: its demand, capped by its slice target.

    ``offered_mbps`` is the demand realised in the same window the throughput
    was measured over, so short-window arrival noise cancels.

    Comparing achieved throughput against an absolute slice target punishes a
    UE that simply has little to send - a 0.5 Mbps UE in a 5 Mbps eMBB slice
    can never reach 5 Mbps however well it is served. QoS is satisfaction of
    demand, so the requirement is ``min(slice target, offered load)``.

    This is a definitional choice the baseline never states; it is recorded in
    the report because every violation number depends on it.
    """
    return np.minimum(targets_mbps, offered_mbps)


def throughput_violation(throughput_mbps: np.ndarray, targets_mbps: np.ndarray) -> np.ndarray:
    """Per-UE normalised throughput shortfall, ``max(0, 1 - q/Q)``, in [0, 1].

    A UE with no demand in the window (``Q == 0``) cannot be under-served and
    contributes zero. This case is common, not an edge case: a 0.1 Mbps UE
    generates less than one packet per 100 ms window, so many windows see no
    arrivals at all. Without this guard those windows each count as a full
    violation and put a constant penalty on light traffic that no policy can
    remove.
    """
    q = np.asarray(throughput_mbps, dtype=float)
    t = np.asarray(targets_mbps, dtype=float)
    has_demand = t > 1e-9
    shortfall = np.maximum(0.0, 1.0 - q / np.where(has_demand, t, 1.0))
    return np.where(has_demand, shortfall, 0.0)


def delay_violation(
    delay_ms: np.ndarray, targets_ms: np.ndarray, clip_factor: float = 2.0
) -> np.ndarray:
    """Per-UE normalised delay excess, ``max(0, d_hat/D - 1)``.

    ``d_hat = min(d, clip_factor * D)`` reproduces the paper's guard against a
    handful of very large delay samples dominating the objective.
    """
    d_hat = np.minimum(delay_ms, clip_factor * targets_ms)
    return np.maximum(0.0, d_hat / np.maximum(targets_ms, 1e-9) - 1.0)


def qos_violation_ratio(
    throughput_mbps: np.ndarray,
    delay_ms: np.ndarray,
    targets_mbps: np.ndarray,
    targets_ms: np.ndarray,
    tolerance: float = 0.95,
) -> float:
    """Fraction of UEs failing either QoS constraint in this step.

    This is the Figure 7 metric. Averaged over steps it gives the violation
    ratio for a scenario.
    """
    if len(throughput_mbps) == 0:
        return 0.0
    bad = (throughput_mbps < tolerance * targets_mbps) | (delay_ms > targets_ms)
    return float(bad.mean())


def per_slice_violation_ratio(
    throughput_mbps: np.ndarray,
    delay_ms: np.ndarray,
    targets_mbps: np.ndarray,
    targets_ms: np.ndarray,
    slice_ids: np.ndarray,
    n_slices: int,
    tolerance: float = 0.95,
) -> np.ndarray:
    """Violation ratio broken out per slice, for the per-slice plots."""
    bad = (throughput_mbps < tolerance * targets_mbps) | (delay_ms > targets_ms)
    out = np.zeros(n_slices)
    for s in range(n_slices):
        mask = slice_ids == s
        out[s] = float(bad[mask].mean()) if mask.any() else 0.0
    return out


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


def energy_saved_fraction(telemetry: StepTelemetry) -> float:
    """Network-wide energy saved against an always-active deployment, in [0, 1].

    Aggregated over RUs rather than averaged per RU, so a large cell that
    cannot sleep is not masked by a small idle one.
    """
    baseline = telemetry.baseline_energy_j.sum()
    if baseline <= 0:
        return 0.0
    return float(1.0 - telemetry.energy_j.sum() / baseline)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """The reward and every term that went into it, for logging and ablation."""

    total: float
    energy_term: float
    throughput_penalty: float
    delay_penalty: float
    coverage_penalty: float

    # The baseline's decomposition, kept so the dual-actor split still works.
    r_energy: float
    r_qos: float

    def as_dict(self) -> dict:
        return {
            "reward": self.total,
            "energy_term": self.energy_term,
            "throughput_penalty": self.throughput_penalty,
            "delay_penalty": self.delay_penalty,
            "coverage_penalty": self.coverage_penalty,
            "r_energy": self.r_energy,
            "r_qos": self.r_qos,
        }


def compute_reward(
    telemetry: StepTelemetry,
    slice_ids: np.ndarray,
    slices: List[SliceSpec],
    weights: RewardWeights,
) -> RewardBreakdown:
    """Evaluate the Lagrangian reward for one step.

    Returns both the scalar total and the decomposition into an energy part and
    a QoS part, because the dual-actor architecture trains its two actors on
    those two components separately.
    """
    targets_mbps = effective_targets_mbps(
        np.array([slices[s].throughput_target_mbps for s in slice_ids]),
        telemetry.offered_mbps,
    )
    targets_ms = np.array([slices[s].delay_target_ms for s in slice_ids])

    # Energy term: the paper's sleep ratio b_t / N_ts (averaged over RUs), or
    # the power-model saving. See RewardWeights.energy_metric.
    if weights.energy_metric == "sleep_ratio":
        e_term = float(np.mean(telemetry.sleep_ratio))
    elif weights.energy_metric == "power_model":
        e_term = energy_saved_fraction(telemetry)
    else:
        raise ValueError(f"unknown energy_metric {weights.energy_metric!r}")

    # Penalties are summed over UEs as in the paper, then rescaled to the
    # paper's 8-UE reference so the energy/QoS balance does not drift with
    # network size. For an 8-UE network the scale factor is exactly 1.
    n_ue = max(len(telemetry.throughput_mbps), 1)
    scale = weights.reference_n_ue / n_ue

    thr_pen = scale * float(throughput_violation(telemetry.throughput_mbps, targets_mbps).sum())
    del_pen = scale * float(
        delay_violation(telemetry.delay_ms, targets_ms, weights.delay_clip_factor).sum()
    )
    cov_pen = scale * float(np.asarray(telemetry.uncovered_fraction).sum())

    energy_term = weights.w_energy * e_term
    r_energy = energy_term
    r_qos = -(
        weights.lambda_throughput * thr_pen
        + weights.lambda_delay * del_pen
        + weights.lambda_coverage * cov_pen
    )

    return RewardBreakdown(
        total=r_energy + r_qos,
        energy_term=energy_term,
        throughput_penalty=weights.lambda_throughput * thr_pen,
        delay_penalty=weights.lambda_delay * del_pen,
        coverage_penalty=weights.lambda_coverage * cov_pen,
        r_energy=r_energy,
        r_qos=r_qos,
    )
