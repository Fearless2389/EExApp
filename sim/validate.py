"""
validate.py - Sanity checks for the simulator.
==============================================

These are not unit tests of code paths; they are checks that the *physics*
behaves. A simulator that runs without raising but produces impossible numbers
is worse than one that crashes, so every claim we make downstream depends on
these passing.

Run with:  python -m sim.validate

Checks
------
1. Observation contract: shape, range, feature order.
2. Link budget: a UE at the cell edge has lower SINR than one at the centre,
   and peak throughput is in the right order of magnitude for 40 MHz.
3. Sleep monotonicity: more sleep slots -> more energy saved, less throughput.
4. Energy accounting: never negative, never above the always-active baseline
   by more than the transition cost.
5. Delay responds to load: heavy traffic produces higher delay than light.
6. Coverage term: identically zero with one RU, positive when two RUs both
   sleep with overlapping coverage.
7. Action repair: invalid sleep partitions are repaired and counted.
8. Determinism: same seed gives the same trajectory.
"""

from __future__ import annotations

import sys

import numpy as np

from .env import OranSimEnv
from .params import (
    N_OBS_FEATURES,
    OBS_FEATURE_NAMES,
    RADIO_DEFAULT,
    ScenarioConfig,
)
from .radio import bits_per_prb, downlink_sinr_db, rsrp_dbm, sinr_to_cqi, transmission_rank

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, PASS if condition else FAIL, detail))


def _run_fixed(env: OranSimEnv, sleep_b: int, n_steps: int = 12) -> dict:
    """Run a policy that always sleeps ``sleep_b`` slots, return mean metrics."""
    n_dl = env.n_dl_slots
    b = int(np.clip(sleep_b, 0, n_dl))
    a = (n_dl - b) // 2
    c = n_dl - b - a
    action = {
        "sleep": np.tile([a, b, c], (env.n_ru, 1)),
        "slices": np.full((env.n_ru, env.n_slices), 1.0 / env.n_slices),
    }
    env.reset()
    acc = {"energy_saved": [], "throughput": [], "delay": [], "violation": [], "uncovered": []}
    for _ in range(n_steps):
        _, _, _, info = env.step(action)
        acc["energy_saved"].append(info["energy_saved_fraction"])
        acc["throughput"].append(float(np.mean(info["throughput_mbps"])))
        acc["delay"].append(float(np.mean(info["delay_ms"])))
        acc["violation"].append(info["qos_violation_ratio"])
        acc["uncovered"].append(float(np.sum(info["uncovered_fraction"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------


def test_observation_contract() -> None:
    env = OranSimEnv(ScenarioConfig(name="obs", n_ue=8, n_slices=3, n_ru=1, seed=1))
    obs = env.reset()
    check("obs shape is [n_ue, 17]", obs.shape == (8, N_OBS_FEATURES), str(obs.shape))
    check("obs is normalised to [0,1]", bool(obs.min() >= 0.0 and obs.max() <= 1.0),
          f"min={obs.min():.3f} max={obs.max():.3f}")
    check("obs has no NaN", bool(np.isfinite(obs).all()))
    check("feature order matches baseline", OBS_FEATURE_NAMES[0] == "dl_curr_tbs"
          and OBS_FEATURE_NAMES[9] == "ul_bler" and OBS_FEATURE_NAMES[-1] == "prb_tot_ul")


def test_link_budget() -> None:
    radio = RADIO_DEFAULT
    near = rsrp_dbm(np.array([20.0]), radio)
    far = rsrp_dbm(np.array([300.0]), radio)
    check("RSRP falls with distance", bool(near[0] > far[0]), f"{near[0]:.1f} vs {far[0]:.1f} dBm")

    sinr_near = downlink_sinr_db(near, np.empty((1, 0)), radio)
    sinr_far = downlink_sinr_db(far, np.empty((1, 0)), radio)
    check("SINR falls with distance", bool(sinr_near[0] > sinr_far[0]),
          f"{sinr_near[0]:.1f} vs {sinr_far[0]:.1f} dB")

    # Peak throughput for 106 PRBs at CQI 15 over 14 of 20 slots. A single
    # layer should give ~120 Mbps for 40 MHz; four layers roughly four times
    # that, which is the right order for a 4T4R n78 cell.
    slots_per_s = 2000 * (14 / 20)  # mu=1 -> 2000 slots/s, 14 DL of every 20
    peak_1 = radio.n_prb * float(bits_per_prb(np.array([15]), 1)[0]) * slots_per_s / 1e6
    peak_r = (
        radio.n_prb * float(bits_per_prb(np.array([15]), radio.mimo_layers)[0]) * slots_per_s / 1e6
    )
    check("single-layer peak in 50-250 Mbps for 40 MHz", 50.0 < peak_1 < 250.0, f"{peak_1:.1f} Mbps")
    check("4-layer peak in 300-700 Mbps for 40 MHz 4T4R", 300.0 < peak_r < 700.0, f"{peak_r:.1f} Mbps")

    rank_near = int(transmission_rank(sinr_near, radio)[0])
    rank_edge = int(transmission_rank(np.array([5.0]), radio)[0])
    check("rank adapts to SINR", rank_near == radio.mimo_layers and rank_edge == 1,
          f"near={rank_near} edge(5dB)={rank_edge}")

    cqi_near = int(sinr_to_cqi(sinr_near)[0])
    check("cell-centre UE reaches a high CQI", cqi_near >= 12, f"CQI {cqi_near}")


def test_sleep_monotonicity() -> None:
    # Heavy traffic and near-total sleep: with 4-layer MIMO the cell has enough
    # headroom that lighter settings never violate, which would let the
    # throughput and violation checks below pass without testing anything.
    cfg = ScenarioConfig(name="sleep", traffic_level="heavy", n_ue=8, n_slices=3, n_ru=1, seed=7)
    none_ = _run_fixed(OranSimEnv(cfg), 0)
    half = _run_fixed(OranSimEnv(cfg), cfg.numerology.n_dl_slots // 2)
    most = _run_fixed(OranSimEnv(cfg), cfg.numerology.n_dl_slots - 1)

    check("more sleep saves more energy",
          none_["energy_saved"] < half["energy_saved"] < most["energy_saved"],
          f"{none_['energy_saved']:.3f} -> {half['energy_saved']:.3f} -> {most['energy_saved']:.3f}")
    check("more sleep lowers throughput",
          none_["throughput"] >= half["throughput"] >= most["throughput"],
          f"{none_['throughput']:.2f} -> {half['throughput']:.2f} -> {most['throughput']:.2f} Mbps")
    check("more sleep raises delay",
          none_["delay"] <= half["delay"] <= most["delay"],
          f"{none_['delay']:.1f} -> {half['delay']:.1f} -> {most['delay']:.1f} ms")
    check("more sleep raises QoS violations (strictly)",
          none_["violation"] < most["violation"],
          f"{none_['violation']:.3f} -> {most['violation']:.3f}")
    check("near-total sleep actually cuts throughput",
          most["throughput"] < 0.95 * none_["throughput"],
          f"{none_['throughput']:.2f} -> {most['throughput']:.2f} Mbps")


def test_energy_bounds() -> None:
    cfg = ScenarioConfig(name="energy", traffic_level="medium", n_ue=8, n_ru=1, seed=3)
    env = OranSimEnv(cfg)
    env.reset()
    saved = []
    for b in range(0, cfg.numerology.n_dl_slots + 1, 2):
        saved.append(_run_fixed(OranSimEnv(cfg), b, n_steps=6)["energy_saved"])
    check("energy saved never negative", all(s >= -1e-6 for s in saved), f"min={min(saved):.4f}")
    check("energy saved never exceeds 1", all(s <= 1.0 for s in saved), f"max={max(saved):.4f}")
    check("zero sleep saves nothing", abs(saved[0]) < 1e-6, f"{saved[0]:.6f}")


def test_delay_responds_to_load() -> None:
    def mean_delay(level: str) -> float:
        cfg = ScenarioConfig(name=level, traffic_level=level, n_ue=8, n_slices=3, n_ru=1, seed=11)
        return _run_fixed(OranSimEnv(cfg), cfg.numerology.n_dl_slots // 2)["delay"]

    light, heavy = mean_delay("light"), mean_delay("heavy")
    check("heavy traffic delays more than light", heavy > light, f"{light:.1f} vs {heavy:.1f} ms")


def test_coverage_term() -> None:
    cfg1 = ScenarioConfig(name="cov1", n_ue=8, n_slices=3, n_ru=1, seed=5)
    single = _run_fixed(OranSimEnv(cfg1), cfg1.numerology.n_dl_slots - 2)
    check("coverage penalty is zero with one RU", single["uncovered"] < 1e-9,
          f"{single['uncovered']:.6f}")

    cfg2 = ScenarioConfig(name="cov2", n_ue=16, n_slices=3, n_ru=3, seed=5,
                          inter_site_distance_m=120.0)
    multi = _run_fixed(OranSimEnv(cfg2), cfg2.numerology.n_dl_slots - 2)
    check("coverage penalty fires when all RUs sleep together", multi["uncovered"] > 0.0,
          f"{multi['uncovered']:.4f}")


def test_action_repair() -> None:
    env = OranSimEnv(ScenarioConfig(name="repair", n_ue=6, n_slices=3, n_ru=1, seed=2))
    env.reset()
    # Deliberately invalid: does not sum to n_dl_slots.
    _, _, _, info = env.step({"sleep": [[1, 1, 1]], "slices": [[0.5, 0.3, 0.2]]})
    check("invalid sleep partition is repaired", info["action_repairs"] == 1,
          f"repairs={info['action_repairs']}")
    ru = env.net.rus[0]
    check("repaired partition satisfies the constraint",
          ru.a + ru.b + ru.c == env.n_dl_slots, f"({ru.a},{ru.b},{ru.c})")


def test_reward_consistent_with_violations() -> None:
    """With no sleep and light load nothing is violated, so the reward's QoS
    penalty must be ~zero too. Guards against the reward and the reported
    violation ratio disagreeing, e.g. by penalising UEs with no demand."""
    cfg = ScenarioConfig(name="consistency", traffic_level="light", n_ue=8, n_slices=3, n_ru=1, seed=9)
    env = OranSimEnv(cfg)
    env.reset()
    n_dl = env.n_dl_slots
    penalties, violations = [], []
    for _ in range(30):
        _, (_, r_q), _, info = env.step({"sleep": [[n_dl, 0, 0]], "slices": [[1 / 3, 1 / 3, 1 / 3]]})
        penalties.append(-r_q)
        violations.append(info["qos_violation_ratio"])
    check("no violations -> no QoS penalty", max(violations) == 0.0 and np.mean(penalties) < 0.01,
          f"violation={np.mean(violations):.3f} penalty={np.mean(penalties):.4f}")


def test_determinism() -> None:
    cfg = ScenarioConfig(name="det", traffic_level="medium", n_ue=8, n_ru=2, seed=42)
    a = _run_fixed(OranSimEnv(cfg), 4, n_steps=8)
    b = _run_fixed(OranSimEnv(cfg), 4, n_steps=8)
    same = all(abs(a[k] - b[k]) < 1e-12 for k in a)
    check("same seed reproduces the trajectory", same)


def test_slice_allocation_matters() -> None:
    """Starving a slice should raise its violation rate relative to a fair split."""
    cfg = ScenarioConfig(name="slice", traffic_level="heavy", n_ue=9, n_slices=3, n_ru=1, seed=13)
    n_dl = cfg.numerology.n_dl_slots

    def run(fractions) -> float:
        env = OranSimEnv(cfg)
        env.reset()
        act = {"sleep": [[n_dl, 0, 0]], "slices": [fractions]}
        vio = []
        for _ in range(12):
            _, _, _, info = env.step(act)
            vio.append(info["per_slice_violation"][0])
        return float(np.mean(vio))

    fair = run([1 / 3, 1 / 3, 1 / 3])
    starved = run([0.02, 0.49, 0.49])
    check("starving a slice raises its violation rate", starved >= fair,
          f"fair={fair:.3f} starved={starved:.3f}")


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_observation_contract,
        test_link_budget,
        test_sleep_monotonicity,
        test_energy_bounds,
        test_delay_responds_to_load,
        test_coverage_term,
        test_action_repair,
        test_reward_consistent_with_violations,
        test_determinism,
        test_slice_allocation_matters,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # surface the failure, never swallow it
            _results.append((t.__name__, FAIL, f"raised {type(exc).__name__}: {exc}"))

    width = max(len(n) for n, _, _ in _results) + 2
    print("=" * (width + 40))
    print("SIMULATOR VALIDATION")
    print("=" * (width + 40))
    for name, status, detail in _results:
        print(f"  [{status}] {name:<{width}} {detail}")

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print("-" * (width + 40))
    print(f"  {len(_results) - n_fail}/{len(_results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
