"""
env.py - The simulated O-RAN environment.
=========================================

A gym-style environment that replaces the baseline's hardware-in-the-loop
``algorithms/env.py``. No SQLite, no FlexRIC, no binary control file, no
testbed: it runs anywhere NumPy runs.

Three design commitments
------------------------
1. **Observation compatibility.** The per-UE observation is the baseline's
   exact 17 features, in the baseline's order, normalised by the baseline's
   ranges. A policy trained here can be moved to the real E2 interface without
   touching the state definition.

2. **Multi-RU from day one.** ``n_ru = 1`` reproduces the single-cell baseline
   through the same code path used for the multi-agent setting, so nothing has
   to be rewritten in Phase 3.

3. **No torch dependency.** The environment is pure NumPy, so it is usable on
   Python versions that torch has not shipped wheels for yet, and so that
   environment and agent stay cleanly separable.

Action format
-------------
``step`` accepts a dict, or a flat array for the single-RU case:

    {"sleep": [[a, b, c], ...],        # one row per RU, summing to n_dl_slots
     "slices": [[f1, ..., fI], ...]}   # one row per RU, summing to 1

The flat form ``[f1..fI, a, b, c]`` matches the baseline's ordering so its own
agent can drive this environment unmodified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import (
    RewardBreakdown,
    compute_reward,
    effective_targets_mbps,
    per_slice_violation_ratio,
    qos_violation_ratio,
)
from .network import Network
from .params import (
    N_OBS_FEATURES,
    OBS_FEATURE_NAMES,
    OBS_NORMALISATION,
    ScenarioConfig,
)
from .radio import power_headroom_db, uplink_snr_db
from .scheduler import StepTelemetry, run_step


class OranSimEnv:
    """Simulated 5G O-RAN environment for joint sleep scheduling and slicing."""

    def __init__(self, cfg: Optional[ScenarioConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or ScenarioConfig(name="default")
        self.cfg.validate()
        self.seed = self.cfg.seed if seed is None else seed

        self.rng = np.random.default_rng(self.seed)
        self.net = Network(self.cfg, self.rng)

        self.n_ue = len(self.net.ues)
        self.n_ru = len(self.net.rus)
        self.n_slices = self.net.n_slices
        self.n_dl_slots = self.net.n_dl_slots

        self.step_count = 0
        self.last_telemetry: Optional[StepTelemetry] = None

    # ------------------------------------------------------------------
    # Spaces, described rather than imported so there is no gym dependency
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> Tuple[int, int]:
        """``[n_ue, 17]`` - one 17-feature row per UE, as in the baseline."""
        return (self.n_ue, N_OBS_FEATURES)

    @property
    def sleep_action_sum(self) -> int:
        """The value ``a + b + c`` must equal. The baseline calls this N_sf."""
        return self.n_dl_slots

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to a fresh episode and return the initial observation."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.net = Network(self.cfg, self.rng)
        self.step_count = 0

        # Warm up with one fully-active step so the first observation carries
        # real measurements rather than zeros.
        self._apply_action(self._default_action())
        self.last_telemetry = run_step(self.net)
        return self._observation()

    def _default_action(self) -> Dict[str, np.ndarray]:
        """Fully active, equal slice shares."""
        sleep = np.tile(np.array([self.n_dl_slots, 0, 0]), (self.n_ru, 1))
        slices = np.full((self.n_ru, self.n_slices), 1.0 / self.n_slices)
        return {"sleep": sleep, "slices": slices}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action) -> Tuple[np.ndarray, Tuple[float, float], bool, dict]:
        """Apply one joint action and advance the network by one decision step.

        Returns ``(observation, (r_energy, r_qos), done, info)``. The reward is
        returned decomposed so the dual-actor architecture can train its two
        actors on their own objectives, matching the baseline's interface.
        """
        parsed = self._parse_action(action)
        self._apply_action(parsed)

        telemetry = run_step(self.net)
        self.last_telemetry = telemetry
        self.step_count += 1

        slice_ids = np.array([u.slice_id for u in self.net.ues])
        breakdown: RewardBreakdown = compute_reward(
            telemetry, slice_ids, self.net.slices, self.cfg.reward
        )

        info = self._build_info(telemetry, breakdown, slice_ids)
        return self._observation(), (breakdown.r_energy, breakdown.r_qos), False, info

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def _parse_action(self, action) -> Dict[str, np.ndarray]:
        """Accept either the dict form or the baseline's flat vector."""
        if isinstance(action, dict):
            sleep = np.atleast_2d(np.asarray(action["sleep"], dtype=int))
            slices = np.atleast_2d(np.asarray(action["slices"], dtype=float))
            return {"sleep": sleep, "slices": slices}

        flat = np.asarray(action, dtype=float).ravel()
        expected = self.n_ru * (self.n_slices + 3)
        if flat.size != expected:
            raise ValueError(
                f"flat action has {flat.size} elements, expected {expected} "
                f"({self.n_ru} RUs x ({self.n_slices} slice shares + 3 sleep values))"
            )
        flat = flat.reshape(self.n_ru, self.n_slices + 3)
        return {
            "slices": flat[:, : self.n_slices],
            "sleep": np.round(flat[:, self.n_slices :]).astype(int),
        }

    def _apply_action(self, parsed: Dict[str, np.ndarray]) -> None:
        """Install the actions on the RUs, repairing the sleep partition if needed.

        The partition constraint ``a + b + c = n_dl_slots`` is repaired rather
        than rejected: an agent whose logits drift is corrected towards a valid
        action instead of crashing the rollout. Repairs are counted in ``info``
        so a policy that constantly produces invalid actions is visible rather
        than silently patched.
        """
        self._n_repairs = 0
        for ru_idx, ru in enumerate(self.net.rus):
            a, b, c = (int(x) for x in parsed["sleep"][ru_idx])
            a, b, c = max(0, a), max(0, b), max(0, c)
            total = a + b + c
            if total != self.n_dl_slots:
                self._n_repairs += 1
                if total == 0:
                    a, b, c = self.n_dl_slots, 0, 0
                else:
                    # Preserve the proportions, then fix rounding on the tail.
                    scale = self.n_dl_slots / total
                    a, b = int(round(a * scale)), int(round(b * scale))
                    a = min(a, self.n_dl_slots)
                    b = min(b, self.n_dl_slots - a)
                    c = self.n_dl_slots - a - b
            ru.set_sleep_schedule(a, b, c)
            ru.set_slice_allocation(parsed["slices"][ru_idx])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """Build the ``[n_ue, 17]`` normalised observation.

        Feature order and normalisation ranges are the baseline's, so this
        array is drop-in compatible with its state encoder.
        """
        t = self.last_telemetry
        cfg = self.cfg
        n_ue = self.n_ue

        distances = self.net.distances_m()
        serving = np.array([u.serving_ru for u in self.net.ues])
        serving_distance = distances[np.arange(n_ue), serving]
        serving_shadow = self.net.shadowing_db[np.arange(n_ue), serving]

        ul_snr = uplink_snr_db(serving_distance, cfg.radio, serving_shadow)
        phr = power_headroom_db(serving_distance, cfg.radio)

        duration_s = cfg.step_duration_ms / 1000.0
        # The baseline's KPM volumes are reported per 100 ms window.
        kpm_scale = 100.0 / cfg.step_duration_ms

        raw: Dict[str, np.ndarray] = {
            # MAC
            "dl_curr_tbs": t.tbs_bytes,
            "dl_sched_rb": t.prb_dl,
            "pusch_snr": np.clip(ul_snr, 0.0, 70.0),
            "pucch_snr": np.clip(ul_snr - 3.0, 0.0, 50.0),
            "wb_cqi": t.mean_cqi,
            "dl_mcs1": t.mean_mcs,
            "ul_mcs1": np.clip(t.mean_mcs - 2.0, 0.0, 28.0),
            "phr": phr,
            "dl_bler": t.mean_bler,
            "ul_bler": np.clip(t.mean_bler * 0.8, 0.0, 0.5),
            # KPM
            "pdcp_sdu_volume_dl": t.served_bytes * kpm_scale,
            "pdcp_sdu_volume_ul": t.served_bytes * kpm_scale * 0.15,  # light uplink
            "rlc_sdu_delay_dl": t.delay_ms,
            "ue_thp_dl": t.throughput_mbps * 1000.0,  # kbps, the baseline's unit
            "ue_thp_ul": t.throughput_mbps * 1000.0 * 0.15,
            "prb_tot_dl": t.prb_dl * self.n_dl_slots * cfg.frames_per_step,
            "prb_tot_ul": t.prb_dl * self.n_dl_slots * cfg.frames_per_step * 0.2,
        }

        obs = np.zeros((n_ue, N_OBS_FEATURES), dtype=np.float32)
        for j, name in enumerate(OBS_FEATURE_NAMES):
            lo, hi = OBS_NORMALISATION[name]
            obs[:, j] = np.clip((raw[name] - lo) / (hi - lo), 0.0, 1.0)
        return obs

    # ------------------------------------------------------------------
    # Info / metrics
    # ------------------------------------------------------------------

    def _build_info(
        self, t: StepTelemetry, breakdown: RewardBreakdown, slice_ids: np.ndarray
    ) -> dict:
        targets_mbps = effective_targets_mbps(
            np.array([self.net.slices[s].throughput_target_mbps for s in slice_ids]),
            t.offered_mbps,
        )
        targets_ms = np.array([self.net.slices[s].delay_target_ms for s in slice_ids])

        info = dict(breakdown.as_dict())
        info.update(
            {
                "step": self.step_count,
                "qos_violation_ratio": qos_violation_ratio(
                    t.throughput_mbps, t.delay_ms, targets_mbps, targets_ms
                ),
                "per_slice_violation": per_slice_violation_ratio(
                    t.throughput_mbps, t.delay_ms, targets_mbps, targets_ms, slice_ids, self.n_slices
                ),
                "energy_saved_fraction": float(1.0 - t.energy_j.sum() / max(t.baseline_energy_j.sum(), 1e-9)),
                "mean_sleep_ratio": float(t.sleep_ratio.mean()),
                "mean_power_w": float(t.energy_j.sum() / (self.cfg.step_duration_ms / 1000.0)),
                "transitions": int(t.transitions.sum()),
                "throughput_mbps": t.throughput_mbps.copy(),
                "delay_ms": t.delay_ms.copy(),
                "queued_bytes": t.queued_bytes.copy(),
                "dropped_bytes": t.dropped_bytes.copy(),
                "prb_utilisation": t.prb_utilisation.copy(),
                "uncovered_fraction": np.asarray(t.uncovered_fraction).copy(),
                "action_repairs": self._n_repairs,
            }
        )
        return info

    # ------------------------------------------------------------------
    # Introspection helpers used by the graph builder in later phases
    # ------------------------------------------------------------------

    def slice_ids(self) -> np.ndarray:
        return np.array([u.slice_id for u in self.net.ues])

    def serving_ru_ids(self) -> np.ndarray:
        return np.array([u.serving_ru for u in self.net.ues])

    def ru_adjacency(self) -> np.ndarray:
        """Weighted RU-RU overlap matrix - the coordination edges of the GNN."""
        return self.net.neighbour_matrix()

    def rsrp_matrix(self) -> np.ndarray:
        return self.net.rsrp_matrix_dbm()
