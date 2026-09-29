"""
scheduler.py - The slot-level MAC loop.
=======================================

This is where the actions actually take effect. One decision step covers
``frames_per_step`` radio frames; each frame is walked slot by slot, and in
every downlink slot each RU either sleeps or schedules PRBs to its UEs.

Per downlink slot, for each awake RU:

    1. work out which other RUs are awake  -> interference
    2. SINR per UE -> CQI -> bits per PRB
    3. split the PRB pool across slices by the RS action (beta)
    4. share each slice's PRBs among its backlogged UEs
    5. serve bytes from the UE queues, discounted by BLER
    6. charge energy for the slot at the resulting PRB utilisation

Sleeping RUs transmit nothing, so their UEs' queues age - which is exactly the
tension the policy has to resolve.

The intra-slice discipline is an equal share among backlogged UEs. That is
deliberately simple and neutral; swapping in proportional-fair is a one-function
change at ``_share_prbs_within_slice``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .network import Network
from .radio import (
    block_error_rate,
    bits_per_prb,
    cqi_to_mcs,
    db_to_lin,
    downlink_sinr_db,
    lin_to_db,
    sinr_to_cqi,
    transmission_rank,
)


# ---------------------------------------------------------------------------
# Per-step accumulators
# ---------------------------------------------------------------------------


@dataclass
class StepTelemetry:
    """Everything the environment needs after a step, per UE unless stated."""

    served_bytes: np.ndarray
    arrived_bytes: np.ndarray
    dropped_bytes: np.ndarray
    queued_bytes: np.ndarray
    delay_ms: np.ndarray
    throughput_mbps: np.ndarray
    offered_mbps: np.ndarray  # demand realised in THIS window (from arrivals)

    mean_cqi: np.ndarray
    mean_mcs: np.ndarray
    mean_sinr_db: np.ndarray
    mean_bler: np.ndarray
    prb_dl: np.ndarray  # mean PRBs per scheduled slot
    tbs_bytes: np.ndarray  # mean transport block size per scheduled slot

    # Per RU
    energy_j: np.ndarray
    baseline_energy_j: np.ndarray
    sleep_ratio: np.ndarray
    transitions: np.ndarray
    prb_utilisation: np.ndarray

    # Per UE, coverage continuity
    uncovered_fraction: np.ndarray = field(default=None)


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def _share_prbs_within_slice(prb_budget: float, backlogged: np.ndarray) -> np.ndarray:
    """Split a slice's PRB budget equally among its backlogged UEs.

    Parameters
    ----------
    prb_budget
        PRBs available to this slice in this slot.
    backlogged
        Boolean array over the slice's UEs; a UE with an empty queue is skipped
        so its share is not wasted.

    Returns
    -------
    PRBs per UE of the slice, same length as ``backlogged``.
    """
    out = np.zeros(len(backlogged), dtype=float)
    n_active = int(backlogged.sum())
    if n_active == 0 or prb_budget <= 0:
        return out
    out[backlogged] = prb_budget / n_active
    return out


def _slice_prb_budgets(fractions: np.ndarray, n_prb: int) -> np.ndarray:
    """Convert the RS action into an integer-ish PRB budget per slice."""
    raw = np.asarray(fractions, dtype=float) * n_prb
    return raw


# ---------------------------------------------------------------------------
# The step loop
# ---------------------------------------------------------------------------


def run_step(net: Network) -> StepTelemetry:
    """Advance the network by one decision step and return telemetry.

    The sleep schedules and slice allocations already sitting on the RUs are
    the actions being applied; the caller installs them before calling this.
    """
    cfg = net.cfg
    radio = cfg.radio
    num = cfg.numerology

    n_ue = len(net.ues)
    n_ru = len(net.rus)
    n_dl = num.n_dl_slots
    slot_ms = num.slot_duration_ms

    # --- per-step geometry, held fixed within the step ---------------------
    net.move_ues(cfg.step_duration_ms)
    net.associate()
    rsrp = net.rsrp_matrix_dbm()  # [n_ue, n_ru]
    rsrp_lin = db_to_lin(rsrp)
    serving = np.array([u.serving_ru for u in net.ues])
    serving_rsrp_lin = rsrp_lin[np.arange(n_ue), serving]
    noise_lin = db_to_lin(radio.thermal_noise_dbm_per_prb(radio.ue_noise_figure_db))

    # Coverage candidates, as a boolean matrix. Only UEs that could be served
    # by two or more RUs can suffer a coordination failure (see below).
    candidate = np.zeros((n_ue, n_ru), dtype=bool)
    for i, cands in enumerate(net.covered_by()):
        candidate[i, cands] = True
    has_choice = candidate.sum(axis=1) >= 2

    # Slice membership per RU, computed once per step rather than per slot.
    # Association only changes between steps, so this is exact.
    members: List[List[np.ndarray]] = [
        [
            np.array(
                [u.ue_id for u in net.ues if u.serving_ru == ru.ru_id and u.slice_id == s],
                dtype=int,
            )
            for s in range(net.n_slices)
        ]
        for ru in net.rus
    ]

    # Which RUs are awake in each downlink slot: [n_dl, n_ru]. The sleep
    # schedule repeats identically in every frame of the step.
    awake_pattern = np.array(
        [[not ru.is_sleeping_in_dl_slot(d) for ru in net.rus] for d in range(n_dl)]
    )

    # Link quality depends only on which RUs are awake, and a step has very
    # few distinct awake patterns, so compute each pattern's link state once.
    link_cache: Dict[bytes, tuple] = {}

    def link_state(awake: np.ndarray) -> tuple:
        key = awake.tobytes()
        if key not in link_cache:
            awake_lin = rsrp_lin * awake[None, :]
            own = awake_lin[np.arange(n_ue), serving]
            interference = np.maximum(awake_lin.sum(axis=1) - own, 0.0)
            sinr = lin_to_db(serving_rsrp_lin / (interference + noise_lin))
            cqi = sinr_to_cqi(sinr)
            rank = transmission_rank(sinr, radio)
            bler = block_error_rate(sinr, cqi)
            link_cache[key] = (sinr, cqi, cqi_to_mcs(cqi), bits_per_prb(cqi, rank), bler)
        return link_cache[key]

    # --- accumulators ------------------------------------------------------
    cqi_sum = np.zeros(n_ue)
    mcs_sum = np.zeros(n_ue)
    sinr_sum = np.zeros(n_ue)
    bler_sum = np.zeros(n_ue)
    prb_sum = np.zeros(n_ue)
    tbs_sum = np.zeros(n_ue)
    scheduled_slots = np.zeros(n_ue)

    prb_used_per_ru = np.zeros(n_ru)
    prb_slots_per_ru = np.zeros(n_ru)
    uncovered_slots = np.zeros(n_ue)
    total_dl_slots = 0

    for ru in net.rus:
        ru.energy.reset()
        ru._was_sleeping = False

    # --- frame / slot walk -------------------------------------------------
    for _frame in range(cfg.frames_per_step):
        # Traffic arrives once per frame, in 10 ms worth of packets.
        for ue in net.ues:
            n_pkt = ue.traffic.arrivals(10.0, net.rng)
            ue.queue.enqueue(n_pkt, ue.traffic.packet_bytes, net.time_ms)

        for dl_idx in range(n_dl):
            total_dl_slots += 1
            net.time_ms += slot_ms

            awake = awake_pattern[dl_idx]

            # Coverage continuity. This penalises a *coordination* failure, not
            # sleep in general: a UE is uncovered when it has data queued, more
            # than one RU could serve it, and every one of them chose to sleep
            # in this slot. Restricting it to UEs with a choice is what keeps
            # the term identically zero for a single RU and stops it
            # double-counting the delay penalty, which already prices ordinary
            # sleep-induced starvation.
            if has_choice.any():
                queued_now = np.array([u.queue.queued_bytes > 0 for u in net.ues])
                no_candidate_awake = ~(candidate & awake[None, :]).any(axis=1)
                uncovered_slots += has_choice & queued_now & no_candidate_awake

            # Interference seen by every UE is the received power from all awake
            # RUs other than its serving RU; see ``link_state``.
            sinr_db, cqi, mcs, per_prb_bits, bler = link_state(awake)

            for ru in net.rus:
                sleeping = not awake[ru.ru_id]

                if sleeping:
                    ru.energy.account_slot(is_sleeping=True, prb_utilisation=0.0)
                    ru._was_sleeping = True
                    prb_slots_per_ru[ru.ru_id] += 1
                    continue

                # Charge the wake-up if the previous downlink slot was asleep.
                if ru._was_sleeping:
                    ru.energy.account_transition()
                    ru._was_sleeping = False

                budgets = _slice_prb_budgets(ru.slice_prb_fraction, radio.n_prb)
                prb_used_this_slot = 0.0

                for slice_id in range(net.n_slices):
                    idx = members[ru.ru_id][slice_id]
                    if idx.size == 0:
                        continue
                    slice_ues = [net.ues[g] for g in idx]
                    backlogged = np.array([u.queue.queued_bytes > 0 for u in slice_ues])
                    shares = _share_prbs_within_slice(budgets[slice_id], backlogged)

                    for local_i, ue in enumerate(slice_ues):
                        n_prb_ue = shares[local_i]
                        if n_prb_ue <= 0:
                            continue
                        gi = idx[local_i]
                        # Bytes offered this slot, after block errors.
                        offered_bits = n_prb_ue * per_prb_bits[gi] * (1.0 - bler[gi])
                        offered_bytes = offered_bits / 8.0
                        served = ue.queue.serve(offered_bytes, net.time_ms)

                        # PRBs are only consumed in proportion to what was used.
                        if offered_bytes > 0:
                            used_frac = min(1.0, served / offered_bytes)
                        else:
                            used_frac = 0.0
                        prb_used_this_slot += n_prb_ue * used_frac

                        cqi_sum[gi] += cqi[gi]
                        mcs_sum[gi] += mcs[gi]
                        sinr_sum[gi] += sinr_db[gi]
                        bler_sum[gi] += bler[gi]
                        prb_sum[gi] += n_prb_ue
                        tbs_sum[gi] += offered_bytes
                        scheduled_slots[gi] += 1

                utilisation = prb_used_this_slot / radio.n_prb
                ru.energy.account_slot(is_sleeping=False, prb_utilisation=utilisation)
                prb_used_per_ru[ru.ru_id] += prb_used_this_slot
                prb_slots_per_ru[ru.ru_id] += 1

    # --- collect -----------------------------------------------------------
    served = np.zeros(n_ue)
    arrived = np.zeros(n_ue)
    dropped = np.zeros(n_ue)
    queued = np.zeros(n_ue)
    delay = np.zeros(n_ue)

    for i, ue in enumerate(net.ues):
        st = ue.queue.collect(net.time_ms)
        served[i] = st["served_bytes"]
        arrived[i] = st["arrived_bytes"]
        dropped[i] = st["dropped_bytes"]
        queued[i] = st["queued_bytes"]
        delay[i] = st["delay_ms"]

    duration_s = cfg.step_duration_ms / 1000.0
    throughput_mbps = served * 8.0 / duration_s / 1e6

    safe = np.maximum(scheduled_slots, 1.0)
    telemetry = StepTelemetry(
        served_bytes=served,
        arrived_bytes=arrived,
        dropped_bytes=dropped,
        queued_bytes=queued,
        delay_ms=delay,
        throughput_mbps=throughput_mbps,
        # Demand is measured over the same window as throughput, so that
        # Poisson arrival variance cancels instead of being mistaken for a
        # QoS failure. Using each UE's configured mean rate here makes a
        # lightly loaded UE look permanently in violation, because in any
        # short window its realised arrivals differ from the mean.
        offered_mbps=arrived * 8.0 / duration_s / 1e6,
        mean_cqi=cqi_sum / safe,
        mean_mcs=mcs_sum / safe,
        mean_sinr_db=np.where(scheduled_slots > 0, sinr_sum / safe, sinr_db),
        mean_bler=bler_sum / safe,
        prb_dl=prb_sum / safe,
        tbs_bytes=tbs_sum / safe,
        energy_j=np.array([r.energy.energy_j for r in net.rus]),
        baseline_energy_j=np.array([r.energy.baseline_energy_j for r in net.rus]),
        sleep_ratio=np.array([r.energy.sleep_ratio for r in net.rus]),
        transitions=np.array([r.energy.transitions for r in net.rus]),
        prb_utilisation=prb_used_per_ru / np.maximum(prb_slots_per_ru * radio.n_prb, 1.0),
        uncovered_fraction=uncovered_slots / max(total_dl_slots, 1),
    )

    # Cache the latest radio measurements on the UEs for the observation builder.
    for i, ue in enumerate(net.ues):
        ue.last_cqi = int(round(telemetry.mean_cqi[i]))
        ue.last_sinr_db = float(telemetry.mean_sinr_db[i])
        ue.last_dl_bler = float(telemetry.mean_bler[i])
        ue.last_prb_dl = float(telemetry.prb_dl[i])
        ue.last_tbs_bytes = float(telemetry.tbs_bytes[i])

    return telemetry
