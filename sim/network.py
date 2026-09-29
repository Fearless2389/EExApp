"""
network.py - Topology, radio units, user equipment and association.
===================================================================

This is the module the whole multi-agent contribution rests on. The baseline
has no RU entity at all: its state is a flat list of UEs attached to a single
implicit cell. Here an RU is a first-class object with a position, a sleep
schedule, a slice allocation and an energy account, and UEs associate to
whichever RU serves them best.

Everything is written for N radio units from the start. Running with
``n_ru = 1`` reproduces the baseline's single-cell setting through exactly the
same code path, so Phase 2 and Phase 3 share an implementation.

Classes
-------
``RadioUnit``   one RU: position, sleep schedule, slice split, energy account
``UserEquipment``  one UE: position, slice membership, traffic source, queue
``Network``     the deployment: geometry, association, mobility, neighbours
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .params import ScenarioConfig, SliceSpec, TRAFFIC_LEVELS, UDP_PACKET_BYTES, build_slices
from .power import RuEnergyAccounting
from .radio import path_loss_db, rsrp_dbm
from .traffic import TrafficSource, UeQueue, assign_traffic_rates


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class RadioUnit:
    """One Radio Unit and the state an agent controls on it.

    The sleep action ``(a, b, c)`` partitions the frame's downlink slots into
    an active prefix of length ``a``, a contiguous sleep window of length
    ``b``, and an active suffix of length ``c``, with ``a + b + c = n_dl_slots``.

    ``slice_prb_fraction`` is the RS action: the share of the PRB pool given to
    each slice, summing to one.
    """

    ru_id: int
    position: np.ndarray  # (x, y) in metres
    n_dl_slots: int
    n_slices: int

    a: int = 0
    b: int = 0
    c: int = 0
    slice_prb_fraction: np.ndarray = field(default=None, repr=False)

    energy: Optional[RuEnergyAccounting] = field(default=None, repr=False)
    _was_sleeping: bool = False

    def __post_init__(self) -> None:
        if self.slice_prb_fraction is None:
            self.slice_prb_fraction = np.full(self.n_slices, 1.0 / self.n_slices)
        if self.a + self.b + self.c != self.n_dl_slots:
            # Default to fully active.
            self.a, self.b, self.c = self.n_dl_slots, 0, 0

    # ---- action application ----------------------------------------------

    def set_sleep_schedule(self, a: int, b: int, c: int) -> None:
        """Install a sleep schedule, validating the partition constraint."""
        a, b, c = int(a), int(b), int(c)
        if a < 0 or b < 0 or c < 0:
            raise ValueError(f"sleep schedule must be non-negative, got ({a}, {b}, {c})")
        if a + b + c != self.n_dl_slots:
            raise ValueError(
                f"sleep schedule ({a}, {b}, {c}) sums to {a + b + c}, expected {self.n_dl_slots}"
            )
        self.a, self.b, self.c = a, b, c

    def set_slice_allocation(self, fractions: np.ndarray) -> None:
        """Install the per-slice PRB shares, renormalising to sum to one."""
        f = np.asarray(fractions, dtype=float)
        if f.shape != (self.n_slices,):
            raise ValueError(f"expected {self.n_slices} slice fractions, got shape {f.shape}")
        f = np.clip(f, 0.0, None)
        total = f.sum()
        self.slice_prb_fraction = f / total if total > 0 else np.full(self.n_slices, 1.0 / self.n_slices)

    # ---- per-slot queries -------------------------------------------------

    def is_sleeping_in_dl_slot(self, dl_slot_index: int) -> bool:
        """True when downlink slot ``dl_slot_index`` falls inside the sleep window."""
        return self.a <= dl_slot_index < self.a + self.b

    @property
    def sleep_ratio(self) -> float:
        return self.b / self.n_dl_slots if self.n_dl_slots else 0.0


@dataclass
class UserEquipment:
    """One UE: where it is, which slice it belongs to, and its buffer."""

    ue_id: int
    position: np.ndarray  # (x, y) in metres
    slice_id: int
    traffic: TrafficSource
    queue: UeQueue = field(default_factory=UeQueue, repr=False)

    serving_ru: int = 0
    heading_rad: float = 0.0

    # Most recent per-step radio measurements, filled in by the scheduler.
    last_cqi: int = 0
    last_sinr_db: float = 0.0
    last_dl_bler: float = 0.0
    last_prb_dl: float = 0.0
    last_tbs_bytes: float = 0.0


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


class Network:
    """The physical deployment: RU placement, UE placement, association.

    Layout
    ------
    One RU sits at the origin. Additional RUs are placed on a ring at
    ``inter_site_distance_m`` so that coverage regions overlap, which is what
    makes joint sleep scheduling non-trivial. UEs are scattered uniformly over
    the union of the cells.

    Shadowing is drawn once per (UE, RU) pair at reset and held fixed for the
    episode, which is the usual treatment for slow fading over short episodes.
    """

    def __init__(self, cfg: ScenarioConfig, rng: np.random.Generator):
        cfg.validate()
        self.cfg = cfg
        self.rng = rng

        self.slices: List[SliceSpec] = build_slices(cfg.n_slices)
        self.n_slices = cfg.n_slices
        self.n_dl_slots = cfg.numerology.n_dl_slots

        self.rus: List[RadioUnit] = self._place_rus()
        self.ues: List[UserEquipment] = self._place_ues()

        # Slow fading, fixed per episode: shape [n_ue, n_ru].
        self.shadowing_db = rng.normal(0.0, cfg.radio.shadowing_sigma_db, size=(len(self.ues), len(self.rus)))

        self.time_ms: float = 0.0
        self.associate()

    # ---- construction -----------------------------------------------------

    def _place_rus(self) -> List[RadioUnit]:
        cfg = self.cfg
        positions = [np.array([0.0, 0.0])]
        if cfg.n_ru > 1:
            # Remaining RUs on a ring around the first.
            for k in range(cfg.n_ru - 1):
                angle = 2.0 * np.pi * k / (cfg.n_ru - 1)
                positions.append(
                    cfg.inter_site_distance_m * np.array([np.cos(angle), np.sin(angle)])
                )
        return [
            RadioUnit(
                ru_id=i,
                position=p,
                n_dl_slots=self.n_dl_slots,
                n_slices=self.n_slices,
                energy=RuEnergyAccounting(cfg.power, cfg.numerology.slot_duration_ms),
            )
            for i, p in enumerate(positions)
        ]

    def _place_ues(self) -> List[UserEquipment]:
        cfg = self.cfg
        rates = assign_traffic_rates(cfg.n_ue, TRAFFIC_LEVELS[cfg.traffic_level], self.rng)

        # Scatter UEs over a disc covering every cell.
        span = cfg.cell_radius_m + (cfg.inter_site_distance_m if cfg.n_ru > 1 else 0.0)
        radii = span * np.sqrt(self.rng.uniform(0.02, 1.0, size=cfg.n_ue))
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.n_ue)

        ues: List[UserEquipment] = []
        for i in range(cfg.n_ue):
            ues.append(
                UserEquipment(
                    ue_id=i,
                    position=np.array([radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])]),
                    # Even distribution across slices, matching the paper.
                    slice_id=i % cfg.n_slices,
                    traffic=TrafficSource(rate_mbps=float(rates[i]), packet_bytes=UDP_PACKET_BYTES),
                    heading_rad=float(self.rng.uniform(0.0, 2.0 * np.pi)),
                )
            )
        return ues

    # ---- geometry ---------------------------------------------------------

    def distances_m(self) -> np.ndarray:
        """Euclidean UE-to-RU distances, shape ``[n_ue, n_ru]``."""
        ue_pos = np.stack([u.position for u in self.ues])  # [n_ue, 2]
        ru_pos = np.stack([r.position for r in self.rus])  # [n_ru, 2]
        return np.linalg.norm(ue_pos[:, None, :] - ru_pos[None, :, :], axis=2)

    def rsrp_matrix_dbm(self) -> np.ndarray:
        """Received power from every RU at every UE, shape ``[n_ue, n_ru]``."""
        return rsrp_dbm(self.distances_m(), self.cfg.radio, self.shadowing_db)

    def associate(self) -> None:
        """Attach each UE to the RU with the strongest received power."""
        best = np.argmax(self.rsrp_matrix_dbm(), axis=1)
        for ue, ru_idx in zip(self.ues, best):
            ue.serving_ru = int(ru_idx)

    def neighbour_matrix(self, overlap_threshold_db: float = 6.0) -> np.ndarray:
        """Adjacency over RUs, shape ``[n_ru, n_ru]``, used for the RU graph edges.

        Two RUs are neighbours when at least one UE sees them within
        ``overlap_threshold_db`` of each other, which is the operational
        definition of a coverage overlap: a UE in that region could be served
        by either, so their sleep decisions are coupled.

        The returned matrix is weighted by the number of such UEs, normalised
        by the UE count, giving an overlap strength in [0, 1].
        """
        rsrp = self.rsrp_matrix_dbm()
        n_ru = len(self.rus)
        adj = np.zeros((n_ru, n_ru))
        for i in range(n_ru):
            for j in range(i + 1, n_ru):
                overlap = np.abs(rsrp[:, i] - rsrp[:, j]) <= overlap_threshold_db
                weight = overlap.sum() / max(len(self.ues), 1)
                adj[i, j] = adj[j, i] = weight
        return adj

    def covered_by(self, rsrp_margin_db: float = 6.0) -> List[np.ndarray]:
        """For each UE, the RU indices able to serve it.

        A UE is servable by any RU within ``rsrp_margin_db`` of its best.
        This is what the coverage-continuity constraint is evaluated against:
        a UE with queued data is covered only if at least one RU in this set
        is awake.
        """
        rsrp = self.rsrp_matrix_dbm()
        best = rsrp.max(axis=1, keepdims=True)
        return [np.flatnonzero(row) for row in (rsrp >= best - rsrp_margin_db)]

    # ---- dynamics ---------------------------------------------------------

    def move_ues(self, duration_ms: float) -> None:
        """Advance UE positions by a random-walk mobility model.

        Each UE keeps a heading that is perturbed slightly each step, so tracks
        are smooth rather than jittery, and is reflected back towards the
        origin if it wanders outside the deployment area.
        """
        cfg = self.cfg
        if cfg.ue_speed_mps <= 0.0:
            return

        span = cfg.cell_radius_m + (cfg.inter_site_distance_m if cfg.n_ru > 1 else 0.0)
        step_m = cfg.ue_speed_mps * duration_ms / 1000.0

        for ue in self.ues:
            ue.heading_rad += float(self.rng.normal(0.0, 0.3))
            ue.position = ue.position + step_m * np.array(
                [np.cos(ue.heading_rad), np.sin(ue.heading_rad)]
            )
            # Reflect at the boundary by turning back towards the centre.
            if np.linalg.norm(ue.position) > span:
                ue.heading_rad = float(np.arctan2(-ue.position[1], -ue.position[0]))
                ue.position = ue.position * (span / np.linalg.norm(ue.position))

    def ues_of_ru(self, ru_id: int) -> List[UserEquipment]:
        return [u for u in self.ues if u.serving_ru == ru_id]

    def ues_of_slice(self, ru_id: int, slice_id: int) -> List[UserEquipment]:
        return [u for u in self.ues if u.serving_ru == ru_id and u.slice_id == slice_id]

    # ---- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        for ue in self.ues:
            ue.queue.reset()
        for ru in self.rus:
            ru.energy.reset()
            ru.set_sleep_schedule(self.n_dl_slots, 0, 0)
            ru.set_slice_allocation(np.full(self.n_slices, 1.0 / self.n_slices))
        self.time_ms = 0.0
        self.associate()
