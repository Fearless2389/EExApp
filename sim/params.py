"""
params.py - All tunable constants and scenario presets for the O-RAN simulator.
=============================================================================

Everything that is a "magic number" anywhere else in the simulator lives here,
so that a reader can audit the physical assumptions in one place and a
researcher can sweep them without touching logic.

Sections
--------
1. Numerology and frame structure  (3GPP TS 38.211)
2. Radio / link adaptation tables  (3GPP TS 38.214)
3. Cell and channel parameters     (3GPP TR 38.901, n78 band)
4. RU power model
5. Slice definitions and QoS targets
6. Traffic profiles
7. Reward weights
8. Observation normalisation (kept byte-identical to the baseline repo)
9. Scenario presets (the paper's 3 traffic levels x 3 slice counts)

Units are stated on every field. Where the baseline paper or its released code
is ambiguous, the ambiguity is called out in a comment so it can be defended
in the report rather than silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. Numerology and frame structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Numerology:
    """5G NR frame structure.

    A radio frame is always 10 ms. It contains ``10 * 2**mu`` slots, so the
    slot duration shrinks as the subcarrier spacing grows.

    ``tdd_pattern`` is one character per slot: 'D' downlink, 'U' uplink,
    'S' special/flexible. Its length must equal ``slots_per_frame``.

    The sleep action (a_t, b_t, c_t) partitions the *downlink* slots, which is
    what the baseline's ``N_sf`` counts.
    """

    mu: int = 1  # numerology index -> 30 kHz SCS, the paper's n78 configuration
    tdd_pattern: str = "DDDDDDDSUU" * 2  # 20 slots = 14 D, 2 S, 4 U

    @property
    def scs_khz(self) -> float:
        return 15.0 * (2**self.mu)

    @property
    def slots_per_frame(self) -> int:
        return 10 * (2**self.mu)

    @property
    def slot_duration_ms(self) -> float:
        return 1.0 / (2**self.mu)

    @property
    def dl_slot_indices(self) -> np.ndarray:
        """Absolute slot indices within the frame that carry downlink data."""
        return np.array([i for i, c in enumerate(self.tdd_pattern) if c == "D"], dtype=int)

    @property
    def n_dl_slots(self) -> int:
        """The baseline calls this ``N_sf``. Sleep scheduling partitions these."""
        return len(self.dl_slot_indices)

    def validate(self) -> None:
        if len(self.tdd_pattern) != self.slots_per_frame:
            raise ValueError(
                f"tdd_pattern has {len(self.tdd_pattern)} slots but mu={self.mu} "
                f"implies {self.slots_per_frame} slots per frame"
            )
        if set(self.tdd_pattern) - set("DUS"):
            raise ValueError("tdd_pattern may only contain the characters D, U and S")


# Preset that reproduces the released baseline configuration exactly
# (config.ENV['N_sf'] == 7). Note this requires mu=0, which contradicts the
# paper's stated 30 kHz subcarrier spacing - one of several paper/code
# inconsistencies documented in the analysis report.
NUMEROLOGY_PAPER_CODE = Numerology(mu=0, tdd_pattern="DDDDDDDSUU")

# Our default: consistent with the paper's stated 30 kHz SCS and with the
# baseline's own PRB normalisation range of 0-106.
NUMEROLOGY_DEFAULT = Numerology(mu=1, tdd_pattern="DDDDDDDSUU" * 2)


# ---------------------------------------------------------------------------
# 2. Radio / link adaptation tables
# ---------------------------------------------------------------------------

# 3GPP TS 38.214 Table 5.2.2.1-2 - CQI index -> spectral efficiency (bits/RE).
# Index 0 means "out of range", i.e. no transmission is possible.
CQI_SPECTRAL_EFFICIENCY: np.ndarray = np.array(
    [
        0.0000,  # CQI 0 - out of range
        0.1523,  # CQI 1
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,  # CQI 15
    ]
)

# Lower SINR bound (dB) at which each CQI index becomes usable at ~10% BLER.
# Widely used approximation of the 38.214 CQI selection procedure.
CQI_SINR_THRESHOLDS_DB: np.ndarray = np.array(
    [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
)

# Resource elements available for data in one PRB in one slot.
# A PRB is 12 subcarriers; a slot is 14 OFDM symbols, of which we reserve 2 for
# control and reference signals.
N_RE_PER_PRB_PER_SLOT: int = 12 * 12

MAX_CQI: int = 15
MAX_MCS: int = 28  # matches the baseline's dl_mcs1 normalisation range


# ---------------------------------------------------------------------------
# 3. Cell and channel parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadioParams:
    """Physical layer and propagation constants for an n78 deployment."""

    carrier_freq_ghz: float = 3.5  # n78
    bandwidth_mhz: float = 40.0
    n_prb: int = 106  # 40 MHz at 30 kHz SCS; matches the baseline's 0-106 range

    ru_tx_power_dbm: float = 46.0  # total conducted power across all PRBs
    ru_antenna_gain_dbi: float = 8.0
    ue_tx_power_dbm: float = 23.0  # typical handset maximum
    ue_noise_figure_db: float = 7.0
    ru_noise_figure_db: float = 5.0

    shadowing_sigma_db: float = 4.0  # log-normal shadowing standard deviation
    min_coupling_loss_db: float = 45.0  # floor, prevents absurd SINR very close in

    # Path loss model: 3GPP TR 38.901 UMi Street Canyon LOS, simplified to
    #   PL(d) = 32.4 + 21*log10(d_m) + 20*log10(f_GHz)
    pathloss_intercept_db: float = 32.4
    pathloss_exponent_factor: float = 21.0

    # Single-user MIMO. The baseline testbed's Pegatron PR1450 is 4T4R and
    # n78 handsets receive on 4 antennas, so up to 4 spatial layers are
    # available. The number actually used (the transmission rank) is chosen per
    # UE from its SINR by ``radio.transmission_rank``. Modelling a single layer
    # makes every traffic level roughly four times heavier relative to cell
    # capacity than in the baseline, which would put our scenarios in a
    # different operating regime from the figures we compare against.
    mimo_layers: int = 4
    # Minimum SINR (dB) for rank 2, 3 and 4 respectively.
    rank_sinr_thresholds_db: tuple = (8.0, 15.0, 20.0)

    # Load-coupled interference. An awake RU only interferes on the PRBs it
    # actually schedules, so each interferer is weighted by its PRB
    # utilisation over its awake slots in the previous step (the standard
    # load-coupling model). Setting this False gives the "full-buffer" worst
    # case, in which every awake RU transmits on every PRB regardless of load;
    # that made a 4-cell layout interference-limited to ~0 dB at the cell edge
    # and left half the UEs in violation before any RU slept. Only matters
    # with more than one RU.
    load_coupled_interference: bool = True
    min_interference_load: float = 0.05  # control and reference signals never stop

    # Propagation environment.
    #   "los"        UMi Street Canyon LOS at every distance. Matches the
    #                baseline's 12 x 16 m indoor lab, where every link is
    #                line-of-sight. Default, used for the single-RU study.
    #   "umi_mixed"  3GPP TR 38.901 UMi with the distance-dependent LOS
    #                probability P_LOS(d) = min(18/d, 1)(1 - e^{-d/36}) + e^{-d/36};
    #                the path gain is the P_LOS-weighted mean of the LOS and
    #                NLOS gains (NLOS: 22.4 + 35.3 log10 d + 21.3 log10 f).
    #                Used for multi-cell deployments: with LOS everywhere the
    #                exponent of 2.1 lets interference from a neighbour 100 m
    #                away barely decay, which drove SINR to ~0 dB even for UEs
    #                30 m from their own RU.
    propagation: str = "los"

    @property
    def prb_bandwidth_hz(self) -> float:
        """A PRB is 12 subcarriers wide."""
        return 12.0 * self._scs_hz

    _scs_hz: float = 30_000.0

    def thermal_noise_dbm_per_prb(self, noise_figure_db: float) -> float:
        """kTB in dBm for one PRB, plus the receiver noise figure."""
        return -174.0 + 10.0 * np.log10(self.prb_bandwidth_hz) + noise_figure_db


RADIO_DEFAULT = RadioParams()


# ---------------------------------------------------------------------------
# 4. RU power model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerParams:
    """Radio Unit power consumption.

    Linear load-dependent model in the style of EARTH / GreenTouch:

        P_active(u) = p_static + p_dynamic * u        (u = PRB utilisation)
        P_sleep     = p_sleep
        E_wake      = e_transition_j  per sleep -> active transition

    The transition energy is what discourages the policy from thrashing between
    sleep and active every slot. The baseline has no such term.

    Defaults are representative of a small-cell / indoor 4x4 RU comparable to
    the Pegatron PR1450 used in the baseline testbed. They are parameters, not
    measurements - every energy number we report must state these values.
    """

    p_static_w: float = 80.0  # consumed whenever the RU is awake
    p_dynamic_w: float = 40.0  # additional, scales with PRB utilisation
    p_sleep_w: float = 30.0  # micro-sleep (3GPP SM1): RF chain off, rest warm
    # One wake-up costs roughly the active power held over the SM1 transition
    # time of ~35 us: 120 W * 35 us ~= 0.004 J. Setting this to a deep-sleep
    # value instead makes transitions dominate slot-level energy entirely.
    e_transition_j: float = 0.004


POWER_DEFAULT = PowerParams()


# ---------------------------------------------------------------------------
# 5. Slice definitions and QoS targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceSpec:
    """One network slice and the QoS its UEs expect.

    NOTE ON UNITS. The baseline's config lists throughput targets of 100/50/20
    and delay targets of 10/5/20 without units, while normalising throughput by
    1000 kbps and delay by 1000 ms. Those targets are not self-consistent with
    the 0.1-10 Mbps traffic the paper generates, so we define our own targets
    in explicit physical units and state them in the report. The *relative*
    structure of the baseline is preserved: eMBB wants throughput, URLLC wants
    low delay, mMTC is tolerant of both.
    """

    name: str
    throughput_target_mbps: float
    delay_target_ms: float


SLICE_TEMPLATES: List[SliceSpec] = [
    SliceSpec("embb", throughput_target_mbps=5.0, delay_target_ms=50.0),
    SliceSpec("urllc", throughput_target_mbps=1.0, delay_target_ms=5.0),
    SliceSpec("mmtc", throughput_target_mbps=0.2, delay_target_ms=100.0),
]


def build_slices(n_slices: int) -> List[SliceSpec]:
    """Return ``n_slices`` slices, cycling the three 5G service templates.

    The paper evaluates 2, 4 and 8 slices. With more than three, the templates
    repeat and are suffixed so that names stay unique.
    """
    out: List[SliceSpec] = []
    for i in range(n_slices):
        tpl = SLICE_TEMPLATES[i % len(SLICE_TEMPLATES)]
        suffix = "" if i < len(SLICE_TEMPLATES) else f"_{i // len(SLICE_TEMPLATES)}"
        out.append(SliceSpec(tpl.name + suffix, tpl.throughput_target_mbps, tpl.delay_target_ms))
    return out


# ---------------------------------------------------------------------------
# 6. Traffic profiles
# ---------------------------------------------------------------------------

# The paper generates per-UE UDP traffic with iPerf at three levels.
# Values are (min_mbps, max_mbps); each UE draws a fixed rate in that range.
TRAFFIC_LEVELS: Dict[str, Tuple[float, float]] = {
    "light": (0.1, 1.0),
    "medium": (1.0, 5.0),
    "heavy": (5.0, 10.0),
}

UDP_PACKET_BYTES: int = 1400  # typical iPerf UDP payload


# ---------------------------------------------------------------------------
# 7. Reward weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    """Lagrangian weights for the relaxed constrained problem.

    ``w_energy`` scales the energy-saving term; the three lambdas price the
    throughput, delay and coverage violations respectively.

    ``lambda_coverage`` has no counterpart in the baseline: coverage continuity
    is identically satisfied when there is only one RU, which is why the
    baseline never needed it.
    """

    w_energy: float = 1.0
    lambda_throughput: float = 0.7  # baseline config's lambda_p
    lambda_delay: float = 0.3  # baseline config's lambda_d
    lambda_coverage: float = 1.0
    delay_clip_factor: float = 2.0  # the paper's d_hat = min(d, 2*D)

    # Which quantity the energy term of the *training* reward uses.
    #   "sleep_ratio"  - the paper's b_t / N_ts, averaged over RUs. Default, so
    #                    that learning curves sit on the same scale as the
    #                    paper's Figure 5 and the comparison is like-for-like.
    #   "power_model"  - fraction of energy actually saved under PowerParams.
    # Energy saved under the power model is always *reported* as an evaluation
    # metric either way; this switch only decides what the agent optimises.
    energy_metric: str = "sleep_ratio"

    # The paper sums QoS penalties over UEs, so the penalty scale grows with
    # the number of UEs while the energy term does not. Penalties are scaled
    # by ``reference_n_ue / n_ue``, which is exactly the paper's form for its
    # 8-UE testbed and keeps the energy/QoS balance fixed as networks grow.
    # Without this, a policy trained on 8 UEs would face a different objective
    # when transferred to 32 - which would confound the generalisation test.
    reference_n_ue: int = 8


REWARD_DEFAULT = RewardWeights()


# ---------------------------------------------------------------------------
# 8. Observation normalisation
# ---------------------------------------------------------------------------

# Kept byte-identical to the baseline repo's config.NORMALIZATION so that a
# policy trained here remains portable to the real E2 interface without any
# change to the state definition. Order matters: this is the feature order of
# the 17-dimensional per-UE observation vector.
#
# 10 MAC features followed by 7 KPM features.
OBS_FEATURE_NAMES: List[str] = [
    # MAC
    "dl_curr_tbs",
    "dl_sched_rb",
    "pusch_snr",
    "pucch_snr",
    "wb_cqi",
    "dl_mcs1",
    "ul_mcs1",
    "phr",
    "dl_bler",
    "ul_bler",
    # KPM
    "pdcp_sdu_volume_dl",
    "pdcp_sdu_volume_ul",
    "rlc_sdu_delay_dl",
    "ue_thp_dl",
    "ue_thp_ul",
    "prb_tot_dl",
    "prb_tot_ul",
]

OBS_NORMALISATION: Dict[str, Tuple[float, float]] = {
    "dl_curr_tbs": (0.0, 3000.0),  # bytes
    "dl_sched_rb": (0.0, 106.0),  # PRBs
    "pusch_snr": (0.0, 70.0),  # dB
    "pucch_snr": (0.0, 50.0),  # dB
    "wb_cqi": (0.0, 15.0),
    "dl_mcs1": (0.0, 28.0),
    "ul_mcs1": (0.0, 28.0),
    "phr": (20.0, 70.0),  # dB
    "dl_bler": (0.0, 0.5),
    "ul_bler": (0.0, 0.5),
    "pdcp_sdu_volume_dl": (0.0, 20000.0),  # bytes per reporting period
    "pdcp_sdu_volume_ul": (0.0, 20000.0),
    "rlc_sdu_delay_dl": (0.0, 1000.0),  # ms
    "ue_thp_dl": (0.0, 1000.0),  # kbps
    "ue_thp_ul": (0.0, 1000.0),
    "prb_tot_dl": (0.0, 1000.0),
    "prb_tot_ul": (0.0, 1000.0),
}

N_OBS_FEATURES: int = len(OBS_FEATURE_NAMES)
assert N_OBS_FEATURES == 17, "The baseline observation is 17-dimensional"


# ---------------------------------------------------------------------------
# 9. Scenario presets
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """One evaluation scenario.

    The paper's grid is 3 traffic levels x 3 slice counts with 8 UEs spread
    evenly across slices, on a single RU. ``n_ru`` generalises that to the
    multi-RU setting this project introduces.
    """

    name: str
    traffic_level: str = "medium"
    n_slices: int = 3
    n_ue: int = 8
    n_ru: int = 1

    cell_radius_m: float = 150.0
    ue_speed_mps: float = 1.0  # pedestrian mobility
    inter_site_distance_m: float = 200.0

    numerology: Numerology = field(default_factory=lambda: NUMEROLOGY_DEFAULT)
    radio: RadioParams = field(default_factory=lambda: RADIO_DEFAULT)
    power: PowerParams = field(default_factory=lambda: POWER_DEFAULT)
    reward: RewardWeights = field(default_factory=lambda: REWARD_DEFAULT)

    frames_per_step: int = 10  # a decision applies to this many 10 ms frames
    seed: int = 0

    @property
    def step_duration_ms(self) -> float:
        return 10.0 * self.frames_per_step

    def validate(self) -> None:
        self.numerology.validate()
        if self.traffic_level not in TRAFFIC_LEVELS:
            raise ValueError(f"unknown traffic level {self.traffic_level!r}")
        if self.n_ue < self.n_slices:
            raise ValueError("need at least one UE per slice")


def paper_scenarios(n_ru: int = 1, n_ue: int = 8) -> List[ScenarioConfig]:
    """The nine scenarios used in the baseline's Figures 5 and 7.

    Three traffic levels crossed with slice counts 2, 4 and 8.
    """
    out: List[ScenarioConfig] = []
    for level in ("light", "medium", "heavy"):
        for n_slices in (2, 4, 8):
            out.append(
                ScenarioConfig(
                    name=f"{level}_{n_slices}slice",
                    traffic_level=level,
                    n_slices=n_slices,
                    n_ue=n_ue,
                    n_ru=n_ru,
                )
            )
    return out
