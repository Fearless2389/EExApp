"""
sim - A simulated 5G/6G O-RAN environment for energy-efficiency research.
=========================================================================

Replaces the hardware-in-the-loop environment of the EExApp baseline, which
cannot be run without a physical testbed, with a slot-level simulator that
runs anywhere NumPy runs.

Quick start
-----------
    from sim import OranSimEnv, ScenarioConfig

    env = OranSimEnv(ScenarioConfig(name="demo", traffic_level="medium",
                                    n_slices=3, n_ue=8, n_ru=1))
    obs = env.reset()                       # [n_ue, 17]
    obs, (r_e, r_q), done, info = env.step({"sleep":  [[10, 4, 0]],
                                            "slices": [[0.4, 0.3, 0.3]]})

Design notes are in ``sim/README.md``.
"""

from .env import OranSimEnv
from .metrics import compute_reward, energy_saved_fraction, qos_violation_ratio
from .network import Network, RadioUnit, UserEquipment
from .params import (
    NUMEROLOGY_DEFAULT,
    NUMEROLOGY_PAPER_CODE,
    OBS_FEATURE_NAMES,
    POWER_DEFAULT,
    RADIO_DEFAULT,
    REWARD_DEFAULT,
    Numerology,
    PowerParams,
    RadioParams,
    RewardWeights,
    ScenarioConfig,
    SliceSpec,
    build_slices,
    paper_scenarios,
)
from .scheduler import StepTelemetry, run_step

__all__ = [
    "OranSimEnv",
    "Network",
    "RadioUnit",
    "UserEquipment",
    "ScenarioConfig",
    "Numerology",
    "RadioParams",
    "PowerParams",
    "RewardWeights",
    "SliceSpec",
    "StepTelemetry",
    "run_step",
    "build_slices",
    "paper_scenarios",
    "compute_reward",
    "qos_violation_ratio",
    "energy_saved_fraction",
    "OBS_FEATURE_NAMES",
    "NUMEROLOGY_DEFAULT",
    "NUMEROLOGY_PAPER_CODE",
    "RADIO_DEFAULT",
    "POWER_DEFAULT",
    "REWARD_DEFAULT",
]
