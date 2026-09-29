"""
power.py - Radio Unit energy model.
===================================

The baseline never measures energy. It uses the sleep ratio ``b_t / N_sf`` as a
proxy throughout, so none of its results can be stated in watts or in a
percentage energy saving. This module replaces that proxy with an explicit
load-dependent power model, which is what lets us report real numbers and what
makes the sleep-transition cost expressible.

Model
-----
Per downlink slot, an RU consumes

    active:  p_static + p_dynamic * prb_utilisation
    sleep:   p_sleep

and each sleep -> active transition additionally costs ``e_transition_j``.

The transition term is the reason a policy cannot win by toggling sleep on and
off every slot; the baseline has no such term and therefore no defence against
that behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import PowerParams


@dataclass
class RuEnergyAccounting:
    """Accumulates the energy an RU consumes over a decision step."""

    params: PowerParams
    slot_duration_ms: float

    energy_j: float = 0.0
    baseline_energy_j: float = 0.0  # energy if the RU had stayed fully active
    active_slots: int = 0
    sleep_slots: int = 0
    transitions: int = 0

    def account_slot(self, is_sleeping: bool, prb_utilisation: float) -> None:
        """Record one downlink slot.

        ``prb_utilisation`` is in [0, 1] and only matters when awake.
        """
        dt_s = self.slot_duration_ms / 1000.0
        u = float(np.clip(prb_utilisation, 0.0, 1.0))

        if is_sleeping:
            self.energy_j += self.params.p_sleep_w * dt_s
            self.sleep_slots += 1
        else:
            self.energy_j += (self.params.p_static_w + self.params.p_dynamic_w * u) * dt_s
            self.active_slots += 1

        # The comparison point is always a fully active RU carrying the same load.
        self.baseline_energy_j += (self.params.p_static_w + self.params.p_dynamic_w * u) * dt_s

    def account_transition(self) -> None:
        """Charge one sleep -> active wake-up."""
        self.energy_j += self.params.e_transition_j
        self.transitions += 1

    # ---- readouts ---------------------------------------------------------

    @property
    def energy_saved_fraction(self) -> float:
        """Fraction of energy saved against an always-active RU, in [0, 1].

        Negative values are possible in principle if transition costs exceed
        the sleep saving; that is a real effect and is not clipped away.
        """
        if self.baseline_energy_j <= 0.0:
            return 0.0
        return 1.0 - (self.energy_j / self.baseline_energy_j)

    @property
    def sleep_ratio(self) -> float:
        """The baseline's proxy metric, kept so the two can be compared."""
        total = self.active_slots + self.sleep_slots
        return self.sleep_slots / total if total else 0.0

    @property
    def mean_power_w(self) -> float:
        total_slots = self.active_slots + self.sleep_slots
        if total_slots == 0:
            return 0.0
        duration_s = total_slots * self.slot_duration_ms / 1000.0
        return self.energy_j / duration_s

    def reset(self) -> None:
        self.energy_j = 0.0
        self.baseline_energy_j = 0.0
        self.active_slots = 0
        self.sleep_slots = 0
        self.transitions = 0
