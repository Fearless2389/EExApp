"""
radio.py - Propagation, SINR and link adaptation.
=================================================

Turns geometry into bits. The chain implemented here is the standard one:

    distance -> path loss -> received power -> SINR -> CQI -> MCS -> TBS

Everything is vectorised over UEs so a step never loops in Python over users.

Public entry points
-------------------
``path_loss_db``            free-space-like UMi LOS path loss
``rsrp_dbm``                received reference power from one RU at one UE
``downlink_sinr_db``        SINR including interference from awake neighbours
``uplink_snr_db``           PUSCH/PUCCH SNR at the RU
``sinr_to_cqi``             CQI selection at ~10% BLER
``cqi_to_mcs``              CQI index mapped onto the 0-28 MCS scale
``bits_per_prb``            payload bits carried by one PRB in one slot
``block_error_rate``        residual BLER after link adaptation
``power_headroom_db``       UE power headroom report
"""

from __future__ import annotations

import numpy as np

from .params import (
    CQI_SINR_THRESHOLDS_DB,
    CQI_SPECTRAL_EFFICIENCY,
    MAX_CQI,
    MAX_MCS,
    N_RE_PER_PRB_PER_SLOT,
    RadioParams,
)

# ---------------------------------------------------------------------------
# dB helpers
# ---------------------------------------------------------------------------


def db_to_lin(x_db: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin_to_db(x_lin: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(np.asarray(x_lin, dtype=float), 1e-30))


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def path_loss_db(distance_m: np.ndarray, radio: RadioParams, shadowing_db: np.ndarray | float = 0.0) -> np.ndarray:
    """3GPP TR 38.901 UMi Street Canyon LOS path loss, plus shadowing.

        PL = 32.4 + 21*log10(d) + 20*log10(f_GHz) + X_sigma

    A minimum coupling loss floor is applied so that a UE standing at the mast
    does not receive an unphysical amount of power.
    """
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    pl = (
        radio.pathloss_intercept_db
        + radio.pathloss_exponent_factor * np.log10(d)
        + 20.0 * np.log10(radio.carrier_freq_ghz)
        + np.asarray(shadowing_db, dtype=float)
    )
    return np.maximum(pl, radio.min_coupling_loss_db)


def rsrp_dbm(distance_m: np.ndarray, radio: RadioParams, shadowing_db: np.ndarray | float = 0.0) -> np.ndarray:
    """Per-PRB received power at the UE from one RU transmitting on all PRBs.

    The RU's total conducted power is shared across its PRBs, so the per-PRB
    transmit power is ``P_total - 10*log10(n_prb)``.
    """
    tx_per_prb = radio.ru_tx_power_dbm - 10.0 * np.log10(radio.n_prb)
    return tx_per_prb + radio.ru_antenna_gain_dbi - path_loss_db(distance_m, radio, shadowing_db)


# ---------------------------------------------------------------------------
# SINR
# ---------------------------------------------------------------------------


def downlink_sinr_db(
    serving_rsrp_dbm: np.ndarray,
    interferer_rsrp_dbm: np.ndarray,
    radio: RadioParams,
) -> np.ndarray:
    """Downlink SINR per UE.

    Parameters
    ----------
    serving_rsrp_dbm
        Shape ``[n_ue]``. Received power from each UE's serving RU.
    interferer_rsrp_dbm
        Shape ``[n_ue, n_interferers]``. Received power from every *awake*
        non-serving RU. Pass an empty second axis when there is only one RU,
        which reduces the expression to plain SNR.

    Returns
    -------
    SINR in dB, shape ``[n_ue]``.
    """
    noise_lin = db_to_lin(radio.thermal_noise_dbm_per_prb(radio.ue_noise_figure_db))
    signal_lin = db_to_lin(serving_rsrp_dbm)

    interferer_rsrp_dbm = np.atleast_2d(np.asarray(interferer_rsrp_dbm, dtype=float))
    if interferer_rsrp_dbm.size == 0:
        interference_lin = np.zeros_like(signal_lin)
    else:
        interference_lin = db_to_lin(interferer_rsrp_dbm).sum(axis=1)

    return lin_to_db(signal_lin / (interference_lin + noise_lin))


def uplink_snr_db(distance_m: np.ndarray, radio: RadioParams, shadowing_db: np.ndarray | float = 0.0) -> np.ndarray:
    """SNR of a UE's uplink transmission as measured at its serving RU.

    Used to populate the PUSCH and PUCCH SNR observation features. PUCCH is
    reported a few dB lower than PUSCH because control is sent over far fewer
    resource blocks; the offset is applied by the caller.
    """
    noise_dbm = radio.thermal_noise_dbm_per_prb(radio.ru_noise_figure_db)
    rx_dbm = radio.ue_tx_power_dbm + radio.ru_antenna_gain_dbi - path_loss_db(distance_m, radio, shadowing_db)
    return rx_dbm - noise_dbm


def power_headroom_db(distance_m: np.ndarray, radio: RadioParams, target_rx_dbm: float = -90.0) -> np.ndarray:
    """UE power headroom: how much transmit power remains unused.

    Modelled as the difference between the UE's maximum transmit power and the
    power it would need to reach a target received level at the RU.
    """
    required_tx = target_rx_dbm + path_loss_db(distance_m, radio) - radio.ru_antenna_gain_dbi
    return np.clip(radio.ue_tx_power_dbm - required_tx, -20.0, 70.0)


# ---------------------------------------------------------------------------
# Link adaptation
# ---------------------------------------------------------------------------


def sinr_to_cqi(sinr_db: np.ndarray) -> np.ndarray:
    """Select the highest CQI whose SINR threshold the UE meets.

    Returns an integer array in ``[0, 15]``. Zero means the channel cannot
    support any transmission.
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    # searchsorted counts how many thresholds the SINR exceeds, which is
    # exactly the CQI index.
    return np.searchsorted(CQI_SINR_THRESHOLDS_DB, sinr_db, side="right").astype(int)


def cqi_to_mcs(cqi: np.ndarray) -> np.ndarray:
    """Map a CQI index onto the 0-28 MCS scale used by the MAC observation."""
    cqi = np.asarray(cqi, dtype=float)
    return np.clip(np.round(cqi * MAX_MCS / MAX_CQI), 0, MAX_MCS).astype(int)


def spectral_efficiency(cqi: np.ndarray) -> np.ndarray:
    """Bits per resource element for the given CQI indices."""
    idx = np.clip(np.asarray(cqi, dtype=int), 0, MAX_CQI)
    return CQI_SPECTRAL_EFFICIENCY[idx]


def transmission_rank(sinr_db: np.ndarray, radio: RadioParams) -> np.ndarray:
    """Number of spatial layers used for each UE, from 1 to ``radio.mimo_layers``.

    Rank adaptation: higher SINR supports more parallel streams. The thresholds
    are a coarse stand-in for the rank indicator a UE would report, and are
    capped by the antenna configuration.
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    thresholds = np.asarray(radio.rank_sinr_thresholds_db, dtype=float)
    rank = 1 + np.searchsorted(thresholds, sinr_db, side="right")
    return np.clip(rank, 1, max(1, radio.mimo_layers)).astype(int)


def bits_per_prb(cqi: np.ndarray, rank: np.ndarray | int = 1) -> np.ndarray:
    """Payload bits carried by one PRB in one slot at the given CQI and rank."""
    return spectral_efficiency(cqi) * N_RE_PER_PRB_PER_SLOT * np.asarray(rank)


def block_error_rate(sinr_db: np.ndarray, cqi: np.ndarray) -> np.ndarray:
    """Residual block error rate after link adaptation.

    Link adaptation targets 10% BLER at the SINR threshold of the selected
    CQI. Above that threshold the error rate falls off; the logistic below is
    a smooth stand-in for a real BLER curve and is clipped to the baseline's
    observation range of 0 to 0.5.
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    cqi = np.clip(np.asarray(cqi, dtype=int), 0, MAX_CQI)

    # SINR margin above the threshold of the chosen CQI.
    thresholds = np.concatenate(([-np.inf], CQI_SINR_THRESHOLDS_DB))
    margin = sinr_db - thresholds[cqi]
    margin = np.where(np.isfinite(margin), margin, 0.0)

    bler = 0.1 / (1.0 + np.exp(1.2 * margin))
    # A UE with no usable CQI loses everything it is sent.
    bler = np.where(cqi == 0, 0.5, bler)
    return np.clip(bler, 0.0, 0.5)
