"""
obs.py - Turn simulator state into the tensors the agents consume.
==================================================================

Every agent - baseline or ours - reads the same observation dictionary. What
differs is how much of it an encoder is *allowed* to use:

  * The baseline's encoders (GRU, MLP, Transformer) see only ``ue_x``, the
    17 E2 KPIs per UE. That is the paper's state definition, and it carries no
    information about which UE belongs to which slice.
  * The "+slice features" Transformer additionally sees each UE's slice
    one-hot and slice QoS targets, concatenated onto its KPIs. It is the
    strong baseline: it has the same information as the GNN, but as flat
    features rather than as graph structure.
  * The GNN sees the typed graph: UE, slice and RU nodes, and the edges
    between them.

Keeping one observation format for all of them means any difference in
results comes from the architecture, not from what the environment exposed.

Observation dictionary (numpy, one step)
----------------------------------------
    ue_x      [K, 17]   per-UE KPIs, the baseline's exact features
    ue_ru     [K]       index of each UE's serving RU
    ue_slice  [K]       index of each UE's slice
    sl_x      [R, I, 7] one node per (RU, slice) pair - see SLICE_FEATURES
    ru_x      [R, 3]    one node per RU - see RU_FEATURES
    ru_adj    [R, R]    RU-RU coverage-overlap weights, 0 on the diagonal

Every field is something a Near-RT RIC xApp can obtain over E2 or already
knows from its own configuration and last action.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

SLICE_FEATURES = [
    "throughput_target_mbps / 10",
    "delay_target_ms / 100",
    "class_embb",
    "class_urllc",
    "class_mmtc",
    "current_prb_share",
    "members_on_this_ru / 8",
]
RU_FEATURES = [
    "last_sleep_ratio",
    "last_prb_utilisation",
    "served_ues / 8",
]

N_SLICE_FEATURES = len(SLICE_FEATURES)
N_RU_FEATURES = len(RU_FEATURES)

_CLASS_INDEX = {"embb": 0, "urllc": 1, "mmtc": 2}


def build_obs(env, ue_obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Assemble the observation dictionary from an ``OranSimEnv``.

    ``ue_obs`` is what ``env.reset()`` / ``env.step()`` returned; everything
    else is read from the environment's current network state.
    """
    net = env.net
    n_ru, n_sl = len(net.rus), net.n_slices

    ue_ru = np.array([u.serving_ru for u in net.ues], dtype=np.int64)
    ue_slice = np.array([u.slice_id for u in net.ues], dtype=np.int64)

    sl_x = np.zeros((n_ru, n_sl, N_SLICE_FEATURES), dtype=np.float32)
    for r, ru in enumerate(net.rus):
        for i, spec in enumerate(net.slices):
            base = spec.name.split("_")[0]
            sl_x[r, i, 0] = spec.throughput_target_mbps / 10.0
            sl_x[r, i, 1] = spec.delay_target_ms / 100.0
            sl_x[r, i, 2 + _CLASS_INDEX[base]] = 1.0
            sl_x[r, i, 5] = ru.slice_prb_fraction[i]
            sl_x[r, i, 6] = np.sum((ue_ru == r) & (ue_slice == i)) / 8.0

    tel = env.last_telemetry
    ru_x = np.zeros((n_ru, N_RU_FEATURES), dtype=np.float32)
    for r, ru in enumerate(net.rus):
        ru_x[r, 0] = ru.sleep_ratio
        ru_x[r, 1] = float(tel.prb_utilisation[r]) if tel is not None else 0.0
        ru_x[r, 2] = np.sum(ue_ru == r) / 8.0

    ru_adj = env.ru_adjacency().astype(np.float32) if n_ru > 1 else np.zeros((1, 1), np.float32)
    np.fill_diagonal(ru_adj, 0.0)

    return {
        "ue_x": np.asarray(ue_obs, dtype=np.float32),
        "ue_ru": ue_ru,
        "ue_slice": ue_slice,
        "sl_x": sl_x,
        "ru_x": ru_x,
        "ru_adj": ru_adj,
    }


def collate(obs_list: List[Dict[str, np.ndarray]], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Stack a list of same-shaped observations into a batch of tensors.

    Within one training run the number of UEs, RUs and slices is fixed, so
    plain stacking is exact; no padding is needed.
    """
    out = {}
    for key in obs_list[0]:
        arr = np.stack([o[key] for o in obs_list])
        dtype = torch.long if arr.dtype.kind in "iu" else torch.float32
        out[key] = torch.as_tensor(arr, dtype=dtype, device=device)
    return out
