"""
encoders.py - State encoders: the paper's, the strong baseline, and ours.
========================================================================

Every encoder maps a batch of observations (see ``obs.py``) to the same three
outputs, so everything downstream - actors, critics, PPO - is shared and the
encoder is the only thing that differs between the compared methods:

    h_ru     [B, R, d]      one embedding per RU; the actors act on this
    h_slice  [B, R, I, d]   one embedding per (RU, slice) pair, or None
    h_glob   [B, d]         whole-network summary; the critics read this

Encoders
--------
``TransformerSetEncoder``  The paper's encoder (Sec. III-B): project each UE's
                           17 KPIs to d=64, two self-attention layers with 4
                           heads and FFN 128, mean-pool. Permutation-invariant,
                           but it has no way to know which UE is in which
                           slice - that is not in the paper's state.
``TransformerPlusEncoder`` The same, with each UE's slice one-hot and slice QoS
                           targets appended to its features. Same information
                           as the GNN, presented as flat features. This is the
                           comparison that separates "the GNN wins because it
                           has more information" from "the GNN wins because of
                           how it structures that information".
``MLPSetEncoder``          The paper's "w/o Trans" ablation: pad to 10 UEs,
                           flatten, 2-layer MLP. Order-dependent.
``GRUSetEncoder``          What the released code actually ships instead of a
                           Transformer. Included for completeness; the
                           released-code comparison runs the authors' own
                           module rather than this re-implementation.
``HeteroGNNEncoder``       Ours. Typed message passing over UE, slice and RU
                           nodes with relation-specific GATv2 attention.

Multi-RU note: set encoders pool only the UEs served by each RU, so each RU's
agent sees its own cell and nothing else. That is the independent-learner
baseline. The GNN additionally passes messages between neighbouring RUs.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import N_RU_FEATURES, N_SLICE_FEATURES

N_UE_FEATURES = 17
MAX_SLICES = 8  # the paper's largest slice count; sizes the one-hot
MAX_UES_PADDED = 10  # the baseline's config.ENV['max_ues']


class EncOut(NamedTuple):
    h_ru: torch.Tensor
    h_slice: Optional[torch.Tensor]
    h_glob: torch.Tensor


def ru_membership(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Boolean ``[B, R, K]``: UE k is served by RU r."""
    n_ru = batch["ru_x"].shape[1]
    ru_ids = torch.arange(n_ru, device=batch["ue_ru"].device)
    return batch["ue_ru"][:, None, :] == ru_ids[None, :, None]


def _padded_sequences(x: torch.Tensor, mask: torch.Tensor, max_len: int) -> torch.Tensor:
    """Gather each set's members in index order, zero-pad or truncate to ``max_len``.

    x: [N, K, F], mask: [N, K] -> [N, max_len, F]. This is the baseline's
    padding scheme: padded rows are zeros and are fed to the network as if
    they were real UEs.
    """
    n, k, f = x.shape
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    take = min(k, max_len)
    idx = order[:, :take]
    gathered = x.gather(1, idx[..., None].expand(-1, -1, f))
    valid = mask.gather(1, idx)
    gathered = gathered * valid[..., None]
    if take < max_len:
        gathered = torch.cat([gathered, x.new_zeros(n, max_len - take, f)], dim=1)
    return gathered


# ---------------------------------------------------------------------------
# Set encoders (the paper's family)
# ---------------------------------------------------------------------------


class _SetEncoder(nn.Module):
    """Shared plumbing: one pooled vector per RU from that RU's UEs."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def ue_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["ue_x"]

    def encode_sets(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> EncOut:
        x = self.ue_features(batch)  # [B, K, F]
        mask = ru_membership(batch)  # [B, R, K]
        b, r, k = mask.shape
        xs = x[:, None].expand(b, r, k, x.shape[-1]).reshape(b * r, k, x.shape[-1])
        h = self.encode_sets(xs, mask.reshape(b * r, k)).reshape(b, r, self.d_model)
        return EncOut(h_ru=h, h_slice=None, h_glob=h.mean(dim=1))


class TransformerSetEncoder(_SetEncoder):
    """The paper's encoder: 2 layers, 4 heads, d=64, FFN 128, mean pooling."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, in_features: int = N_UE_FEATURES):
        super().__init__(d_model)
        self.proj = nn.Linear(in_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def encode_sets(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pad = ~mask
        # An RU with no UEs would give an all-masked row and NaN attention.
        # Unmask one position for those rows; its output is zeroed by the pool.
        empty = ~mask.any(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        h = self.transformer(self.proj(x), src_key_padding_mask=pad)
        m = mask[..., None].to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class TransformerPlusEncoder(TransformerSetEncoder):
    """The paper's Transformer, given each UE's slice identity and targets."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__(d_model, n_layers, n_heads, in_features=N_UE_FEATURES + MAX_SLICES + 2)

    def ue_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ue_slice = batch["ue_slice"]
        onehot = F.one_hot(ue_slice, MAX_SLICES).to(batch["ue_x"].dtype)
        # Slice targets are features 0 and 1 of the slice nodes (same on every RU).
        targets = batch["sl_x"][:, 0, :, 0:2]  # [B, I, 2]
        per_ue = targets.gather(1, ue_slice[..., None].expand(-1, -1, 2))
        return torch.cat([batch["ue_x"], onehot, per_ue], dim=-1)


class MLPSetEncoder(_SetEncoder):
    """The paper's "w/o Trans" ablation: pad to 10 UEs, flatten, 2-layer MLP."""

    def __init__(self, d_model: int = 64, hidden: int = 128):
        super().__init__(d_model)
        self.net = nn.Sequential(
            nn.Linear(MAX_UES_PADDED * N_UE_FEATURES, hidden), nn.ReLU(), nn.Linear(hidden, d_model)
        )

    def encode_sets(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq = _padded_sequences(x, mask, MAX_UES_PADDED)
        return self.net(seq.flatten(1))


class GRUSetEncoder(_SetEncoder):
    """The released code's encoder: linear projection + 2-layer GRU, final state."""

    def __init__(self, d_model: int = 64):
        super().__init__(d_model)
        self.proj = nn.Linear(N_UE_FEATURES, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)

    def encode_sets(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq = _padded_sequences(x, mask, MAX_UES_PADDED)
        _, h_n = self.gru(self.proj(seq))
        return h_n[-1]


# ---------------------------------------------------------------------------
# Heterogeneous GNN (ours)
# ---------------------------------------------------------------------------


class RelationAttention(nn.Module):
    """GATv2-style attention for one relation type, dense and masked.

    For destination node i and source node j with an edge between them:

        e_ij  = a^T LeakyReLU(W_q h_i + W_k h_j [+ W_e x_ij])
        alpha = softmax_j(e_ij)             over i's neighbours only
        m_i   = sum_j alpha_ij W_v h_j

    Dense rather than sparse because the graphs are small (tens of nodes) and
    dense batched attention is faster on CPU than scatter operations at this
    size. It is mathematically the same operator as PyTorch Geometric's
    ``GATv2Conv``; a node with no neighbours of this type receives zero.
    """

    def __init__(self, d: int, n_heads: int = 4, edge_dim: int = 0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_e = nn.Linear(edge_dim, d, bias=False) if edge_dim else None
        self.att = nn.Parameter(torch.randn(n_heads, self.dh) * (self.dh**-0.5))

    def forward(
        self,
        h_dst: torch.Tensor,
        h_src: torch.Tensor,
        mask: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, nd, d = h_dst.shape
        ns = h_src.shape[1]
        z = self.w_q(h_dst)[:, :, None, :] + self.w_k(h_src)[:, None, :, :]
        if self.w_e is not None and edge is not None:
            z = z + self.w_e(edge)
        z = F.leaky_relu(z, 0.2).view(b, nd, ns, self.h, self.dh)
        score = (z * self.att).sum(-1)  # [B, Nd, Ns, H]

        m = mask[..., None].expand_as(score)
        has_any = m.any(dim=2, keepdim=True)
        score = score.masked_fill(~m, float("-inf"))
        score = torch.where(has_any, score, torch.zeros_like(score))
        alpha = torch.softmax(score, dim=2) * m

        v = self.w_v(h_src).view(b, ns, self.h, self.dh)
        return torch.einsum("bnsh,bshd->bnhd", alpha, v).reshape(b, nd, d)


class _NodeUpdate(nn.Module):
    """Residual update of a node from its aggregated messages."""

    def __init__(self, d: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.ffn(torch.cat([h, msg], dim=-1)))


class HeteroLayer(nn.Module):
    """One round of typed message passing across all seven relations."""

    RELATIONS = ("sl<-ue", "sl<-ru", "ue<-sl", "ue<-ru", "ru<-sl", "ru<-ue", "ru<-ru")

    def __init__(self, d: int, n_heads: int, use_ru_edges: bool):
        super().__init__()
        self.use_ru_edges = use_ru_edges
        self.rel = nn.ModuleDict(
            {
                name.replace("<-", "_from_"): RelationAttention(d, n_heads, edge_dim=1 if name == "ru<-ru" else 0)
                for name in self.RELATIONS
                if use_ru_edges or name != "ru<-ru"
            }
        )
        self.upd_ue, self.upd_sl, self.upd_ru = _NodeUpdate(d), _NodeUpdate(d), _NodeUpdate(d)

    def forward(self, h_ue, h_sl, h_ru, masks):
        r = self.rel
        m_sl = r["sl_from_ue"](h_sl, h_ue, masks["sl_ue"]) + r["sl_from_ru"](h_sl, h_ru, masks["sl_ru"])
        m_ue = r["ue_from_sl"](h_ue, h_sl, masks["ue_sl"]) + r["ue_from_ru"](h_ue, h_ru, masks["ue_ru"])
        m_ru = r["ru_from_sl"](h_ru, h_sl, masks["ru_sl"]) + r["ru_from_ue"](h_ru, h_ue, masks["ru_ue"])
        if self.use_ru_edges:
            m_ru = m_ru + r["ru_from_ru"](h_ru, h_ru, masks["ru_ru"], masks["ru_ru_w"])
        return self.upd_ue(h_ue, m_ue), self.upd_sl(h_sl, m_sl), self.upd_ru(h_ru, m_ru)


class HeteroGNNEncoder(nn.Module):
    """Our encoder: typed message passing over the RAN graph.

    Nodes
      UE     - the 17 E2 KPIs
      slice  - one node per (RU, slice) pair: QoS targets, service class,
               current PRB share, member count
      RU     - last sleep ratio, PRB utilisation, served-UE count

    Edges (each with its own attention parameters)
      UE <-> slice   membership, restricted to the UE's serving RU
      UE <-> RU      serving association
      slice <-> RU   the slice instance lives on that RU
      RU <-> RU      coverage overlap, weighted; the coordination channel

    ``use_ru_edges=False`` removes the RU-RU relation. With it removed, RUs
    cannot exchange information and the multi-RU system degenerates to
    independent agents - which is the ablation that isolates coordination.
    """

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, use_ru_edges: bool = True):
        super().__init__()
        self.d_model = d_model
        self.emb_ue = nn.Linear(N_UE_FEATURES, d_model)
        self.emb_sl = nn.Linear(N_SLICE_FEATURES, d_model)
        self.emb_ru = nn.Linear(N_RU_FEATURES, d_model)
        self.layers = nn.ModuleList([HeteroLayer(d_model, n_heads, use_ru_edges) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    @staticmethod
    def build_masks(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ue_ru, ue_slice = batch["ue_ru"], batch["ue_slice"]
        b = ue_ru.shape[0]
        n_ru, n_sl = batch["sl_x"].shape[1:3]
        dev = ue_ru.device

        # Slice-instance node j stands for (RU sl_ru[j], slice sl_id[j]).
        sl_ru = torch.arange(n_ru, device=dev).repeat_interleave(n_sl)
        sl_id = torch.arange(n_sl, device=dev).repeat(n_ru)

        sl_ue = (ue_ru[:, None, :] == sl_ru[None, :, None]) & (ue_slice[:, None, :] == sl_id[None, :, None])
        ru_ue = ue_ru[:, None, :] == torch.arange(n_ru, device=dev)[None, :, None]
        ru_sl = (torch.arange(n_ru, device=dev)[:, None] == sl_ru[None, :]).expand(b, -1, -1)
        adj = batch["ru_adj"]
        return {
            "sl_ue": sl_ue,
            "ue_sl": sl_ue.transpose(1, 2),
            "ru_ue": ru_ue,
            "ue_ru": ru_ue.transpose(1, 2),
            "ru_sl": ru_sl,
            "sl_ru": ru_sl.transpose(1, 2),
            "ru_ru": adj > 0,
            "ru_ru_w": adj[..., None],
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> EncOut:
        b = batch["ue_x"].shape[0]
        n_ru, n_sl = batch["sl_x"].shape[1:3]
        masks = self.build_masks(batch)

        h_ue = self.emb_ue(batch["ue_x"])
        h_sl = self.emb_sl(batch["sl_x"].reshape(b, n_ru * n_sl, -1))
        h_ru = self.emb_ru(batch["ru_x"])
        for layer in self.layers:
            h_ue, h_sl, h_ru = layer(h_ue, h_sl, h_ru, masks)

        h_glob = self.readout(torch.cat([h_ru.mean(dim=1), h_sl.mean(dim=1)], dim=-1))
        return EncOut(h_ru=h_ru, h_slice=h_sl.view(b, n_ru, n_sl, self.d_model), h_glob=h_glob)


# ---------------------------------------------------------------------------


def make_encoder(name: str, d_model: int = 64, **kwargs) -> nn.Module:
    if name == "transformer":
        return TransformerSetEncoder(d_model)
    if name == "transformer_plus":
        return TransformerPlusEncoder(d_model)
    if name == "mlp":
        return MLPSetEncoder(d_model)
    if name == "gru":
        return GRUSetEncoder(d_model)
    if name == "gnn":
        return HeteroGNNEncoder(d_model, **kwargs)
    raise ValueError(f"unknown encoder {name!r}")
