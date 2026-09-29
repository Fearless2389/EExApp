"""
policy.py - Actors, critics and the bipartite critic GAT.
=========================================================

One ``Policy`` class covers every method in the comparison; a ``PolicySpec``
switches its parts on and off. Keeping one implementation is what makes the
comparison fair: EExApp, its ablations and our GNN differ *only* in the
fields of the spec.

    PolicySpec(encoder="transformer", dual=True,  gat=True)   EExApp (paper)
    PolicySpec(encoder="transformer", dual=False, gat=False)  SASC
    PolicySpec(encoder="mlp",         dual=True,  gat=True)   w/o Trans
    PolicySpec(encoder="transformer", dual=True,  gat=False)  w/o GAT
    PolicySpec(encoder="mlp",         dual=True,  gat=False)  w/o Both
    PolicySpec(encoder="gnn",         dual=True,  gat=True)   ours

Corrections relative to the released code
-----------------------------------------
Sleep head. The released EE actor emits 3 logits and slices them as if there
were N+1, which confines the sleep length b to {0, 1}. Here b has N+1 logits
over {0..N}, and the position a has N+1 logits masked to {0..N-b}, so every
valid (a, b, c) is reachable and the log-probability is exact.

Slice head. The released RS actor samples a Gaussian, passes it through a
softmax and clamps, then scores the result under the *untransformed* Gaussian,
which makes the PPO ratio meaningless. Here the slice split is a Dirichlet,
whose support is the simplex, so the log-probability is exact.

Critic GAT. The released GAT is never trained, its attention softmax leaks
onto non-edges, and its learned adjacency has no effect. Here it is trained
and the bipartite graph is complete, so no mask is needed. See
``BipartiteCriticGAT`` for how it is trained, which the paper does not state.

For the GNN, the slice head scores each (RU, slice) node individually, so one
set of weights handles any number of slices. The set encoders must use a
fixed-width output layer instead, sized for the largest slice count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from .encoders import MAX_SLICES, EncOut, make_encoder

_NEG = -1e9  # masked-logit value; finite so entropy stays well defined


@dataclass(frozen=True)
class PolicySpec:
    encoder: str = "transformer"
    dual: bool = True  # separate EE and RS actors (EExApp) vs one shared actor (SASC)
    gat: bool = True  # bipartite GAT over the two critics (dual only)
    ru_edges: bool = True  # GNN only: RU-RU coordination edges
    d_model: int = 64
    hidden: int = 64


def mlp(sizes, act=nn.Tanh, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Action distributions
# ---------------------------------------------------------------------------


class SleepDist:
    """Distribution over sleep schedules (a, b, c) with a + b + c = N, per RU.

    Factorised as p(b) p(a | b): first the sleep length, then where the
    window starts. c follows from the constraint.
    """

    def __init__(self, logits_b: torch.Tensor, logits_a: torch.Tensor, n_dl: int):
        self.n = n_dl
        self.logits_b = logits_b  # [B, R, N+1]
        self.logits_a = logits_a  # [B, R, N+1]
        self._idx = torch.arange(n_dl + 1, device=logits_b.device)

    def _dist_b(self) -> Categorical:
        return Categorical(logits=self.logits_b)

    def _dist_a(self, b: torch.Tensor) -> Categorical:
        allowed = self._idx <= (self.n - b)[..., None]
        return Categorical(logits=self.logits_a.masked_fill(~allowed, _NEG))

    def sample(self):
        b = self._dist_b().sample()
        a = self._dist_a(b).sample()
        return a, b

    def mode(self):
        b = self.logits_b.argmax(-1)
        allowed = self._idx <= (self.n - b)[..., None]
        a = self.logits_a.masked_fill(~allowed, _NEG).argmax(-1)
        return a, b

    def log_prob(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._dist_b().log_prob(b) + self._dist_a(b).log_prob(a)

    def entropy(self, b: torch.Tensor) -> torch.Tensor:
        return self._dist_b().entropy() + self._dist_a(b).entropy()


class SliceDist:
    """Dirichlet over per-slice PRB shares, per RU. Support is the simplex."""

    def __init__(self, concentration: torch.Tensor):
        self.conc = concentration.clamp(1e-3, 500.0)  # [B, R, I]
        self.dist = Dirichlet(self.conc)

    @staticmethod
    def _clean(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=1e-6)
        return x / x.sum(-1, keepdim=True)

    def sample(self) -> torch.Tensor:
        return self._clean(self.dist.sample())

    def mode(self) -> torch.Tensor:
        # The mean rather than the mode: the mode is undefined for
        # concentrations below one, the mean always exists.
        return self.conc / self.conc.sum(-1, keepdim=True)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(self._clean(x))

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()


# ---------------------------------------------------------------------------
# Bipartite critic GAT
# ---------------------------------------------------------------------------


class BipartiteCriticGAT(nn.Module):
    """The paper's bipartite GAT (Eq. 14), implemented so that it can learn.

    Source nodes are the two critics, carrying their value estimates V_alpha
    and V_beta. Target nodes are the two actors. Each actor j attends over both
    critics and receives an actor-specific value

        V_hat_j = w_s * sum_i gamma_ij V_i,   gamma_ij = softmax_i(e_ij)

    with e_ij computed GATv2-style from the critic value, a learned query for
    actor j, and the network state - so the weighting can change with
    conditions, which is the paper's stated motivation.

    Training signal. The paper does not say what trains the GAT, and the
    released code never trains it. Here V_hat_j is regressed onto the total
    return and used as actor j's baseline. The critic values enter detached,
    so the two decomposed critics stay pure estimates of the energy and QoS
    returns and the GAT learns only how to combine them. ``w_s`` starts at 2
    so that equal attention reproduces V_alpha + V_beta, the unweighted sum.
    """

    def __init__(self, d_state: int, d_hidden: int = 16):
        super().__init__()
        self.src = nn.Linear(1, d_hidden)
        self.queries = nn.Parameter(torch.randn(2, d_hidden) * 0.1)
        self.state = nn.Linear(d_state, d_hidden)
        self.att = nn.Parameter(torch.randn(d_hidden) * 0.1)
        self.w_s = nn.Parameter(torch.tensor(2.0))

    def forward(self, v: torch.Tensor, h_glob: torch.Tensor):
        """v: [B, 2] critic values -> (V_hat [B, 2], gamma [B, 2 targets, 2 sources])."""
        v = v.detach()
        s = self.src(v[..., None])  # [B, 2 src, h]
        q = self.queries[None] + self.state(h_glob)[:, None]  # [B, 2 tgt, h]
        z = F.leaky_relu(q[:, :, None, :] + s[:, None, :, :], 0.2)
        gamma = torch.softmax((z * self.att).sum(-1), dim=-1)  # over sources
        v_hat = self.w_s * (gamma * v[:, None, :]).sum(-1)
        return v_hat, gamma


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PolicyOut(NamedTuple):
    sleep: SleepDist
    slices: SliceDist
    values: Dict[str, torch.Tensor]
    gamma: Optional[torch.Tensor]


class Policy(nn.Module):
    def __init__(self, spec: PolicySpec, n_dl: int):
        super().__init__()
        self.spec = spec
        self.n_dl = n_dl
        d, hid = spec.d_model, spec.hidden
        kwargs = {"use_ru_edges": spec.ru_edges} if spec.encoder == "gnn" else {}
        self.encoder = make_encoder(spec.encoder, d, **kwargs)
        self.slice_from_nodes = spec.encoder == "gnn"

        # Actors. Dual: independent EE and RS trunks. SASC: one shared trunk.
        if spec.dual:
            self.ee_trunk = mlp([d, hid, hid], out_act=nn.Tanh)
            self.rs_trunk = mlp([d, hid, hid], out_act=nn.Tanh)
        else:
            self.shared_trunk = mlp([d, hid, hid], out_act=nn.Tanh)
        self.head_b = nn.Linear(hid, n_dl + 1)
        self.head_a = nn.Linear(hid, n_dl + 1)
        if self.slice_from_nodes:
            self.slice_scorer = mlp([d + hid, hid, 1])
        else:
            self.slice_head = nn.Linear(hid, MAX_SLICES)

        # Critics.
        if spec.dual:
            self.v_alpha = mlp([d, hid, hid, 1])
            self.v_beta = mlp([d, hid, hid, 1])
            self.gat = BipartiteCriticGAT(d) if spec.gat else None
        else:
            self.v = mlp([d, hid, hid, 1])
            self.gat = None

    # -- parameter groups, so actors and critics can use different rates --

    def actor_parameters(self):
        mods = [self.head_b, self.head_a]
        mods += [self.ee_trunk, self.rs_trunk] if self.spec.dual else [self.shared_trunk]
        mods += [self.slice_scorer] if self.slice_from_nodes else [self.slice_head]
        return [p for m in mods for p in m.parameters()]

    def critic_parameters(self):
        if self.spec.dual:
            mods = [self.v_alpha, self.v_beta] + ([self.gat] if self.gat is not None else [])
        else:
            mods = [self.v]
        return [p for m in mods for p in m.parameters()]

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> PolicyOut:
        enc: EncOut = self.encoder(batch)
        n_sl = batch["sl_x"].shape[2]

        if self.spec.dual:
            t_ee, t_rs = self.ee_trunk(enc.h_ru), self.rs_trunk(enc.h_ru)
        else:
            t_ee = t_rs = self.shared_trunk(enc.h_ru)

        sleep = SleepDist(self.head_b(t_ee), self.head_a(t_ee), self.n_dl)

        if self.slice_from_nodes:
            ctx = t_rs[:, :, None, :].expand(-1, -1, n_sl, -1)
            logits = self.slice_scorer(torch.cat([enc.h_slice, ctx], dim=-1)).squeeze(-1)
        else:
            logits = self.slice_head(t_rs)[..., :n_sl]
        slices = SliceDist(F.softplus(logits) + 0.5)

        gamma = None
        if self.spec.dual:
            v_a = self.v_alpha(enc.h_glob).squeeze(-1)
            v_b = self.v_beta(enc.h_glob).squeeze(-1)
            if self.gat is not None:
                v_hat, gamma = self.gat(torch.stack([v_a, v_b], dim=-1), enc.h_glob)
                values = {"v_a": v_a, "v_b": v_b, "vhat_a": v_hat[:, 0], "vhat_b": v_hat[:, 1]}
            else:
                values = {"v_a": v_a, "v_b": v_b, "vhat_a": v_a + v_b, "vhat_b": v_a + v_b}
        else:
            values = {"v": self.v(enc.h_glob).squeeze(-1)}
        return PolicyOut(sleep, slices, values, gamma)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
