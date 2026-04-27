"""
DeepSet Q-value network for permutation-invariant vehicle representations.

Vanilla MLP Q-networks are sensitive to the order in which vehicles appear in
the observation row.  A DeepSet applies a shared encoder φ to every vehicle
independently, aggregates with sum-pooling, then maps the fixed-size embedding
together with the action to a Q-value via ρ.

Architecture
------------
    φ : (n_features,) → (embed_dim,)   [shared per-vehicle MLP]
    Σ : sum-pool over N_OBS_VEHICLES vehicles
    ρ : (embed_dim + action_dim,) → (1,) [final Q-value MLP]

Reference: Zaheer et al., "Deep Sets" (NeurIPS 2017).
"""

import torch
import torch.nn as nn
from torch import Tensor

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)


def _mlp(in_dim: int, hidden: list[int], out_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    if out_dim is not None:
        layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class DeepSetQNetwork(QValueNetwork):
    """
    Permutation-invariant Q-network using a DeepSet architecture.

    Parameters
    ----------
    n_vehicles   : number of vehicles in the observation (rows)
    n_features   : features per vehicle (columns)
    action_dim   : dimension of the action representation (one-hot size)
    phi_hidden   : hidden layer widths for the per-vehicle encoder φ
    rho_hidden   : hidden layer widths for the Q-value head ρ
    """

    def __init__(
        self,
        n_vehicles: int,
        n_features: int,
        action_dim: int,
        phi_hidden: list[int] | None = None,
        rho_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._n_vehicles = n_vehicles
        self._n_features = n_features
        self._action_dim_val = action_dim

        phi_hidden = phi_hidden or [64, 64]
        rho_hidden = rho_hidden or [256, 256]
        embed_dim = phi_hidden[-1]

        self.phi = _mlp(n_features, phi_hidden)
        self.rho = _mlp(embed_dim + action_dim, rho_hidden, out_dim=1)

    # ── QValueNetwork interface ───────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self._n_vehicles * self._n_features

    @property
    def action_dim(self) -> int:
        return self._action_dim_val

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        """
        Args
        ----
        state_batch  : (B, state_dim)
        action_batch : (B, action_dim)  or  (B, n_actions, action_dim)

        Returns
        -------
        Q-values: (B,)  or  (B, n_actions)
        """
        batch_size = state_batch.shape[0]
        embedding = self._embed(state_batch, batch_size)  # (B, embed_dim)

        if action_batch.dim() == 2:
            # Single action per state: (B, action_dim)
            x = torch.cat([embedding, action_batch], dim=-1)
            return self.rho(x).squeeze(-1)  # (B,)
        else:
            # Multiple actions per state: (B, n_actions, action_dim)
            n_actions = action_batch.shape[1]
            emb_exp = embedding.unsqueeze(1).expand(-1, n_actions, -1)
            x = torch.cat([emb_exp, action_batch], dim=-1)
            return self.rho(x).squeeze(-1)  # (B, n_actions)

    # ── nn.Module forward ─────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., state_dim + action_dim) — split and delegate to get_q_values
        state = x[..., : self.state_dim]
        action = x[..., self.state_dim :]
        if state.dim() == 3:
            # (B, n_actions, state_dim) — use first row (all rows identical)
            state = state[:, 0, :]
        return self.get_q_values(state, action)

    # ── internal ──────────────────────────────────────────────────────────────

    def _embed(self, state_batch: Tensor, batch_size: int) -> Tensor:
        """Apply φ to every vehicle then sum-pool → (B, embed_dim)."""
        vehicles = state_batch.view(batch_size, self._n_vehicles, self._n_features)
        flat = vehicles.reshape(batch_size * self._n_vehicles, self._n_features)
        encoded = self.phi(flat)
        embed_dim = encoded.shape[-1]
        return encoded.view(batch_size, self._n_vehicles, embed_dim).sum(dim=1)
