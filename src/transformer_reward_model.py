#!/usr/bin/env python3
"""Dynamic Transformer reward model aligned with wentou_bid_dm_ddpo_v2."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)]


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = torch.matmul(hidden_states, self.query) / self.scale
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if (~valid_mask.any(dim=1)).any():
                raise ValueError("every reward-model sequence needs at least one valid token")
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


class TransformerRewardModel(nn.Module):
    """Sequence-length-independent scalar RM used by the internal reference."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        out_dim: int = 1,
        traj_add_a: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.traj_add_a = bool(traj_add_a)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.pooling = MaskedAttentionPooling(hidden_dim)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, time_steps, feature_dim = x.shape
        if feature_dim != self.in_dim:
            raise ValueError(f"expected feature dim {self.in_dim}, got {feature_dim}")
        if padding_mask is None and lengths is not None:
            padding_mask = (
                torch.arange(time_steps, device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None
        hidden = self.pos_encoder(self.input_proj(x))
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        pooled = self.pooling(hidden, valid_mask=padding_mask)
        return self.reward_head(pooled)


class TrajectoryTransformerRewardModel(TransformerRewardModel):
    """Adapter for the local ASPO `(states, actions, masks)` RM contract."""

    def __init__(self, state_dim: int, action_dim: int = 1, **kwargs):
        traj_add_a = bool(kwargs.pop("traj_add_a", False))
        super().__init__(
            in_dim=state_dim + action_dim if traj_add_a else state_dim,
            traj_add_a=traj_add_a,
            **kwargs,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        *,
        ignore_mask: bool = False,
    ) -> torch.Tensor:
        if self.traj_add_a:
            if actions is None:
                raise ValueError("actions are required when traj_add_a=True")
            features = torch.cat([actions, states], dim=-1)
        else:
            features = states
        padding_mask = None if ignore_mask else masks
        return super().forward(features, padding_mask=padding_mask).squeeze(-1)


def build_transformer_reward_model(
    state_dim: int,
    action_dim: int = 1,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    traj_add_a: bool = False,
) -> TrajectoryTransformerRewardModel:
    return TrajectoryTransformerRewardModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        traj_add_a=traj_add_a,
    )
