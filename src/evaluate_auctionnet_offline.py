#!/usr/bin/env python3
"""Evaluate bid policies in AuctionNet's held-out market-price replay environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_offline_bid_diffusion import (
    Config,
    DiffusionPolicy,
    Normalizer,
    RewardModel,
    robust_scores,
)


STATE_DIM = 16
NUM_TICKS = 48
DEFAULT_POLICIES = [
    "logged",
    "fixed_cpa",
    "pid",
    "bc",
    "iql",
    "cql",
    "td3_bc",
    "bcq",
    "mopo",
    "combo",
    "diffusion_bc_single",
    "diffusion_bc_best",
    "diffusion_ddpo_best",
]


@dataclass
class TickData:
    time_index: int
    pvalues: np.ndarray
    pvalue_sigmas: np.ndarray
    market_prices: np.ndarray
    logged_bids: np.ndarray
    potential_conversions: np.ndarray
    drop_priority: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--state-checkpoint-dir", default=None)
    parser.add_argument("--idm-checkpoint-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--best-of-n", type=int, default=16)
    parser.add_argument("--advertiser-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--read-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--comparison-target", default="diffusion_ddpo_best")
    return parser.parse_args()


def stable_seed(base_seed: int, *values: Any) -> int:
    payload = ":".join([str(base_seed), *map(str, values)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def competition_score(reward: float, cost: float, cpa_constraint: float) -> float:
    cpa = cost / (reward + 1e-10)
    if cpa <= cpa_constraint:
        return float(reward)
    return float(reward * (cpa_constraint / (cpa + 1e-10)) ** 2)


def mean_over_ticks(history: list[np.ndarray]) -> float:
    return float(np.mean([np.mean(values) for values in history])) if history else 0.0


def build_state(
    time_index: int,
    budget: float,
    remaining_budget: float,
    pvalues: np.ndarray,
    history_pvalue_info: list[np.ndarray],
    history_bids: list[np.ndarray],
    history_auction_result: list[np.ndarray],
    history_impression_result: list[np.ndarray],
    history_market_price: list[np.ndarray],
) -> np.ndarray:
    history_xi = [result[:, 0] for result in history_auction_result]
    history_pvalues = [result[:, 0] for result in history_pvalue_info]
    history_conversion = [result[:, 1] for result in history_impression_result]

    def last_three(history: list[np.ndarray]) -> float:
        return mean_over_ticks(history[-3:])

    state = np.asarray(
        [
            (NUM_TICKS - time_index) / NUM_TICKS,
            remaining_budget / budget if budget > 0 else 0.0,
            mean_over_ticks(history_bids),
            last_three(history_bids),
            mean_over_ticks(history_market_price),
            mean_over_ticks(history_pvalues),
            mean_over_ticks(history_conversion),
            mean_over_ticks(history_xi),
            last_three(history_market_price),
            last_three(history_pvalues),
            last_three(history_conversion),
            last_three(history_xi),
            float(np.mean(pvalues)),
            float(len(pvalues)),
            float(sum(len(values) for values in history_bids[-3:])),
            float(sum(len(values) for values in history_bids)),
        ],
        dtype=np.float32,
    )
    if state.shape != (STATE_DIM,):
        raise RuntimeError(f"Unexpected state shape: {state.shape}")
    return state


def action_from_tick(bids: np.ndarray, pvalues: np.ndarray) -> float:
    denominator = float(np.sum(pvalues))
    return float(np.sum(bids) / denominator) if denominator > 0 else 0.0


def continuous_reward_from_tick(pvalue_info: np.ndarray, impression_result: np.ndarray) -> float:
    return float(np.sum(pvalue_info[:, 0] * impression_result[:, 0]))


def load_normalizer(path: Path) -> Normalizer:
    payload = json.loads(path.read_text())
    normalizer = Normalizer()
    for key, value in payload.items():
        setattr(normalizer, key, np.asarray(value, dtype=np.float32) if isinstance(value, list) else value)
    return normalizer


class DiffusionAuctionStrategy:
    def __init__(
        self,
        name: str,
        policy: DiffusionPolicy,
        reward_models: list[RewardModel],
        normalizer: Normalizer,
        cfg: Config,
        fallback: Any,
        device: torch.device,
        best_of_n: int,
        seed: int,
    ):
        self.name = name
        self.policy = policy
        self.reward_models = reward_models
        self.normalizer = normalizer
        self.cfg = cfg
        self.fallback = fallback
        self.device = device
        self.best_of_n = best_of_n
        self.seed = seed
        self.budget = 100.0
        self.remaining_budget = 100.0
        self.cpa = 2.0
        self.category = 1
        self.period = 0
        self.advertiser = 0
        self.state_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.remaining_budget = self.budget
        self.state_history = []
        self.fallback.budget = self.budget
        self.fallback.remaining_budget = self.budget
        self.fallback.cpa = self.cpa
        self.fallback.category = self.category
        self.fallback.reset()

    def bidding(
        self,
        time_index: int,
        pvalues: np.ndarray,
        pvalue_sigmas: np.ndarray,
        history_pvalue_info: list[np.ndarray],
        history_bids: list[np.ndarray],
        history_auction_result: list[np.ndarray],
        history_impression_result: list[np.ndarray],
        history_market_price: list[np.ndarray],
    ) -> np.ndarray:
        state = build_state(
            time_index,
            self.budget,
            self.remaining_budget,
            pvalues,
            history_pvalue_info,
            history_bids,
            history_auction_result,
            history_impression_result,
            history_market_price,
        )
        self.state_history.append(state)
        if time_index < self.cfg.history_length:
            self.fallback.remaining_budget = self.remaining_budget
            return self.fallback.bidding(
                time_index,
                pvalues,
                pvalue_sigmas,
                history_pvalue_info,
                history_bids,
                history_auction_result,
                history_impression_result,
                history_market_price,
            )

        states = np.stack(self.state_history[-self.cfg.history_length :])
        recent_pvalues = history_pvalue_info[-self.cfg.history_length :]
        recent_bids = history_bids[-self.cfg.history_length :]
        recent_impressions = history_impression_result[-self.cfg.history_length :]
        past_actions = np.asarray(
            [action_from_tick(bids, info[:, 0]) for bids, info in zip(recent_bids, recent_pvalues)],
            dtype=np.float32,
        )
        past_rewards = np.asarray(
            [continuous_reward_from_tick(info, result) for info, result in zip(recent_pvalues, recent_impressions)],
            dtype=np.float32,
        )
        conditions = np.asarray(
            [self.budget, self.cpa, self.category, time_index / 47.0], dtype=np.float32
        )

        states = (states - self.normalizer.state_mean[None]) / self.normalizer.state_std[None]
        actions = self.normalizer.encode_actions(past_actions)
        signed_reward = np.sign(past_rewards) * np.log1p(np.abs(past_rewards))
        rewards = (signed_reward - self.normalizer.reward_mean) / self.normalizer.reward_std
        conditions = (conditions - self.normalizer.condition_mean) / self.normalizer.condition_std
        cond_np = np.concatenate([states.reshape(-1), actions, rewards, conditions]).astype(np.float32)
        cond = torch.from_numpy(cond_np[None]).to(self.device)
        previous = torch.tensor([actions[-1]], dtype=torch.float32, device=self.device)

        torch.manual_seed(stable_seed(self.seed, self.name, self.period, self.advertiser, time_index))
        with torch.inference_mode():
            if self.best_of_n == 1:
                selected, _ = self.policy.sample(cond)
            else:
                repeated_cond = cond.repeat(self.best_of_n, 1)
                repeated_previous = previous.repeat(self.best_of_n)
                candidates, _ = self.policy.sample(repeated_cond)
                robust, _, _, _, _ = robust_scores(
                    self.cfg, self.reward_models, repeated_cond, candidates, repeated_previous
                )
                selected = candidates[robust.argmax()][None]
        alpha = float(self.normalizer.decode_actions(selected[:, :1].cpu().numpy())[0, 0])
        return alpha * pvalues


class FixedCpaStrategy:
    def __init__(self) -> None:
        self.name = "FixedCPA"
        self.budget = 100.0
        self.remaining_budget = 100.0
        self.cpa = 2.0
        self.category = 1

    def reset(self) -> None:
        self.remaining_budget = self.budget

    def bidding(
        self,
        time_index: int,
        pvalues: np.ndarray,
        pvalue_sigmas: np.ndarray,
        history_pvalue_info: list[np.ndarray],
        history_bids: list[np.ndarray],
        history_auction_result: list[np.ndarray],
        history_impression_result: list[np.ndarray],
        history_market_price: list[np.ndarray],
    ) -> np.ndarray:
        del (
            time_index,
            pvalue_sigmas,
            history_pvalue_info,
            history_bids,
            history_auction_result,
            history_impression_result,
            history_market_price,
        )
        return self.cpa * pvalues


class StateTargetedAuctionStrategy:
    """Generate clean future states, invert them to bids, then rank candidates with RM."""

    def __init__(
        self,
        state_policy: DiffusionPolicy,
        state_cfg: Any,
        state_normalizer: Any,
        idm_model: nn.Module,
        idm_normalizer: Any,
        bid_normalizer: Normalizer,
        bid_cfg: Config,
        reward_models: list[RewardModel],
        fallback: Any,
        device: torch.device,
        best_of_n: int,
        seed: int,
        keep_indices: np.ndarray,
    ):
        self.name = "state_idm_best"
        self.state_policy = state_policy
        self.state_cfg = state_cfg
        self.state_normalizer = state_normalizer
        self.idm_model = idm_model
        self.idm_normalizer = idm_normalizer
        self.bid_normalizer = bid_normalizer
        self.bid_cfg = bid_cfg
        self.reward_models = reward_models
        self.fallback = fallback
        self.device = device
        self.best_of_n = best_of_n
        self.seed = seed
        self.keep_indices = np.asarray(keep_indices, dtype=np.int64)
        self.budget = 100.0
        self.remaining_budget = 100.0
        self.cpa = 2.0
        self.category = 1
        self.period = 0
        self.advertiser = 0
        self.state_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.remaining_budget = self.budget
        self.state_history = []
        self.fallback.budget = self.budget
        self.fallback.remaining_budget = self.budget
        self.fallback.cpa = self.cpa
        self.fallback.category = self.category
        self.fallback.reset()

    def _build_state_condition(
        self,
        states: np.ndarray,
        past_actions: np.ndarray,
        past_rewards: np.ndarray,
        conditions: np.ndarray,
    ) -> np.ndarray:
        normalized_states = self.state_normalizer.encode_state(states)
        normalized_actions = (
            (np.log1p(np.maximum(past_actions, 0.0)) - self.state_normalizer.action_mean)
            / self.state_normalizer.action_std
        ).astype(np.float32)
        signed_reward = np.sign(past_rewards) * np.log1p(np.abs(past_rewards))
        normalized_rewards = (
            (signed_reward - self.state_normalizer.reward_mean)
            / self.state_normalizer.reward_std
        ).astype(np.float32)
        normalized_conditions = self.state_normalizer.encode_condition(conditions)
        return np.concatenate(
            [
                normalized_states.reshape(1, -1),
                normalized_actions.reshape(1, -1),
                normalized_rewards.reshape(1, -1),
                normalized_conditions.reshape(1, -1),
            ],
            axis=1,
        ).astype(np.float32)

    def _idm_to_alpha(
        self, current_state: np.ndarray, future_states: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        all_states = np.concatenate(
            [np.broadcast_to(current_state, (len(future_states), 1, STATE_DIM)), future_states],
            axis=1,
        )
        transformed = self.idm_normalizer.encode_states(all_states)
        transformed[:, 1:, 2] = 0.0
        transformed[:, 1:, 3] = 0.0
        normalized_conditions = self.idm_normalizer.encode_conditions(
            np.broadcast_to(conditions, (len(future_states), len(conditions)))
        )
        idm_input = np.concatenate(
            [transformed[:, 0], transformed[:, 1:].reshape(len(future_states), -1), normalized_conditions],
            axis=1,
        ).astype(np.float32)
        output = (
            self.idm_model(torch.from_numpy(idm_input).to(self.device))
            .detach()
            .cpu()
            .numpy()
        )
        return self.idm_normalizer.decode_actions(output)

    def bidding(
        self,
        time_index: int,
        pvalues: np.ndarray,
        pvalue_sigmas: np.ndarray,
        history_pvalue_info: list[np.ndarray],
        history_bids: list[np.ndarray],
        history_auction_result: list[np.ndarray],
        history_impression_result: list[np.ndarray],
        history_market_price: list[np.ndarray],
    ) -> np.ndarray:
        state = build_state(
            time_index,
            self.budget,
            self.remaining_budget,
            pvalues,
            history_pvalue_info,
            history_bids,
            history_auction_result,
            history_impression_result,
            history_market_price,
        )
        self.state_history.append(state)
        if time_index < self.state_cfg.history_length:
            self.fallback.remaining_budget = self.remaining_budget
            return self.fallback.bidding(
                time_index,
                pvalues,
                pvalue_sigmas,
                history_pvalue_info,
                history_bids,
                history_auction_result,
                history_impression_result,
                history_market_price,
            )

        recent_pvalues = history_pvalue_info[-self.state_cfg.history_length :]
        recent_bids = history_bids[-self.state_cfg.history_length :]
        recent_impressions = history_impression_result[-self.state_cfg.history_length :]
        past_actions = np.asarray(
            [action_from_tick(bids, info[:, 0]) for bids, info in zip(recent_bids, recent_pvalues)],
            dtype=np.float32,
        )
        past_rewards = np.asarray(
            [continuous_reward_from_tick(info, result) for info, result in zip(recent_pvalues, recent_impressions)],
            dtype=np.float32,
        )
        conditions = np.asarray(
            [self.budget, self.cpa, self.category, time_index / 47.0], dtype=np.float32
        )
        state_condition = self._build_state_condition(
            np.stack(self.state_history[-self.state_cfg.history_length :]),
            past_actions,
            past_rewards,
            conditions,
        )
        cond = torch.from_numpy(state_condition).to(self.device)
        torch.manual_seed(stable_seed(self.seed, self.name, self.period, self.advertiser, time_index))
        with torch.inference_mode():
            repeated_cond = cond.repeat(self.best_of_n, 1)
            generated, _ = self.state_policy.sample(repeated_cond)
            generated = generated.reshape(self.best_of_n, self.state_cfg.horizon, len(self.keep_indices))

        full_norm = torch.zeros(
            self.best_of_n, self.state_cfg.horizon, STATE_DIM, device=self.device
        )
        full_norm[:, :, self.keep_indices] = generated
        future_raw = self.state_normalizer.decode_state(full_norm.cpu().numpy())
        current_state = self.state_history[-1]
        alpha_candidates = self._idm_to_alpha(current_state, future_raw, conditions)

        # A single candidate needs no ranking. Bypassing the bid RM also lets
        # horizon ablations use their native action-chunk width instead of the
        # four-step width expected by the current reward-model checkpoint.
        if self.best_of_n == 1:
            return alpha_candidates[0, :1][0] * pvalues

        # Rebuild the condition with the bid-policy normalization used by the RM.
        bid_states = (np.stack(self.state_history[-self.state_cfg.history_length :]) - self.bid_normalizer.state_mean[None, None]) / self.bid_normalizer.state_std[None, None]
        bid_actions = self.bid_normalizer.encode_actions(past_actions)
        signed_reward = np.sign(past_rewards) * np.log1p(np.abs(past_rewards))
        bid_rewards = (signed_reward - self.bid_normalizer.reward_mean) / self.bid_normalizer.reward_std
        bid_conditions = (conditions - self.bid_normalizer.condition_mean) / self.bid_normalizer.condition_std
        bid_cond_np = np.concatenate([bid_states.reshape(-1), bid_actions, bid_rewards, bid_conditions]).astype(np.float32)
        bid_cond = torch.from_numpy(np.repeat(bid_cond_np[None], self.best_of_n, axis=0)).to(self.device)
        bid_candidates = torch.from_numpy(self.bid_normalizer.encode_actions(alpha_candidates)).to(self.device)
        previous = torch.tensor(
            [self.bid_normalizer.encode_actions(np.asarray([past_actions[-1]], dtype=np.float32))[0]] * self.best_of_n,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            robust, _, _, _, _ = robust_scores(
                self.bid_cfg, self.reward_models, bid_cond, bid_candidates, previous
            )
            selected = int(robust.argmax().item())
        return alpha_candidates[selected, :1][0] * pvalues


def load_state_targeted_strategy(
    state_checkpoint_dir: Path,
    idm_checkpoint_dir: Path,
    bid_checkpoint_dir: Path,
    fallback_factory: Any,
    device: torch.device,
    best_of_n: int,
    seed: int,
) -> StateTargetedAuctionStrategy:
    state_root = state_checkpoint_dir.parent.parent
    idm_root = idm_checkpoint_dir.parent.parent
    for root in [state_root, idm_root]:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from train_state_diffusion import (
        KEEP_STATE_INDICES,
        Config as StateConfig,
        StateNormalizer,
    )
    from train_auctionnet_idm import ChunkActionMLP, Normalizer as IDMNormalizer

    state_cfg = StateConfig(**json.loads((state_checkpoint_dir / "config.json").read_text()))
    state_payload = json.loads((state_checkpoint_dir / "normalization.json").read_text())
    state_normalizer = StateNormalizer()
    for key, value in state_payload.items():
        if isinstance(value, list):
            setattr(state_normalizer, key, np.asarray(value, dtype=np.float32))
        else:
            setattr(state_normalizer, key, value)
    state_policy = DiffusionPolicy(
        state_cfg.history_length * STATE_DIM + state_cfg.history_length * 2 + 4,
        state_cfg.horizon * len(KEEP_STATE_INDICES),
        state_cfg.hidden_dim,
        state_cfg.diffusion_steps,
    ).to(device)
    state_policy.load_state_dict(
        torch.load(state_checkpoint_dir / "state_diffusion.pt", map_location=device, weights_only=True)
    )
    state_policy.eval()

    idm_cfg = json.loads((idm_checkpoint_dir / "config.json").read_text())
    idm_payload = json.loads((idm_checkpoint_dir / "normalization.json").read_text())
    idm_normalizer = IDMNormalizer()
    for key, value in idm_payload.items():
        if isinstance(value, list):
            setattr(idm_normalizer, key, np.asarray(value, dtype=np.float32))
        else:
            setattr(idm_normalizer, key, value)
    idm_input_dim = STATE_DIM * (1 + idm_cfg["horizon"]) + 4
    idm_model = ChunkActionMLP(
        idm_input_dim, idm_cfg["hidden_dim"], idm_cfg["horizon"]
    ).to(device)
    idm_model.load_state_dict(
        torch.load(idm_checkpoint_dir / "oracle_idm_no_future_bid_stats.pt", map_location=device, weights_only=True)
    )
    idm_model.eval()

    bid_strategies = load_diffusion_strategies(
        bid_checkpoint_dir, fallback_factory, device, best_of_n, seed
    )
    bid_reference = bid_strategies["diffusion_bc_best"]
    return StateTargetedAuctionStrategy(
        state_policy,
        state_cfg,
        state_normalizer,
        idm_model,
        idm_normalizer,
        bid_reference.normalizer,
        bid_reference.cfg,
        bid_reference.reward_models,
        fallback_factory(),
        device,
        best_of_n,
        seed,
        KEEP_STATE_INDICES,
    )


def load_diffusion_strategies(
    checkpoint_dir: Path,
    fallback_factory: Any,
    device: torch.device,
    best_of_n: int,
    seed: int,
) -> dict[str, DiffusionAuctionStrategy]:
    cfg = Config(**json.loads((checkpoint_dir / "config.json").read_text()))
    normalizer = load_normalizer(checkpoint_dir / "normalization.json")
    cond_dim = cfg.history_length * STATE_DIM + cfg.history_length * 2 + 4

    policies = {}
    for checkpoint_name in ["bc", "ddpo"]:
        model = DiffusionPolicy(cond_dim, cfg.horizon, cfg.hidden_dim, cfg.diffusion_steps).to(device)
        model.load_state_dict(
            torch.load(checkpoint_dir / f"bid_diffusion_{checkpoint_name}.pt", map_location=device, weights_only=True)
        )
        policies[checkpoint_name] = model.eval()

    reward_models = []
    for index in range(cfg.ensemble_size):
        model = RewardModel(cond_dim, cfg.horizon).to(device)
        model.load_state_dict(
            torch.load(checkpoint_dir / f"reward_model_{index}.pt", map_location=device, weights_only=True)
        )
        reward_models.append(model.eval())

    def make(name: str, policy_name: str, candidates: int) -> DiffusionAuctionStrategy:
        return DiffusionAuctionStrategy(
            name,
            policies[policy_name],
            reward_models,
            normalizer,
            cfg,
            fallback_factory(),
            device,
            candidates,
            seed,
        )

    return {
        "diffusion_bc_single": make("diffusion_bc_single", "bc", 1),
        "diffusion_bc_best": make("diffusion_bc_best", "bc", best_of_n),
        "diffusion_ddpo_best": make("diffusion_ddpo_best", "ddpo", best_of_n),
    }


def import_class(module_name: str, class_name: str) -> Any:
    return getattr(importlib.import_module(module_name), class_name)


def load_official_strategies(auctionnet_root: Path, requested: set[str]) -> dict[str, Any]:
    sys.path.insert(0, str(auctionnet_root))
    specs = {
        "pid": ("simul_bidding_env.strategy.pid_bidding_strategy", "PidBiddingStrategy"),
        "bc": ("simul_bidding_env.strategy.bc_bidding_strategy", "BcBiddingStrategy"),
        "iql": ("simul_bidding_env.strategy.iql_bidding_strategy", "IqlBiddingStrategy"),
        "cql": ("simul_bidding_env.strategy.cql_bidding_strategy", "CqlBiddingStrategy"),
        "td3_bc": ("simul_bidding_env.strategy.td3_bc_bidding_strategy", "TD3_BCBiddingStrategy"),
        "bcq": ("simul_bidding_env.strategy.bcq_bidding_strategy", "BcqBiddingStrategy"),
        "mopo": ("simul_bidding_env.strategy.mbrl_mopo_bidding_strategy", "MbrlMopoBiddingStrategy"),
        "combo": (
            "simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy",
            "MbrlComboMicroBiddingStrategy",
        ),
    }
    strategies = {"fixed_cpa": FixedCpaStrategy()} if "fixed_cpa" in requested else {}
    strategies.update(
        {name: import_class(*spec)() for name, spec in specs.items() if name in requested}
    )
    return strategies


def reset_strategy(strategy: Any, budget: float, cpa: float, category: int, period: int, advertiser: int) -> None:
    strategy.budget = budget
    strategy.remaining_budget = budget
    strategy.cpa = cpa
    strategy.category = category
    if hasattr(strategy, "period"):
        strategy.period = period
    if hasattr(strategy, "advertiser"):
        strategy.advertiser = advertiser
    if hasattr(strategy, "last_remaining_budget"):
        strategy.last_remaining_budget = budget
    if hasattr(strategy, "alpha"):
        strategy.alpha = None
    strategy.reset()


def enforce_budget(
    bids: np.ndarray,
    market_prices: np.ndarray,
    remaining_budget: float,
    drop_priority: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_bids = np.asarray(bids, dtype=np.float64).copy()
    final_bids[~np.isfinite(final_bids)] = 0.0
    final_bids[final_bids < 0] = 0.0
    status = final_bids >= market_prices
    costs = market_prices * status
    while costs.sum() > remaining_budget + 1e-8 and status.any():
        winners = np.flatnonzero(status)
        over_ratio = max((float(costs.sum()) - remaining_budget) / (float(costs.sum()) + 1e-4), 0.0)
        drop_count = max(1, int(math.ceil(len(winners) * over_ratio)))
        to_drop = winners[np.argsort(drop_priority[winners])[:drop_count]]
        final_bids[to_drop] = 0.0
        status = final_bids >= market_prices
        costs = market_prices * status
    return final_bids.astype(np.float32), status, costs.astype(np.float32)


def build_ticks(group: pd.DataFrame, base_seed: int, period: int, advertiser: int) -> list[TickData]:
    ticks = []
    for time_index, tick in group.groupby("timeStepIndex", sort=True):
        pvalues = tick["pValue"].to_numpy(dtype=np.float32)
        sigmas = tick["pValueSigma"].to_numpy(dtype=np.float32)
        rng = np.random.default_rng(stable_seed(base_seed, period, advertiser, int(time_index)))
        latent_values = rng.normal(pvalues, sigmas)
        conversions = (rng.random(len(pvalues)) < np.clip(latent_values, 0.0, 1.0)).astype(np.int8)
        ticks.append(
            TickData(
                time_index=int(time_index),
                pvalues=pvalues,
                pvalue_sigmas=sigmas,
                market_prices=tick["leastWinningCost"].to_numpy(dtype=np.float32),
                logged_bids=tick["bid"].to_numpy(dtype=np.float32),
                potential_conversions=conversions,
                drop_priority=rng.random(len(pvalues)),
            )
        )
    return ticks


def evaluate_episode(
    policy_name: str,
    strategy: Any,
    ticks: list[TickData],
    period: int,
    advertiser: int,
    budget: float,
    cpa_constraint: float,
    category: int,
) -> dict[str, Any]:
    if strategy is not None:
        reset_strategy(strategy, budget, cpa_constraint, category, period, advertiser)
    remaining_budget = budget
    history_pvalue_info: list[np.ndarray] = []
    history_bids: list[np.ndarray] = []
    history_auction_result: list[np.ndarray] = []
    history_impression_result: list[np.ndarray] = []
    history_market_price: list[np.ndarray] = []
    total_reward = total_continuous_reward = total_cost = 0.0
    total_wins = total_pvs = 0
    action_values = []
    last_active_tick = -1

    for tick in ticks:
        if remaining_budget < 0.1:
            proposed_bids = np.zeros_like(tick.pvalues)
        elif policy_name == "logged":
            proposed_bids = tick.logged_bids
        else:
            strategy.remaining_budget = remaining_budget
            proposed_bids = strategy.bidding(
                tick.time_index,
                tick.pvalues,
                tick.pvalue_sigmas,
                history_pvalue_info,
                history_bids,
                history_auction_result,
                history_impression_result,
                history_market_price,
            )

        bids, status, costs = enforce_budget(
            proposed_bids, tick.market_prices, remaining_budget, tick.drop_priority
        )
        conversions = tick.potential_conversions * status
        tick_cost = float(costs.sum())
        tick_reward = float(conversions.sum())
        tick_continuous_reward = float(np.sum(tick.pvalues * status))
        remaining_budget = max(0.0, remaining_budget - tick_cost)
        if strategy is not None:
            strategy.remaining_budget = remaining_budget

        total_cost += tick_cost
        total_reward += tick_reward
        total_continuous_reward += tick_continuous_reward
        total_wins += int(status.sum())
        total_pvs += len(status)
        action_values.append(action_from_tick(bids, tick.pvalues))
        if status.any():
            last_active_tick = tick.time_index

        history_pvalue_info.append(np.stack([tick.pvalues, tick.pvalue_sigmas], axis=1))
        history_bids.append(bids)
        history_auction_result.append(np.stack([status, status, costs], axis=1).astype(np.float32))
        history_impression_result.append(np.stack([status, conversions], axis=1).astype(np.float32))
        history_market_price.append(tick.market_prices)

    cpa = total_cost / (total_reward + 1e-10)
    continuous_cpa = total_cost / (total_continuous_reward + 1e-10)
    return {
        "policy": policy_name,
        "period": period,
        "advertiser": advertiser,
        "category": category,
        "budget": budget,
        "cpa_constraint": cpa_constraint,
        "reward": total_reward,
        "continuous_reward": total_continuous_reward,
        "cost": total_cost,
        "cpa": cpa,
        "continuous_cpa": continuous_cpa,
        "score": competition_score(total_reward, total_cost, cpa_constraint),
        "continuous_score": competition_score(total_continuous_reward, total_cost, cpa_constraint),
        "budget_utilization": total_cost / budget if budget > 0 else 0.0,
        "cpa_violation": float(cpa > cpa_constraint),
        "continuous_cpa_violation": float(continuous_cpa > cpa_constraint),
        "win_rate": total_wins / max(total_pvs, 1),
        "mean_action": float(np.mean(action_values)),
        "last_active_tick": last_active_tick,
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in results.groupby("policy", sort=False):
        total_reward = float(group["reward"].sum())
        total_continuous_reward = float(group["continuous_reward"].sum())
        total_cost = float(group["cost"].sum())
        rows.append(
            {
                "policy": policy,
                "episodes": len(group),
                "mean_score": float(group["score"].mean()),
                "total_score": float(group["score"].sum()),
                "official_normalized_score": float(group["score"].sum() / 20000.0),
                "mean_continuous_score": float(group["continuous_score"].mean()),
                "total_reward": total_reward,
                "total_continuous_reward": total_continuous_reward,
                "total_cost": total_cost,
                "aggregate_cpa": total_cost / (total_reward + 1e-10),
                "aggregate_continuous_cpa": total_cost / (total_continuous_reward + 1e-10),
                "mean_budget_utilization": float(group["budget_utilization"].mean()),
                "cpa_violation_rate": float(group["cpa_violation"].mean()),
                "continuous_cpa_violation_rate": float(group["continuous_cpa_violation"].mean()),
                "mean_win_rate": float(group["win_rate"].mean()),
                "mean_action": float(group["mean_action"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_continuous_score", ascending=False).reset_index(drop=True)


def paired_comparisons(results: pd.DataFrame, target: str = "diffusion_ddpo_best") -> list[dict[str, Any]]:
    pivot = results.pivot_table(index=["period", "advertiser"], columns="policy", values="continuous_score")
    if target not in pivot:
        return []
    comparisons = []
    for baseline in pivot.columns:
        if baseline == target:
            continue
        valid = pivot[[target, baseline]].dropna()
        delta = valid[target] - valid[baseline]
        comparisons.append(
            {
                "target": target,
                "baseline": baseline,
                "episodes": len(valid),
                "mean_continuous_score_delta": float(delta.mean()),
                "median_continuous_score_delta": float(delta.median()),
                "win_rate": float((delta > 0).mean()),
                "tie_rate": float((delta == 0).mean()),
            }
        )
    return sorted(comparisons, key=lambda row: row["mean_continuous_score_delta"], reverse=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    auctionnet_root = Path(args.auctionnet_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else auctionnet_root / "strategy_train_env/data/traffic"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    requested = set(args.policies)
    official = load_official_strategies(auctionnet_root, requested)
    diffusion = load_diffusion_strategies(
        checkpoint_dir,
        FixedCpaStrategy,
        device,
        args.best_of_n,
        args.seed,
    )
    strategies = {**official, **diffusion}
    if "state_idm_best" in requested:
        if not args.state_checkpoint_dir or not args.idm_checkpoint_dir:
            raise ValueError(
                "state_idm_best requires --state-checkpoint-dir and --idm-checkpoint-dir"
            )
        strategies["state_idm_best"] = load_state_targeted_strategy(
            Path(args.state_checkpoint_dir).resolve(),
            Path(args.idm_checkpoint_dir).resolve(),
            checkpoint_dir,
            FixedCpaStrategy,
            device,
            args.best_of_n,
            args.seed,
        )
    if "state_idm_single" in requested:
        if not args.state_checkpoint_dir or not args.idm_checkpoint_dir:
            raise ValueError(
                "state_idm_single requires --state-checkpoint-dir and --idm-checkpoint-dir"
            )
        strategies["state_idm_single"] = load_state_targeted_strategy(
            Path(args.state_checkpoint_dir).resolve(),
            Path(args.idm_checkpoint_dir).resolve(),
            checkpoint_dir,
            FixedCpaStrategy,
            device,
            1,
            args.seed,
        )
        strategies["state_idm_single"].name = "state_idm_single"
    unknown = sorted(set(args.policies) - ({"logged"} | set(strategies)))
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")

    usecols = [
        "deliveryPeriodIndex",
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint",
        "timeStepIndex",
        "pValue",
        "pValueSigma",
        "bid",
        "leastWinningCost",
    ]
    float_type = np.float32 if args.read_dtype == "float32" else np.float64
    dtypes = {name: float_type for name in usecols}
    rows = []
    started = time.time()
    for period in args.periods:
        path = data_dir / f"period-{period}.csv"
        print(f"READ_START period={period} path={path}", flush=True)
        frame = pd.read_csv(path, usecols=usecols, dtype=dtypes)
        print(f"READ_DONE period={period} rows={len(frame)} seconds={time.time() - started:.1f}", flush=True)
        grouped_advertisers = frame.groupby("advertiserNumber", sort=True)
        for advertiser_offset, (advertiser_value, group) in enumerate(grouped_advertisers):
            if args.advertiser_limit is not None and advertiser_offset >= args.advertiser_limit:
                break
            advertiser = int(advertiser_value)
            first = group.iloc[0]
            budget = float(first["budget"])
            cpa_constraint = float(first["CPAConstraint"])
            category = int(first["advertiserCategoryIndex"])
            ticks = build_ticks(group, args.seed, period, advertiser)
            for policy_name in args.policies:
                result = evaluate_episode(
                    policy_name,
                    None if policy_name == "logged" else strategies[policy_name],
                    ticks,
                    period,
                    advertiser,
                    budget,
                    cpa_constraint,
                    category,
                )
                rows.append(result)
                print("EPISODE " + json.dumps(result, sort_keys=True), flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "episode_results.partial.csv", index=False)
        del frame

    results = pd.DataFrame(rows)
    summary = summarize(results)
    comparisons = paired_comparisons(results, args.comparison_target)
    results.to_csv(output_dir / "episode_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "summary": summary.to_dict(orient="records"),
        "paired_comparisons": comparisons,
        "methodology": {
            "market": "AuctionNet held-out logged leastWinningCost replay",
            "periods": args.periods,
            "randomness": "common potential conversions and budget-drop priority across policies",
            "primary_metric": "continuous_score using pValue-weighted wins and official CPA penalty",
            "secondary_metric": "sampled conversion score using common random outcomes",
            "warm_start": "Diffusion policies use alpha=CPA constraint for ticks 0-3",
            "state_targeted_policy": (
                "State Diffusion -> no-bid-stat IDM -> Ensemble RM -> Best-of-N"
                if "state_idm_best" in requested
                else None
            ),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print("FINAL_SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
