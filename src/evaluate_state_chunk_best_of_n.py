#!/usr/bin/env python3
"""Evaluate state-space Best-of-N followed by a single-step IDM in AuctionNet replay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
for root in [HERE, HERE.parent / "remote_bid_diffusion", HERE.parent / "remote_inverse_dynamics"]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import (  # noqa: E402
    STATE_DIM,
    FixedCpaStrategy,
    action_from_tick,
    build_state,
    build_ticks,
    continuous_reward_from_tick,
    evaluate_episode,
    paired_comparisons,
    stable_seed,
    summarize,
)
from train_auctionnet_idm import Normalizer as IDMNormalizer  # noqa: E402
from train_offline_bid_diffusion import DiffusionPolicy  # noqa: E402
try:  # Optional research branch; not required by the released Transformer RM path.
    from train_policy_aligned_episode_q_model import EpisodeQModel  # noqa: E402
except ImportError:
    EpisodeQModel = None
from train_single_step_idm import SingleActionMLP  # noqa: E402
from train_state_chunk_reward_model import (  # noqa: E402
    StateChunkRewardModel,
    load_state_normalizer,
    robust_state_chunk_cpa_scores,
    robust_state_chunk_scores,
)
from train_transformer_state_chunk_rm import (  # noqa: E402
    MODEL_NAME as TRANSFORMER_RM_NAME,
    DynamicStateTransformerRewardModel,
)
from train_state_diffusion import KEEP_STATE_INDICES  # noqa: E402


def robust_transformer_state_chunk_scores(
    models: list[DynamicStateTransformerRewardModel],
    cond: torch.Tensor,
    state_chunk: torch.Tensor,
    metadata: dict,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history_length = int(metadata["history_length"])
    state_dim = int(metadata["state_dim"])
    history_full = cond[:, : history_length * STATE_DIM].reshape(
        -1, history_length, STATE_DIM
    )
    keep_indices = torch.as_tensor(
        metadata["keep_state_indices"], device=cond.device, dtype=torch.long
    )
    history = history_full.index_select(-1, keep_indices)
    if state_chunk.ndim != 2 or state_chunk.shape[1] % state_dim:
        raise ValueError(
            f"state chunk width {state_chunk.shape} is incompatible with state_dim={state_dim}"
        )
    horizon = state_chunk.shape[1] // state_dim
    future = state_chunk.reshape(state_chunk.shape[0], horizon, state_dim)
    sequence = torch.cat([history, future], dim=1)
    valid_mask = torch.ones(
        sequence.shape[:2], dtype=torch.bool, device=sequence.device
    )
    predictions = torch.stack(
        [model(sequence, valid_mask) for model in models], dim=0
    )
    mean = predictions.mean(0)
    uncertainty = predictions.std(0, unbiased=False)
    support = F.relu(state_chunk.abs() - float(metadata["state_clip"])).mean(-1)
    score = mean - uncertainty_beta * uncertainty - support_penalty * support
    return score, {"mean": mean, "uncertainty": uncertainty, "support": support}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--idm-checkpoint-dir", required=True)
    parser.add_argument("--state-rm-checkpoint-dir", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--uncertainty-beta", type=float, default=0.5)
    parser.add_argument("--support-penalty", type=float, default=0.2)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--ensemble-limit", type=int, default=None)
    parser.add_argument("--ensemble-members", type=int, nargs="+", default=None)
    parser.add_argument(
        "--ranking-objective",
        choices=["reward", "cpa_score", "episode_score"],
        default="reward",
    )
    parser.add_argument("--advertiser-limit", type=int, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--read-dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def load_payload(path: Path, target: Any, skip: set[str] | None = None) -> Any:
    skip = skip or set()
    for key, value in json.loads(path.read_text()).items():
        if key in skip:
            continue
        setattr(target, key, np.asarray(value, dtype=np.float32) if isinstance(value, list) else value)
    return target


class StateChunkBestOfNStrategy:
    """Rank future state chunks, decode one action, then replan next tick."""

    def __init__(
        self,
        state_policy: DiffusionPolicy,
        state_cfg: Any,
        state_normalizer: Any,
        idm: SingleActionMLP,
        idm_normalizer: IDMNormalizer,
        reward_models: list[torch.nn.Module],
        rm_metadata: dict,
        fallback: Any,
        device: torch.device,
        best_of_n: int,
        candidate_pool_size: int,
        uncertainty_beta: float,
        support_penalty: float,
        ranking_objective: str,
        action_scale: float,
        seed: int,
        rm_type: str = "mlp",
    ):
        self.name = f"state_chunk_rm_n{best_of_n}"
        self.state_policy = state_policy
        self.state_cfg = state_cfg
        self.state_normalizer = state_normalizer
        self.idm = idm
        self.idm_normalizer = idm_normalizer
        self.reward_models = reward_models
        self.rm_metadata = rm_metadata
        self.fallback = fallback
        self.device = device
        self.best_of_n = best_of_n
        self.candidate_pool_size = candidate_pool_size
        self.uncertainty_beta = uncertainty_beta
        self.support_penalty = support_penalty
        self.ranking_objective = ranking_objective
        self.action_scale = action_scale
        self.seed = seed
        self.rm_type = rm_type
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

    def build_condition(
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

    def decode_action(
        self,
        current_state: np.ndarray,
        selected_chunk: np.ndarray,
        conditions: np.ndarray,
    ) -> float:
        all_states = np.concatenate([current_state[None, None], selected_chunk[None]], axis=1)
        normalized_states = self.idm_normalizer.encode_states(all_states)
        normalized_states[:, 1:, 2] = 0.0
        normalized_states[:, 1:, 3] = 0.0
        normalized_conditions = self.idm_normalizer.encode_conditions(conditions[None])
        inputs = np.concatenate(
            [normalized_states.reshape(1, -1), normalized_conditions], axis=1
        ).astype(np.float32)
        with torch.inference_mode():
            action_norm = self.idm(torch.from_numpy(inputs).to(self.device)).cpu().numpy()
        return float(self.idm_normalizer.decode_actions(action_norm)[0, 0])

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
        cond_np = self.build_condition(
            np.stack(self.state_history[-self.state_cfg.history_length :]),
            past_actions,
            past_rewards,
            conditions,
        )
        cond = torch.from_numpy(cond_np).to(self.device)
        torch.manual_seed(
            stable_seed(
                self.seed,
                "shared-state-chunk-candidates",
                self.period,
                self.advertiser,
                time_index,
            )
        )
        with torch.inference_mode():
            pool_context = cond.repeat(self.candidate_pool_size, 1)
            candidate_pool, _ = self.state_policy.sample(pool_context)
            repeated = pool_context[: self.best_of_n]
            generated = candidate_pool[: self.best_of_n]

        generated_view = generated.reshape(
            self.best_of_n, self.state_cfg.horizon, len(KEEP_STATE_INDICES)
        )
        full_norm = np.zeros(
            (self.best_of_n, self.state_cfg.horizon, STATE_DIM), dtype=np.float32
        )
        full_norm[:, :, KEEP_STATE_INDICES] = generated_view.cpu().numpy()
        future_states = self.state_normalizer.decode_state(full_norm)

        selected = 0
        if self.best_of_n > 1:
            with torch.inference_mode():
                if self.ranking_objective == "episode_score":
                    score, _ = policy_aligned_episode_scores(
                        self.reward_models,
                        self.rm_metadata,
                        repeated,
                        generated,
                        self.uncertainty_beta,
                        self.support_penalty,
                    )
                elif self.ranking_objective == "cpa_score":
                    predicted_cost = np.maximum(
                        (state[1] - future_states[:, -1, 1]) * self.budget,
                        0.0,
                    ).astype(np.float32)
                    score, _ = robust_state_chunk_cpa_scores(
                        self.reward_models,
                        repeated,
                        generated,
                        torch.from_numpy(predicted_cost).to(self.device),
                        torch.full(
                            (self.best_of_n,), self.cpa, device=self.device
                        ),
                        self.rm_metadata["return_mean"],
                        self.rm_metadata["return_std"],
                        self.rm_metadata["state_clip"],
                        self.uncertainty_beta,
                        self.support_penalty,
                    )
                else:
                    if self.rm_type == "transformer":
                        score, _ = robust_transformer_state_chunk_scores(
                            self.reward_models,
                            repeated,
                            generated,
                            self.rm_metadata,
                            self.uncertainty_beta,
                            self.support_penalty,
                        )
                    else:
                        score, _ = robust_state_chunk_scores(
                            self.reward_models,
                            repeated,
                            generated,
                            self.rm_metadata["state_clip"],
                            self.uncertainty_beta,
                            self.support_penalty,
                        )
                selected = int(score.argmax().item())

        action = self.decode_action(state, future_states[selected], conditions)
        return action * self.action_scale * pvalues


def policy_aligned_episode_scores(
    models: list[EpisodeQModel],
    metadata: dict,
    policy_context: torch.Tensor,
    state_chunks: torch.Tensor,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank chunks with the policy-aligned ensemble's direct episode-Score head."""

    features = torch.cat([policy_context, state_chunks], dim=-1)
    expected_dim = int(metadata["input_dim"])
    if features.shape[-1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} RM features, got {features.shape[-1]}")
    mean = torch.as_tensor(metadata["feature_mean"], device=features.device)
    std = torch.as_tensor(metadata["feature_std"], device=features.device)
    normalized = (features - mean) / std
    member_scores = torch.stack([model(normalized)[..., 2] for model in models])
    ensemble_mean = member_scores.mean(0)
    uncertainty = member_scores.std(0, unbiased=False)
    normalized_chunks = normalized[..., int(metadata["context_dim"]) :]
    support = F.relu(normalized_chunks.abs() - 3.0).mean(-1)
    robust_score = (
        ensemble_mean
        - uncertainty_beta * uncertainty
        - support_penalty * support
    )
    return robust_score, {
        "mean": ensemble_mean,
        "uncertainty": uncertainty,
        "support": support,
    }


def load_strategy_components(args: argparse.Namespace, device: torch.device) -> dict:
    state_dir = Path(args.state_checkpoint_dir)
    state_cfg, state_normalizer = load_state_normalizer(state_dir)
    state_policy = DiffusionPolicy(
        state_cfg.history_length * STATE_DIM + state_cfg.history_length * 2 + 4,
        state_cfg.horizon * len(KEEP_STATE_INDICES),
        state_cfg.hidden_dim,
        state_cfg.diffusion_steps,
    ).to(device)
    state_policy.load_state_dict(
        torch.load(state_dir / "state_diffusion.pt", map_location=device, weights_only=True)
    )
    state_policy.eval()

    idm_dir = Path(args.idm_checkpoint_dir)
    idm_cfg = json.loads((idm_dir / "config.json").read_text())
    idm_normalizer = load_payload(
        idm_dir / "normalization.json", IDMNormalizer()
    )
    idm_input_dim = STATE_DIM * (1 + idm_cfg["horizon"]) + 4
    idm = SingleActionMLP(idm_input_dim, idm_cfg["hidden_dim"]).to(device)
    idm.load_state_dict(
        torch.load(idm_dir / "single_step_idm.pt", map_location=device, weights_only=True)
    )
    idm.eval()

    rm_dir = Path(args.state_rm_checkpoint_dir)
    rm_cfg = json.loads((rm_dir / "config.json").read_text())
    rm_metadata = json.loads((rm_dir / "normalization.json").read_text())
    reward_models = []
    rm_type = "mlp"
    if rm_cfg.get("model") == "EpisodeQModel":
        if EpisodeQModel is None:
            raise ImportError(
                "EpisodeQModel support requires train_policy_aligned_episode_q_model.py"
            )
        if args.ranking_objective != "episode_score":
            raise ValueError(
                "Policy-aligned EpisodeQModel requires --ranking-objective episode_score"
            )
        for member in range(rm_cfg["ensemble_size"]):
            model = EpisodeQModel(rm_metadata["input_dim"], rm_cfg["hidden_dim"]).to(device)
            model.load_state_dict(
                torch.load(
                    rm_dir / f"state_chunk_episode_q_{member}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            reward_models.append(model.eval())
    elif rm_cfg.get("model") == TRANSFORMER_RM_NAME:
        if args.ranking_objective != "reward":
            raise ValueError("Transformer state RM currently supports reward ranking")
        rm_type = "transformer"
        for member in range(rm_cfg["ensemble_size"]):
            model = DynamicStateTransformerRewardModel(
                state_dim=rm_metadata["state_dim"],
                hidden_dim=rm_cfg["hidden_dim"],
                num_layers=rm_cfg["num_layers"],
                num_heads=rm_cfg["num_heads"],
                ff_dim=rm_cfg["ff_dim"],
                dropout=rm_cfg["dropout"],
            ).to(device)
            model.load_state_dict(
                torch.load(
                    rm_dir / f"transformer_state_rm_{member}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            reward_models.append(model.eval())
    else:
        if args.ranking_objective == "episode_score":
            raise ValueError("episode_score ranking requires a policy-aligned EpisodeQModel")
        for member in range(rm_cfg["ensemble_size"]):
            model = StateChunkRewardModel(
                rm_metadata["cond_dim"], rm_metadata["chunk_dim"], rm_cfg["hidden_dim"]
            ).to(device)
            model.load_state_dict(
                torch.load(
                    rm_dir / f"state_chunk_rm_{member}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            reward_models.append(model.eval())
    if args.ensemble_limit is not None and args.ensemble_members is not None:
        raise ValueError("ensemble-limit and ensemble-members are mutually exclusive")
    if args.ensemble_members is not None:
        members = args.ensemble_members
        if len(set(members)) != len(members):
            raise ValueError("ensemble members must be unique")
        if not members or min(members) < 0 or max(members) >= len(reward_models):
            raise ValueError(
                f"ensemble members must be unique indices in [0, {len(reward_models) - 1}]"
            )
        reward_models = [reward_models[index] for index in members]
    elif args.ensemble_limit is not None:
        if args.ensemble_limit < 1 or args.ensemble_limit > len(reward_models):
            raise ValueError(
                f"ensemble limit must be in [1, {len(reward_models)}]"
            )
        reward_models = reward_models[: args.ensemble_limit]
    return {
        "state_policy": state_policy,
        "state_cfg": state_cfg,
        "state_normalizer": state_normalizer,
        "idm": idm,
        "idm_normalizer": idm_normalizer,
        "reward_models": reward_models,
        "rm_metadata": rm_metadata,
        "rm_type": rm_type,
    }


def main() -> None:
    args = parse_args()
    if any(value < 1 for value in args.candidate_counts):
        raise ValueError("candidate counts must be positive")
    if args.candidate_pool_size is not None and args.candidate_pool_size < max(args.candidate_counts):
        raise ValueError("candidate pool size must cover every candidate count")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    components = load_strategy_components(args, device)
    strategies = {}
    candidate_pool_size = args.candidate_pool_size or max(args.candidate_counts)
    for count in sorted(set(args.candidate_counts)):
        strategy = StateChunkBestOfNStrategy(
            **components,
            fallback=FixedCpaStrategy(),
            device=device,
            best_of_n=count,
            candidate_pool_size=candidate_pool_size,
            uncertainty_beta=args.uncertainty_beta,
            support_penalty=args.support_penalty,
            ranking_objective=args.ranking_objective,
            action_scale=args.action_scale,
            seed=args.seed,
        )
        strategies[strategy.name] = strategy
    if args.include_baselines:
        strategies["fixed_cpa"] = FixedCpaStrategy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auctionnet_root = Path(args.auctionnet_root)
    data_dir = Path(args.data_dir) if args.data_dir else auctionnet_root / "strategy_train_env/data/traffic"
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
    dtype = np.float32 if args.read_dtype == "float32" else np.float64
    rows = []
    started = time.time()
    policy_names = list(strategies)
    if args.include_baselines:
        policy_names = ["logged", *policy_names]
    for period in args.periods:
        frame = pd.read_csv(
            data_dir / f"period-{period}.csv",
            usecols=usecols,
            dtype={name: dtype for name in usecols},
        )
        for offset, (advertiser_value, group) in enumerate(
            frame.groupby("advertiserNumber", sort=True)
        ):
            if args.advertiser_limit is not None and offset >= args.advertiser_limit:
                break
            advertiser = int(advertiser_value)
            first = group.iloc[0]
            ticks = build_ticks(group, args.seed, period, advertiser)
            for name in policy_names:
                result = evaluate_episode(
                    name,
                    None if name == "logged" else strategies[name],
                    ticks,
                    period,
                    advertiser,
                    float(first["budget"]),
                    float(first["CPAConstraint"]),
                    int(first["advertiserCategoryIndex"]),
                )
                rows.append(result)
                print("EPISODE " + json.dumps(result, sort_keys=True), flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "episode_results.partial.csv", index=False)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    target = "state_chunk_rm_n1"
    payload = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "summary": summary.to_dict(orient="records"),
        "paired_comparisons": paired_comparisons(results, target),
        "methodology": {
            "policy": "State Diffusion -> state-chunk ensemble RM -> single-step IDM",
            "execution": "execute one bid and replan at the next tick",
            "candidate_randomness": "candidate prefixes shared across N",
            "primary_metric": "continuous_score",
        },
    }
    results.to_csv(output_dir / "episode_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print("FINAL_SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
