#!/usr/bin/env python3
"""Train an episode-return ensemble for state-chunk candidate ranking.

Each candidate replaces the action executed at the current decision. Every
later action is generated after observing the replayed next state, so labels
match receding-horizon deployment under the policy checkpoint being evaluated.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
for root in [
    HERE,
    HERE.parent / "remote_bid_diffusion",
    HERE.parent / "bid_diffusion",
    HERE.parent / "remote_inverse_dynamics",
    HERE.parent / "inverse_dynamics",
]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import (  # noqa: E402
    build_state,
    competition_score,
    enforce_budget,
    stable_seed,
)
from train_state_diffusion import KEEP_STATE_INDICES  # noqa: E402
from train_state_replay_ddpo import (  # noqa: E402
    ReplayState,
    decode_actions,
    load_models,
    load_templates,
    state_condition,
)
from train_policy_aligned_reward_cost_model import append_environment_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--idm-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-files", nargs="+", default=None)
    parser.add_argument("--train-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--val-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--test-periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--rollouts-per-advertiser", type=int, default=2)
    parser.add_argument("--advertiser-limit", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--decision-stride", type=int, default=1)
    parser.add_argument("--max-groups-per-period", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--target-mode", choices=["absolute", "advantage"], default="absolute")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpisodeQModel(nn.Module):
    """Residual MLP with reward, cost, and direct episode-score heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def candidate_episode_outcomes(
    replay: ReplayState,
    decision_time: int,
    candidate_alphas: np.ndarray,
    base_alphas: np.ndarray,
    prefix_reward: float,
    prefix_cost: float,
    remaining_budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay each first-action intervention under a shared base continuation."""

    rewards = np.full(len(candidate_alphas), prefix_reward, dtype=np.float64)
    costs = np.full(len(candidate_alphas), prefix_cost, dtype=np.float64)
    budgets = np.full(len(candidate_alphas), remaining_budget, dtype=np.float64)
    for time_index in range(decision_time, len(replay.template.ticks)):
        tick = replay.template.ticks[time_index]
        alphas = candidate_alphas if time_index == decision_time else base_alphas[:, time_index]
        for candidate_index, alpha in enumerate(alphas):
            if budgets[candidate_index] < 0.1:
                continue
            proposed = float(alpha) * tick.pvalues
            _, status, step_costs = enforce_budget(
                proposed,
                tick.market_prices,
                float(budgets[candidate_index]),
                tick.drop_priority,
            )
            step_cost = float(step_costs.sum())
            rewards[candidate_index] += float(np.sum(tick.pvalues * status))
            costs[candidate_index] += step_cost
            budgets[candidate_index] = max(0.0, budgets[candidate_index] - step_cost)
    scores = np.asarray(
        [competition_score(r, c, replay.template.cpa) for r, c in zip(rewards, costs)],
        dtype=np.float32,
    )
    return rewards.astype(np.float32), costs.astype(np.float32), scores


def clone_replay_prefix(replay: ReplayState) -> ReplayState:
    """Clone mutable episode history while sharing the immutable market template."""

    return ReplayState(
        template=replay.template,
        group_id=replay.group_id,
        remaining_budget=float(replay.remaining_budget),
        state_history=list(replay.state_history),
        history_pvalue_info=list(replay.history_pvalue_info),
        history_bids=list(replay.history_bids),
        history_auction_result=list(replay.history_auction_result),
        history_impression_result=list(replay.history_impression_result),
        history_market_price=list(replay.history_market_price),
        total_continuous_reward=float(replay.total_continuous_reward),
        total_cost=float(replay.total_cost),
    )


def append_decision_state(replay: ReplayState, time_index: int) -> None:
    tick = replay.template.ticks[time_index]
    replay.state_history.append(
        build_state(
            time_index,
            replay.template.budget,
            replay.remaining_budget,
            tick.pvalues,
            replay.history_pvalue_info,
            replay.history_bids,
            replay.history_auction_result,
            replay.history_impression_result,
            replay.history_market_price,
        )
    )


def rollout_candidate_outcomes(
    replay_prefix: ReplayState,
    decision_time: int,
    candidate_alphas: np.ndarray,
    continuation_actions: Callable[[list[ReplayState], int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute each candidate, then obtain every later action from a callback."""

    rewards, costs, scores = rollout_candidate_group_outcomes(
        [replay_prefix],
        decision_time,
        np.asarray(candidate_alphas, dtype=np.float32).reshape(1, -1),
        continuation_actions,
    )
    return rewards[0], costs[0], scores[0]


def rollout_candidate_group_outcomes(
    replay_prefixes: list[ReplayState],
    decision_time: int,
    candidate_alphas: np.ndarray,
    continuation_actions: Callable[[list[ReplayState], int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize closed-loop candidate replay across contexts at one decision time."""

    candidate_alphas = np.asarray(candidate_alphas, dtype=np.float32)
    if candidate_alphas.ndim != 2 or len(candidate_alphas) != len(replay_prefixes):
        raise ValueError("Candidate actions must have shape [groups, candidates]")
    if not replay_prefixes:
        raise ValueError("At least one replay prefix is required")
    tick_counts = {len(replay.template.ticks) for replay in replay_prefixes}
    if len(tick_counts) != 1:
        raise ValueError("Batched replay prefixes must have the same episode length")
    candidate_count = candidate_alphas.shape[1]
    replays = []
    for group_index, replay_prefix in enumerate(replay_prefixes):
        for _ in range(candidate_count):
            replay = clone_replay_prefix(replay_prefix)
            replay.counterfactual_group = group_index
            replays.append(replay)
    for replay, alpha in zip(replays, candidate_alphas.reshape(-1)):
        append_environment_step(replay, float(alpha), decision_time)

    for time_index in range(decision_time + 1, tick_counts.pop()):
        for replay in replays:
            append_decision_state(replay, time_index)
        active = [replay for replay in replays if replay.remaining_budget >= 0.1]
        active_alphas = np.asarray(
            continuation_actions(active, time_index) if active else [],
            dtype=np.float32,
        ).reshape(-1)
        if len(active_alphas) != len(active):
            raise ValueError(
                "Continuation policy must return exactly one action per active replay"
            )
        action_by_id = {
            id(replay): float(alpha) for replay, alpha in zip(active, active_alphas)
        }
        for replay in replays:
            append_environment_step(replay, action_by_id.get(id(replay), 0.0), time_index)

    rewards = np.asarray(
        [replay.total_continuous_reward for replay in replays], dtype=np.float32
    )
    costs = np.asarray([replay.total_cost for replay in replays], dtype=np.float32)
    scores = np.asarray(
        [
            competition_score(reward, cost, replay.template.cpa)
            for replay, reward, cost in zip(replays, rewards, costs)
        ],
        dtype=np.float32,
    )
    shape = (len(replay_prefixes), candidate_count)
    return rewards.reshape(shape), costs.reshape(shape), scores.reshape(shape)


@torch.inference_mode()
def sample_with_grouped_noise(
    policy,
    cond: torch.Tensor,
    group_ids: np.ndarray,
) -> torch.Tensor:
    """Sample with common random numbers inside each counterfactual group."""

    group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=cond.device)
    if group_ids.shape != (len(cond),):
        raise ValueError("group_ids must contain one id per policy condition")
    _, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_count = int(inverse.max().item()) + 1 if len(inverse) else 0
    x = torch.randn(group_count, policy.bid_dim, device=cond.device)[inverse]
    for step in range(policy.steps - 1, -1, -1):
        timesteps = torch.full(
            (len(cond),), step, device=cond.device, dtype=torch.long
        )
        mean, variance, _ = policy.model_stats(x, timesteps, cond)
        if step == 0:
            x = mean
            continue
        grouped_noise = torch.randn(
            group_count, policy.bid_dim, device=cond.device
        )[inverse]
        x = mean + variance.sqrt() * grouped_noise
    return x


@torch.inference_mode()
def closed_loop_candidate_outcomes(
    replay_prefix: ReplayState,
    decision_time: int,
    candidate_alphas: np.ndarray,
    policy,
    state_cfg,
    state_normalizer,
    idm,
    idm_normalizer,
    device: torch.device,
    seed: int,
    seed_namespace: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate candidates under fresh receding-horizon replanning by `policy`."""

    rewards, costs, scores = closed_loop_candidate_group_outcomes(
        [replay_prefix],
        decision_time,
        np.asarray(candidate_alphas, dtype=np.float32).reshape(1, -1),
        policy,
        state_cfg,
        state_normalizer,
        idm,
        idm_normalizer,
        device,
        seed,
        seed_namespace,
    )
    return rewards[0], costs[0], scores[0]


@torch.inference_mode()
def closed_loop_candidate_group_outcomes(
    replay_prefixes: list[ReplayState],
    decision_time: int,
    candidate_alphas: np.ndarray,
    policy,
    state_cfg,
    state_normalizer,
    idm,
    idm_normalizer,
    device: torch.device,
    seed: int,
    seed_namespace: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate same-time candidate groups in one batched policy rollout."""

    def continuation_actions(
        active_replays: list[ReplayState], time_index: int
    ) -> np.ndarray:
        if not active_replays:
            return np.empty(0, dtype=np.float32)
        built = [
            state_condition(
                replay, time_index, state_normalizer, state_cfg.history_length
            )
            for replay in active_replays
        ]
        cond = np.stack([item[0] for item in built])
        current = np.stack([item[1] for item in built])
        conditions = np.stack([item[2] for item in built])
        torch.manual_seed(
            stable_seed(
                seed,
                seed_namespace,
                decision_time,
                time_index,
            )
        )
        group_ids = np.asarray(
            [replay.counterfactual_group for replay in active_replays],
            dtype=np.int64,
        )
        generated = sample_with_grouped_noise(
            policy,
            torch.from_numpy(cond).to(device),
            group_ids,
        )
        alpha_chunks = decode_actions(
            generated,
            current,
            conditions,
            state_cfg,
            state_normalizer,
            idm,
            idm_normalizer,
            device,
        )
        return np.asarray(alpha_chunks[:, 0], dtype=np.float32)

    return rollout_candidate_group_outcomes(
        replay_prefixes,
        decision_time,
        candidate_alphas,
        continuation_actions,
    )


@torch.inference_mode()
def collect_period(
    period: int,
    args: argparse.Namespace,
    policy,
    state_cfg,
    state_normalizer,
    idm,
    idm_normalizer,
    device: torch.device,
    group_offset: int,
) -> tuple[dict[str, np.ndarray], int]:
    templates = load_templates(Path(args.auctionnet_root), period, args.seed)
    if args.advertiser_limit is not None:
        templates = templates[: args.advertiser_limit]
    replays = [
        ReplayState.create(template, advertiser_index)
        for advertiser_index, template in enumerate(templates)
        for _ in range(args.rollouts_per_advertiser)
    ]
    base_alphas = np.stack(
        [np.full(48, replay.template.cpa, dtype=np.float32) for replay in replays]
    )
    snapshots: list[dict] = []
    policy.eval()

    for time_index in range(48):
        for replay in replays:
            append_decision_state(replay, time_index)

        if time_index >= state_cfg.history_length:
            active = np.asarray(
                [index for index, replay in enumerate(replays) if replay.remaining_budget >= 0.1],
                dtype=np.int64,
            )
            if len(active):
                collect_at_time = (
                    (time_index - state_cfg.history_length) % args.decision_stride == 0
                    and (
                        args.max_groups_per_period is None
                        or len(snapshots) < args.max_groups_per_period
                    )
                )
                sample_count = args.candidate_count if collect_at_time else 1
                built = [
                    state_condition(
                        replays[index], time_index, state_normalizer, state_cfg.history_length
                    )
                    for index in active
                ]
                cond_np = np.stack([item[0] for item in built])
                current_np = np.stack([item[1] for item in built])
                condition_np = np.stack([item[2] for item in built])
                candidate_cond = np.repeat(cond_np, sample_count, axis=0)
                candidate_current = np.repeat(current_np, sample_count, axis=0)
                candidate_condition = np.repeat(condition_np, sample_count, axis=0)
                torch.manual_seed(stable_seed(args.seed, "episode-q-v4", period, time_index))
                generated, _ = policy.sample(torch.from_numpy(candidate_cond).to(device))
                alpha_chunks = decode_actions(
                    generated,
                    candidate_current,
                    candidate_condition,
                    state_cfg,
                    state_normalizer,
                    idm,
                    idm_normalizer,
                    device,
                )
                generated_np = generated.cpu().numpy().astype(np.float32)
                features = np.concatenate(
                    [
                        candidate_cond,
                        generated_np,
                        np.log1p(np.maximum(alpha_chunks, 0.0)),
                    ],
                    axis=1,
                ).reshape(len(active), sample_count, -1)
                alpha_by_replay = alpha_chunks.reshape(
                    len(active), sample_count, state_cfg.horizon
                )
                for position, replay_index in enumerate(active):
                    replay = replays[replay_index]
                    should_collect = (
                        collect_at_time
                        and (
                            args.max_groups_per_period is None
                            or len(snapshots) < args.max_groups_per_period
                        )
                    )
                    prefix = np.asarray(
                        [
                            np.log1p(max(replay.total_continuous_reward, 0.0)),
                            np.log1p(max(replay.total_cost, 0.0)),
                        ],
                        dtype=np.float32,
                    )
                    group_features = np.concatenate(
                        [
                            features[position],
                            np.broadcast_to(prefix, (sample_count, 2)),
                        ],
                        axis=1,
                    ).astype(np.float32)
                    if should_collect:
                        snapshots.append(
                            {
                                "replay_prefix": clone_replay_prefix(replay),
                                "time_index": time_index,
                                "features": group_features,
                                "candidate_alphas": alpha_by_replay[position, :, 0].copy(),
                            }
                        )
                    base_alphas[replay_index, time_index] = alpha_by_replay[position, 0, 0]

        for replay_index, replay in enumerate(replays):
            append_environment_step(replay, float(base_alphas[replay_index, time_index]), time_index)

    feature_groups = []
    reward_groups = []
    cost_groups = []
    score_groups = []
    period_groups = []
    group_ids = []
    if not snapshots:
        raise ValueError(f"Period {period} produced no active candidate groups")
    outcomes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for decision_time in sorted({snapshot["time_index"] for snapshot in snapshots}):
        indices = [
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot["time_index"] == decision_time
        ]
        batched = closed_loop_candidate_group_outcomes(
            [snapshots[index]["replay_prefix"] for index in indices],
            decision_time,
            np.stack([snapshots[index]["candidate_alphas"] for index in indices]),
            policy,
            state_cfg,
            state_normalizer,
            idm,
            idm_normalizer,
            device,
            args.seed,
            f"episode-q-closed-loop-t{decision_time}",
        )
        for position, snapshot_index in enumerate(indices):
            outcomes[snapshot_index] = tuple(values[position] for values in batched)
    for local_group, snapshot in enumerate(snapshots):
        rewards, costs, scores = outcomes[local_group]
        feature_groups.append(snapshot["features"])
        reward_groups.append(rewards)
        cost_groups.append(costs)
        score_groups.append(scores)
        period_groups.append(period)
        group_ids.append(group_offset + local_group)

    result = {
        "features": np.stack(feature_groups).astype(np.float32),
        "rewards": np.stack(reward_groups).astype(np.float32),
        "costs": np.stack(cost_groups).astype(np.float32),
        "scores": np.stack(score_groups).astype(np.float32),
        "periods": np.asarray(period_groups, dtype=np.int64),
        "groups": np.asarray(group_ids, dtype=np.int64),
    }
    print(
        "COLLECT_EPISODE_Q_PERIOD "
        + json.dumps(
            {
                "period": period,
                "groups": len(result["features"]),
                "rows": int(np.prod(result["features"].shape[:2])),
                "feature_dim": int(result["features"].shape[-1]),
            }
        ),
        flush=True,
    )
    return result, group_offset + len(snapshots)


def collect_data(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    parts = []
    group_offset = 0
    if args.dataset_files:
        for dataset_file in args.dataset_files:
            with np.load(dataset_file) as payload:
                part = {
                    key: payload[key]
                    for key in ["features", "rewards", "costs", "scores", "periods", "groups"]
                }
            part["groups"] = np.arange(
                group_offset, group_offset + len(part["features"]), dtype=np.int64
            )
            group_offset += len(part["features"])
            parts.append(part)
        state_config = json.loads(
            (Path(args.state_checkpoint_dir) / "config.json").read_text()
        )
        horizon = int(state_config["horizon"])
    else:
        model_args = SimpleNamespace(
            state_checkpoint_dir=args.state_checkpoint_dir,
            idm_checkpoint_dir=args.idm_checkpoint_dir,
            step_rm_dir=None,
        )
        policy, state_cfg, state_normalizer, idm, idm_normalizer, _, _ = load_models(
            model_args, device
        )
        horizon = int(state_cfg.horizon)
        for period in sorted(set(args.train_periods + args.val_periods + args.test_periods)):
            part, group_offset = collect_period(
                period,
                args,
                policy,
                state_cfg,
                state_normalizer,
                idm,
                idm_normalizer,
                device,
                group_offset,
            )
            parts.append(part)
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    data["features"] = transform_group_features(data["features"], args.target_mode)
    train_mask = np.isin(data["periods"], args.train_periods)
    val_mask = np.isin(data["periods"], args.val_periods)
    test_mask = np.isin(data["periods"], args.test_periods)
    flat_train = data["features"][train_mask].reshape(-1, data["features"].shape[-1])
    feature_mean = flat_train.mean(0)
    feature_std = np.maximum(flat_train.std(0), 1e-6)
    data["inputs"] = ((data["features"] - feature_mean) / feature_std).astype(np.float32)

    raw_targets = np.stack([data["rewards"], data["costs"], data["scores"]], axis=-1)
    if args.target_mode == "advantage":
        raw_targets = raw_targets - raw_targets[:, :1]
        transformed_targets = np.sign(raw_targets) * np.log1p(np.abs(raw_targets))
        target_transform = "signed_log1p"
    else:
        transformed_targets = np.log1p(np.maximum(raw_targets, 0.0))
        target_transform = "log1p_nonnegative"
    train_targets = transformed_targets[train_mask].reshape(-1, 3)
    target_mean = train_targets.mean(0)
    target_std = np.maximum(train_targets.std(0), 1e-6)
    data["targets"] = ((transformed_targets - target_mean) / target_std).astype(np.float32)
    data["train_mask"] = train_mask
    data["val_mask"] = val_mask
    data["test_mask"] = test_mask
    metadata = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_names": ["episode_reward", "episode_cost", "competition_score"],
        "target_transform": target_transform,
        "target_mode": args.target_mode,
        "input_dim": int(data["inputs"].shape[-1]),
        "horizon": horizon,
        "state_chunk_dim": int(horizon * len(KEEP_STATE_INDICES)),
        "candidate_count": int(args.candidate_count),
        "continuation": "candidate first action, then current-policy closed-loop replanning",
    }
    return data, metadata


def transform_group_features(features: np.ndarray, target_mode: str) -> np.ndarray:
    """Build candidate-relative features without discarding the policy context."""

    features = np.asarray(features, dtype=np.float32)
    if target_mode != "advantage":
        return features
    if features.shape[-1] != 138:
        raise ValueError(f"Advantage features expect 138 raw dimensions, got {features.shape[-1]}")
    context = features[..., :76]
    candidate_payload = features[..., 76:136]
    prefix = features[..., 136:138]
    base_payload = candidate_payload[..., :1, :]
    payload_delta = candidate_payload - base_payload
    return np.concatenate([context, prefix, candidate_payload, payload_delta], axis=-1).astype(np.float32)


class GroupDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, indices: np.ndarray):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        group = self.indices[index]
        return self.inputs[group], self.targets[group]


def ranking_loss(predicted_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    pred_delta = predicted_scores[:, :, None] - predicted_scores[:, None, :]
    target_delta = target_scores[:, :, None] - target_scores[:, None, :]
    mask = target_delta.abs() > 1e-6
    if not mask.any():
        return pred_delta.sum() * 0.0
    direction = target_delta.sign()
    return F.softplus(-direction[mask] * pred_delta[mask]).mean()


def make_loader(
    data: dict,
    indices: np.ndarray,
    batch_groups: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        GroupDataset(data["inputs"], data["targets"], indices),
        batch_size=batch_groups,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def train_member(args: argparse.Namespace, data: dict, seed: int, device: torch.device):
    seed_everything(seed)
    train_indices = np.flatnonzero(data["train_mask"])
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(train_indices, len(train_indices), replace=True)
    val_indices = np.flatnonzero(data["val_mask"])
    train_loader = make_loader(data, bootstrap, args.batch_groups, seed, True)
    val_loader = make_loader(data, val_indices, args.batch_groups, seed, False)
    model = EpisodeQModel(data["inputs"].shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            prediction = model(inputs)
            regression = F.smooth_l1_loss(prediction, targets, beta=0.5)
            rank = ranking_loss(prediction[..., 2], targets[..., 2])
            loss = regression + args.rank_weight * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total = count = 0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                prediction = model(inputs)
                loss = F.mse_loss(prediction, targets)
                total += float(loss) * len(inputs)
                count += len(inputs)
        val_loss = total / max(count, 1)
        if val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(
                "EPISODE_Q_MEMBER "
                + json.dumps({"seed": seed, "epoch": epoch, "val_mse": val_loss}),
                flush=True,
            )
    model.load_state_dict(best_state)
    return model.eval(), best


def decode_predictions(prediction: np.ndarray, metadata: dict) -> np.ndarray:
    mean = np.asarray(metadata["target_mean"], dtype=np.float32)
    std = np.asarray(metadata["target_std"], dtype=np.float32)
    transformed = np.clip(prediction * std + mean, -20.0, 20.0)
    if metadata.get("target_transform") == "signed_log1p":
        return (np.sign(transformed) * np.expm1(np.abs(transformed))).astype(np.float32)
    decoded = np.expm1(transformed)
    return np.maximum(decoded, 0.0).astype(np.float32)


def pairwise_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_delta = prediction[:, :, None] - prediction[:, None, :]
    target_delta = target[:, :, None] - target[:, None, :]
    upper = np.triu(np.ones(target_delta.shape[1:], dtype=bool), k=1)[None]
    mask = upper & (target_delta != 0)
    return float(((pred_delta * target_delta > 0)[mask]).mean()) if mask.any() else float("nan")


def competition_scores_from_heads(
    rewards: np.ndarray,
    costs: np.ndarray,
    cpa_constraints: np.ndarray,
) -> np.ndarray:
    cpa = costs / np.maximum(rewards, 1e-10)
    constraints = np.asarray(cpa_constraints, dtype=np.float32).reshape(-1, 1)
    penalty = np.minimum(1.0, (constraints / np.maximum(cpa, 1e-10)) ** 2)
    return (rewards * penalty).astype(np.float32)


@torch.inference_mode()
def evaluate(models: list[EpisodeQModel], data: dict, mask: np.ndarray, metadata: dict, device: torch.device) -> dict:
    inputs = torch.from_numpy(data["inputs"][mask]).to(device)
    member_prediction = torch.stack([model(inputs) for model in models]).cpu().numpy()
    decoded_members = decode_predictions(member_prediction, metadata)
    prediction = decoded_members.mean(0)
    targets = np.stack([data["rewards"][mask], data["costs"][mask], data["scores"][mask]], axis=-1)
    if metadata.get("target_mode") == "advantage":
        targets = targets - targets[:, :1]
    names = metadata["target_names"]
    metrics = {"groups": int(mask.sum()), "rows": int(mask.sum() * inputs.shape[1])}
    for index, name in enumerate(names):
        error = prediction[..., index] - targets[..., index]
        centered = targets[..., index] - targets[..., index].mean()
        metrics[f"{name}_mae"] = float(np.abs(error).mean())
        metrics[f"{name}_r2"] = float(1.0 - np.sum(error**2) / max(np.sum(centered**2), 1e-8))
        metrics[f"{name}_pairwise_accuracy"] = pairwise_accuracy(
            prediction[..., index], targets[..., index]
        )
    selected = prediction[..., 2].argmax(axis=1)
    target_scores = targets[..., 2]
    metrics["score_top1_regret"] = float(
        np.mean(target_scores.max(axis=1) - target_scores[np.arange(len(selected)), selected])
    )
    if "cpa_constraints" in data:
        derived_scores = competition_scores_from_heads(
            prediction[..., 0],
            prediction[..., 1],
            data["cpa_constraints"][mask],
        )
        derived_selected = derived_scores.argmax(axis=1)
        derived_error = derived_scores - data["scores"][mask]
        metrics["derived_score_mae"] = float(np.abs(derived_error).mean())
        metrics["derived_score_pairwise_accuracy"] = pairwise_accuracy(
            derived_scores, data["scores"][mask]
        )
        metrics["derived_score_top1_regret"] = float(
            np.mean(
                data["scores"][mask].max(axis=1)
                - data["scores"][mask][
                    np.arange(len(derived_selected)), derived_selected
                ]
            )
        )
    normalized_error = np.abs(member_prediction.mean(0) - data["targets"][mask])
    uncertainty = member_prediction.std(0)
    metrics["score_uncertainty_error_corr"] = float(
        np.corrcoef(uncertainty[..., 2].reshape(-1), normalized_error[..., 2].reshape(-1))[0, 1]
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.decision_stride < 1:
        raise ValueError("decision-stride must be positive")
    if args.max_groups_per_period is not None and args.max_groups_per_period < 1:
        raise ValueError("max-groups-per-period must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = collect_data(args, device)
    np.savez_compressed(
        output_dir / "policy_aligned_episode_q_dataset.npz",
        features=data["features"],
        rewards=data["rewards"],
        costs=data["costs"],
        scores=data["scores"],
        periods=data["periods"],
        groups=data["groups"],
    )
    models = []
    members = []
    for member in range(args.ensemble_size):
        model, best = train_member(args, data, args.seed + 100 + member, device)
        torch.save(model.state_dict(), output_dir / f"episode_q_model_{member}.pt")
        models.append(model)
        members.append({"member": member, "best_val_mse": best})
    metrics = {
        "config": vars(args),
        "normalization": metadata,
        "members": members,
        "split_groups": {
            "train": int(data["train_mask"].sum()),
            "validation": int(data["val_mask"].sum()),
            "test": int(data["test_mask"].sum()),
        },
        "validation": evaluate(models, data, data["val_mask"], metadata, device),
        "test": evaluate(models, data, data["test_mask"], metadata, device),
    }
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL_POLICY_ALIGNED_EPISODE_Q " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
