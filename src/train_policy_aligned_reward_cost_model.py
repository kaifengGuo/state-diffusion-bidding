#!/usr/bin/env python3
"""Train a policy-aligned reward/cost ensemble on closed-loop AuctionNet replay."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
for root in [
    HERE,
    HERE.parent / "remote_bid_diffusion",
    HERE.parent / "bid_diffusion",
    HERE.parent / "remote_inverse_dynamics",
    HERE.parent / "inverse_dynamics",
]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import build_state, competition_score, enforce_budget, stable_seed  # noqa: E402
from train_state_diffusion import KEEP_STATE_INDICES  # noqa: E402
from train_state_replay_ddpo import (  # noqa: E402
    ReplayState,
    decode_actions,
    load_models,
    load_templates,
    state_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--idm-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--val-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--test-periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--rollouts-per-advertiser", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RewardCostModel(nn.Module):
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
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def append_environment_step(replay: ReplayState, alpha: float, time_index: int) -> tuple[float, float]:
    tick = replay.template.ticks[time_index]
    proposed = alpha * tick.pvalues
    if replay.remaining_budget < 0.1:
        proposed = np.zeros_like(tick.pvalues)
    bids, status, costs = enforce_budget(
        proposed, tick.market_prices, replay.remaining_budget, tick.drop_priority
    )
    conversions = tick.potential_conversions * status
    tick_cost = float(costs.sum())
    tick_reward = float(np.sum(tick.pvalues * status))
    replay.remaining_budget = max(0.0, replay.remaining_budget - tick_cost)
    replay.total_cost += tick_cost
    replay.total_continuous_reward += tick_reward
    replay.history_pvalue_info.append(np.stack([tick.pvalues, tick.pvalue_sigmas], axis=1))
    replay.history_bids.append(bids)
    replay.history_auction_result.append(
        np.stack([status, status, costs], axis=1).astype(np.float32)
    )
    replay.history_impression_result.append(
        np.stack([status, conversions], axis=1).astype(np.float32)
    )
    replay.history_market_price.append(tick.market_prices)
    return tick_reward, tick_cost


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
    replays = [
        ReplayState.create(template, advertiser_index)
        for advertiser_index, template in enumerate(templates)
        for _ in range(args.rollouts_per_advertiser)
    ]
    feature_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    cost_rows: list[float] = []
    cpa_rows: list[float] = []
    period_rows: list[int] = []
    group_rows: list[int] = []
    group_id = group_offset
    policy.eval()

    for time_index in range(48):
        for replay in replays:
            tick = replay.template.ticks[time_index]
            state = build_state(
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
            replay.state_history.append(state)

        executed_alpha = np.asarray([replay.template.cpa for replay in replays], dtype=np.float32)
        if time_index >= state_cfg.history_length:
            active = np.asarray(
                [index for index, replay in enumerate(replays) if replay.remaining_budget >= 0.1],
                dtype=np.int64,
            )
            if len(active):
                built = [
                    state_condition(
                        replays[index], time_index, state_normalizer, state_cfg.history_length
                    )
                    for index in active
                ]
                cond_np = np.stack([item[0] for item in built])
                current_np = np.stack([item[1] for item in built])
                condition_np = np.stack([item[2] for item in built])
                candidate_cond = np.repeat(cond_np, args.candidate_count, axis=0)
                candidate_current = np.repeat(current_np, args.candidate_count, axis=0)
                candidate_condition = np.repeat(condition_np, args.candidate_count, axis=0)
                torch.manual_seed(stable_seed(args.seed, "rm-v3", period, time_index))
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
                    [candidate_cond, generated_np, np.log1p(np.maximum(alpha_chunks, 0.0))],
                    axis=1,
                ).astype(np.float32)
                alpha_by_replay = alpha_chunks.reshape(
                    len(active), args.candidate_count, state_cfg.horizon
                )
                feature_by_replay = features.reshape(len(active), args.candidate_count, -1)

                for position, replay_index in enumerate(active):
                    replay = replays[replay_index]
                    tick = replay.template.ticks[time_index]
                    for candidate_index in range(args.candidate_count):
                        alpha = float(alpha_by_replay[position, candidate_index, 0])
                        proposed = alpha * tick.pvalues
                        _, status, costs = enforce_budget(
                            proposed,
                            tick.market_prices,
                            replay.remaining_budget,
                            tick.drop_priority,
                        )
                        feature_rows.append(feature_by_replay[position, candidate_index])
                        reward_rows.append(float(np.sum(tick.pvalues * status)))
                        cost_rows.append(float(costs.sum()))
                        cpa_rows.append(replay.template.cpa)
                        period_rows.append(period)
                        group_rows.append(group_id)
                    executed_alpha[replay_index] = alpha_by_replay[position, 0, 0]
                    group_id += 1

        for replay_index, replay in enumerate(replays):
            append_environment_step(replay, float(executed_alpha[replay_index]), time_index)

    result = {
        "features": np.stack(feature_rows).astype(np.float32),
        "rewards": np.asarray(reward_rows, dtype=np.float32),
        "costs": np.asarray(cost_rows, dtype=np.float32),
        "cpa_constraints": np.asarray(cpa_rows, dtype=np.float32),
        "periods": np.asarray(period_rows, dtype=np.int64),
        "groups": np.asarray(group_rows, dtype=np.int64),
    }
    print(
        "COLLECT_PERIOD "
        + json.dumps(
            {
                "period": period,
                "rows": len(result["features"]),
                "groups": int(len(np.unique(result["groups"]))),
                "feature_dim": int(result["features"].shape[1]),
            }
        ),
        flush=True,
    )
    return result, group_id


def collect_data(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    model_args = SimpleNamespace(
        state_checkpoint_dir=args.state_checkpoint_dir,
        idm_checkpoint_dir=args.idm_checkpoint_dir,
        step_rm_dir=None,
    )
    policy, state_cfg, state_normalizer, idm, idm_normalizer, _, _ = load_models(
        model_args, device
    )
    parts = []
    group_offset = 0
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
    train_mask = np.isin(data["periods"], args.train_periods)
    val_mask = np.isin(data["periods"], args.val_periods)
    test_mask = np.isin(data["periods"], args.test_periods)
    feature_mean = data["features"][train_mask].mean(0)
    feature_std = np.maximum(data["features"][train_mask].std(0), 1e-6)
    log_reward = np.log1p(np.maximum(data["rewards"], 0.0))
    log_cost = np.log1p(np.maximum(data["costs"], 0.0))
    reward_mean = float(log_reward[train_mask].mean())
    reward_std = float(max(log_reward[train_mask].std(), 1e-6))
    cost_mean = float(log_cost[train_mask].mean())
    cost_std = float(max(log_cost[train_mask].std(), 1e-6))
    data["inputs"] = ((data["features"] - feature_mean) / feature_std).astype(np.float32)
    data["targets"] = np.stack(
        [
            (log_reward - reward_mean) / reward_std,
            (log_cost - cost_mean) / cost_std,
        ],
        axis=1,
    ).astype(np.float32)
    data["train_mask"] = train_mask
    data["val_mask"] = val_mask
    data["test_mask"] = test_mask
    metadata = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "cost_mean": cost_mean,
        "cost_std": cost_std,
        "target_transform": "dual_log1p_nonnegative",
        "input_dim": int(data["inputs"].shape[1]),
        "horizon": int(state_cfg.horizon),
        "state_chunk_dim": int(state_cfg.horizon * len(KEEP_STATE_INDICES)),
        "candidate_count": int(args.candidate_count),
    }
    return data, metadata


def make_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def train_member(args: argparse.Namespace, data: dict, seed: int, device: torch.device):
    seed_everything(seed)
    train_groups = np.unique(data["groups"][data["train_mask"]])
    lookup = {
        int(group): np.flatnonzero(data["groups"] == group) for group in train_groups
    }
    rng = np.random.default_rng(seed)
    sampled_groups = rng.choice(train_groups, len(train_groups), replace=True)
    bootstrap = np.concatenate([lookup[int(group)] for group in sampled_groups])
    loader = make_loader(data["inputs"][bootstrap], data["targets"][bootstrap], args.batch_size, seed)
    val_inputs = torch.from_numpy(data["inputs"][data["val_mask"]]).to(device)
    val_targets = torch.from_numpy(data["targets"][data["val_mask"]]).to(device)
    model = RewardCostModel(data["inputs"].shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in loader:
            prediction = model(inputs.to(device))
            loss = F.smooth_l1_loss(prediction, targets.to(device), beta=0.5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.mse_loss(model(val_inputs), val_targets))
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
                "REWARD_COST_MEMBER "
                + json.dumps({"seed": seed, "epoch": epoch, "val_mse": val_loss}),
                flush=True,
            )
    model.load_state_dict(best_state)
    return model.eval(), best


def decode_predictions(prediction: np.ndarray, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    reward = np.expm1(
        np.clip(prediction[..., 0] * metadata["reward_std"] + metadata["reward_mean"], -20.0, 20.0)
    )
    cost = np.expm1(
        np.clip(prediction[..., 1] * metadata["cost_std"] + metadata["cost_mean"], -20.0, 20.0)
    )
    return np.maximum(reward, 0.0), np.maximum(cost, 0.0)


def competition_scores(reward: np.ndarray, cost: np.ndarray, constraint: np.ndarray) -> np.ndarray:
    cpa = cost / (reward + 1e-10)
    return np.where(
        cpa <= constraint,
        reward,
        reward * (constraint / (cpa + 1e-10)) ** 2,
    ).astype(np.float32)


def pairwise_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    correct = total = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            target_delta = target[left] - target[right]
            if target_delta == 0:
                continue
            total += 1
            correct += int((prediction[left] - prediction[right]) * target_delta > 0)
    return correct / total if total else float("nan")


@torch.inference_mode()
def evaluate(models: list[RewardCostModel], data: dict, mask: np.ndarray, metadata: dict, device: torch.device) -> dict:
    inputs = torch.from_numpy(data["inputs"][mask]).to(device)
    member_prediction = torch.stack([model(inputs) for model in models]).cpu().numpy()
    prediction = member_prediction.mean(0)
    pred_reward, pred_cost = decode_predictions(prediction, metadata)
    reward = data["rewards"][mask]
    cost = data["costs"][mask]
    constraint = data["cpa_constraints"][mask]
    groups = data["groups"][mask]
    pred_score = competition_scores(pred_reward, pred_cost, constraint)
    true_score = competition_scores(reward, cost, constraint)
    reward_pairwise = []
    score_pairwise = []
    top1_regret = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        reward_acc = pairwise_accuracy(pred_reward[index], reward[index])
        score_acc = pairwise_accuracy(pred_score[index], true_score[index])
        if np.isfinite(reward_acc):
            reward_pairwise.append(reward_acc)
        if np.isfinite(score_acc):
            score_pairwise.append(score_acc)
        selected = index[int(np.argmax(pred_score[index]))]
        top1_regret.append(float(np.max(true_score[index]) - true_score[selected]))
    reward_error = pred_reward - reward
    cost_error = pred_cost - cost
    true_constraint = cost - constraint * reward
    pred_constraint = pred_cost - constraint * pred_reward
    target = data["targets"][mask]
    normalized_std = member_prediction.std(0)
    reward_uncertainty_corr = float(
        np.corrcoef(normalized_std[:, 0], np.abs(prediction[:, 0] - target[:, 0]))[0, 1]
    )
    cost_uncertainty_corr = float(
        np.corrcoef(normalized_std[:, 1], np.abs(prediction[:, 1] - target[:, 1]))[0, 1]
    )
    return {
        "rows": int(mask.sum()),
        "groups": int(len(np.unique(groups))),
        "reward_raw_mae": float(np.abs(reward_error).mean()),
        "reward_raw_r2": float(
            1.0 - np.sum(reward_error**2) / max(np.sum((reward - reward.mean()) ** 2), 1e-8)
        ),
        "cost_raw_mae": float(np.abs(cost_error).mean()),
        "cost_raw_r2": float(
            1.0 - np.sum(cost_error**2) / max(np.sum((cost - cost.mean()) ** 2), 1e-8)
        ),
        "constraint_residual_mae": float(np.abs(pred_constraint - true_constraint).mean()),
        "constraint_sign_accuracy": float(
            ((pred_constraint > 0) == (true_constraint > 0)).mean()
        ),
        "reward_pairwise_accuracy": float(np.mean(reward_pairwise)),
        "score_pairwise_accuracy": float(np.mean(score_pairwise)),
        "score_top1_regret": float(np.mean(top1_regret)),
        "reward_uncertainty_error_corr": reward_uncertainty_corr,
        "cost_uncertainty_error_corr": cost_uncertainty_corr,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = collect_data(args, device)
    np.savez_compressed(
        output_dir / "policy_aligned_dataset.npz",
        features=data["features"],
        rewards=data["rewards"],
        costs=data["costs"],
        cpa_constraints=data["cpa_constraints"],
        periods=data["periods"],
        groups=data["groups"],
    )
    models = []
    members = []
    for member in range(args.ensemble_size):
        model, best = train_member(args, data, args.seed + 100 + member, device)
        torch.save(model.state_dict(), output_dir / f"reward_cost_model_{member}.pt")
        models.append(model)
        members.append({"member": member, "best_val_mse": best})
    metrics = {
        "config": vars(args),
        "normalization": metadata,
        "members": members,
        "split_rows": {
            "train": int(data["train_mask"].sum()),
            "validation": int(data["val_mask"].sum()),
            "test": int(data["test_mask"].sum()),
        },
        "validation": evaluate(models, data, data["val_mask"], metadata, device),
        "test": evaluate(models, data, data["test_mask"], metadata, device),
    }
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL_POLICY_ALIGNED_RM " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
