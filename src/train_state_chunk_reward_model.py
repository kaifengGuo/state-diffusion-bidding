#!/usr/bin/env python3
"""Train an ensemble reward model that scores generated future state chunks."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_state_diffusion import (
    KEEP_STATE_INDICES,
    Config as StateConfig,
    StateNormalizer,
    build_condition,
    build_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--member-index", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class StateChunkRewardModel(nn.Module):
    def __init__(self, cond_dim: int, chunk_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.cond_dim = cond_dim
        self.chunk_dim = chunk_dim
        self.input = nn.Linear(cond_dim + chunk_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(3)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cond: torch.Tensor, state_chunk: torch.Tensor) -> torch.Tensor:
        features = torch.cat([cond, state_chunk], dim=-1)
        return self.output(self.blocks(self.input(features))).squeeze(-1)


def ensemble_predictions(
    models: list[StateChunkRewardModel],
    cond: torch.Tensor,
    state_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.stack([model(cond, state_chunk) for model in models], dim=0)
    return predictions.mean(0), predictions.std(0, unbiased=False)


def robust_state_chunk_scores(
    models: list[StateChunkRewardModel],
    cond: torch.Tensor,
    state_chunk: torch.Tensor,
    state_clip: float,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean, uncertainty = ensemble_predictions(models, cond, state_chunk)
    support = F.relu(state_chunk.abs() - state_clip).mean(-1)
    score = mean - uncertainty_beta * uncertainty - support_penalty * support
    return score, {"mean": mean, "uncertainty": uncertainty, "support": support}


def robust_state_chunk_cpa_scores(
    models: list[StateChunkRewardModel],
    cond: torch.Tensor,
    state_chunk: torch.Tensor,
    predicted_cost: torch.Tensor,
    cpa_constraint: torch.Tensor,
    return_mean: float,
    return_std: float,
    state_clip: float,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized = torch.stack([model(cond, state_chunk) for model in models], dim=0)
    rewards = torch.expm1(normalized * return_std + return_mean).clamp_min(0.0)
    cost = predicted_cost.unsqueeze(0).expand_as(rewards)
    constraint = cpa_constraint.unsqueeze(0).expand_as(rewards)
    cpa = cost / rewards.clamp_min(1e-6)
    member_scores = torch.where(
        cpa <= constraint,
        rewards,
        rewards * (constraint / cpa.clamp_min(1e-6)).square(),
    )
    mean = member_scores.mean(0)
    uncertainty = member_scores.std(0, unbiased=False)
    support = F.relu(state_chunk.abs() - state_clip).mean(-1)
    score = (
        mean
        - uncertainty_beta * uncertainty
        - support_penalty * mean.detach().abs() * support
    )
    return score, {
        "mean": mean,
        "uncertainty": uncertainty,
        "support": support,
        "predicted_reward": rewards.mean(0),
        "predicted_cost": predicted_cost,
    }


def load_state_normalizer(checkpoint_dir: Path) -> tuple[StateConfig, StateNormalizer]:
    cfg = StateConfig(**json.loads((checkpoint_dir / "config.json").read_text()))
    payload = json.loads((checkpoint_dir / "normalization.json").read_text())
    normalizer = StateNormalizer()
    for key, value in payload.items():
        if key == "keep_state_indices":
            continue
        setattr(
            normalizer,
            key,
            np.asarray(value, dtype=np.float32) if isinstance(value, list) else value,
        )
    return cfg, normalizer


def prepare_data(args: argparse.Namespace) -> tuple[dict, dict, StateConfig]:
    state_dir = Path(args.state_checkpoint_dir)
    state_cfg, normalizer = load_state_normalizer(state_dir)
    arrays, stats = build_windows(pd.read_csv(args.csv_path), state_cfg)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    train_periods, val_periods, test_periods = periods[:-4], periods[-4:-2], periods[-2:]
    train_mask = np.isin(arrays["periods"], train_periods)
    val_mask = np.isin(arrays["periods"], val_periods)
    test_mask = np.isin(arrays["periods"], test_periods)
    cond = build_condition(
        arrays["states"],
        arrays["past_actions"],
        arrays["past_rewards"],
        arrays["conditions"],
        normalizer,
        normalizer.action_mean,
        normalizer.action_std,
    )
    state_chunk = normalizer.encode_state(arrays["future_states"])[
        :, :, KEEP_STATE_INDICES
    ].reshape(len(cond), -1)
    raw_return = arrays["future_returns"].astype(np.float32)
    log_return = np.log1p(np.maximum(raw_return, 0.0))
    return_mean = float(log_return[train_mask].mean())
    return_std = float(max(log_return[train_mask].std(), 1e-6))
    target = ((log_return - return_mean) / return_std).astype(np.float32)
    state_clip = float(max(np.percentile(np.abs(state_chunk[train_mask]), 99.5), 2.5))
    data = {
        "cond": cond,
        "state_chunk": state_chunk.astype(np.float32),
        "target": target,
        "raw_return": raw_return,
        "periods": arrays["periods"],
        "metadata": arrays["metadata"],
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
    metadata = {
        "data": stats,
        "train": int(train_mask.sum()),
        "validation": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "periods": [train_periods, val_periods, test_periods],
        "cond_dim": int(cond.shape[1]),
        "chunk_dim": int(state_chunk.shape[1]),
        "history_length": state_cfg.history_length,
        "horizon": state_cfg.horizon,
        "keep_state_indices": KEEP_STATE_INDICES.tolist(),
        "return_mean": return_mean,
        "return_std": return_std,
        "state_clip": state_clip,
        "target": "log1p_standardized_h_step_continuous_return",
    }
    return data, metadata, state_cfg


def make_loader(
    cond: np.ndarray,
    state_chunk: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(cond),
            torch.from_numpy(state_chunk),
            torch.from_numpy(target),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: StateChunkRewardModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = count = 0
    for cond, state_chunk, target in loader:
        cond = cond.to(device, non_blocking=True)
        state_chunk = state_chunk.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss = F.smooth_l1_loss(model(cond, state_chunk), target, beta=0.5)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(cond)
        count += len(cond)
    return total / max(count, 1)


def train_member(
    args: argparse.Namespace,
    data: dict,
    metadata: dict,
    member: int,
    device: torch.device,
) -> tuple[StateChunkRewardModel, list[dict], float]:
    seed = args.seed + 100 + member
    seed_everything(seed)
    train_indices = np.flatnonzero(data["train_mask"])
    bootstrap = np.random.default_rng(seed).choice(
        train_indices, len(train_indices), replace=True
    )
    train_loader = make_loader(
        data["cond"][bootstrap],
        data["state_chunk"][bootstrap],
        data["target"][bootstrap],
        args.batch_size,
        True,
        args.num_workers,
        seed,
    )
    val_loader = make_loader(
        data["cond"][data["val_mask"]],
        data["state_chunk"][data["val_mask"]],
        data["target"][data["val_mask"]],
        args.batch_size,
        False,
        args.num_workers,
        seed,
    )
    model = StateChunkRewardModel(
        metadata["cond_dim"], metadata["chunk_dim"], args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        row = {
            "epoch": epoch,
            "train_huber": run_epoch(model, train_loader, device, optimizer, scaler),
            "val_huber": run_epoch(model, val_loader, device),
        }
        history.append(row)
        print(f"STATE_CHUNK_RM_{member} " + json.dumps(row), flush=True)
        if row["val_huber"] < best:
            best = row["val_huber"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history, best


@torch.inference_mode()
def predict_ensemble(
    models: list[StateChunkRewardModel],
    cond: np.ndarray,
    state_chunk: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for start in range(0, len(cond), batch_size):
        batch_cond = torch.from_numpy(cond[start : start + batch_size]).to(device)
        batch_chunk = torch.from_numpy(state_chunk[start : start + batch_size]).to(device)
        mean, std = ensemble_predictions(models, batch_cond, batch_chunk)
        means.append(mean.cpu().numpy())
        stds.append(std.cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def decode_return(values: np.ndarray, metadata: dict) -> np.ndarray:
    return np.expm1(values * metadata["return_std"] + metadata["return_mean"]).clip(
        min=0.0
    )


def evaluate_ensemble(
    models: list[StateChunkRewardModel],
    data: dict,
    metadata: dict,
    device: torch.device,
    batch_size: int,
) -> dict:
    mask = data["test_mask"]
    cond = data["cond"][mask]
    chunks = data["state_chunk"][mask]
    target = data["target"][mask]
    mean, std = predict_ensemble(models, cond, chunks, device, batch_size)
    residual = target - mean
    denominator = max(float(((target - target.mean()) ** 2).sum()), 1e-8)
    rng = np.random.default_rng(20260806)
    shuffled = chunks[rng.permutation(len(chunks))]
    shuffled_mean, _ = predict_ensemble(models, cond, shuffled, device, batch_size)
    shuffled_residual = target - shuffled_mean
    return {
        "normalized_mae": float(np.abs(residual).mean()),
        "normalized_rmse": float(np.sqrt((residual**2).mean())),
        "r2": 1.0 - float((residual**2).sum()) / denominator,
        "raw_return_mae": float(
            np.abs(decode_return(mean, metadata) - data["raw_return"][mask]).mean()
        ),
        "ensemble_std_mean": float(std.mean()),
        "ensemble_std_p90": float(np.percentile(std, 90)),
        "abs_error_uncertainty_corr": float(np.corrcoef(np.abs(residual), std)[0, 1]),
        "shuffled_state_r2": 1.0 - float((shuffled_residual**2).sum()) / denominator,
        "shuffled_state_prediction_change": float(np.abs(mean - shuffled_mean).mean()),
    }


def load_models(
    args: argparse.Namespace,
    metadata: dict,
    device: torch.device,
) -> list[StateChunkRewardModel]:
    output_dir = Path(args.output_dir)
    models = []
    for member in range(args.ensemble_size):
        model = StateChunkRewardModel(
            metadata["cond_dim"], metadata["chunk_dim"], args.hidden_dim
        ).to(device)
        model.load_state_dict(
            torch.load(
                output_dir / f"state_chunk_rm_{member}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        models.append(model.eval())
    return models


def main() -> None:
    args = parse_args()
    if args.member_index is not None and not 0 <= args.member_index < args.ensemble_size:
        raise ValueError("member-index must be within the ensemble range")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata, _ = prepare_data(args)
    config = {
        **vars(args),
        "model": "StateChunkRewardModel",
        "input_contract": "policy_condition_plus_generated_state_chunk",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    print("DATA " + json.dumps(metadata), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.evaluate_only:
        members = (
            [args.member_index]
            if args.member_index is not None
            else list(range(args.ensemble_size))
        )
        for member in members:
            model, history, best = train_member(args, data, metadata, member, device)
            torch.save(model.state_dict(), output_dir / f"state_chunk_rm_{member}.pt")
            (output_dir / f"history_{member}.json").write_text(
                json.dumps(history, indent=2)
            )
            (output_dir / f"member_{member}_metrics.json").write_text(
                json.dumps({"best_validation_huber": best}, indent=2)
            )

    expected = [output_dir / f"state_chunk_rm_{i}.pt" for i in range(args.ensemble_size)]
    if all(path.exists() for path in expected):
        models = load_models(args, metadata, device)
        metrics = {
            "split": {
                key: metadata[key]
                for key in ["data", "train", "validation", "test", "periods"]
            },
            "test": evaluate_ensemble(models, data, metadata, device, args.batch_size),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
