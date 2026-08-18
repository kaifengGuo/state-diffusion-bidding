#!/usr/bin/env python3
"""Train an ensemble per-decision reward model for dense DDPO advantages."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


STATE_DIM = 16
LOG_STATE_DIMS = (13, 14, 15)


class StepRewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_states(series: pd.Series) -> np.ndarray:
    return np.stack(
        [np.fromstring(value.strip().strip("()"), sep=",", dtype=np.float32) for value in series]
    )


def transform_states(states: np.ndarray) -> np.ndarray:
    result = states.astype(np.float32, copy=True)
    result[:, list(LOG_STATE_DIMS)] = np.log1p(np.maximum(result[:, list(LOG_STATE_DIMS)], 0.0))
    return result


def build_data(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    states = parse_states(frame["state"])
    transformed_states = transform_states(states)
    actions = frame["action"].to_numpy(np.float32)
    conditions = frame[["budget", "CPAConstraint", "advertiserCategoryIndex", "timeStepIndex"]].to_numpy(np.float32)
    conditions[:, 3] /= 47.0
    rewards = frame["reward_continuous"].to_numpy(np.float32)
    periods = frame["deliveryPeriodIndex"].to_numpy(np.int64)
    train_mask = np.isin(periods, list(range(7, 24)))
    val_mask = np.isin(periods, [24, 25])
    test_mask = np.isin(periods, [26, 27])
    state_mean = transformed_states[train_mask].mean(0)
    state_std = np.maximum(transformed_states[train_mask].std(0), 1e-6)
    condition_mean = conditions[train_mask].mean(0)
    condition_std = np.maximum(conditions[train_mask].std(0), 1e-6)
    log_actions = np.log1p(np.maximum(actions[train_mask], 0.0))
    action_mean = float(log_actions.mean())
    action_std = float(max(log_actions.std(), 1e-6))
    signed_rewards = np.sign(rewards) * np.log1p(np.abs(rewards))
    reward_mean = float(signed_rewards[train_mask].mean())
    reward_std = float(max(signed_rewards[train_mask].std(), 1e-6))
    inputs = np.concatenate(
        [
            (transformed_states - state_mean) / state_std,
            ((np.log1p(np.maximum(actions, 0.0)) - action_mean) / action_std)[:, None],
            (conditions - condition_mean) / condition_std,
        ],
        axis=1,
    ).astype(np.float32)
    targets = ((signed_rewards - reward_mean) / reward_std).astype(np.float32)
    normalizer = {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "condition_mean": condition_mean.tolist(),
        "condition_std": condition_std.tolist(),
        "action_mean": action_mean,
        "action_std": action_std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }
    return {
        "inputs": inputs,
        "targets": targets,
        "periods": periods,
        "raw_rewards": rewards,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }, {"normalizer": normalizer, "input_dim": int(inputs.shape[1])}


def make_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def train_member(args, data, seed: int, device: torch.device):
    seed_everything(seed)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train_mask)
    bootstrap = rng.choice(train_indices, len(train_indices), replace=True)
    loader = make_loader(data["inputs"][bootstrap], data["targets"][bootstrap], args.batch_size, True, seed)
    val_inputs = torch.from_numpy(data["inputs"][val_mask]).to(device)
    val_targets = torch.from_numpy(data["targets"][val_mask]).to(device)
    model = StepRewardModel(data["inputs"].shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for inputs, targets in loader:
            prediction = model(inputs.to(device))
            loss = F.smooth_l1_loss(prediction, targets.to(device), beta=0.5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.mse_loss(model(val_inputs), val_targets))
        row = {"epoch": epoch, "train_huber": total / max(count, 1), "val_mse": val_loss}
        history.append(row)
        if val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(f"STEP_RM member={seed} " + json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    return model.eval(), best, history


@torch.inference_mode()
def evaluate(models, data, mask, device):
    inputs = torch.from_numpy(data["inputs"][mask]).to(device)
    target = torch.from_numpy(data["targets"][mask]).to(device)
    predictions = torch.stack([model(inputs) for model in models])
    mean = predictions.mean(0)
    std = predictions.std(0, unbiased=False)
    residual = target - mean
    return {
        "normalized_mae": float(residual.abs().mean()),
        "normalized_rmse": float(residual.square().mean().sqrt()),
        "r2": float(1.0 - residual.square().sum() / (target - target.mean()).square().sum().clamp_min(1e-8)),
        "ensemble_std_mean": float(std.mean()),
        "abs_error_uncertainty_corr": float(np.corrcoef(residual.abs().cpu().numpy(), std.cpu().numpy())[0, 1]),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.csv_path)
    data, metadata = build_data(frame)
    device = torch.device(args.device)
    models = []
    histories = []
    metrics = {"config": vars(args), "data": metadata}
    for member in range(args.ensemble_size):
        model, best, history = train_member(args, data, args.seed + 100 + member, device)
        torch.save(model.state_dict(), output_dir / f"step_reward_model_{member}.pt")
        models.append(model)
        histories.append(history)
        metrics.setdefault("members", []).append({"member": member, "best_val_mse": best})
    metrics["validation"] = evaluate(models, data, data["val_mask"], device)
    metrics["test"] = evaluate(models, data, data["test_mask"], device)
    (output_dir / "normalization.json").write_text(json.dumps(metadata["normalizer"], indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2))
    print("FINAL_STEP_RM " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
