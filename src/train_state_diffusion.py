#!/usr/bin/env python3
"""Train a clean state-chunk Diffusion Policy for IDM-guided bidding."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from train_offline_bid_diffusion import DiffusionPolicy, STATE_NAMES, parse_state
except ImportError:  # Local tests run without an explicit PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote_bid_diffusion"))
    from train_offline_bid_diffusion import DiffusionPolicy, STATE_NAMES, parse_state


LOG_STATE_DIMS = (13, 14, 15)
KEEP_STATE_INDICES = np.asarray([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


@dataclass
class Config:
    csv_path: str
    output_dir: str
    history_length: int = 4
    horizon: int = 4
    reward_column: str = "reward_continuous"
    diffusion_steps: int = 20
    hidden_dim: int = 512
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    seed: int = 42
    num_workers: int = 4


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--reward-column", default="reward_continuous")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return Config(**vars(parser.parse_args()))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_states(states: np.ndarray) -> np.ndarray:
    result = states.astype(np.float32, copy=True)
    result[..., list(LOG_STATE_DIMS)] = np.log1p(
        np.maximum(result[..., list(LOG_STATE_DIMS)], 0.0)
    )
    return result


def inverse_transform_states(states: np.ndarray) -> np.ndarray:
    result = states.astype(np.float32, copy=True)
    result[..., list(LOG_STATE_DIMS)] = np.expm1(result[..., list(LOG_STATE_DIMS)])
    return result


def build_windows(df: pd.DataFrame, cfg: Config) -> tuple[dict[str, np.ndarray], dict]:
    required = {
        "deliveryPeriodIndex",
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint",
        "timeStepIndex",
        "state",
        "action",
        "done",
        cfg.reward_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    state_histories, past_actions, past_rewards = [], [], []
    future_states, future_rewards, conditions, periods, metadata = [], [], [], [], []
    candidates = skipped_done = skipped_gap = 0
    for (period, advertiser), group in df.groupby(
        ["deliveryPeriodIndex", "advertiserNumber"], sort=True
    ):
        rows = group.sort_values("timeStepIndex").reset_index(drop=True)
        times = rows["timeStepIndex"].to_numpy(dtype=np.int64)
        done = rows["done"].to_numpy()
        if pd.isna(done).any() or not np.isin(done, [0, 1, False, True]).all():
            raise ValueError(f"Invalid done values for {period}/{advertiser}")
        done = done.astype(bool)
        states = np.stack([parse_state(value) for value in rows["state"]])
        for index in range(cfg.history_length, len(rows) - cfg.horizon + 1):
            candidates += 1
            if done[index : index + cfg.horizon].any():
                skipped_done += 1
                continue
            expected = np.arange(times[index], times[index] + cfg.horizon + 1)
            if not np.array_equal(times[index : index + cfg.horizon + 1], expected):
                skipped_gap += 1
                continue
            state_histories.append(states[index - cfg.history_length + 1 : index + 1])
            past_actions.append(
                rows.iloc[index - cfg.history_length : index]["action"].to_numpy(
                    dtype=np.float32
                )
            )
            past_rewards.append(
                rows.iloc[index - cfg.history_length : index][cfg.reward_column].to_numpy(
                    dtype=np.float32
                )
            )
            future_states.append(states[index + 1 : index + cfg.horizon + 1])
            future_rewards.append(
                rows.iloc[index : index + cfg.horizon][cfg.reward_column].to_numpy(
                    dtype=np.float32
                )
            )
            row = rows.iloc[index]
            conditions.append(
                [
                    float(row["budget"]),
                    float(row["CPAConstraint"]),
                    float(row["advertiserCategoryIndex"]),
                    float(row["timeStepIndex"]) / 47.0,
                ]
            )
            periods.append(int(period))
            metadata.append([int(period), int(advertiser), int(times[index])])

    arrays = {
        "states": np.asarray(state_histories, dtype=np.float32),
        "past_actions": np.asarray(past_actions, dtype=np.float32),
        "past_rewards": np.asarray(past_rewards, dtype=np.float32),
        "future_states": np.asarray(future_states, dtype=np.float32),
        "future_rewards": np.asarray(future_rewards, dtype=np.float32),
        "future_returns": np.asarray(future_rewards, dtype=np.float32).sum(axis=1),
        "conditions": np.asarray(conditions, dtype=np.float32),
        "periods": np.asarray(periods, dtype=np.int64),
        "metadata": np.asarray(metadata, dtype=np.int64),
    }
    stats = {
        "candidate_windows": candidates,
        "valid_windows": len(periods),
        "skipped_done": skipped_done,
        "skipped_time_gap": skipped_gap,
    }
    return arrays, stats


class StateNormalizer:
    def fit(self, arrays: dict[str, np.ndarray], train_mask: np.ndarray) -> "StateNormalizer":
        state_pool = transform_states(
            np.concatenate(
                [arrays["states"][train_mask], arrays["future_states"][train_mask]], axis=1
            )
        ).reshape(-1, len(STATE_NAMES))
        self.state_mean = state_pool.mean(0)
        self.state_std = np.maximum(state_pool.std(0), 1e-6)
        self.condition_mean = arrays["conditions"][train_mask].mean(0)
        self.condition_std = np.maximum(arrays["conditions"][train_mask].std(0), 1e-6)
        signed_reward = np.sign(arrays["past_rewards"]) * np.log1p(
            np.abs(arrays["past_rewards"])
        )
        self.reward_mean = float(signed_reward[train_mask].mean())
        self.reward_std = float(max(signed_reward[train_mask].std(), 1e-6))
        return self

    def encode_state(self, states: np.ndarray) -> np.ndarray:
        return (
            (transform_states(states) - self.state_mean[None, None])
            / self.state_std[None, None]
        ).astype(np.float32)

    def decode_state(self, states: np.ndarray) -> np.ndarray:
        return inverse_transform_states(
            states * self.state_std[None, None] + self.state_mean[None, None]
        )

    def encode_condition(self, conditions: np.ndarray) -> np.ndarray:
        return ((conditions - self.condition_mean) / self.condition_std).astype(np.float32)

    def state_dict(self) -> dict:
        result = {"keep_state_indices": KEEP_STATE_INDICES.tolist()}
        for key, value in vars(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result


def build_condition(
    states: np.ndarray,
    past_actions: np.ndarray,
    past_rewards: np.ndarray,
    conditions: np.ndarray,
    normalizer: StateNormalizer,
    action_mean: float,
    action_std: float,
) -> np.ndarray:
    normalized_states = normalizer.encode_state(states)
    normalized_actions = (
        (np.log1p(np.maximum(past_actions, 0.0)) - action_mean) / action_std
    ).astype(np.float32)
    signed_reward = np.sign(past_rewards) * np.log1p(np.abs(past_rewards))
    normalized_rewards = (
        (signed_reward - normalizer.reward_mean) / normalizer.reward_std
    ).astype(np.float32)
    normalized_conditions = normalizer.encode_condition(conditions)
    return np.concatenate(
        [normalized_states.reshape(len(states), -1), normalized_actions, normalized_rewards, normalized_conditions],
        axis=1,
    ).astype(np.float32)


def make_loader(inputs, targets, cfg: Config, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def train_model(cfg: Config, inputs, targets, train_mask, val_mask, device):
    model = DiffusionPolicy(
        inputs.shape[1],
        cfg.horizon * len(KEEP_STATE_INDICES),
        cfg.hidden_dim,
        cfg.diffusion_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_loader = make_loader(inputs[train_mask], targets[train_mask], cfg, True)
    val_loader = make_loader(inputs[val_mask], targets[val_mask], cfg, False)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        row = {"epoch": epoch}
        for name, loader, training in [("train", train_loader, True), ("val", val_loader, False)]:
            model.train(training)
            total = count = 0
            for cond, target in loader:
                cond, target = cond.to(device), target.to(device)
                loss = model.training_loss(target, cond)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.detach()) * len(cond)
                count += len(cond)
            row[f"{name}_noise_mse"] = total / max(count, 1)
        history.append(row)
        print("STATE_DIFFUSION " + json.dumps(row), flush=True)
        if row["val_noise_mse"] < best:
            best = row["val_noise_mse"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history, best


@torch.inference_mode()
def evaluate_state_model(model, inputs, target_states, normalizer, test_mask, device, cfg):
    predictions = []
    for start in range(0, int(test_mask.sum()), cfg.batch_size):
        cond = torch.from_numpy(inputs[test_mask][start : start + cfg.batch_size]).to(device)
        samples, _ = model.sample(cond)
        predictions.append(samples.cpu().numpy())
    pred_norm = np.concatenate(predictions).reshape(-1, cfg.horizon, len(KEEP_STATE_INDICES))
    target_raw = target_states[test_mask]
    target_norm_full = normalizer.encode_state(target_raw)
    target_norm = target_norm_full[:, :, KEEP_STATE_INDICES]
    err = np.abs(pred_norm - target_norm)
    return {
        "normalized_mae": float(err.mean()),
        "normalized_rmse": float(np.sqrt((err**2).mean())),
        "per_horizon_normalized_mae": err.mean((0, 2)).tolist(),
        "target_raw": target_raw,
        "pred_norm": pred_norm,
    }


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dataframe = pd.read_csv(cfg.csv_path)
    arrays, data_stats = build_windows(dataframe, cfg)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    train_periods, val_periods, test_periods = periods[:-4], periods[-4:-2], periods[-2:]
    train_mask = np.isin(arrays["periods"], train_periods)
    val_mask = np.isin(arrays["periods"], val_periods)
    test_mask = np.isin(arrays["periods"], test_periods)

    normalizer = StateNormalizer().fit(arrays, train_mask)
    action_pool = arrays["past_actions"][train_mask].reshape(-1)
    action_log = np.log1p(np.maximum(action_pool, 0.0))
    action_mean = float(action_log.mean())
    action_std = float(max(action_log.std(), 1e-6))
    cond = build_condition(
        arrays["states"], arrays["past_actions"], arrays["past_rewards"], arrays["conditions"],
        normalizer, action_mean, action_std,
    )
    future_norm = normalizer.encode_state(arrays["future_states"])[:, :, KEEP_STATE_INDICES]
    targets = future_norm.reshape(len(future_norm), -1).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DATA " + json.dumps({
        "data": data_stats,
        "train": int(train_mask.sum()),
        "validation": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "periods": [train_periods, val_periods, test_periods],
        "condition_dim": int(cond.shape[1]),
        "state_chunk_dim": int(targets.shape[1]),
        "device": str(device),
    }), flush=True)
    (output_dir / "normalization.json").write_text(json.dumps({
        **normalizer.state_dict(), "action_mean": action_mean, "action_std": action_std,
    }, indent=2))

    model, history, best_val = train_model(cfg, cond, targets, train_mask, val_mask, device)
    torch.save(model.state_dict(), output_dir / "state_diffusion.pt")
    eval_result = evaluate_state_model(
        model, cond, arrays["future_states"], normalizer, test_mask, device, cfg
    )
    metrics = {
        "data": {
            "data": data_stats,
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
            "periods": [train_periods, val_periods, test_periods],
        },
        "best_val_noise_mse": best_val,
        "test_state_diffusion": {
            key: value for key, value in eval_result.items() if isinstance(value, (float, int, list))
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        metadata=arrays["metadata"][test_mask],
        target_future_states=eval_result["target_raw"],
        predicted_future_states_norm=eval_result["pred_norm"],
    )
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
