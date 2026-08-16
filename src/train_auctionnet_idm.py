#!/usr/bin/env python3
"""Train an AuctionNet chunk inverse-dynamics model on held-out period splits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


STATE_NAMES = [
    "time_left",
    "budget_left",
    "avg_bid_all",
    "avg_bid_last3",
    "avg_lwc_all",
    "avg_pvalue_all",
    "avg_conversion_all",
    "avg_win_all",
    "avg_lwc_last3",
    "avg_pvalue_last3",
    "avg_conversion_last3",
    "avg_win_last3",
    "current_pvalue",
    "current_pv_num",
    "last3_pv_num",
    "historical_pv_num",
]
LOG_STATE_DIMS = (13, 14, 15)


@dataclass
class Config:
    csv_path: str
    output_dir: str
    horizon: int = 4
    hidden_dim: int = 512
    batch_size: int = 512
    epochs: int = 120
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42
    num_workers: int = 4


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return Config(**vars(parser.parse_args()))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_state(value: str) -> np.ndarray:
    result = np.fromstring(value.strip().strip("()"), sep=",", dtype=np.float32)
    if len(result) != len(STATE_NAMES):
        raise ValueError(f"Expected {len(STATE_NAMES)} state values, got {len(result)}")
    return result


def transform_states(states: np.ndarray) -> np.ndarray:
    result = states.astype(np.float32, copy=True)
    result[..., list(LOG_STATE_DIMS)] = np.log1p(
        np.maximum(result[..., list(LOG_STATE_DIMS)], 0.0)
    )
    return result


def build_windows(df: pd.DataFrame, horizon: int) -> tuple[dict[str, np.ndarray], dict]:
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
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    state_chunks = []
    action_chunks = []
    previous_actions = []
    conditions = []
    periods = []
    metadata = []
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
        actions = rows["action"].to_numpy(dtype=np.float32)
        for start in range(1, len(rows) - horizon):
            candidates += 1
            stop = start + horizon
            if done[start:stop].any():
                skipped_done += 1
                continue
            expected_times = np.arange(times[start], times[start] + horizon + 1)
            if not np.array_equal(times[start : stop + 1], expected_times):
                skipped_gap += 1
                continue
            state_chunks.append(states[start : stop + 1])
            action_chunks.append(actions[start:stop])
            previous_actions.append(actions[start - 1])
            row = rows.iloc[start]
            conditions.append(
                [
                    float(row["budget"]),
                    float(row["CPAConstraint"]),
                    float(row["advertiserCategoryIndex"]),
                    float(row["timeStepIndex"]) / 47.0,
                ]
            )
            periods.append(int(period))
            metadata.append([int(period), int(advertiser), int(times[start])])

    arrays = {
        "states": np.asarray(state_chunks, dtype=np.float32),
        "actions": np.asarray(action_chunks, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
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


class Normalizer:
    def fit(self, arrays: dict[str, np.ndarray], train_mask: np.ndarray) -> "Normalizer":
        transformed = transform_states(arrays["states"])
        state_pool = transformed[train_mask].reshape(-1, len(STATE_NAMES))
        self.state_mean = state_pool.mean(0)
        self.state_std = np.maximum(state_pool.std(0), 1e-6)
        self.condition_mean = arrays["conditions"][train_mask].mean(0)
        self.condition_std = np.maximum(arrays["conditions"][train_mask].std(0), 1e-6)
        action_pool = np.concatenate(
            [
                arrays["actions"][train_mask].reshape(-1),
                arrays["previous_actions"][train_mask],
            ]
        )
        log_actions = np.log1p(np.maximum(action_pool, 0.0))
        self.action_mean = float(log_actions.mean())
        self.action_std = float(max(log_actions.std(), 1e-6))
        return self

    def encode_states(self, states: np.ndarray) -> np.ndarray:
        transformed = transform_states(states)
        return ((transformed - self.state_mean) / self.state_std).astype(np.float32)

    def encode_conditions(self, conditions: np.ndarray) -> np.ndarray:
        return ((conditions - self.condition_mean) / self.condition_std).astype(np.float32)

    def encode_actions(self, actions: np.ndarray) -> np.ndarray:
        return (
            (np.log1p(np.maximum(actions, 0.0)) - self.action_mean) / self.action_std
        ).astype(np.float32)

    def decode_actions(self, actions: np.ndarray) -> np.ndarray:
        return np.expm1(actions * self.action_std + self.action_mean).clip(min=0.0).astype(
            np.float32
        )

    def state_dict(self) -> dict:
        result = {}
        for key, value in vars(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result


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


class ChunkActionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(4)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(inputs)))


def make_inputs(
    normalized_states: np.ndarray,
    normalized_conditions: np.ndarray,
    use_future_states: bool,
) -> np.ndarray:
    if use_future_states:
        states = normalized_states.reshape(len(normalized_states), -1)
    else:
        states = normalized_states[:, 0]
    return np.concatenate([states, normalized_conditions], axis=1).astype(np.float32)


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction = model(inputs)
            loss = F.smooth_l1_loss(prediction, targets, beta=0.5)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * len(inputs)
        count += len(inputs)
    return total / max(count, 1)


def train_model(
    cfg: Config,
    name: str,
    train_inputs: np.ndarray,
    val_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, list[dict], float]:
    seed_everything(cfg.seed)
    train_loader = make_loader(
        train_inputs,
        train_targets,
        cfg.batch_size,
        True,
        cfg.num_workers,
        cfg.seed,
    )
    val_loader = make_loader(
        val_inputs,
        val_targets,
        cfg.batch_size,
        False,
        cfg.num_workers,
        cfg.seed,
    )
    model = ChunkActionMLP(
        train_inputs.shape[1], cfg.hidden_dim, cfg.horizon
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train_huber": train_loss, "val_huber": val_loss}
        history.append(row)
        print(f"{name.upper()} " + json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history, best


@torch.inference_mode()
def predict(
    model: nn.Module,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def action_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.abs(prediction - target)
    log_error = np.abs(np.log1p(prediction) - np.log1p(target))
    denominator = np.maximum(np.abs(target), 1.0)
    residual = np.log1p(target) - np.log1p(prediction)
    centered = np.log1p(target) - np.log1p(target).mean()
    return {
        "raw_mae": float(error.mean()),
        "raw_median_ae": float(np.median(error)),
        "raw_p90_ae": float(np.percentile(error, 90)),
        "relative_mae": float((error / denominator).mean()),
        "log_mae": float(log_error.mean()),
        "log_r2": 1.0
        - float((residual**2).sum() / max((centered**2).sum(), 1e-8)),
        "per_horizon_raw_mae": error.mean(0).tolist(),
        "per_horizon_log_mae": log_error.mean(0).tolist(),
    }


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    dataframe = pd.read_csv(cfg.csv_path)
    arrays, data_stats = build_windows(dataframe, cfg.horizon)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    train_periods, val_periods, test_periods = periods[:-4], periods[-4:-2], periods[-2:]
    train_mask = np.isin(arrays["periods"], train_periods)
    val_mask = np.isin(arrays["periods"], val_periods)
    test_mask = np.isin(arrays["periods"], test_periods)

    normalizer = Normalizer().fit(arrays, train_mask)
    normalized_states = normalizer.encode_states(arrays["states"])
    normalized_conditions = normalizer.encode_conditions(arrays["conditions"])
    normalized_actions = normalizer.encode_actions(arrays["actions"])
    current_inputs = make_inputs(normalized_states, normalized_conditions, False)
    oracle_inputs = make_inputs(normalized_states, normalized_conditions, True)
    no_future_bid_stats_states = normalized_states.copy()
    no_future_bid_stats_states[:, 1:, 2] = 0.0
    no_future_bid_stats_states[:, 1:, 3] = 0.0
    no_future_bid_stats_inputs = make_inputs(
        no_future_bid_stats_states, normalized_conditions, True
    )

    split_stats = {
        "data": data_stats,
        "train": int(train_mask.sum()),
        "validation": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "periods": [train_periods, val_periods, test_periods],
        "current_input_dim": int(current_inputs.shape[1]),
        "oracle_idm_input_dim": int(oracle_inputs.shape[1]),
    }
    print("DATA " + json.dumps(split_stats), flush=True)
    (output_dir / "normalization.json").write_text(
        json.dumps(normalizer.state_dict(), indent=2)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_model, current_history, current_val = train_model(
        cfg,
        "current_only",
        current_inputs[train_mask],
        current_inputs[val_mask],
        normalized_actions[train_mask],
        normalized_actions[val_mask],
        device,
    )
    oracle_model, oracle_history, oracle_val = train_model(
        cfg,
        "oracle_idm",
        oracle_inputs[train_mask],
        oracle_inputs[val_mask],
        normalized_actions[train_mask],
        normalized_actions[val_mask],
        device,
    )
    no_bid_stats_model, no_bid_stats_history, no_bid_stats_val = train_model(
        cfg,
        "oracle_idm_no_future_bid_stats",
        no_future_bid_stats_inputs[train_mask],
        no_future_bid_stats_inputs[val_mask],
        normalized_actions[train_mask],
        normalized_actions[val_mask],
        device,
    )
    torch.save(current_model.state_dict(), output_dir / "current_only.pt")
    torch.save(oracle_model.state_dict(), output_dir / "oracle_idm.pt")
    torch.save(
        no_bid_stats_model.state_dict(),
        output_dir / "oracle_idm_no_future_bid_stats.pt",
    )

    current_norm = predict(
        current_model, current_inputs[test_mask], device, cfg.batch_size
    )
    oracle_norm = predict(oracle_model, oracle_inputs[test_mask], device, cfg.batch_size)
    no_bid_stats_norm = predict(
        no_bid_stats_model,
        no_future_bid_stats_inputs[test_mask],
        device,
        cfg.batch_size,
    )
    current_raw = normalizer.decode_actions(current_norm)
    oracle_raw = normalizer.decode_actions(oracle_norm)
    no_bid_stats_raw = normalizer.decode_actions(no_bid_stats_norm)
    target_raw = arrays["actions"][test_mask]

    rng = np.random.default_rng(cfg.seed + 1000)
    shuffled_states = normalized_states[test_mask].copy()
    permutation = rng.permutation(len(shuffled_states))
    shuffled_states[:, 1:] = shuffled_states[permutation, 1:]
    shuffled_inputs = make_inputs(
        shuffled_states, normalized_conditions[test_mask], True
    )
    shuffled_norm = predict(oracle_model, shuffled_inputs, device, cfg.batch_size)
    shuffled_raw = normalizer.decode_actions(shuffled_norm)

    previous_raw = np.repeat(
        arrays["previous_actions"][test_mask, None], cfg.horizon, axis=1
    )
    cpa_raw = np.repeat(
        arrays["conditions"][test_mask, 1:2], cfg.horizon, axis=1
    )
    train_mean = float(arrays["actions"][train_mask].mean())
    mean_raw = np.full_like(target_raw, train_mean)

    metrics = {
        "split": split_stats,
        "best_validation_huber": {
            "current_only": current_val,
            "oracle_idm": oracle_val,
            "oracle_idm_no_future_bid_stats": no_bid_stats_val,
        },
        "test": {
            "oracle_idm_ground_truth_future_states": action_metrics(
                oracle_raw, target_raw
            ),
            "oracle_idm_without_future_bid_stats": action_metrics(
                no_bid_stats_raw, target_raw
            ),
            "oracle_idm_shuffled_future_states": action_metrics(
                shuffled_raw, target_raw
            ),
            "current_state_only": action_metrics(current_raw, target_raw),
            "copy_previous_action": action_metrics(previous_raw, target_raw),
            "alpha_equals_cpa": action_metrics(cpa_raw, target_raw),
            "train_mean_action": action_metrics(mean_raw, target_raw),
        },
        "future_state_information": {
            "raw_mae_improvement_vs_current_only": float(
                action_metrics(current_raw, target_raw)["raw_mae"]
                - action_metrics(oracle_raw, target_raw)["raw_mae"]
            ),
            "shuffled_future_state_prediction_change": float(
                np.abs(shuffled_raw - oracle_raw).mean()
            ),
            "raw_mae_gain_from_future_bid_stats": float(
                action_metrics(no_bid_stats_raw, target_raw)["raw_mae"]
                - action_metrics(oracle_raw, target_raw)["raw_mae"]
            ),
        },
    }
    histories = {
        "current_only": current_history,
        "oracle_idm": oracle_history,
        "oracle_idm_no_future_bid_stats": no_bid_stats_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2))
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        metadata=arrays["metadata"][test_mask],
        target_action=target_raw,
        oracle_idm_action=oracle_raw,
        oracle_idm_no_future_bid_stats_action=no_bid_stats_raw,
        current_only_action=current_raw,
        shuffled_future_action=shuffled_raw,
    )
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
