#!/usr/bin/env python3
"""Train a trajectory-conditioned IDM that predicts only the next bid multiplier."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_auctionnet_idm import (
    Normalizer,
    ResidualBlock,
    action_metrics,
    build_windows,
    make_inputs,
)


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


class SingleActionMLP(nn.Module):
    """Map the current state and full desired state chunk to one action."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(4)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(inputs)))


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    cfg: Config,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
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
            loss = F.smooth_l1_loss(model(inputs), targets, beta=0.5)
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
    inputs: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    device: torch.device,
) -> tuple[SingleActionMLP, list[dict], float]:
    model = SingleActionMLP(inputs.shape[1], cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = make_loader(inputs[train_mask], targets[train_mask], cfg, True)
    val_loader = make_loader(inputs[val_mask], targets[val_mask], cfg, False)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        row = {
            "epoch": epoch,
            "train_huber": run_epoch(model, train_loader, device, optimizer, scaler),
            "val_huber": run_epoch(model, val_loader, device),
        }
        history.append(row)
        print("SINGLE_STEP_IDM " + json.dumps(row), flush=True)
        if row["val_huber"] < best:
            best = row["val_huber"]
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


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "output_dim": 1, "target": "action_t"}, indent=2)
    )

    arrays, data_stats = build_windows(pd.read_csv(cfg.csv_path), cfg.horizon)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    train_periods, val_periods, test_periods = periods[:-4], periods[-4:-2], periods[-2:]
    train_mask = np.isin(arrays["periods"], train_periods)
    val_mask = np.isin(arrays["periods"], val_periods)
    test_mask = np.isin(arrays["periods"], test_periods)

    normalizer = Normalizer().fit(arrays, train_mask)
    normalized_states = normalizer.encode_states(arrays["states"])
    normalized_states[:, 1:, 2] = 0.0
    normalized_states[:, 1:, 3] = 0.0
    normalized_conditions = normalizer.encode_conditions(arrays["conditions"])
    inputs = make_inputs(normalized_states, normalized_conditions, True)
    targets = normalizer.encode_actions(arrays["actions"][:, :1])
    (output_dir / "normalization.json").write_text(
        json.dumps(normalizer.state_dict(), indent=2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = {
        "data": data_stats,
        "train": int(train_mask.sum()),
        "validation": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "periods": [train_periods, val_periods, test_periods],
        "input_dim": int(inputs.shape[1]),
        "output_dim": 1,
        "device": str(device),
    }
    print("DATA " + json.dumps(split), flush=True)
    model, history, best_val = train_model(
        cfg, inputs, targets, train_mask, val_mask, device
    )
    torch.save(model.state_dict(), output_dir / "single_step_idm.pt")

    prediction = normalizer.decode_actions(
        predict(model, inputs[test_mask], device, cfg.batch_size)
    )
    target = arrays["actions"][test_mask, :1]
    current_only_states = normalized_states[test_mask, :1]
    current_only_inputs = make_inputs(
        current_only_states,
        normalized_conditions[test_mask],
        True,
    )
    # Pad current-only inputs so the same trained network can measure its dependence
    # on the desired future trajectory.
    current_only_padded = np.zeros_like(inputs[test_mask])
    current_only_padded[:, :16] = current_only_inputs[:, :16]
    current_only_padded[:, -4:] = current_only_inputs[:, -4:]
    current_only_prediction = normalizer.decode_actions(
        predict(model, current_only_padded, device, cfg.batch_size)
    )
    metrics = {
        "split": split,
        "best_validation_huber": best_val,
        "test": {
            "single_step_idm": action_metrics(prediction, target),
            "copy_previous_action": action_metrics(
                arrays["previous_actions"][test_mask, None], target
            ),
            "alpha_equals_cpa": action_metrics(
                arrays["conditions"][test_mask, 1:2], target
            ),
            "future_chunk_ablated": action_metrics(current_only_prediction, target),
        },
        "future_state_information": {
            "prediction_change_when_chunk_ablated": float(
                np.abs(prediction - current_only_prediction).mean()
            )
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
