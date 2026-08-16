#!/usr/bin/env python3
"""Train the internal-style dynamic Transformer RM on the proven state policy."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from transformer_reward_model import TransformerRewardModel
except ImportError:  # Local tests keep the aligned RM in a sibling directory.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote_dynamic_rm_aspo"))
    from transformer_reward_model import TransformerRewardModel

from train_state_chunk_reward_model import load_state_normalizer
from train_state_diffusion import KEEP_STATE_INDICES, build_windows


MODEL_NAME = "DynamicStateTransformerRewardModel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--member-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DynamicStateTransformerRewardModel(TransformerRewardModel):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            out_dim=1,
        )

    def forward(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(states, padding_mask=valid_mask).squeeze(-1)


def build_transformer_sequences(
    history_states: np.ndarray,
    future_states: np.ndarray,
    future_rewards: np.ndarray,
    history_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand every logged window into aligned future lengths 1 through H."""

    if history_states.shape[1] != history_length:
        raise ValueError("history state length does not match the policy config")
    if future_states.shape[1] != horizon or future_rewards.shape[1] != horizon:
        raise ValueError("future state/reward length does not match the policy config")

    sample_count = len(history_states)
    sequence_length = history_length + horizon
    state_dim = history_states.shape[-1]
    sequences = np.zeros(
        (sample_count * horizon, sequence_length, state_dim), dtype=np.float32
    )
    masks = np.zeros((sample_count * horizon, sequence_length), dtype=bool)
    raw_returns = np.zeros(sample_count * horizon, dtype=np.float32)
    source_indices = np.repeat(np.arange(sample_count, dtype=np.int64), horizon)

    for offset, future_length in enumerate(range(1, horizon + 1)):
        rows = np.arange(sample_count, dtype=np.int64) * horizon + offset
        valid_length = history_length + future_length
        sequences[rows, :history_length] = history_states
        sequences[rows, history_length:valid_length] = future_states[:, :future_length]
        masks[rows, :valid_length] = True
        raw_returns[rows] = future_rewards[:, :future_length].sum(axis=1)
    return sequences, masks, raw_returns, source_indices


def prepare_data(args: argparse.Namespace) -> tuple[dict, dict]:
    state_dir = Path(args.state_checkpoint_dir)
    state_cfg, normalizer = load_state_normalizer(state_dir)
    arrays, stats = build_windows(pd.read_csv(args.csv_path), state_cfg)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    if len(periods) < 5:
        raise ValueError("need train, RM validation, policy selection, and test periods")
    train_periods = periods[:-4]
    rm_validation_periods = [periods[-4]]
    policy_selection_periods = [periods[-3]]
    test_periods = periods[-2:]

    history = normalizer.encode_state(arrays["states"])[..., KEEP_STATE_INDICES]
    future = normalizer.encode_state(arrays["future_states"])[..., KEEP_STATE_INDICES]
    rewards = arrays["future_rewards"].astype(np.float32)
    sequences, masks, raw_returns, sources = build_transformer_sequences(
        history,
        future,
        rewards,
        state_cfg.history_length,
        state_cfg.horizon,
    )
    expanded_periods = arrays["periods"][sources]
    train_mask = np.isin(expanded_periods, train_periods)
    rm_val_mask = np.isin(expanded_periods, rm_validation_periods)
    selection_mask = np.isin(expanded_periods, policy_selection_periods)

    log_returns = np.log1p(np.maximum(raw_returns, 0.0))
    return_mean = float(log_returns[train_mask].mean())
    return_std = float(max(log_returns[train_mask].std(), 1e-6))
    targets = ((log_returns - return_mean) / return_std).astype(np.float32)
    future_only = sequences[:, state_cfg.history_length :]
    state_clip = float(
        max(np.percentile(np.abs(future_only[train_mask]), 99.5), 2.5)
    )
    data = {
        "sequences": sequences,
        "masks": masks,
        "targets": targets,
        "raw_returns": raw_returns,
        "periods": expanded_periods,
        "train_mask": train_mask,
        "rm_val_mask": rm_val_mask,
        "selection_mask": selection_mask,
    }
    metadata = {
        "data": stats,
        "train": int(train_mask.sum()),
        "rm_validation": int(rm_val_mask.sum()),
        "policy_selection": int(selection_mask.sum()),
        "periods": [
            train_periods,
            rm_validation_periods,
            policy_selection_periods,
            test_periods,
        ],
        "state_dim": int(sequences.shape[-1]),
        "history_length": int(state_cfg.history_length),
        "horizon": int(state_cfg.horizon),
        "sequence_length": int(sequences.shape[1]),
        "keep_state_indices": KEEP_STATE_INDICES.tolist(),
        "return_mean": return_mean,
        "return_std": return_std,
        "state_clip": state_clip,
        "target": "visible_future_log1p_continuous_return",
        "input_contract": "normalized_history_states_plus_candidate_future_states",
    }
    return data, metadata


def make_loader(
    data: dict,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data["sequences"][indices]),
            torch.from_numpy(data["masks"][indices]),
            torch.from_numpy(data["targets"][indices]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: DynamicStateTransformerRewardModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    squared = absolute = count = 0.0
    for sequences, masks, targets in loader:
        sequences = sequences.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(sequences, masks)
        loss = F.mse_loss(predictions, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        residual = predictions.detach() - targets
        squared += float(residual.square().sum().cpu())
        absolute += float(residual.abs().sum().cpu())
        count += len(targets)
    return {
        "mse": squared / max(count, 1.0),
        "mae": absolute / max(count, 1.0),
    }


@torch.inference_mode()
def evaluate_split(
    model: DynamicStateTransformerRewardModel,
    data: dict,
    mask: np.ndarray,
    metadata: dict,
    device: torch.device,
    batch_size: int,
) -> dict:
    indices = np.flatnonzero(mask)
    predictions = []
    targets = []
    loader = make_loader(data, indices, batch_size, False, 0, 0)
    model.eval()
    for sequences, masks, batch_targets in loader:
        predictions.append(model(sequences.to(device), masks.to(device)).cpu().numpy())
        targets.append(batch_targets.numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    residual = pred - target
    denominator = max(float(((target - target.mean()) ** 2).sum()), 1e-8)
    decoded = np.expm1(
        pred * metadata["return_std"] + metadata["return_mean"]
    ).clip(min=0.0)
    return {
        "normalized_mse": float(np.mean(residual**2)),
        "normalized_mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - float((residual**2).sum()) / denominator,
        "raw_return_mae": float(np.mean(np.abs(decoded - data["raw_returns"][mask]))),
    }


def build_model(args: argparse.Namespace, state_dim: int) -> DynamicStateTransformerRewardModel:
    return DynamicStateTransformerRewardModel(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


def train_member(
    args: argparse.Namespace,
    data: dict,
    metadata: dict,
    member: int,
    device: torch.device,
) -> tuple[DynamicStateTransformerRewardModel, list[dict], float]:
    seed = args.seed + member * 1009
    seed_everything(seed)
    train_indices = np.flatnonzero(data["train_mask"])
    train_loader = make_loader(
        data, train_indices, args.batch_size, True, args.num_workers, seed
    )
    val_loader = make_loader(
        data,
        np.flatnonzero(data["rm_val_mask"]),
        args.batch_size,
        False,
        args.num_workers,
        seed,
    )
    model = build_model(args, metadata["state_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        print(f"TRANSFORMER_STATE_RM_{member} " + json.dumps(row), flush=True)
        if row["val_mse"] < best:
            best = row["val_mse"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("reward model did not complete an epoch")
    model.load_state_dict(best_state)
    return model.eval(), history, best


def main() -> None:
    args = parse_args()
    if args.member_index is not None and not 0 <= args.member_index < args.ensemble_size:
        raise ValueError("member-index must be within ensemble-size")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = prepare_data(args)
    config = {**vars(args), "model": MODEL_NAME}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    print("DATA " + json.dumps(metadata), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = (
        [args.member_index]
        if args.member_index is not None
        else list(range(args.ensemble_size))
    )
    metrics = {"members": {}}
    for member in members:
        model, history, best = train_member(args, data, metadata, member, device)
        torch.save(model.state_dict(), output_dir / f"transformer_state_rm_{member}.pt")
        (output_dir / f"history_{member}.json").write_text(json.dumps(history, indent=2))
        metrics["members"][str(member)] = {
            "best_rm_validation_mse": best,
            "rm_validation": evaluate_split(
                model,
                data,
                data["rm_val_mask"],
                metadata,
                device,
                args.batch_size,
            ),
            "policy_selection": evaluate_split(
                model,
                data,
                data["selection_mask"],
                metadata,
                device,
                args.batch_size,
            ),
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
