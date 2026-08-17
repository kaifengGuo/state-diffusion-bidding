#!/usr/bin/env python3
"""Train a policy-conditioned Transformer Episode-Q ensemble."""

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
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
for root in [HERE, HERE.parent / "remote_bid_diffusion"]:
    sys.path.insert(0, str(root))
try:
    from transformer_reward_model import MaskedAttentionPooling, SinusoidalPositionalEncoding
except ImportError:
    sys.path.insert(0, str(HERE.parent / "remote_dynamic_rm_aspo"))
    from transformer_reward_model import MaskedAttentionPooling, SinusoidalPositionalEncoding

from train_policy_aligned_state_chunk_rm import load_dataset  # noqa: E402
from train_policy_aligned_episode_q_model import (  # noqa: E402
    competition_scores_from_heads,
    decode_predictions,
    pairwise_accuracy,
    ranking_loss,
)
from train_state_diffusion import KEEP_STATE_INDICES  # noqa: E402


MODEL_NAME = "PolicyConditionedTransformerEpisodeQ"
FULL_STATE_DIM = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--dataset-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-periods", type=int, nargs="+", default=list(range(7, 25)))
    parser.add_argument("--val-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--test-periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--policy-version", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-groups", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--member-index", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument("--reward-cost-weight", type=float, default=1.0)
    parser.add_argument("--score-reg-weight", type=float, default=1.0)
    parser.add_argument(
        "--score-target-mode",
        choices=["absolute", "within_group_advantage"],
        default="absolute",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mse", "score_pairwise"],
        default="mse",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_within_group(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Grouped values must have [groups, candidates] shape")
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return (centered / np.maximum(scale, epsilon)).astype(np.float32)


def listwise_ranking_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Listwise temperature must be positive")
    target_distribution = F.softmax(target_scores / temperature, dim=-1)
    predicted_log_distribution = F.log_softmax(
        predicted_scores / temperature, dim=-1
    )
    return -(target_distribution * predicted_log_distribution).sum(-1).mean()


class PolicyConditionedTransformerEpisodeQ(nn.Module):
    """Encode history/future state tokens and policy context into three Q heads."""

    def __init__(
        self,
        state_dim: int,
        aux_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.state_dim = int(state_dim)
        self.aux_dim = int(aux_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.token_type = nn.Embedding(2, hidden_dim)
        self.position = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.pooling = MaskedAttentionPooling(hidden_dim)
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
        auxiliary: torch.Tensor,
        history_length: int,
    ) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError("Unexpected Transformer Episode-Q state shape")
        if valid_mask.shape != states.shape[:2]:
            raise ValueError("State mask must align with sequence tokens")
        if auxiliary.shape != (len(states), self.aux_dim):
            raise ValueError("Auxiliary policy context has the wrong shape")
        positions = torch.arange(states.shape[1], device=states.device)
        token_types = (positions >= history_length).long()[None].expand(len(states), -1)
        hidden = self.position(self.input_proj(states) + self.token_type(token_types))
        hidden = self.transformer(hidden, src_key_padding_mask=~valid_mask.bool())
        pooled = self.pooling(hidden, valid_mask.bool())
        return self.output(torch.cat([pooled, self.aux_encoder(auxiliary)], dim=-1))


def build_transformer_inputs(
    features: np.ndarray,
    policy_versions: np.ndarray,
    context_dim: int,
    state_chunk_dim: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("Episode-Q features must have [groups, candidates, dims]")
    state_dim = len(KEEP_STATE_INDICES)
    if state_chunk_dim % state_dim:
        raise ValueError("State chunk width is incompatible with kept state dimensions")
    history_width = history_length * FULL_STATE_DIM
    history_full = features[..., :history_width].reshape(
        *features.shape[:2], history_length, FULL_STATE_DIM
    )
    history = history_full[..., KEEP_STATE_INDICES]
    future = features[..., context_dim : context_dim + state_chunk_dim].reshape(
        *features.shape[:2], state_chunk_dim // state_dim, state_dim
    )
    states = np.concatenate([history, future], axis=-2).astype(np.float32)
    masks = np.ones(states.shape[:-1], dtype=bool)
    context_aux = features[..., history_width:context_dim]
    versions = np.broadcast_to(
        np.asarray(policy_versions, dtype=np.float32)[:, None, None],
        (*features.shape[:2], 1),
    )
    auxiliary = np.concatenate([context_aux, versions], axis=-1).astype(np.float32)
    return states, masks, auxiliary


def prepare_data(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    loader_args = SimpleNamespace(
        dataset_files=args.dataset_files,
        collect_only=False,
        train_periods=args.train_periods,
        val_periods=args.val_periods,
        test_periods=args.test_periods,
        state_checkpoint_dir=args.state_checkpoint_dir,
        candidate_count=args.candidate_count,
        include_policy_version=False,
        policy_version=args.policy_version,
    )
    data, base_metadata = load_dataset(loader_args, device)
    state_config = json.loads((Path(args.state_checkpoint_dir) / "config.json").read_text())
    history_length = int(state_config["history_length"])
    states, masks, auxiliary = build_transformer_inputs(
        data["features"],
        data["policy_versions"],
        int(base_metadata["context_dim"]),
        int(base_metadata["state_chunk_dim"]),
        history_length,
    )
    train_aux = auxiliary[data["train_mask"]].reshape(-1, auxiliary.shape[-1])
    aux_mean = train_aux.mean(0)
    aux_std = np.maximum(train_aux.std(0), 1e-6)
    data["states"] = states
    data["masks"] = masks
    data["auxiliary"] = ((auxiliary - aux_mean) / aux_std).astype(np.float32)
    if args.score_target_mode == "within_group_advantage":
        data["targets"] = data["targets"].copy()
        data["targets"][..., 2] = standardize_within_group(data["scores"])
    metadata = {
        **base_metadata,
        "model": MODEL_NAME,
        "state_dim": int(states.shape[-1]),
        "history_length": history_length,
        "sequence_length": int(states.shape[-2]),
        "aux_dim": int(auxiliary.shape[-1]),
        "aux_mean": aux_mean.tolist(),
        "aux_std": aux_std.tolist(),
        "keep_state_indices": KEEP_STATE_INDICES.tolist(),
        "policy_version_dim": 1,
        "current_policy_version": float(args.policy_version),
        "score_target_mode": args.score_target_mode,
        "input_contract": "history_and_future_state_tokens_plus_policy_context_and_version",
    }
    return data, metadata


class GroupDataset(Dataset):
    def __init__(self, data: dict, indices: np.ndarray):
        self.states = torch.from_numpy(data["states"])
        self.masks = torch.from_numpy(data["masks"])
        self.auxiliary = torch.from_numpy(data["auxiliary"])
        self.targets = torch.from_numpy(data["targets"])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        group = self.indices[index]
        return (
            self.states[group],
            self.masks[group],
            self.auxiliary[group],
            self.targets[group],
        )


def make_loader(
    data: dict,
    indices: np.ndarray,
    batch_groups: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        GroupDataset(data, indices),
        batch_size=batch_groups,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def build_model(args: argparse.Namespace, metadata: dict) -> PolicyConditionedTransformerEpisodeQ:
    return PolicyConditionedTransformerEpisodeQ(
        state_dim=int(metadata["state_dim"]),
        aux_dim=int(metadata["aux_dim"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


def train_member(args: argparse.Namespace, data: dict, metadata: dict, member: int, device: torch.device):
    seed = args.seed + 100 + member
    seed_everything(seed)
    train_indices = np.flatnonzero(data["train_mask"])
    bootstrap = np.random.default_rng(seed).choice(
        train_indices, len(train_indices), replace=True
    )
    train_loader = make_loader(data, bootstrap, args.batch_groups, seed, True)
    val_loader = make_loader(
        data, np.flatnonzero(data["val_mask"]), args.batch_groups, seed, False
    )
    model = build_model(args, metadata).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best = float("inf")
    best_state = None
    best_metrics = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, masks, auxiliary, targets in train_loader:
            shape = states.shape
            states = states.reshape(-1, shape[-2], shape[-1]).to(device)
            masks = masks.reshape(-1, shape[-2]).to(device)
            auxiliary = auxiliary.reshape(-1, auxiliary.shape[-1]).to(device)
            targets = targets.to(device)
            prediction = model(states, masks, auxiliary, int(metadata["history_length"]))
            prediction = prediction.reshape(*targets.shape)
            if args.score_target_mode == "within_group_advantage":
                reward_cost_regression = F.smooth_l1_loss(
                    prediction[..., :2], targets[..., :2], beta=0.5
                )
                score_regression = F.smooth_l1_loss(
                    prediction[..., 2], targets[..., 2], beta=0.5
                )
                regression = (
                    args.reward_cost_weight * reward_cost_regression
                    + args.score_reg_weight * score_regression
                )
            else:
                regression = F.smooth_l1_loss(prediction, targets, beta=0.5)
            rank = ranking_loss(prediction[..., 2], targets[..., 2])
            listwise = listwise_ranking_loss(
                prediction[..., 2],
                targets[..., 2],
                args.listwise_temperature,
            )
            loss = (
                regression
                + args.rank_weight * rank
                + args.listwise_weight * listwise
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total = count = 0
        score_predictions = []
        score_targets = []
        with torch.inference_mode():
            for states, masks, auxiliary, targets in val_loader:
                shape = states.shape
                prediction = model(
                    states.reshape(-1, shape[-2], shape[-1]).to(device),
                    masks.reshape(-1, shape[-2]).to(device),
                    auxiliary.reshape(-1, auxiliary.shape[-1]).to(device),
                    int(metadata["history_length"]),
                ).reshape(*targets.shape)
                loss = F.mse_loss(prediction, targets.to(device))
                total += float(loss) * len(states)
                count += len(states)
                score_predictions.append(prediction[..., 2].cpu().numpy())
                score_targets.append(targets[..., 2].numpy())
        val_loss = total / max(count, 1)
        val_pairwise = pairwise_accuracy(
            np.concatenate(score_predictions), np.concatenate(score_targets)
        )
        history.append(
            {
                "epoch": epoch,
                "val_mse": val_loss,
                "val_score_pairwise_accuracy": val_pairwise,
            }
        )
        selection_value = (
            val_loss if args.selection_metric == "mse" else -val_pairwise
        )
        if selection_value < best:
            best = selection_value
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "best_epoch": epoch,
                "best_val_mse": val_loss,
                "best_val_score_pairwise_accuracy": val_pairwise,
                "selection_metric": args.selection_metric,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(
                "TRANSFORMER_EPISODE_Q_MEMBER "
                + json.dumps(
                    {
                        "member": member,
                        "epoch": epoch,
                        "val_mse": val_loss,
                        "val_score_pairwise_accuracy": val_pairwise,
                    }
                ),
                flush=True,
            )
    if best_state is None or best_metrics is None:
        raise RuntimeError("Transformer Episode-Q did not finish an epoch")
    model.load_state_dict(best_state)
    return model.eval(), history, best_metrics


@torch.inference_mode()
def predict_models(models, data: dict, mask: np.ndarray, metadata: dict, device: torch.device):
    indices = np.flatnonzero(mask)
    loader = make_loader(data, indices, 64, 0, False)
    predictions = [[] for _ in models]
    for states, masks, auxiliary, _ in loader:
        shape = states.shape
        flat_states = states.reshape(-1, shape[-2], shape[-1]).to(device)
        flat_masks = masks.reshape(-1, shape[-2]).to(device)
        flat_aux = auxiliary.reshape(-1, auxiliary.shape[-1]).to(device)
        for index, model in enumerate(models):
            output = model(
                flat_states, flat_masks, flat_aux, int(metadata["history_length"])
            ).reshape(shape[0], shape[1], 3)
            predictions[index].append(output.cpu().numpy())
    return np.stack([np.concatenate(parts) for parts in predictions])


def evaluate(models, data: dict, mask: np.ndarray, metadata: dict, device: torch.device) -> dict:
    normalized_members = predict_models(models, data, mask, metadata, device)
    decoded_members = decode_predictions(normalized_members, metadata)
    prediction = decoded_members.mean(0)
    targets = np.stack(
        [data["rewards"][mask], data["costs"][mask], data["scores"][mask]], axis=-1
    )
    metrics = {"groups": int(mask.sum()), "rows": int(mask.sum() * prediction.shape[1])}
    score_target_mode = metadata.get("score_target_mode", "absolute")
    regression_head_count = 2 if score_target_mode == "within_group_advantage" else 3
    for index, name in enumerate(metadata["target_names"][:regression_head_count]):
        error = prediction[..., index] - targets[..., index]
        centered = targets[..., index] - targets[..., index].mean()
        metrics[f"{name}_mae"] = float(np.abs(error).mean())
        metrics[f"{name}_r2"] = float(
            1.0 - np.sum(error**2) / max(np.sum(centered**2), 1e-8)
        )
        metrics[f"{name}_pairwise_accuracy"] = pairwise_accuracy(
            prediction[..., index], targets[..., index]
        )
    if score_target_mode == "within_group_advantage":
        score_prediction = normalized_members[..., 2].mean(0)
        advantage_target = data["targets"][mask][..., 2]
        advantage_error = score_prediction - advantage_target
        advantage_centered = advantage_target - advantage_target.mean()
        metrics["score_advantage_mae"] = float(np.abs(advantage_error).mean())
        metrics["score_advantage_r2"] = float(
            1.0
            - np.sum(advantage_error**2)
            / max(np.sum(advantage_centered**2), 1e-8)
        )
        metrics["competition_score_pairwise_accuracy"] = pairwise_accuracy(
            score_prediction, targets[..., 2]
        )
    else:
        score_prediction = prediction[..., 2]
    selected = score_prediction.argmax(axis=1)
    target_scores = targets[..., 2]
    metrics["score_top1_regret"] = float(
        np.mean(target_scores.max(1) - target_scores[np.arange(len(selected)), selected])
    )
    derived = competition_scores_from_heads(
        prediction[..., 0], prediction[..., 1], data["cpa_constraints"][mask]
    )
    metrics["derived_score_pairwise_accuracy"] = pairwise_accuracy(
        derived, target_scores
    )
    derived_selected = derived.argmax(1)
    metrics["derived_score_top1_regret"] = float(
        np.mean(
            target_scores.max(1)
            - target_scores[np.arange(len(derived_selected)), derived_selected]
        )
    )
    normalized_error = np.abs(normalized_members.mean(0) - data["targets"][mask])
    uncertainty = normalized_members.std(0)
    metrics["score_uncertainty_error_corr"] = float(
        np.corrcoef(
            uncertainty[..., 2].reshape(-1), normalized_error[..., 2].reshape(-1)
        )[0, 1]
    )
    return metrics


def load_models(args: argparse.Namespace, metadata: dict, device: torch.device):
    models = []
    for member in range(args.ensemble_size):
        model = build_model(args, metadata).to(device)
        model.load_state_dict(
            torch.load(
                Path(args.output_dir) / f"transformer_episode_q_{member}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        models.append(model.eval())
    return models


def main() -> None:
    args = parse_args()
    if min(
        args.rank_weight,
        args.listwise_weight,
        args.reward_cost_weight,
        args.score_reg_weight,
    ) < 0:
        raise ValueError("Episode-Q loss weights must be nonnegative")
    if args.listwise_temperature <= 0:
        raise ValueError("Listwise temperature must be positive")
    if args.member_index is not None and not 0 <= args.member_index < args.ensemble_size:
        raise ValueError("member-index must be within ensemble-size")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = prepare_data(args, device)
    config = {**vars(args), "model": MODEL_NAME}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    if not args.evaluate_only:
        members = [args.member_index] if args.member_index is not None else range(args.ensemble_size)
        for member in members:
            model, history, best = train_member(args, data, metadata, member, device)
            torch.save(model.state_dict(), output_dir / f"transformer_episode_q_{member}.pt")
            (output_dir / f"member_{member}_metrics.json").write_text(
                json.dumps({**best, "history": history}, indent=2)
            )
    checkpoints = [output_dir / f"transformer_episode_q_{i}.pt" for i in range(args.ensemble_size)]
    if args.member_index is None and all(path.exists() for path in checkpoints):
        models = load_models(args, metadata, device)
        payload = {
            "config": config,
            "normalization": metadata,
            "split_groups": {
                "train": int(data["train_mask"].sum()),
                "validation": int(data["val_mask"].sum()),
                "test": int(data["test_mask"].sum()),
            },
            "validation": evaluate(models, data, data["val_mask"], metadata, device),
            "test": evaluate(models, data, data["test_mask"], metadata, device),
        }
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        print("FINAL_TRANSFORMER_EPISODE_Q " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
