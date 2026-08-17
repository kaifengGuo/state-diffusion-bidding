#!/usr/bin/env python3
"""DDPO-IS post-training for State Diffusion with a Transformer RM ensemble."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from evaluate_auctionnet_offline import STATE_DIM
from train_offline_bid_diffusion import DiffusionPolicy, gaussian_log_prob
from train_state_chunk_reward_model import load_state_normalizer
from train_state_diffusion import (
    KEEP_STATE_INDICES,
    build_condition,
    build_windows,
)
from train_transformer_state_chunk_rm import (
    MODEL_NAME,
    DynamicStateTransformerRewardModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--state-rm-checkpoint-dir", required=True)
    parser.add_argument("--ensemble-members", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rl-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--ntp-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--validation-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--contexts-per-iteration", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ntp-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--ppo-clip", type=float, default=0.05)
    parser.add_argument(
        "--reference-weight",
        "--kl-weight",
        dest="reference_weight",
        type=float,
        default=0.1,
        help="Weight for reference-denoiser epsilon MSE (not an exact KL).",
    )
    parser.add_argument("--ntp-base-weight", type=float, default=0.5)
    parser.add_argument("--ntp-risk-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-scale", type=float, default=2.0)
    parser.add_argument("--support-scale", type=float, default=1.0)
    parser.add_argument("--rm-uncertainty-beta", type=float, default=0.0)
    parser.add_argument("--support-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adaptive_loss_weights(
    uncertainty: torch.Tensor,
    support: torch.Tensor,
    uncertainty_scale: float,
    support_scale: float,
    ntp_base_weight: float,
    ntp_risk_weight: float,
) -> tuple[float, float]:
    """Reduce the RL weight and strengthen NTP anchoring as RM risk rises."""

    risk = uncertainty_scale * uncertainty.mean() + support_scale * support.mean()
    confidence = float(torch.exp(-risk).clamp(0.1, 1.0))
    ntp_weight = ntp_base_weight + (1.0 - confidence) * ntp_risk_weight
    return confidence, ntp_weight


def grouped_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Normalize terminal RM rewards among samples from the same context."""

    if len(rewards) % group_size:
        raise ValueError("Reward batch must contain complete candidate groups")
    groups = rewards.reshape(-1, group_size)
    advantages = (groups - groups.mean(1, keepdim=True)) / (
        groups.std(1, keepdim=True, unbiased=False) + 1e-6
    )
    return advantages.reshape(-1)


def clipped_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DDPO-IS clipped surrogate over denoising transitions."""

    ratio = (new_log_prob - old_log_prob).clamp(-8, 8).exp()
    surrogate = torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip, 1 + clip) * advantage,
    )
    return -surrogate.mean(), ratio


def load_policy(checkpoint_dir: Path, device: torch.device):
    cfg, normalizer = load_state_normalizer(checkpoint_dir)
    cond_dim = cfg.history_length * STATE_DIM + cfg.history_length * 2 + 4
    policy = DiffusionPolicy(
        cond_dim,
        cfg.horizon * len(KEEP_STATE_INDICES),
        cfg.hidden_dim,
        cfg.diffusion_steps,
    ).to(device)
    policy.load_state_dict(
        torch.load(
            checkpoint_dir / "state_diffusion.pt",
            map_location=device,
            weights_only=True,
        )
    )
    return policy, cfg, normalizer


def load_ntp_data(
    csv_path: str,
    state_cfg,
    normalizer,
    train_periods: list[int],
    validation_periods: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays, _ = build_windows(pd.read_csv(csv_path), state_cfg)
    contexts = build_condition(
        arrays["states"],
        arrays["past_actions"],
        arrays["past_rewards"],
        arrays["conditions"],
        normalizer,
        normalizer.action_mean,
        normalizer.action_std,
    )
    chunks = normalizer.encode_state(arrays["future_states"])[
        :, :, KEEP_STATE_INDICES
    ].reshape(len(contexts), -1).astype(np.float32)
    train = np.isin(arrays["periods"], train_periods)
    validation = np.isin(arrays["periods"], validation_periods)
    if not train.any() or not validation.any():
        raise ValueError("NTP train and validation periods must both contain windows")
    return contexts[train], chunks[train], contexts[validation], chunks[validation]


@torch.inference_mode()
def ntp_validation_loss(
    policy: DiffusionPolicy,
    contexts: np.ndarray,
    chunks: np.ndarray,
    device: torch.device,
    seed: int,
    sample_size: int = 1024,
) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(contexts), min(sample_size, len(contexts)), replace=False)
    torch.manual_seed(seed)
    return float(
        policy.training_loss(
            torch.from_numpy(chunks[indices]).to(device),
            torch.from_numpy(contexts[indices]).to(device),
        )
    )


def load_transformer_rm(
    checkpoint_dir: Path,
    device: torch.device,
    ensemble_members: list[int] | None = None,
):
    config = json.loads((checkpoint_dir / "config.json").read_text())
    metadata = json.loads((checkpoint_dir / "normalization.json").read_text())
    if config.get("model") != MODEL_NAME:
        raise ValueError(f"expected {MODEL_NAME}, got {config.get('model')}")
    available = int(config["ensemble_size"])
    members = list(range(available)) if ensemble_members is None else ensemble_members
    if not members or len(set(members)) != len(members):
        raise ValueError("ensemble members must be non-empty and unique")
    if min(members) < 0 or max(members) >= available:
        raise ValueError(f"ensemble members must be in [0, {available - 1}]")

    models = []
    for member in members:
        model = DynamicStateTransformerRewardModel(
            state_dim=int(metadata["state_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            ff_dim=int(config["ff_dim"]),
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(
            torch.load(
                checkpoint_dir / f"transformer_state_rm_{member}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        models.append(model)
    return models, metadata


@torch.no_grad()
def transformer_rm_scores(
    models: list[DynamicStateTransformerRewardModel],
    metadata: dict,
    context: torch.Tensor,
    chunks: torch.Tensor,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history_length = int(metadata["history_length"])
    state_dim = int(metadata["state_dim"])
    history_full = context[:, : history_length * STATE_DIM].reshape(
        -1, history_length, STATE_DIM
    )
    keep = torch.as_tensor(
        metadata["keep_state_indices"], dtype=torch.long, device=context.device
    )
    history = history_full.index_select(-1, keep)
    if chunks.ndim != 2 or chunks.shape[1] % state_dim:
        raise ValueError(
            f"state chunk width {chunks.shape} is incompatible with state_dim={state_dim}"
        )
    horizon = chunks.shape[1] // state_dim
    future = chunks.reshape(chunks.shape[0], horizon, state_dim)
    sequence = torch.cat([history, future], dim=1)
    mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
    members = torch.stack([model(sequence, mask) for model in models], dim=0)
    mean = members.mean(0)
    uncertainty = members.std(0, unbiased=False)
    support = F.relu(chunks.abs() - float(metadata["state_clip"])).mean(-1)
    robust = mean - uncertainty_beta * uncertainty - support_penalty * support
    return robust, mean, uncertainty, support


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    state_dir = Path(args.state_checkpoint_dir)
    rm_dir = Path(args.state_rm_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, state_cfg, state_normalizer = load_policy(state_dir, device)
    reference = copy.deepcopy(policy.denoiser).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    reward_models, rm_metadata = load_transformer_rm(
        rm_dir, device, args.ensemble_members
    )

    rl_contexts, _, val_contexts, val_chunks = load_ntp_data(
        args.csv_path,
        state_cfg,
        state_normalizer,
        args.rl_periods,
        args.validation_periods,
    )
    ntp_contexts, ntp_chunks, _, _ = load_ntp_data(
        args.csv_path,
        state_cfg,
        state_normalizer,
        args.ntp_periods,
        args.validation_periods,
    )
    optimizer = torch.optim.AdamW(policy.denoiser.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + 999)
    history = []

    initial_validation_ntp = ntp_validation_loss(
        policy, val_contexts, val_chunks, device, args.seed
    )
    for iteration in range(1, args.iterations + 1):
        indices = rng.choice(
            len(rl_contexts),
            args.contexts_per_iteration,
            replace=len(rl_contexts) < args.contexts_per_iteration,
        )
        base_context = torch.from_numpy(rl_contexts[indices]).to(device)
        context = base_context.repeat_interleave(args.group_size, dim=0)

        policy.eval()
        with torch.no_grad():
            chunks, records = policy.sample(context, record=True)
            robust, rm_mean, uncertainty, support = transformer_rm_scores(
                reward_models,
                rm_metadata,
                context,
                chunks,
                args.rm_uncertainty_beta,
                args.support_penalty,
            )
            advantages = grouped_advantages(robust, args.group_size)
            rl_weight, ntp_weight = adaptive_loss_weights(
                uncertainty,
                support,
                args.uncertainty_scale,
                args.support_scale,
                args.ntp_base_weight,
                args.ntp_risk_weight,
            )
            old_log_prob = torch.stack(
                [record["old_log_prob"] for record in records], dim=1
            )

        policy.train()
        updates = []
        for _ in range(args.ppo_epochs):
            new_log_probs = []
            reference_terms = []
            for record in records:
                mean, variance, current_epsilon = policy.model_stats(
                    record["x_t"], record["t"], context
                )
                new_log_probs.append(
                    gaussian_log_prob(record["x_prev"], mean, variance).sum(-1)
                )
                with torch.no_grad():
                    reference_epsilon = reference(
                        record["x_t"], record["t"], context
                    )
                reference_terms.append(
                    (current_epsilon - reference_epsilon).square().mean(-1)
                )

            new_log_prob = torch.stack(new_log_probs, dim=1)
            step_advantage = advantages[:, None].expand_as(new_log_prob)
            rl_loss, ratio = clipped_ppo_loss(
                new_log_prob,
                old_log_prob,
                step_advantage,
                args.ppo_clip,
            )
            reference_mse = torch.stack(reference_terms, dim=1).mean()
            ntp_indices = rng.choice(
                len(ntp_contexts),
                args.ntp_batch_size,
                replace=len(ntp_contexts) < args.ntp_batch_size,
            )
            ntp_loss = policy.training_loss(
                torch.from_numpy(ntp_chunks[ntp_indices]).to(device),
                torch.from_numpy(ntp_contexts[ntp_indices]).to(device),
            )
            loss = (
                rl_weight * rl_loss
                + ntp_weight * ntp_loss
                + args.reference_weight * reference_mse
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.denoiser.parameters(), 1.0)
            optimizer.step()
            updates.append(
                {
                    "loss": float(loss.detach()),
                    "rl_loss": float(rl_loss.detach()),
                    "ntp_loss": float(ntp_loss.detach()),
                    "reference_mse": float(reference_mse.detach()),
                    "clip_fraction": float(
                        (
                            (ratio < 1 - args.ppo_clip)
                            | (ratio > 1 + args.ppo_clip)
                        )
                        .float()
                        .mean()
                    ),
                }
            )

        policy.eval()
        grouped = robust.reshape(-1, args.group_size)
        row = {
            "iteration": iteration,
            "rm_objective_mean": float(robust.mean()),
            "rm_objective_std": float(robust.std(unbiased=False)),
            "rm_candidate_margin": float(
                (grouped.max(1).values - grouped.mean(1)).mean()
            ),
            "rm_mean": float(rm_mean.mean()),
            "rm_uncertainty": float(uncertainty.mean()),
            "support": float(support.mean()),
            "rl_weight": rl_weight,
            "ntp_weight": ntp_weight,
            "advantage_std": float(advantages.std(unbiased=False)),
            "validation_ntp_loss": ntp_validation_loss(
                policy, val_contexts, val_chunks, device, args.seed + iteration
            ),
            **{
                key: float(np.mean([update[key] for update in updates]))
                for key in updates[0]
            },
        }
        history.append(row)
        print("TRANSFORMER_STATE_DDPO " + json.dumps(row), flush=True)
        torch.save(policy.state_dict(), output_dir / "state_diffusion.pt")
        torch.save(
            policy.state_dict(),
            output_dir / f"state_diffusion_iter_{iteration:03d}.pt",
        )

    shutil.copy2(state_dir / "config.json", output_dir / "config.json")
    shutil.copy2(state_dir / "normalization.json", output_dir / "normalization.json")
    payload = {
        "config": vars(args),
        "initial_validation_ntp_loss": initial_validation_ntp,
        "history": history,
    }
    (output_dir / "ddpo_metrics.json").write_text(json.dumps(payload, indent=2))
    print("FINAL_TRANSFORMER_STATE_DDPO " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
