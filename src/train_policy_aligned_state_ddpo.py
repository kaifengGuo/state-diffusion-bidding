#!/usr/bin/env python3
"""Conservative DDPO for a state-chunk policy with uncertainty-adaptive NTP anchoring."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
for root in [HERE, HERE.parent / "remote_bid_diffusion"]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import STATE_DIM  # noqa: E402
from train_offline_bid_diffusion import DiffusionPolicy, gaussian_log_prob  # noqa: E402
from train_policy_aligned_episode_q_model import EpisodeQModel  # noqa: E402
from train_state_chunk_reward_model import load_state_normalizer  # noqa: E402
from train_state_diffusion import (  # noqa: E402
    KEEP_STATE_INDICES,
    build_condition,
    build_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--state-rm-checkpoint-dir", required=True)
    parser.add_argument("--rl-dataset-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rl-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--ntp-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--ntp-validation-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--contexts-per-iteration", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ntp-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--ppo-clip", type=float, default=0.05)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--ntp-base-weight", type=float, default=0.1)
    parser.add_argument("--ntp-risk-weight", type=float, default=0.4)
    parser.add_argument("--uncertainty-scale", type=float, default=2.0)
    parser.add_argument("--support-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-beta", type=float, default=0.5)
    parser.add_argument("--support-penalty", type=float, default=0.2)
    parser.add_argument("--cpa-violation-weight", type=float, default=0.0)
    parser.add_argument("--budget-shortfall-weight", type=float, default=0.0)
    parser.add_argument("--budget-util-target", type=float, default=0.9)
    parser.add_argument(
        "--score-source", choices=["direct", "reward_cost"], default="direct"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    risk = uncertainty_scale * uncertainty.mean() + support_scale * support.mean()
    confidence = float(torch.exp(-risk).clamp(0.1, 1.0))
    ntp_weight = ntp_base_weight + (1.0 - confidence) * ntp_risk_weight
    return confidence, ntp_weight


def grouped_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    if len(rewards) % group_size:
        raise ValueError("Reward batch must contain complete candidate groups")
    groups = rewards.reshape(-1, group_size)
    advantages = (groups - groups.mean(1, keepdim=True)) / (
        groups.std(1, keepdim=True, unbiased=False) + 1e-6
    )
    return advantages.reshape(-1)


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
        torch.load(checkpoint_dir / "state_diffusion.pt", map_location=device, weights_only=True)
    )
    return policy, cfg, normalizer


def load_rm(checkpoint_dir: Path, device: torch.device):
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    metadata = json.loads((checkpoint_dir / "normalization.json").read_text())
    if cfg.get("model") != "EpisodeQModel":
        raise ValueError("DDPO requires the policy-aligned EpisodeQModel ensemble")
    models = []
    for member in range(cfg["ensemble_size"]):
        model = EpisodeQModel(metadata["input_dim"], cfg["hidden_dim"]).to(device)
        model.load_state_dict(
            torch.load(
                checkpoint_dir / f"state_chunk_episode_q_{member}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        models.append(model)
    mean = torch.as_tensor(metadata["feature_mean"], device=device)
    std = torch.as_tensor(metadata["feature_std"], device=device)
    return models, metadata, mean, std


@torch.no_grad()
def robust_rm_scores(
    models: list[EpisodeQModel],
    metadata: dict,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    context: torch.Tensor,
    chunks: torch.Tensor,
    uncertainty_beta: float,
    support_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.cat([context, chunks], dim=-1)
    normalized = (features - feature_mean) / feature_std
    member_scores = torch.stack([model(normalized)[..., 2] for model in models])
    mean = member_scores.mean(0)
    uncertainty = member_scores.std(0, unbiased=False)
    normalized_chunks = normalized[..., int(metadata["context_dim"]) :]
    support = F.relu(normalized_chunks.abs() - 3.0).mean(-1)
    robust = mean - uncertainty_beta * uncertainty - support_penalty * support
    return robust, mean, uncertainty, support


@torch.no_grad()
def constrained_episode_q_scores(
    models: list[EpisodeQModel],
    metadata: dict,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    context: torch.Tensor,
    chunks: torch.Tensor,
    state_normalizer,
    uncertainty_beta: float,
    support_penalty: float,
    cpa_violation_weight: float,
    budget_shortfall_weight: float,
    budget_util_target: float,
    score_source: str = "direct",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a conservative score objective from the RM reward/cost/score heads."""

    if metadata.get("target_mode", "absolute") != "absolute":
        raise ValueError("Constraint penalties require an absolute Episode-Q model")
    features = torch.cat([context, chunks], dim=-1)
    normalized = (features - feature_mean) / feature_std
    member_predictions = torch.stack([model(normalized) for model in models])
    target_mean = torch.as_tensor(
        metadata["target_mean"], dtype=context.dtype, device=context.device
    )
    target_std = torch.as_tensor(
        metadata["target_std"], dtype=context.dtype, device=context.device
    )
    transformed = member_predictions * target_std + target_mean
    transformed_mean = transformed.mean(0)
    if metadata.get("target_transform") != "log1p_nonnegative":
        raise ValueError("Constraint penalties require log1p_nonnegative RM targets")
    member_reward = torch.expm1(transformed[..., 0]).clamp(min=0.0)
    member_cost = torch.expm1(transformed[..., 1]).clamp(min=0.0)
    reward = member_reward.mean(0)
    cost = member_cost.mean(0)

    context_dim = int(metadata["context_dim"])
    normalized_chunks = normalized[..., context_dim:]
    support = F.relu(normalized_chunks.abs() - 3.0).mean(-1)
    condition_mean = torch.as_tensor(
        state_normalizer.condition_mean, dtype=context.dtype, device=context.device
    )
    condition_std = torch.as_tensor(
        state_normalizer.condition_std, dtype=context.dtype, device=context.device
    )
    conditions = context[:, -4:] * condition_std + condition_mean
    budget = conditions[:, 0].clamp(min=1e-6)
    cpa_constraint = conditions[:, 1].clamp(min=1e-6)
    if score_source == "reward_cost":
        member_cpa = member_cost / member_reward.clamp(min=1e-6)
        member_penalty = torch.minimum(
            torch.ones_like(member_cpa),
            (cpa_constraint[None] / member_cpa.clamp(min=1e-6)).square(),
        )
        member_score_log = torch.log1p(member_reward * member_penalty)
    elif score_source == "direct":
        member_score_log = transformed[..., 2]
    else:
        raise ValueError(f"Unknown score source: {score_source}")
    score_log = member_score_log.mean(0)
    uncertainty = member_score_log.std(0, unbiased=False)
    cpa_ratio = cost / reward.clamp(min=1e-6) / cpa_constraint
    cpa_violation = F.relu(cpa_ratio - 1.0)
    budget_utilization = cost / budget
    budget_shortfall = F.relu(budget_util_target - budget_utilization)
    objective = (
        score_log
        - uncertainty_beta * uncertainty
        - support_penalty * support
        - cpa_violation_weight * cpa_violation
        - budget_shortfall_weight * budget_shortfall
    )
    return (
        objective,
        score_log,
        uncertainty,
        support,
        cpa_ratio,
        budget_utilization,
    )


def load_rl_contexts(paths: list[str], periods: list[int], context_dim: int) -> np.ndarray:
    contexts = []
    for path in paths:
        with np.load(path) as payload:
            mask = np.isin(payload["periods"], periods)
            features = payload["features"][mask]
            if len(features):
                candidate_contexts = features[..., :context_dim]
                if not np.allclose(candidate_contexts, candidate_contexts[:, :1], atol=1e-6):
                    raise ValueError("Candidates within an RM group do not share policy context")
                contexts.append(candidate_contexts[:, 0])
    if not contexts:
        raise ValueError("No RL contexts matched the requested periods")
    return np.concatenate(contexts).astype(np.float32)


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


def main() -> None:
    args = parse_args()
    if args.cpa_violation_weight < 0 or args.budget_shortfall_weight < 0:
        raise ValueError("Constraint weights must be nonnegative")
    if not 0.0 <= args.budget_util_target <= 1.0:
        raise ValueError("budget-util-target must be in [0, 1]")
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
    models, rm_metadata, feature_mean, feature_std = load_rm(rm_dir, device)
    contexts = load_rl_contexts(
        args.rl_dataset_files, args.rl_periods, int(rm_metadata["context_dim"])
    )
    ntp_context, ntp_chunks, val_context, val_chunks = load_ntp_data(
        args.csv_path,
        state_cfg,
        state_normalizer,
        args.ntp_periods,
        args.ntp_validation_periods,
    )
    optimizer = torch.optim.AdamW(policy.denoiser.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + 999)
    history = []

    for iteration in range(1, args.iterations + 1):
        base_indices = rng.choice(
            len(contexts),
            args.contexts_per_iteration,
            replace=len(contexts) < args.contexts_per_iteration,
        )
        base_context = torch.from_numpy(contexts[base_indices]).to(device)
        context = base_context.repeat_interleave(args.group_size, dim=0)
        policy.eval()
        with torch.no_grad():
            chunks, records = policy.sample(context, record=True)
            (
                robust,
                rm_mean,
                uncertainty,
                support,
                predicted_cpa_ratio,
                predicted_budget_utilization,
            ) = constrained_episode_q_scores(
                models,
                rm_metadata,
                feature_mean,
                feature_std,
                context,
                chunks,
                state_normalizer,
                args.uncertainty_beta,
                args.support_penalty,
                args.cpa_violation_weight,
                args.budget_shortfall_weight,
                args.budget_util_target,
                args.score_source,
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

        policy.train()
        update_rows = []
        for _ in range(args.ppo_epochs):
            new_log_probs = []
            kl_terms = []
            for record in records:
                mean, variance, current_epsilon = policy.model_stats(
                    record["x_t"], record["t"], context
                )
                new_log_probs.append(
                    gaussian_log_prob(record["x_prev"], mean, variance).sum(-1)
                )
                with torch.no_grad():
                    reference_epsilon = reference(record["x_t"], record["t"], context)
                kl_terms.append((current_epsilon - reference_epsilon).square().mean(-1))
            new_log_prob = torch.stack(new_log_probs, dim=1)
            old_log_prob = torch.stack([record["old_log_prob"] for record in records], dim=1)
            ratio = (new_log_prob - old_log_prob).clamp(-8, 8).exp()
            step_advantage = advantages[:, None].expand_as(ratio)
            surrogate = torch.minimum(
                ratio * step_advantage,
                ratio.clamp(1 - args.ppo_clip, 1 + args.ppo_clip) * step_advantage,
            )
            rl_loss = -surrogate.mean()
            reference_mse = torch.stack(kl_terms, dim=1).mean()

            ntp_indices = rng.choice(
                len(ntp_context),
                args.ntp_batch_size,
                replace=len(ntp_context) < args.ntp_batch_size,
            )
            ntp_loss = policy.training_loss(
                torch.from_numpy(ntp_chunks[ntp_indices]).to(device),
                torch.from_numpy(ntp_context[ntp_indices]).to(device),
            )
            loss = (
                rl_weight * rl_loss
                + ntp_weight * ntp_loss
                + args.kl_weight * reference_mse
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.denoiser.parameters(), 1.0)
            optimizer.step()
            update_rows.append(
                {
                    "rl_loss": float(rl_loss.detach()),
                    "ntp_loss": float(ntp_loss.detach()),
                    "reference_mse": float(reference_mse.detach()),
                    "clip_fraction": float(
                        ((ratio < 1 - args.ppo_clip) | (ratio > 1 + args.ppo_clip))
                        .float()
                        .mean()
                    ),
                }
            )

        policy.eval()
        row = {
            "iteration": iteration,
            "robust_rm_score": float(robust.mean()),
            "rm_score": float(rm_mean.mean()),
            "rm_uncertainty": float(uncertainty.mean()),
            "support": float(support.mean()),
            "predicted_cpa_ratio": float(predicted_cpa_ratio.mean()),
            "predicted_budget_utilization": float(
                predicted_budget_utilization.mean()
            ),
            "rl_weight": rl_weight,
            "ntp_weight": ntp_weight,
            "advantage_std": float(advantages.std(unbiased=False)),
            "validation_ntp_loss": ntp_validation_loss(
                policy, val_context, val_chunks, device, args.seed + iteration
            ),
            **{
                key: float(np.mean([update[key] for update in update_rows]))
                for key in update_rows[0]
            },
        }
        history.append(row)
        print("POLICY_ALIGNED_STATE_DDPO " + json.dumps(row), flush=True)
        torch.save(policy.state_dict(), output_dir / "state_diffusion.pt")
        torch.save(policy.state_dict(), output_dir / f"state_diffusion_iter_{iteration:03d}.pt")

    shutil.copy2(state_dir / "config.json", output_dir / "config.json")
    shutil.copy2(state_dir / "normalization.json", output_dir / "normalization.json")
    payload = {"config": vars(args), "history": history}
    (output_dir / "ddpo_metrics.json").write_text(json.dumps(payload, indent=2))
    print("FINAL_POLICY_ALIGNED_STATE_DDPO " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
