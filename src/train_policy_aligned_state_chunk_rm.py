#!/usr/bin/env python3
"""Train a policy-aligned episode-return RM that only observes context and state chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
for root in [HERE, HERE.parent / "remote_bid_diffusion", HERE.parent / "remote_inverse_dynamics"]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import STATE_DIM, stable_seed  # noqa: E402
from train_auctionnet_idm import Normalizer as IDMNormalizer  # noqa: E402
from train_offline_bid_diffusion import DiffusionPolicy  # noqa: E402
from train_policy_aligned_episode_q_model import (  # noqa: E402
    EpisodeQModel,
    append_decision_state,
    clone_replay_prefix,
    closed_loop_candidate_group_outcomes,
    evaluate,
    train_member,
)
from train_policy_aligned_reward_cost_model import append_environment_step  # noqa: E402
from train_single_step_idm import SingleActionMLP  # noqa: E402
from train_state_chunk_reward_model import load_state_normalizer  # noqa: E402
from train_state_diffusion import KEEP_STATE_INDICES  # noqa: E402
from train_state_replay_ddpo import ReplayState, load_templates, state_condition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--idm-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-files", nargs="+", default=None)
    parser.add_argument("--collect-periods", type=int, nargs="+", default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-periods", type=int, nargs="+", default=[24])
    parser.add_argument("--val-periods", type=int, nargs="+", default=[25])
    parser.add_argument("--test-periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--rollouts-per-advertiser", type=int, default=1)
    parser.add_argument("--advertiser-limit", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--decision-stride", type=int, default=1)
    parser.add_argument("--max-groups-per-period", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--member-index", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_idm(checkpoint_dir: Path, horizon: int, device: torch.device):
    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    if int(cfg["horizon"]) != horizon:
        raise ValueError("State Diffusion and single-step IDM horizons must match")
    normalizer = IDMNormalizer()
    for key, value in json.loads((checkpoint_dir / "normalization.json").read_text()).items():
        setattr(
            normalizer,
            key,
            np.asarray(value, dtype=np.float32) if isinstance(value, list) else value,
        )
    model = SingleActionMLP(STATE_DIM * (1 + horizon) + 4, cfg["hidden_dim"]).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint_dir / "single_step_idm.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model, normalizer


def load_policy(args: argparse.Namespace, device: torch.device):
    state_dir = Path(args.state_checkpoint_dir)
    state_cfg, state_normalizer = load_state_normalizer(state_dir)
    cond_dim = state_cfg.history_length * STATE_DIM + state_cfg.history_length * 2 + 4
    policy = DiffusionPolicy(
        cond_dim,
        state_cfg.horizon * len(KEEP_STATE_INDICES),
        state_cfg.hidden_dim,
        state_cfg.diffusion_steps,
    ).to(device)
    policy.load_state_dict(
        torch.load(state_dir / "state_diffusion.pt", map_location=device, weights_only=True)
    )
    policy.eval()
    idm, idm_normalizer = load_idm(
        Path(args.idm_checkpoint_dir), state_cfg.horizon, device
    )
    return policy, state_cfg, state_normalizer, idm, idm_normalizer


def compose_state_chunk_features(
    policy_context: np.ndarray,
    generated_state_chunks: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Compose the RM input while enforcing that no decoded bid is included."""

    context = np.asarray(policy_context, dtype=np.float32)
    chunks = np.asarray(generated_state_chunks, dtype=np.float32)
    if context.ndim != 2 or chunks.ndim != 2 or len(context) != len(chunks):
        raise ValueError("Policy context and state chunks must be aligned 2-D batches")
    expected_chunk_dim = horizon * len(KEEP_STATE_INDICES)
    if chunks.shape[1] != expected_chunk_dim:
        raise ValueError(
            f"Expected {expected_chunk_dim} state-chunk dimensions, got {chunks.shape[1]}"
        )
    return np.concatenate([context, chunks], axis=1).astype(np.float32)


@torch.inference_mode()
def decode_single_actions(
    generated: torch.Tensor,
    current_states: np.ndarray,
    conditions: np.ndarray,
    state_cfg,
    state_normalizer,
    idm: SingleActionMLP,
    idm_normalizer: IDMNormalizer,
    device: torch.device,
) -> np.ndarray:
    count = len(generated)
    generated_view = generated.reshape(count, state_cfg.horizon, len(KEEP_STATE_INDICES))
    full_norm = np.zeros((count, state_cfg.horizon, STATE_DIM), dtype=np.float32)
    full_norm[:, :, KEEP_STATE_INDICES] = generated_view.cpu().numpy()
    future_states = state_normalizer.decode_state(full_norm)
    all_states = np.concatenate([current_states[:, None], future_states], axis=1)
    normalized_states = idm_normalizer.encode_states(all_states)
    normalized_states[:, 1:, 2] = 0.0
    normalized_states[:, 1:, 3] = 0.0
    normalized_conditions = idm_normalizer.encode_conditions(conditions)
    inputs = np.concatenate(
        [normalized_states.reshape(count, -1), normalized_conditions], axis=1
    ).astype(np.float32)
    outputs = idm(torch.from_numpy(inputs).to(device)).cpu().numpy()
    return idm_normalizer.decode_actions(outputs)[:, 0]


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
) -> dict[str, np.ndarray]:
    templates = load_templates(Path(args.auctionnet_root), period, args.seed)
    if args.advertiser_limit is not None:
        templates = templates[: args.advertiser_limit]
    replays = [
        ReplayState.create(template, index)
        for index, template in enumerate(templates)
        for _ in range(args.rollouts_per_advertiser)
    ]
    base_alphas = np.stack(
        [np.full(48, replay.template.cpa, dtype=np.float32) for replay in replays]
    )
    snapshots: list[dict] = []
    for time_index in range(48):
        for replay in replays:
            append_decision_state(replay, time_index)

        if time_index >= state_cfg.history_length:
            active = np.asarray(
                [i for i, replay in enumerate(replays) if replay.remaining_budget >= 0.1],
                dtype=np.int64,
            )
            if len(active):
                collect_at_time = (
                    (time_index - state_cfg.history_length) % args.decision_stride == 0
                    and (
                        args.max_groups_per_period is None
                        or len(snapshots) < args.max_groups_per_period
                    )
                )
                sample_count = args.candidate_count if collect_at_time else 1
                built = [
                    state_condition(
                        replays[i], time_index, state_normalizer, state_cfg.history_length
                    )
                    for i in active
                ]
                cond = np.stack([item[0] for item in built])
                current = np.stack([item[1] for item in built])
                conditions = np.stack([item[2] for item in built])
                candidate_cond = np.repeat(cond, sample_count, axis=0)
                candidate_current = np.repeat(current, sample_count, axis=0)
                candidate_conditions = np.repeat(conditions, sample_count, axis=0)
                torch.manual_seed(stable_seed(args.seed, "single-idm-state-rm", period, time_index))
                generated, _ = policy.sample(torch.from_numpy(candidate_cond).to(device))
                candidate_alphas = decode_single_actions(
                    generated,
                    candidate_current,
                    candidate_conditions,
                    state_cfg,
                    state_normalizer,
                    idm,
                    idm_normalizer,
                    device,
                )
                features = compose_state_chunk_features(
                    candidate_cond,
                    generated.cpu().numpy(),
                    state_cfg.horizon,
                ).reshape(len(active), sample_count, -1)
                alphas = candidate_alphas.reshape(len(active), sample_count)
                for position, replay_index in enumerate(active):
                    replay = replays[replay_index]
                    should_collect = (
                        collect_at_time
                        and (
                            args.max_groups_per_period is None
                            or len(snapshots) < args.max_groups_per_period
                        )
                    )
                    if should_collect:
                        snapshots.append(
                            {
                                "replay_prefix": clone_replay_prefix(replay),
                                "time_index": time_index,
                                "features": features[position],
                                "candidate_alphas": alphas[position].copy(),
                            }
                        )
                    base_alphas[replay_index, time_index] = alphas[position, 0]

        for replay_index, replay in enumerate(replays):
            append_environment_step(
                replay, float(base_alphas[replay_index, time_index]), time_index
            )

    feature_groups, reward_groups, cost_groups, score_groups = [], [], [], []
    outcomes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for decision_time in sorted({snapshot["time_index"] for snapshot in snapshots}):
        indices = [
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot["time_index"] == decision_time
        ]
        batched = closed_loop_candidate_group_outcomes(
            [snapshots[index]["replay_prefix"] for index in indices],
            decision_time,
            np.stack([snapshots[index]["candidate_alphas"] for index in indices]),
            policy,
            state_cfg,
            state_normalizer,
            idm,
            idm_normalizer,
            device,
            args.seed,
            f"state-chunk-rm-closed-loop-t{decision_time}",
        )
        for position, snapshot_index in enumerate(indices):
            outcomes[snapshot_index] = tuple(values[position] for values in batched)
    for local_group, snapshot in enumerate(snapshots):
        rewards, costs, scores = outcomes[local_group]
        feature_groups.append(snapshot["features"])
        reward_groups.append(rewards)
        cost_groups.append(costs)
        score_groups.append(scores)
    if not feature_groups:
        raise ValueError(f"Period {period} produced no active candidate groups")
    result = {
        "features": np.stack(feature_groups).astype(np.float32),
        "rewards": np.stack(reward_groups).astype(np.float32),
        "costs": np.stack(cost_groups).astype(np.float32),
        "scores": np.stack(score_groups).astype(np.float32),
        "periods": np.full(len(feature_groups), period, dtype=np.int64),
        "groups": np.arange(len(feature_groups), dtype=np.int64),
    }
    print(
        "COLLECT_POLICY_ALIGNED_STATE_RM "
        + json.dumps(
            {
                "period": period,
                "groups": len(feature_groups),
                "rows": len(feature_groups) * args.candidate_count,
                "feature_dim": result["features"].shape[-1],
            }
        ),
        flush=True,
    )
    return result


def load_dataset(args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    parts = []
    if args.dataset_files:
        for path in args.dataset_files:
            with np.load(path) as payload:
                parts.append(
                    {
                        key: payload[key]
                        for key in ["features", "rewards", "costs", "scores", "periods", "groups"]
                    }
                )
    else:
        policy, state_cfg, state_normalizer, idm, idm_normalizer = load_policy(args, device)
        periods = args.collect_periods or sorted(
            set(args.train_periods + args.val_periods + args.test_periods)
        )
        for period in periods:
            parts.append(
                collect_period(
                    period,
                    args,
                    policy,
                    state_cfg,
                    state_normalizer,
                    idm,
                    idm_normalizer,
                    device,
                )
            )
    if not parts:
        raise ValueError("No policy-aligned state-chunk datasets were collected or loaded")
    group_offset = 0
    for part in parts:
        part["groups"] = np.arange(
            group_offset, group_offset + len(part["features"]), dtype=np.int64
        )
        group_offset += len(part["features"])
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if args.collect_only:
        return data, {}
    train_mask = np.isin(data["periods"], args.train_periods)
    val_mask = np.isin(data["periods"], args.val_periods)
    test_mask = np.isin(data["periods"], args.test_periods)
    for name, mask in [
        ("train", train_mask),
        ("validation", val_mask),
        ("test", test_mask),
    ]:
        if not mask.any():
            raise ValueError(f"The {name} split contains no candidate groups")
    flat_train = data["features"][train_mask].reshape(-1, data["features"].shape[-1])
    feature_mean = flat_train.mean(0)
    feature_std = np.maximum(flat_train.std(0), 1e-6)
    data["inputs"] = ((data["features"] - feature_mean) / feature_std).astype(np.float32)
    raw_targets = np.stack([data["rewards"], data["costs"], data["scores"]], axis=-1)
    transformed = np.log1p(np.maximum(raw_targets, 0.0))
    train_targets = transformed[train_mask].reshape(-1, 3)
    target_mean = train_targets.mean(0)
    target_std = np.maximum(train_targets.std(0), 1e-6)
    data["targets"] = ((transformed - target_mean) / target_std).astype(np.float32)
    data["train_mask"] = train_mask
    data["val_mask"] = val_mask
    data["test_mask"] = test_mask
    state_config = json.loads(
        (Path(args.state_checkpoint_dir) / "config.json").read_text()
    )
    horizon = int(state_config["horizon"])
    expected_input_dim = int(data["features"].shape[-1])
    expected_chunk_dim = horizon * len(KEEP_STATE_INDICES)
    if expected_input_dim <= expected_chunk_dim:
        raise ValueError("RM input is missing its policy context")
    context_dim = expected_input_dim - expected_chunk_dim
    _, state_normalizer = load_state_normalizer(Path(args.state_checkpoint_dir))
    normalized_conditions = data["features"][:, 0, context_dim - 4 : context_dim]
    raw_conditions = (
        normalized_conditions * state_normalizer.condition_std
        + state_normalizer.condition_mean
    )
    data["cpa_constraints"] = raw_conditions[:, 1].astype(np.float32)
    metadata = {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_names": ["episode_reward", "episode_cost", "competition_score"],
        "target_transform": "log1p_nonnegative",
        "target_mode": "absolute",
        "input_dim": int(data["inputs"].shape[-1]),
        "horizon": horizon,
        "state_chunk_dim": expected_chunk_dim,
        "context_dim": context_dim,
        "candidate_count": args.candidate_count,
        "input_contract": "normalized_policy_context_plus_generated_state_chunk_only",
        "continuation": "candidate first action, then current-policy closed-loop replanning",
    }
    return data, metadata


def load_episode_models(
    args: argparse.Namespace, input_dim: int, device: torch.device
) -> list[EpisodeQModel]:
    models = []
    for member in range(args.ensemble_size):
        model = EpisodeQModel(input_dim, args.hidden_dim).to(device)
        model.load_state_dict(
            torch.load(
                Path(args.output_dir) / f"state_chunk_episode_q_{member}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        models.append(model.eval())
    return models


def main() -> None:
    args = parse_args()
    if args.decision_stride < 1:
        raise ValueError("decision-stride must be positive")
    if args.max_groups_per_period is not None and args.max_groups_per_period < 1:
        raise ValueError("max-groups-per-period must be positive")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = load_dataset(args, device)
    if not args.dataset_files:
        for period in np.unique(data["periods"]):
            mask = data["periods"] == period
            np.savez_compressed(
                output_dir / f"policy_aligned_state_chunk_period_{int(period)}.npz",
                **{key: data[key][mask] for key in ["features", "rewards", "costs", "scores", "periods", "groups"]},
            )
    if args.collect_only:
        print("COLLECT_ONLY_DONE", flush=True)
        return

    config = {**vars(args), "model": "EpisodeQModel", "feature_dim": metadata["input_dim"]}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "normalization.json").write_text(json.dumps(metadata, indent=2))
    if not args.evaluate_only:
        members = [args.member_index] if args.member_index is not None else range(args.ensemble_size)
        for member in members:
            model, best = train_member(args, data, args.seed + 100 + member, device)
            torch.save(model.state_dict(), output_dir / f"state_chunk_episode_q_{member}.pt")
            (output_dir / f"member_{member}_metrics.json").write_text(
                json.dumps({"best_val_mse": best}, indent=2)
            )
    checkpoints = [output_dir / f"state_chunk_episode_q_{i}.pt" for i in range(args.ensemble_size)]
    if args.member_index is None and all(path.exists() for path in checkpoints):
        models = load_episode_models(args, metadata["input_dim"], device)
        metrics = {
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
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("FINAL_POLICY_ALIGNED_STATE_CHUNK_RM " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
