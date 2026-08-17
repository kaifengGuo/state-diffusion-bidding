#!/usr/bin/env python3
"""Closed-loop AuctionNet replay DDPO for the state-chunk diffusion policy."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
for root in [
    HERE,
    HERE.parent / "remote_bid_diffusion",
    HERE.parent / "bid_diffusion",
    HERE.parent / "remote_inverse_dynamics",
    HERE.parent / "inverse_dynamics",
]:
    sys.path.insert(0, str(root))

from evaluate_auctionnet_offline import (  # noqa: E402
    STATE_DIM,
    TickData,
    action_from_tick,
    build_state,
    build_ticks,
    competition_score,
    enforce_budget,
)
from train_auctionnet_idm import (  # noqa: E402
    ChunkActionMLP,
    Normalizer as IDMNormalizer,
)
from train_single_step_idm import SingleActionMLP  # noqa: E402
from train_offline_bid_diffusion import (  # noqa: E402
    DiffusionPolicy,
    gaussian_log_prob,
)
from train_step_reward_model import (  # noqa: E402
    StepRewardModel,
    transform_states as transform_step_states,
)
from train_state_diffusion import (  # noqa: E402
    KEEP_STATE_INDICES,
    Config as StateConfig,
    StateNormalizer,
    build_condition,
    build_windows,
)


@dataclass
class EpisodeTemplate:
    period: int
    advertiser: int
    budget: float
    cpa: float
    category: int
    ticks: list[TickData]


@dataclass
class ReplayState:
    template: EpisodeTemplate
    group_id: int
    remaining_budget: float
    state_history: list[np.ndarray]
    history_pvalue_info: list[np.ndarray]
    history_bids: list[np.ndarray]
    history_auction_result: list[np.ndarray]
    history_impression_result: list[np.ndarray]
    history_market_price: list[np.ndarray]
    total_continuous_reward: float = 0.0
    total_cost: float = 0.0

    @classmethod
    def create(cls, template: EpisodeTemplate, group_id: int) -> "ReplayState":
        return cls(template, group_id, template.budget, [], [], [], [], [], [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auctionnet-root", required=True)
    parser.add_argument("--state-checkpoint-dir", required=True)
    parser.add_argument("--idm-checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-period", type=int, default=24)
    parser.add_argument("--train-periods", type=int, nargs="+", default=None)
    parser.add_argument("--advertisers-per-iteration", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip", type=float, default=0.1)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--step-rm-dir", default=None)
    parser.add_argument("--step-rm-weight", type=float, default=0.5)
    parser.add_argument("--dense-discount", type=float, default=1.0)
    parser.add_argument(
        "--advantage-mode", choices=["episode", "potential"], default="potential"
    )
    parser.add_argument("--ntp-csv-path", default=None)
    parser.add_argument("--ntp-periods", type=int, nargs="+", default=None)
    parser.add_argument("--ntp-weight", type=float, default=0.5)
    parser.add_argument("--ntp-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_payload(target: object, path: Path) -> object:
    for key, value in json.loads(path.read_text()).items():
        setattr(target, key, np.asarray(value, dtype=np.float32) if isinstance(value, list) else value)
    return target


def load_models(args: argparse.Namespace, device: torch.device):
    state_dir = Path(args.state_checkpoint_dir)
    state_cfg = StateConfig(**json.loads((state_dir / "config.json").read_text()))
    state_normalizer = load_payload(StateNormalizer(), state_dir / "normalization.json")
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

    idm_dir = Path(args.idm_checkpoint_dir)
    idm_cfg = json.loads((idm_dir / "config.json").read_text())
    if idm_cfg["horizon"] != state_cfg.horizon:
        raise ValueError("State Diffusion and IDM horizons must match")
    idm_normalizer = load_payload(IDMNormalizer(), idm_dir / "normalization.json")
    idm_input_dim = STATE_DIM * (1 + idm_cfg["horizon"]) + 4
    single_step_idm = idm_cfg.get("output_dim") == 1 or idm_cfg.get("target") == "action_t"
    if single_step_idm:
        idm = SingleActionMLP(idm_input_dim, idm_cfg["hidden_dim"]).to(device)
        checkpoint_name = "single_step_idm.pt"
    else:
        idm = ChunkActionMLP(
            idm_input_dim,
            idm_cfg["hidden_dim"],
            idm_cfg["horizon"],
        ).to(device)
        checkpoint_name = "oracle_idm_no_future_bid_stats.pt"
    idm.load_state_dict(
        torch.load(
            idm_dir / checkpoint_name,
            map_location=device,
            weights_only=True,
        )
    )
    idm.eval()
    for parameter in idm.parameters():
        parameter.requires_grad = False
    step_models = []
    step_payload = None
    if args.step_rm_dir:
        step_dir = Path(args.step_rm_dir)
        step_payload = json.loads((step_dir / "normalization.json").read_text())
        input_dim = 21
        for index in range(5):
            model = StepRewardModel(input_dim, 384).to(device)
            model.load_state_dict(
                torch.load(
                    step_dir / f"step_reward_model_{index}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            step_models.append(model)
    return policy, state_cfg, state_normalizer, idm, idm_normalizer, step_models, step_payload


def load_ntp_samples(
    csv_path: str,
    periods: list[int],
    state_cfg: StateConfig,
    normalizer: StateNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    arrays, _ = build_windows(pd.read_csv(csv_path), state_cfg)
    mask = np.isin(arrays["periods"], periods)
    if not mask.any():
        raise ValueError("No NTP windows matched the requested periods")
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
    return contexts[mask], chunks[mask]


def load_templates(root: Path, period: int, seed: int) -> list[EpisodeTemplate]:
    path = root / "strategy_train_env/data/traffic" / f"period-{period}.csv"
    usecols = [
        "deliveryPeriodIndex",
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint",
        "timeStepIndex",
        "pValue",
        "pValueSigma",
        "bid",
        "leastWinningCost",
    ]
    frame = pd.read_csv(path, usecols=usecols, dtype=np.float32)
    templates = []
    for advertiser_value, group in frame.groupby("advertiserNumber", sort=True):
        first = group.iloc[0]
        advertiser = int(advertiser_value)
        templates.append(
            EpisodeTemplate(
                period,
                advertiser,
                float(first["budget"]),
                float(first["CPAConstraint"]),
                int(first["advertiserCategoryIndex"]),
                build_ticks(group, seed, period, advertiser),
            )
        )
    return templates


def state_condition(
    replay: ReplayState,
    time_index: int,
    normalizer: StateNormalizer,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.stack(replay.state_history[-history_length:])
    recent_pvalues = replay.history_pvalue_info[-history_length:]
    recent_bids = replay.history_bids[-history_length:]
    recent_impressions = replay.history_impression_result[-history_length:]
    past_actions = np.asarray(
        [action_from_tick(bids, info[:, 0]) for bids, info in zip(recent_bids, recent_pvalues)],
        dtype=np.float32,
    )
    past_rewards = np.asarray(
        [float(np.sum(info[:, 0] * result[:, 0])) for info, result in zip(recent_pvalues, recent_impressions)],
        dtype=np.float32,
    )
    conditions = np.asarray(
        [replay.template.budget, replay.template.cpa, replay.template.category, time_index / 47.0],
        dtype=np.float32,
    )
    normalized_states = normalizer.encode_state(states[None])
    normalized_actions = (
        (np.log1p(np.maximum(past_actions, 0.0)) - normalizer.action_mean)
        / normalizer.action_std
    ).astype(np.float32)
    signed_reward = np.sign(past_rewards) * np.log1p(np.abs(past_rewards))
    normalized_rewards = (
        (signed_reward - normalizer.reward_mean) / normalizer.reward_std
    ).astype(np.float32)
    normalized_conditions = normalizer.encode_condition(conditions[None])
    cond = np.concatenate(
        [
            normalized_states.reshape(1, -1),
            normalized_actions[None],
            normalized_rewards[None],
            normalized_conditions,
        ],
        axis=1,
    ).astype(np.float32)
    return cond[0], replay.state_history[-1], conditions


@torch.inference_mode()
def decode_actions(
    generated: torch.Tensor,
    current_states: np.ndarray,
    conditions: np.ndarray,
    state_cfg: StateConfig,
    state_normalizer: StateNormalizer,
    idm: ChunkActionMLP,
    idm_normalizer: IDMNormalizer,
    device: torch.device,
) -> np.ndarray:
    batch = len(current_states)
    generated = generated.reshape(batch, state_cfg.horizon, len(KEEP_STATE_INDICES))
    full_norm = torch.zeros(batch, state_cfg.horizon, STATE_DIM, device=device)
    full_norm[:, :, KEEP_STATE_INDICES] = generated
    future_raw = state_normalizer.decode_state(full_norm.cpu().numpy())
    all_states = np.concatenate([current_states[:, None], future_raw], axis=1)
    transformed = idm_normalizer.encode_states(all_states)
    transformed[:, 1:, 2] = 0.0
    transformed[:, 1:, 3] = 0.0
    normalized_conditions = idm_normalizer.encode_conditions(conditions)
    inputs = np.concatenate(
        [transformed[:, 0], transformed[:, 1:].reshape(batch, -1), normalized_conditions],
        axis=1,
    ).astype(np.float32)
    prediction = idm(torch.from_numpy(inputs).to(device)).cpu().numpy()
    return idm_normalizer.decode_actions(prediction)


def append_transition(
    store: dict[str, list[torch.Tensor]],
    cond: torch.Tensor,
    episode_ids: torch.Tensor,
    decision_time: int,
    records: list[dict[str, torch.Tensor]],
) -> None:
    for record in records:
        store["x_t"].append(record["x_t"].detach().cpu())
        store["x_prev"].append(record["x_prev"].detach().cpu())
        store["t"].append(record["t"].detach().cpu())
        store["cond"].append(cond.detach().cpu())
        store["old_log_prob"].append(record["old_log_prob"].detach().cpu())
        store["episode_id"].append(episode_ids.detach().cpu())
        store["decision_time"].append(torch.full_like(episode_ids.detach().cpu(), decision_time))


@torch.inference_mode()
def predict_step_rewards(
    models: list[StepRewardModel],
    payload: dict,
    states: np.ndarray,
    actions: np.ndarray,
    conditions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if not models:
        return np.zeros(len(states), dtype=np.float32)
    transformed = transform_step_states(states)
    state_mean = np.asarray(payload["state_mean"], np.float32)
    state_std = np.asarray(payload["state_std"], np.float32)
    cond_mean = np.asarray(payload["condition_mean"], np.float32)
    cond_std = np.asarray(payload["condition_std"], np.float32)
    normalized = np.concatenate(
        [
            (transformed - state_mean) / state_std,
            ((np.log1p(np.maximum(actions, 0.0)) - payload["action_mean"]) / payload["action_std"])[:, None],
            (conditions - cond_mean) / cond_std,
        ],
        axis=1,
    ).astype(np.float32)
    inputs = torch.from_numpy(normalized).to(device)
    prediction = torch.stack([model(inputs) for model in models]).mean(0).cpu().numpy()
    transformed = prediction * payload["reward_std"] + payload["reward_mean"]
    if payload.get("target_transform") == "log1p_nonnegative":
        return np.expm1(np.clip(transformed, -20.0, 20.0)).clip(min=0.0).astype(np.float32)
    signed = transformed
    return (np.sign(signed) * np.expm1(np.abs(signed))).astype(np.float32)


def discounted_returns(step_values: np.ndarray, discount: float) -> np.ndarray:
    returns = np.zeros_like(step_values)
    running = np.zeros(len(step_values), dtype=np.float32)
    for time_index in range(step_values.shape[1] - 1, -1, -1):
        running = step_values[:, time_index] + discount * running
        returns[:, time_index] = running
    return returns


def normalize_group_values(
    values: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    advantages = np.zeros_like(values)
    for group_id in np.unique(group_ids):
        group = np.flatnonzero(group_ids == group_id)
        selected = values[group]
        axis = 0 if selected.ndim == 2 else None
        advantages[group] = (selected - selected.mean(axis=axis, keepdims=True)) / (
            selected.std(axis=axis, keepdims=True) + 1e-6
        )
    return advantages


def score_potential_increments(
    step_rewards: np.ndarray,
    step_costs: np.ndarray,
    cpa_constraints: np.ndarray,
) -> np.ndarray:
    """Decompose final CompetitionScore into exact per-step increments."""

    cumulative_rewards = np.cumsum(step_rewards, axis=1)
    cumulative_costs = np.cumsum(step_costs, axis=1)
    potentials = np.zeros_like(step_rewards)
    for episode in range(len(step_rewards)):
        for time_index in range(step_rewards.shape[1]):
            potentials[episode, time_index] = competition_score(
                float(cumulative_rewards[episode, time_index]),
                float(cumulative_costs[episode, time_index]),
                float(cpa_constraints[episode]),
            )
    return np.diff(potentials, axis=1, prepend=np.zeros((len(potentials), 1), np.float32))


def rollout(
    policy: DiffusionPolicy,
    templates: list[EpisodeTemplate],
    args: argparse.Namespace,
    group_size: int,
    state_cfg: StateConfig,
    state_normalizer: StateNormalizer,
    idm: ChunkActionMLP,
    idm_normalizer: IDMNormalizer,
    step_models: list[StepRewardModel],
    step_payload: dict | None,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    replays = [
        ReplayState.create(template, group_id)
        for group_id, template in enumerate(templates)
        for _ in range(group_size)
    ]
    transition_lists = {key: [] for key in ["x_t", "x_prev", "t", "cond", "old_log_prob", "episode_id", "decision_time"]}
    actual_step_rewards = np.zeros((len(replays), 48), dtype=np.float32)
    actual_step_costs = np.zeros((len(replays), 48), dtype=np.float32)
    predicted_step_rewards = np.zeros((len(replays), 48), dtype=np.float32)
    torch.manual_seed(seed)
    policy.eval()
    for time_index in range(48):
        states_now = []
        for replay in replays:
            tick = replay.template.ticks[time_index]
            state = build_state(
                time_index,
                replay.template.budget,
                replay.remaining_budget,
                tick.pvalues,
                replay.history_pvalue_info,
                replay.history_bids,
                replay.history_auction_result,
                replay.history_impression_result,
                replay.history_market_price,
            )
            replay.state_history.append(state)
            states_now.append(state)

        alphas = np.zeros(len(replays), dtype=np.float32)
        if time_index < state_cfg.history_length:
            alphas[:] = np.asarray([replay.template.cpa for replay in replays], np.float32)
        else:
            active = np.asarray(
                [index for index, replay in enumerate(replays) if replay.remaining_budget >= 0.1],
                dtype=np.int64,
            )
            if len(active):
                built = [
                    state_condition(replays[index], time_index, state_normalizer, state_cfg.history_length)
                    for index in active
                ]
                cond_np = np.stack([item[0] for item in built])
                current_np = np.stack([item[1] for item in built])
                condition_np = np.stack([item[2] for item in built])
                cond = torch.from_numpy(cond_np).to(device)
                generated, records = policy.sample(cond, record=True)
                alpha_chunks = decode_actions(
                    generated,
                    current_np,
                    condition_np,
                    state_cfg,
                    state_normalizer,
                    idm,
                    idm_normalizer,
                    device,
                )
                alphas[active] = alpha_chunks[:, 0]
                if step_payload is not None:
                    predicted_step_rewards[active, time_index] = predict_step_rewards(
                        step_models,
                        step_payload,
                        current_np,
                        alpha_chunks[:, 0],
                        condition_np,
                        device,
                    )
                append_transition(
                    transition_lists,
                    cond,
                    torch.from_numpy(active).to(device),
                    time_index,
                    records,
                )

        for index, replay in enumerate(replays):
            tick = replay.template.ticks[time_index]
            proposed = alphas[index] * tick.pvalues
            if replay.remaining_budget < 0.1:
                proposed = np.zeros_like(tick.pvalues)
            bids, status, costs = enforce_budget(
                proposed, tick.market_prices, replay.remaining_budget, tick.drop_priority
            )
            conversions = tick.potential_conversions * status
            tick_cost = float(costs.sum())
            tick_reward = float(np.sum(tick.pvalues * status))
            replay.remaining_budget = max(0.0, replay.remaining_budget - tick_cost)
            replay.total_cost += tick_cost
            replay.total_continuous_reward += tick_reward
            actual_step_rewards[index, time_index] = tick_reward
            actual_step_costs[index, time_index] = tick_cost
            replay.history_pvalue_info.append(np.stack([tick.pvalues, tick.pvalue_sigmas], axis=1))
            replay.history_bids.append(bids)
            replay.history_auction_result.append(
                np.stack([status, status, costs], axis=1).astype(np.float32)
            )
            replay.history_impression_result.append(
                np.stack([status, conversions], axis=1).astype(np.float32)
            )
            replay.history_market_price.append(tick.market_prices)

    rewards = np.asarray(
        [
            competition_score(
                replay.total_continuous_reward,
                replay.total_cost,
                replay.template.cpa,
            )
            for replay in replays
        ],
        dtype=np.float32,
    )
    group_ids = np.asarray([replay.group_id for replay in replays])
    if args.advantage_mode == "potential":
        increments = score_potential_increments(
            actual_step_rewards,
            actual_step_costs,
            np.asarray([replay.template.cpa for replay in replays], dtype=np.float32),
        )
        if not np.allclose(increments.sum(1), rewards, atol=1e-3):
            raise RuntimeError("Potential shaping does not preserve final CompetitionScore")
        returns_to_go = discounted_returns(increments, args.dense_discount)
        advantages = normalize_group_values(returns_to_go, group_ids)
    elif step_payload is not None:
        dense = (1.0 - args.step_rm_weight) * actual_step_rewards + args.step_rm_weight * predicted_step_rewards
        dense[:, -1] += rewards - actual_step_rewards.sum(axis=1)
        returns_to_go = discounted_returns(dense, args.dense_discount)
        advantages = normalize_group_values(returns_to_go, group_ids)
    else:
        advantages = normalize_group_values(rewards, group_ids)
    transitions = {
        key: torch.cat(values, dim=0) for key, values in transition_lists.items()
    }
    return rewards, advantages, transitions


def ppo_update(
    policy: DiffusionPolicy,
    reference: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: dict[str, torch.Tensor],
    advantages: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    ntp_contexts: np.ndarray | None = None,
    ntp_chunks: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    episode_advantages = torch.from_numpy(advantages)
    if episode_advantages.ndim == 2:
        transition_advantages = episode_advantages[
            transitions["episode_id"].long(), transitions["decision_time"].long()
        ]
    else:
        transition_advantages = episode_advantages[transitions["episode_id"].long()]
    dataset = TensorDataset(
        transitions["x_t"],
        transitions["x_prev"],
        transitions["t"],
        transitions["cond"],
        transitions["old_log_prob"],
        transition_advantages,
    )
    loader = DataLoader(dataset, batch_size=args.ppo_batch_size, shuffle=True)
    policy.train()
    policy_losses = []
    kl_values = []
    clip_values = []
    ntp_values = []
    for _ in range(args.ppo_epochs):
        for x_t, x_prev, timesteps, cond, old_log_prob, advantage in loader:
            x_t = x_t.to(device)
            x_prev = x_prev.to(device)
            timesteps = timesteps.to(device)
            cond = cond.to(device)
            old_log_prob = old_log_prob.to(device)
            advantage = advantage.to(device)
            mean, variance, current_epsilon = policy.model_stats(x_t, timesteps, cond)
            new_log_prob = gaussian_log_prob(x_prev, mean, variance).sum(-1)
            ratio = (new_log_prob - old_log_prob).clamp(-8, 8).exp()
            surrogate = torch.minimum(
                ratio * advantage,
                ratio.clamp(1 - args.ppo_clip, 1 + args.ppo_clip) * advantage,
            )
            policy_loss = -surrogate.mean()
            with torch.no_grad():
                reference_epsilon = reference(x_t, timesteps, cond)
            kl = (current_epsilon - reference_epsilon).square().mean()
            if ntp_contexts is not None and ntp_chunks is not None:
                ntp_indices = rng.choice(
                    len(ntp_contexts),
                    args.ntp_batch_size,
                    replace=len(ntp_contexts) < args.ntp_batch_size,
                )
                ntp_loss = policy.training_loss(
                    torch.from_numpy(ntp_chunks[ntp_indices]).to(device),
                    torch.from_numpy(ntp_contexts[ntp_indices]).to(device),
                )
            else:
                ntp_loss = torch.zeros((), device=device)
            loss = policy_loss + args.kl_coef * kl + args.ntp_weight * ntp_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.denoiser.parameters(), 1.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach()))
            kl_values.append(float(kl.detach()))
            ntp_values.append(float(ntp_loss.detach()))
            clip_values.append(
                float(((ratio < 1 - args.ppo_clip) | (ratio > 1 + args.ppo_clip)).float().mean())
            )
    policy.eval()
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "reference_mse": float(np.mean(kl_values)),
        "ntp_loss": float(np.mean(ntp_values)),
        "clip_fraction": float(np.mean(clip_values)),
        "transitions": len(dataset),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    policy, state_cfg, state_normalizer, idm, idm_normalizer, step_models, step_payload = load_models(args, device)
    reference = copy.deepcopy(policy.denoiser).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(policy.denoiser.parameters(), lr=args.learning_rate)
    train_periods = args.train_periods or [args.train_period]
    rng = np.random.default_rng(args.seed + 999)
    period_cycle = rng.permutation(train_periods).tolist()
    ntp_contexts = ntp_chunks = None
    if args.ntp_csv_path:
        ntp_contexts, ntp_chunks = load_ntp_samples(
            args.ntp_csv_path,
            args.ntp_periods or train_periods,
            state_cfg,
            state_normalizer,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for iteration in range(1, args.iterations + 1):
        if (iteration - 1) % len(period_cycle) == 0 and iteration > 1:
            period_cycle = rng.permutation(train_periods).tolist()
        train_period = int(period_cycle[(iteration - 1) % len(period_cycle)])
        templates = load_templates(
            Path(args.auctionnet_root),
            train_period,
            args.seed + train_period * 100,
        )
        count = min(args.advertisers_per_iteration, len(templates))
        selected = rng.choice(len(templates), count, replace=False)
        batch_templates = [templates[index] for index in selected]
        rewards, advantages, transitions = rollout(
            policy,
            batch_templates,
            args,
            args.group_size,
            state_cfg,
            state_normalizer,
            idm,
            idm_normalizer,
            step_models,
            step_payload,
            device,
            args.seed + iteration * 1000,
        )
        update = ppo_update(
            policy,
            reference,
            optimizer,
            transitions,
            advantages,
            args,
            device,
            ntp_contexts,
            ntp_chunks,
            rng,
        )
        row = {
            "iteration": iteration,
            "train_period": train_period,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            **update,
        }
        history.append(row)
        print("STATE_REPLAY_DDPO " + json.dumps(row), flush=True)
        torch.save(policy.state_dict(), output_dir / "state_diffusion.pt")
        torch.save(policy.state_dict(), output_dir / f"state_diffusion_iter_{iteration:03d}.pt")

    state_dir = Path(args.state_checkpoint_dir)
    shutil.copy2(state_dir / "config.json", output_dir / "config.json")
    shutil.copy2(state_dir / "normalization.json", output_dir / "normalization.json")
    payload = {"config": vars(args), "history": history}
    (output_dir / "ddpo_metrics.json").write_text(json.dumps(payload, indent=2))
    print("FINAL_STATE_REPLAY_DDPO " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
