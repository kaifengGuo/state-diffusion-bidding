#!/usr/bin/env python3
"""Offline multi-step bid Diffusion + Ensemble RM + DDPO + Best-of-N."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


STATE_NAMES = [
    "time_left", "budget_left", "avg_bid_all", "avg_bid_last3",
    "avg_lwc_all", "avg_pvalue_all", "avg_conversion_all", "avg_win_all",
    "avg_lwc_last3", "avg_pvalue_last3", "avg_conversion_last3",
    "avg_win_last3", "current_pvalue", "current_pv_num", "last3_pv_num",
    "historical_pv_num",
]


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
    bc_epochs: int = 60
    rm_epochs: int = 60
    ensemble_size: int = 5
    learning_rate: float = 3e-4
    rm_learning_rate: float = 3e-4
    patience: int = 10
    ddpo_iterations: int = 300
    ddpo_batch_size: int = 256
    ppo_update_iters: int = 2
    ppo_clip: float = 0.1
    ddpo_learning_rate: float = 1e-5
    kl_coef: float = 0.05
    uncertainty_beta: float = 0.5
    support_penalty: float = 0.2
    jump_penalty: float = 0.05
    best_of_n: int = 16
    eval_batch_size: int = 128
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
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--rm-epochs", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ddpo-iterations", type=int, default=300)
    parser.add_argument("--ddpo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-update-iters", type=int, default=2)
    parser.add_argument("--ppo-clip", type=float, default=0.1)
    parser.add_argument("--ddpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--uncertainty-beta", type=float, default=0.5)
    parser.add_argument("--support-penalty", type=float, default=0.2)
    parser.add_argument("--jump-penalty", type=float, default=0.05)
    parser.add_argument("--best-of-n", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
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


def build_windows(df: pd.DataFrame, history_length: int, horizon: int, reward_column: str):
    required = {
        "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex",
        "budget", "CPAConstraint", "timeStepIndex", "state", "action", "done",
        reward_column,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    state_history, past_actions, past_rewards = [], [], []
    future_actions, future_returns, conditions, periods, metadata = [], [], [], [], []
    candidates = skipped_done = 0
    keys = ["deliveryPeriodIndex", "advertiserNumber"]
    for (period, advertiser), group in df.groupby(keys, sort=True):
        rows = group.sort_values("timeStepIndex").reset_index(drop=True)
        states = np.stack([parse_state(v) for v in rows["state"]])
        done = rows["done"].to_numpy()
        if pd.isna(done).any() or not np.isin(done, [0, 1, False, True]).all():
            raise ValueError(f"Invalid done values for {period}/{advertiser}")
        done = done.astype(bool)
        for index in range(history_length, len(rows) - horizon + 1):
            candidates += 1
            if done[index:index + horizon].any():
                skipped_done += 1
                continue
            history_rows = rows.iloc[index - history_length:index]
            future_rows = rows.iloc[index:index + horizon]
            # The state history ends at the current decision state s_t.
            state_history.append(states[index - history_length + 1:index + 1])
            past_actions.append(history_rows["action"].to_numpy(dtype=np.float32))
            past_rewards.append(history_rows[reward_column].to_numpy(dtype=np.float32))
            future_actions.append(future_rows["action"].to_numpy(dtype=np.float32))
            future_returns.append(float(future_rows[reward_column].sum()))
            row = rows.iloc[index]
            conditions.append([
                float(row["budget"]), float(row["CPAConstraint"]),
                float(row["advertiserCategoryIndex"]), float(row["timeStepIndex"]) / 47.0,
            ])
            periods.append(int(period))
            metadata.append([int(period), int(advertiser), int(row["timeStepIndex"])])
    arrays = {
        "states": np.asarray(state_history, dtype=np.float32),
        "past_actions": np.asarray(past_actions, dtype=np.float32),
        "past_rewards": np.asarray(past_rewards, dtype=np.float32),
        "future_actions": np.asarray(future_actions, dtype=np.float32),
        "future_returns": np.asarray(future_returns, dtype=np.float32),
        "conditions": np.asarray(conditions, dtype=np.float32),
        "periods": np.asarray(periods, dtype=np.int64),
        "metadata": np.asarray(metadata, dtype=np.int64),
    }
    stats = {"candidate_windows": candidates, "valid_windows": len(periods), "skipped_done": skipped_done}
    return arrays, stats


class Normalizer:
    def fit(self, arrays: dict[str, np.ndarray], train_mask: np.ndarray) -> "Normalizer":
        self.state_mean = arrays["states"][train_mask].reshape(-1, len(STATE_NAMES)).mean(0)
        self.state_std = np.maximum(arrays["states"][train_mask].reshape(-1, len(STATE_NAMES)).std(0), 1e-6)
        log_actions = np.log1p(np.concatenate([
            arrays["past_actions"][train_mask].reshape(-1),
            arrays["future_actions"][train_mask].reshape(-1),
        ]))
        self.action_mean = float(log_actions.mean())
        self.action_std = float(max(log_actions.std(), 1e-6))
        signed_reward = np.sign(arrays["past_rewards"]) * np.log1p(np.abs(arrays["past_rewards"]))
        self.reward_mean = float(signed_reward[train_mask].mean())
        self.reward_std = float(max(signed_reward[train_mask].std(), 1e-6))
        self.condition_mean = arrays["conditions"][train_mask].mean(0)
        self.condition_std = np.maximum(arrays["conditions"][train_mask].std(0), 1e-6)
        log_returns = np.log1p(np.maximum(arrays["future_returns"][train_mask], 0.0))
        self.return_mean = float(log_returns.mean())
        self.return_std = float(max(log_returns.std(), 1e-6))
        encoded_future = self.encode_actions(arrays["future_actions"][train_mask])
        self.action_clip = float(max(np.percentile(np.abs(encoded_future), 99.5), 2.5))
        return self

    def encode_actions(self, actions: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.maximum(actions, 0.0)) - self.action_mean) / self.action_std).astype(np.float32)

    def decode_actions(self, actions: np.ndarray) -> np.ndarray:
        return np.expm1(actions * self.action_std + self.action_mean).clip(min=0.0).astype(np.float32)

    def encode_returns(self, returns: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.maximum(returns, 0.0)) - self.return_mean) / self.return_std).astype(np.float32)

    def decode_returns(self, returns: np.ndarray) -> np.ndarray:
        return np.expm1(returns * self.return_std + self.return_mean).clip(min=0.0).astype(np.float32)

    def transform(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        states = (arrays["states"] - self.state_mean[None, None]) / self.state_std[None, None]
        past_actions = self.encode_actions(arrays["past_actions"])
        future_actions = self.encode_actions(arrays["future_actions"])
        signed_reward = np.sign(arrays["past_rewards"]) * np.log1p(np.abs(arrays["past_rewards"]))
        past_rewards = (signed_reward - self.reward_mean) / self.reward_std
        conditions = (arrays["conditions"] - self.condition_mean) / self.condition_std
        cond = np.concatenate([
            states.reshape(len(states), -1), past_actions, past_rewards, conditions,
        ], axis=1).astype(np.float32)
        return {
            **arrays,
            "cond": cond,
            "future_actions_norm": future_actions,
            "future_returns_norm": self.encode_returns(arrays["future_returns"]),
            "previous_action_norm": past_actions[:, -1],
        }

    def state_dict(self) -> dict:
        result = {}
        for key, value in vars(self).items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BidDenoiser(nn.Module):
    def __init__(self, cond_dim: int, bid_dim: int, hidden_dim: int):
        super().__init__()
        time_dim = 64
        self.time = nn.Sequential(SinusoidalEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.input = nn.Linear(cond_dim + bid_dim + time_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, bid_dim))

    def forward(self, noisy_bids: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.input(torch.cat([noisy_bids, cond, self.time(timesteps)], dim=-1))
        return self.output(self.blocks(hidden))


def cosine_beta_schedule(steps: int) -> torch.Tensor:
    offset = 0.008
    grid = torch.linspace(0, steps, steps + 1)
    alpha_bar = torch.cos(((grid / steps) + offset) / (1 + offset) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, timesteps).reshape(len(timesteps), *([1] * (len(shape) - 1)))


class DiffusionPolicy(nn.Module):
    def __init__(self, cond_dim: int, bid_dim: int, hidden_dim: int, steps: int):
        super().__init__()
        self.steps = steps
        self.bid_dim = bid_dim
        self.denoiser = BidDenoiser(cond_dim, bid_dim, hidden_dim)
        betas = cosine_beta_schedule(steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())
        self.register_buffer("posterior_variance", betas * (1 - alpha_bar_prev) / (1 - alpha_bar))
        self.register_buffer("posterior_mean_coef1", betas * alpha_bar_prev.sqrt() / (1 - alpha_bar))
        self.register_buffer("posterior_mean_coef2", (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar))

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean + extract(
            self.sqrt_one_minus_alpha_bar, timesteps, clean.shape
        ) * noise

    def model_stats(self, x_t: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor):
        epsilon = self.denoiser(x_t, timesteps, cond)
        alpha_bar = extract(self.alpha_bar, timesteps, x_t.shape)
        x0 = (x_t - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt().clamp(min=1e-5)
        x0 = x0.clamp(-6, 6)
        mean = extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0 + extract(
            self.posterior_mean_coef2, timesteps, x_t.shape
        ) * x_t
        variance = extract(self.posterior_variance, timesteps, x_t.shape).clamp(min=1e-8)
        return mean, variance, epsilon

    def training_loss(self, clean: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.add_noise(clean, noise, timesteps)
        return F.mse_loss(self.denoiser(noisy, timesteps, cond), noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, record: bool = False):
        x = torch.randn(len(cond), self.bid_dim, device=cond.device)
        records = []
        for step in range(self.steps - 1, -1, -1):
            t = torch.full((len(cond),), step, device=cond.device, dtype=torch.long)
            mean, variance, _ = self.model_stats(x, t, cond)
            x_prev = mean if step == 0 else mean + variance.sqrt() * torch.randn_like(x)
            if record and step > 0:
                old_log_prob = gaussian_log_prob(x_prev, mean, variance).sum(-1)
                records.append({
                    "x_t": x.detach(), "x_prev": x_prev.detach(), "t": t.detach(),
                    "old_log_prob": old_log_prob.detach(),
                })
            x = x_prev
        return x, records


def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((value - mean).square() / variance) + variance.log() + math.log(2 * math.pi))


class RewardModel(nn.Module):
    def __init__(self, cond_dim: int, bid_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cond_dim + bid_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cond: torch.Tensor, bids: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([cond, bids], dim=-1)).squeeze(-1)


def make_loader(cond, actions, returns=None, batch_size=512, shuffle=True, num_workers=0, seed=42):
    tensors = [torch.from_numpy(cond), torch.from_numpy(actions)]
    if returns is not None:
        tensors.append(torch.from_numpy(returns))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )


def train_diffusion(cfg: Config, data: dict, train_mask, val_mask, device):
    model = DiffusionPolicy(data["cond"].shape[1], cfg.horizon, cfg.hidden_dim, cfg.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    train_loader = make_loader(data["cond"][train_mask], data["future_actions_norm"][train_mask], batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed)
    val_loader = make_loader(data["cond"][val_mask], data["future_actions_norm"][val_mask], batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg.bc_epochs + 1):
        row = {"epoch": epoch}
        for name, loader, training in [("train", train_loader, True), ("val", val_loader, False)]:
            model.train(training)
            total = count = 0
            for cond, actions in loader:
                cond, actions = cond.to(device), actions.to(device)
                loss = model.training_loss(actions, cond)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.detach()) * len(cond)
                count += len(cond)
            row[f"{name}_noise_mse"] = total / count
        history.append(row)
        print("BC " + json.dumps(row), flush=True)
        if row["val_noise_mse"] < best:
            best, best_state, stale = row["val_noise_mse"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), history, best


def train_reward_ensemble(cfg: Config, data: dict, train_mask, val_mask, device):
    models, histories = [], []
    train_indices = np.flatnonzero(train_mask)
    val_cond = torch.from_numpy(data["cond"][val_mask]).to(device)
    val_actions = torch.from_numpy(data["future_actions_norm"][val_mask]).to(device)
    val_returns = torch.from_numpy(data["future_returns_norm"][val_mask]).to(device)
    for member in range(cfg.ensemble_size):
        seed_everything(cfg.seed + 100 + member)
        rng = np.random.default_rng(cfg.seed + 100 + member)
        bootstrap = rng.choice(train_indices, len(train_indices), replace=True)
        loader = make_loader(
            data["cond"][bootstrap], data["future_actions_norm"][bootstrap],
            data["future_returns_norm"][bootstrap], cfg.batch_size, True, cfg.num_workers,
            cfg.seed + member,
        )
        model = RewardModel(data["cond"].shape[1], cfg.horizon).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rm_learning_rate, weight_decay=1e-4)
        best, best_state, stale, member_history = float("inf"), None, 0, []
        for epoch in range(1, cfg.rm_epochs + 1):
            model.train()
            total = count = 0
            for cond, actions, returns in loader:
                cond, actions, returns = cond.to(device), actions.to(device), returns.to(device)
                loss = F.smooth_l1_loss(model(cond, actions), returns, beta=0.5)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(cond)
                count += len(cond)
            model.eval()
            with torch.no_grad():
                val_loss = float(F.mse_loss(model(val_cond, val_actions), val_returns))
            row = {"epoch": epoch, "train_huber": total / count, "val_mse": val_loss}
            member_history.append(row)
            if epoch == 1 or epoch % 10 == 0:
                print(f"RM{member} " + json.dumps(row), flush=True)
            if val_loss < best:
                best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        models.append(model)
        histories.append(member_history)
    return models, histories


def ensemble_predictions(models, cond: torch.Tensor, bids: torch.Tensor):
    predictions = torch.stack([model(cond, bids) for model in models], dim=0)
    return predictions.mean(0), predictions.std(0, unbiased=False)


def robust_scores(cfg: Config, models, cond, bids, previous_action):
    mean, uncertainty = ensemble_predictions(models, cond, bids)
    support = F.relu(bids.abs() - 3.0).mean(-1)
    jumps = torch.cat([bids[:, :1] - previous_action[:, None], bids[:, 1:] - bids[:, :-1]], dim=1).abs().mean(-1)
    robust = mean - cfg.uncertainty_beta * uncertainty - cfg.support_penalty * support - cfg.jump_penalty * jumps
    return robust, mean, uncertainty, support, jumps


def evaluate_reward_models(models, data, test_mask, normalizer, device, batch_size):
    cond_array = data["cond"][test_mask]
    bid_array = data["future_actions_norm"][test_mask]

    def predict(candidate_bids):
        means, stds = [], []
        for start in range(0, len(cond_array), batch_size):
            cond = torch.from_numpy(cond_array[start:start + batch_size]).to(device)
            bids = torch.from_numpy(candidate_bids[start:start + batch_size]).to(device)
            with torch.no_grad():
                batch_mean, batch_std = ensemble_predictions(models, cond, bids)
            means.append(batch_mean.cpu().numpy()); stds.append(batch_std.cpu().numpy())
        return np.concatenate(means), np.concatenate(stds)

    mean, std = predict(bid_array)
    target = data["future_returns_norm"][test_mask]
    residual = target - mean
    r2 = 1 - float((residual ** 2).sum() / max(((target - target.mean()) ** 2).sum(), 1e-8))
    rng = np.random.default_rng(20260805)
    shuffled_mean, _ = predict(bid_array[rng.permutation(len(bid_array))])
    shuffled_residual = target - shuffled_mean
    shuffled_r2 = 1 - float(
        (shuffled_residual ** 2).sum()
        / max(((target - target.mean()) ** 2).sum(), 1e-8)
    )
    raw_pred = normalizer.decode_returns(mean)
    raw_target = data["future_returns"][test_mask]
    return {
        "normalized_mae": float(np.abs(residual).mean()), "normalized_rmse": float(np.sqrt((residual ** 2).mean())),
        "r2": r2, "raw_return_mae": float(np.abs(raw_pred - raw_target).mean()),
        "ensemble_std_mean": float(std.mean()), "ensemble_std_p90": float(np.percentile(std, 90)),
        "abs_error_uncertainty_corr": float(np.corrcoef(np.abs(residual), std)[0, 1]),
        "shuffled_bid_normalized_mae": float(np.abs(shuffled_residual).mean()),
        "shuffled_bid_r2": shuffled_r2,
        "shuffled_bid_prediction_change": float(np.abs(shuffled_mean - mean).mean()),
    }


def train_ddpo(cfg: Config, base_policy: DiffusionPolicy, models, data, train_mask, device):
    policy = copy.deepcopy(base_policy).to(device).train()
    reference = copy.deepcopy(base_policy.denoiser).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(policy.denoiser.parameters(), lr=cfg.ddpo_learning_rate)
    rng = np.random.default_rng(cfg.seed + 999)
    train_indices = np.flatnonzero(train_mask)
    history = []
    for iteration in range(1, cfg.ddpo_iterations + 1):
        indices = rng.choice(train_indices, cfg.ddpo_batch_size, replace=len(train_indices) < cfg.ddpo_batch_size)
        cond = torch.from_numpy(data["cond"][indices]).to(device)
        previous = torch.from_numpy(data["previous_action_norm"][indices]).to(device)
        with torch.no_grad():
            bids, records = policy.sample(cond, record=True)
            reward, mean, uncertainty, support, jumps = robust_scores(cfg, models, cond, bids, previous)
            advantage = reward - reward.mean()
            advantage = advantage / (advantage.std() + 1e-6)
            old_log_probs = torch.stack([record["old_log_prob"] for record in records], dim=1)
        policy_losses, kl_values, clip_values = [], [], []
        for _ in range(cfg.ppo_update_iters):
            new_log_probs, kl_terms = [], []
            for record in records:
                new_mean, variance, current_epsilon = policy.model_stats(record["x_t"], record["t"], cond)
                new_log_probs.append(gaussian_log_prob(record["x_prev"], new_mean, variance).sum(-1))
                with torch.no_grad():
                    reference_epsilon = reference(record["x_t"], record["t"], cond)
                kl_terms.append((current_epsilon - reference_epsilon).square().mean(-1))
            new_log_probs = torch.stack(new_log_probs, dim=1)
            log_ratio = (new_log_probs - old_log_probs).clamp(-8, 8)
            ratio = log_ratio.exp()
            advantage_steps = advantage[:, None].expand_as(ratio)
            surrogate = torch.minimum(
                ratio * advantage_steps,
                ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * advantage_steps,
            )
            policy_loss = -surrogate.mean()
            kl = torch.stack(kl_terms, dim=1).mean()
            loss = policy_loss + cfg.kl_coef * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.denoiser.parameters(), 1.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach())); kl_values.append(float(kl.detach()))
            clip_values.append(float(((ratio < 1 - cfg.ppo_clip) | (ratio > 1 + cfg.ppo_clip)).float().mean()))
        row = {
            "iteration": iteration, "robust_reward": float(reward.mean()), "rm_mean": float(mean.mean()),
            "uncertainty": float(uncertainty.mean()), "support_penalty": float(support.mean()),
            "jump": float(jumps.mean()), "policy_loss": float(np.mean(policy_losses)),
            "reference_mse": float(np.mean(kl_values)), "clip_fraction": float(np.mean(clip_values)),
        }
        history.append(row)
        if iteration == 1 or iteration % 20 == 0:
            print("DDPO " + json.dumps(row), flush=True)
    return policy.eval(), history


@torch.no_grad()
def evaluate_policy(cfg, name, policy, models, data, test_mask, normalizer, device):
    cond_np = data["cond"][test_mask]
    previous_np = data["previous_action_norm"][test_mask]
    actual_norm = data["future_actions_norm"][test_mask]
    actual_raw = data["future_actions"][test_mask]
    collected = {key: [] for key in ["single", "selected", "selected_robust", "selected_rm", "selected_uncertainty", "selected_support", "selected_jump", "candidate_std", "oracle_min_mae"]}
    selected_raw_all = []
    for start in range(0, len(cond_np), cfg.eval_batch_size):
        cond = torch.from_numpy(cond_np[start:start + cfg.eval_batch_size]).to(device)
        previous = torch.from_numpy(previous_np[start:start + cfg.eval_batch_size]).to(device)
        actual = torch.from_numpy(actual_norm[start:start + cfg.eval_batch_size]).to(device)
        single, _ = policy.sample(cond)
        repeated_cond = cond.repeat_interleave(cfg.best_of_n, dim=0)
        repeated_previous = previous.repeat_interleave(cfg.best_of_n, dim=0)
        candidates, _ = policy.sample(repeated_cond)
        robust, rm_mean, uncertainty, support, jumps = robust_scores(
            cfg, models, repeated_cond, candidates, repeated_previous
        )
        batch = len(cond)
        candidates = candidates.reshape(batch, cfg.best_of_n, cfg.horizon)
        robust = robust.reshape(batch, cfg.best_of_n)
        rm_mean = rm_mean.reshape(batch, cfg.best_of_n)
        uncertainty = uncertainty.reshape(batch, cfg.best_of_n)
        support = support.reshape(batch, cfg.best_of_n)
        jumps = jumps.reshape(batch, cfg.best_of_n)
        best = robust.argmax(1)
        row = torch.arange(batch, device=device)
        selected = candidates[row, best]
        oracle_mae = (candidates - actual[:, None]).abs().mean(-1).min(1).values
        collected["single"].append(single.cpu().numpy())
        collected["selected"].append(selected.cpu().numpy())
        collected["selected_robust"].append(robust[row, best].cpu().numpy())
        collected["selected_rm"].append(rm_mean[row, best].cpu().numpy())
        collected["selected_uncertainty"].append(uncertainty[row, best].cpu().numpy())
        collected["selected_support"].append(support[row, best].cpu().numpy())
        collected["selected_jump"].append(jumps[row, best].cpu().numpy())
        collected["candidate_std"].append(candidates.std(1, unbiased=False).mean(-1).cpu().numpy())
        collected["oracle_min_mae"].append(oracle_mae.cpu().numpy())
        selected_raw_all.append(normalizer.decode_actions(selected.cpu().numpy()))
    values = {key: np.concatenate(items) for key, items in collected.items()}
    selected_raw = np.concatenate(selected_raw_all)
    single_raw = normalizer.decode_actions(values["single"])
    return {
        "name": name,
        "single_logged_bid_mae": float(np.abs(single_raw - actual_raw).mean()),
        "best_of_n_logged_bid_mae": float(np.abs(selected_raw - actual_raw).mean()),
        "oracle_candidate_normalized_mae": float(values["oracle_min_mae"].mean()),
        "selected_robust_score": float(values["selected_robust"].mean()),
        "selected_predicted_return_norm": float(values["selected_rm"].mean()),
        "selected_uncertainty": float(values["selected_uncertainty"].mean()),
        "selected_support_violation": float(values["selected_support"].mean()),
        "selected_jump": float(values["selected_jump"].mean()),
        "candidate_diversity": float(values["candidate_std"].mean()),
    }, {"metadata": data["metadata"][test_mask], "selected_bid": selected_raw, "actual_bid": actual_raw}


def evaluate_logged_baselines(cfg, models, data, test_mask, normalizer, device):
    cond = torch.from_numpy(data["cond"][test_mask]).to(device)
    actual = torch.from_numpy(data["future_actions_norm"][test_mask]).to(device)
    previous = torch.from_numpy(data["previous_action_norm"][test_mask]).to(device)
    copied = previous[:, None].expand(-1, cfg.horizon)
    result = {}
    for name, bids in [("logged_future", actual), ("copy_previous", copied)]:
        robust, mean, uncertainty, support, jumps = robust_scores(cfg, models, cond, bids, previous)
        result[name] = {
            "robust_score": float(robust.mean()), "predicted_return_norm": float(mean.mean()),
            "uncertainty": float(uncertainty.mean()), "support_violation": float(support.mean()),
            "jump": float(jumps.mean()),
        }
    copied_raw = normalizer.decode_actions(copied.cpu().numpy())
    result["copy_previous"]["logged_bid_mae"] = float(np.abs(copied_raw - data["future_actions"][test_mask]).mean())
    return result


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    dataframe = pd.read_csv(cfg.csv_path)
    arrays, data_stats = build_windows(dataframe, cfg.history_length, cfg.horizon, cfg.reward_column)
    periods = sorted(np.unique(arrays["periods"]).tolist())
    train_periods, val_periods, test_periods = periods[:-4], periods[-4:-2], periods[-2:]
    train_mask = np.isin(arrays["periods"], train_periods)
    val_mask = np.isin(arrays["periods"], val_periods)
    test_mask = np.isin(arrays["periods"], test_periods)
    normalizer = Normalizer().fit(arrays, train_mask)
    data = normalizer.transform(arrays)
    (output_dir / "normalization.json").write_text(json.dumps(normalizer.state_dict(), indent=2))
    split_stats = {
        "data": data_stats, "train": int(train_mask.sum()), "validation": int(val_mask.sum()),
        "test": int(test_mask.sum()), "periods": [train_periods, val_periods, test_periods],
        "condition_dim": int(data["cond"].shape[1]),
    }
    print("DATA " + json.dumps(split_stats), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_policy, bc_history, best_bc_val = train_diffusion(cfg, data, train_mask, val_mask, device)
    torch.save(base_policy.state_dict(), output_dir / "bid_diffusion_bc.pt")
    models, rm_histories = train_reward_ensemble(cfg, data, train_mask, val_mask, device)
    for index, model in enumerate(models):
        torch.save(model.state_dict(), output_dir / f"reward_model_{index}.pt")
    rm_metrics = evaluate_reward_models(models, data, test_mask, normalizer, device, cfg.batch_size)
    print("RM_METRICS " + json.dumps(rm_metrics), flush=True)
    ddpo_policy, ddpo_history = train_ddpo(cfg, base_policy, models, data, train_mask, device)
    torch.save(ddpo_policy.state_dict(), output_dir / "bid_diffusion_ddpo.pt")

    baseline_metrics = evaluate_logged_baselines(cfg, models, data, test_mask, normalizer, device)
    bc_metrics, bc_predictions = evaluate_policy(cfg, "bc", base_policy, models, data, test_mask, normalizer, device)
    ddpo_metrics, ddpo_predictions = evaluate_policy(cfg, "ddpo", ddpo_policy, models, data, test_mask, normalizer, device)
    metrics = {
        "split": split_stats, "best_bc_val_noise_mse": best_bc_val, "reward_model": rm_metrics,
        "logged_baselines": baseline_metrics, "bc_policy": bc_metrics, "ddpo_policy": ddpo_metrics,
        "offline_evaluation_warning": "Counterfactual candidate returns are RM predictions; logged-bid distance and support metrics are reported as safeguards.",
    }
    histories = {"bc": bc_history, "reward_models": rm_histories, "ddpo": ddpo_history}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2))
    np.savez_compressed(
        output_dir / "test_predictions.npz", metadata=bc_predictions["metadata"],
        actual_bid=bc_predictions["actual_bid"], bc_selected_bid=bc_predictions["selected_bid"],
        ddpo_selected_bid=ddpo_predictions["selected_bid"],
    )
    print("FINAL_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
