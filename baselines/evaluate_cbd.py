#!/usr/bin/env python3
"""Evaluate PlatformBid CBD with the deterministic AuctionNet replay protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbd-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--auctionnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--periods", type=int, nargs="+", default=[26, 27])
    parser.add_argument("--advertiser-start", type=int, default=0)
    parser.add_argument("--advertiser-limit", type=int)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--best-of-n", type=int, default=4)
    parser.add_argument("--rtg-value", type=float)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--state-clip", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_module(root: Path, name: str) -> Any:
    sys.path.insert(0, str(root.resolve()))
    return importlib.import_module(name)


class CBDStrategy:
    """Adapt the paper CBD strategy to the shared replay strategy contract."""

    def __init__(
        self,
        strategy_class: Any,
        checkpoint: Path,
        n_timesteps: int,
        best_of_n: int,
        rtg_value: float | None,
        action_scale: float,
        state_clip: bool,
        seed: int,
        stable_seed: Any,
    ) -> None:
        self.name = "cbd"
        self.seed = seed
        self.stable_seed = stable_seed
        self.inner = strategy_class(
            sparse_data=False,
            model_name=str(checkpoint),
            model_param={
                "n_timesteps": n_timesteps,
                "model_choice": "Unet",
                "attn_block": "vanilla",
                "state_dim": 9,
                "state_norm_mode": "minmax_m11",
                "state_clip": state_clip,
                "predict_epsilon": False,
                "cond_obs_training": True,
                "pred_one_step": False,
                "rtg_value": rtg_value,
                "action_output_mode": "raw",
                "action_scale": action_scale,
                "action_clip": False,
                "best_of_n": best_of_n,
                "best_of_n_mode": "state_gap",
                "inverse_context_len": 1,
            },
            selective_forward=True,
            advertiser_id=0,
            traj_add_a=False,
            use_RM=False,
            use_IDM=False,
        )
        self.budget = self.remaining_budget = 100.0
        self.cpa = 2.0
        self.category = 1
        self.period = 0
        self.advertiser = 0

    def reset(self) -> None:
        self.inner.budget = self.budget
        self.inner.remaining_budget = self.budget
        self.inner.cpa = self.cpa
        self.inner.category = self.category
        self.inner.advertiser_id = self.advertiser
        self.inner.cpa_condition = torch.clamp(
            (torch.tensor(self.cpa, dtype=torch.float32) - 6.0) / 6.0,
            min=0.0,
            max=1.0,
        )
        self.inner.reset()
        self.remaining_budget = self.budget

    def bidding(self, time_index: int, pvalues: np.ndarray, pvalue_sigmas: np.ndarray, *history):
        self.inner.remaining_budget = self.remaining_budget
        step_seed = self.stable_seed(
            self.seed, self.name, self.period, self.advertiser, time_index
        )
        np.random.seed(step_seed)
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)
        return self.inner.bidding(
            time_index, pvalues, pvalue_sigmas, *history
        )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    evaluator = load_module(args.evaluator_root, "evaluate_auctionnet_offline")
    strategy_module = load_module(
        args.cbd_root / "baselines/platformbid",
        "bidding_train_env.strategy.dd_bidding_strategy",
    )
    strategy = CBDStrategy(
        strategy_module.DdBiddingStrategy,
        args.checkpoint,
        args.n_timesteps,
        args.best_of_n,
        args.rtg_value,
        args.action_scale,
        args.state_clip,
        args.seed,
        evaluator.stable_seed,
    )

    data_dir = args.data_dir or args.auctionnet_root / "strategy_train_env/data/traffic"
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    rows = []
    started = time.time()
    for period in args.periods:
        frame = pd.read_csv(
            data_dir / f"period-{period}.csv",
            usecols=usecols,
            dtype={name: np.float32 for name in usecols},
        )
        for offset, (advertiser_value, group) in enumerate(
            frame.groupby("advertiserNumber", sort=True)
        ):
            if offset < args.advertiser_start:
                continue
            if (
                args.advertiser_limit is not None
                and offset >= args.advertiser_start + args.advertiser_limit
            ):
                break
            advertiser = int(advertiser_value)
            first = group.iloc[0]
            strategy.period = period
            strategy.advertiser = advertiser
            ticks = evaluator.build_ticks(group, args.seed, period, advertiser)
            result = evaluator.evaluate_episode(
                "cbd",
                strategy,
                ticks,
                period,
                advertiser,
                float(first["budget"]),
                float(first["CPAConstraint"]),
                int(first["advertiserCategoryIndex"]),
            )
            rows.append(result)
            print("EPISODE " + json.dumps(result, sort_keys=True), flush=True)
            pd.DataFrame(rows).to_csv(
                args.output_dir / "episode_results.partial.csv", index=False
            )

    results = pd.DataFrame(rows)
    summary = evaluator.summarize(results)
    results.to_csv(args.output_dir / "episode_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time.time() - started,
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("FINAL_SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
