#!/usr/bin/env python3
"""Aggregate sweep summaries and compute advertiser-paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--compare",
        action="append",
        nargs=5,
        metavar=("LABEL", "LEFT_DIR", "LEFT_POLICY", "RIGHT_DIR", "RIGHT_POLICY"),
        default=[],
    )
    return parser.parse_args()


def config_for(result_dir: Path) -> dict:
    path = result_dir / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("config", {})


def serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def aggregate_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "results").glob("**/summary.csv")):
        config = config_for(path.parent)
        for summary in pd.read_csv(path).to_dict(orient="records"):
            row = {
                "experiment": str(path.parent.relative_to(root / "results")),
                "result_dir": str(path.parent),
                **summary,
            }
            row.update({f"config_{key}": serializable(value) for key, value in config.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_comparison(
    label: str,
    left_dir: Path,
    left_policy: str,
    right_dir: Path,
    right_policy: str,
    samples: int,
    seed: int,
) -> dict:
    keys = ["period", "advertiser"]
    metric = "continuous_score"
    left = pd.read_csv(left_dir / "episode_results.csv")
    right = pd.read_csv(right_dir / "episode_results.csv")
    left = left[left.policy == left_policy][keys + [metric]].rename(columns={metric: "left"})
    right = right[right.policy == right_policy][keys + [metric]].rename(columns={metric: "right"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError(f"comparison {label} has no paired episodes")
    delta = (paired.left - paired.right).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "label": label,
        "left_policy": left_policy,
        "right_policy": right_policy,
        "episodes": len(delta),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_win_rate": float((delta > 0).mean()),
        "paired_tie_rate": float((delta == 0).mean()),
    }


def ensemble_subset_summary(root: Path) -> pd.DataFrame:
    rows = []
    base = root / "results/ensemble_subsets"
    for path in sorted(base.glob("*/summary.csv")):
        config = config_for(path.parent)
        members = config.get("ensemble_members")
        if (
            members is None
            or config.get("periods") != [25]
            or config.get("seed") != 20260805
        ):
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        rows.append(
            {
                "members": "_".join(map(str, members)),
                "ensemble_size": len(members),
                **row,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    metrics = [
        "mean_continuous_score",
        "total_continuous_reward",
        "aggregate_continuous_cpa",
        "continuous_cpa_violation_rate",
        "mean_budget_utilization",
    ]
    return frame.groupby("ensemble_size")[metrics].agg(["mean", "std", "min", "max"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = aggregate_summaries(args.root)
    summaries.to_csv(args.output_dir / "all_summaries.csv", index=False)

    subset_summary = ensemble_subset_summary(args.root)
    if not subset_summary.empty:
        subset_summary.to_csv(args.output_dir / "ensemble_size_summary.csv")

    comparisons = []
    for offset, (label, left, left_policy, right, right_policy) in enumerate(args.compare):
        comparisons.append(
            bootstrap_comparison(
                label,
                Path(left),
                left_policy,
                Path(right),
                right_policy,
                args.bootstrap_samples,
                args.seed + offset,
            )
        )
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(f"summaries={len(summaries)} comparisons={len(comparisons)}")


if __name__ == "__main__":
    main()
