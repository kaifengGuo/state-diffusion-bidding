#!/usr/bin/env python3
"""Mix current-policy and replay-policy counterfactual Episode-Q datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = ["features", "rewards", "costs", "scores", "periods", "groups"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-files", nargs="+", required=True)
    parser.add_argument("--recent-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mix-periods", type=int, nargs="+", required=True)
    parser.add_argument("--recent-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_parts(paths: list[str], default_version: float) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        with np.load(path) as payload:
            part = {key: payload[key] for key in ARRAY_KEYS}
            part["policy_versions"] = (
                payload["policy_versions"]
                if "policy_versions" in payload.files
                else np.full(len(part["features"]), default_version, dtype=np.float32)
            )
            parts.append(part)
    if not parts:
        raise ValueError("At least one dataset file is required")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def select_rows(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def concatenate_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    result["groups"] = np.arange(len(result["features"]), dtype=np.int64)
    return result


def main() -> None:
    args = parse_args()
    if not 0.5 < args.recent_fraction <= 1.0:
        raise ValueError("recent-fraction must be in (0.5, 1.0]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_parts(args.base_files, default_version=0.0)
    recent = load_parts(args.recent_files, default_version=1.0)
    periods = sorted(set(base["periods"].tolist()) | set(recent["periods"].tolist()))
    mix_periods = set(args.mix_periods)
    rng = np.random.default_rng(args.seed)
    summary = []

    for period in periods:
        recent_indices = np.flatnonzero(recent["periods"] == period)
        if not len(recent_indices):
            raise ValueError(f"Recent-policy data is missing period {period}")
        selected = [select_rows(recent, recent_indices)]
        base_count = 0
        if period in mix_periods and args.recent_fraction < 1.0:
            base_indices = np.flatnonzero(base["periods"] == period)
            requested = int(
                round(len(recent_indices) * (1.0 - args.recent_fraction) / args.recent_fraction)
            )
            if requested > len(base_indices):
                raise ValueError(
                    f"Period {period} needs {requested} base groups, only {len(base_indices)} available"
                )
            chosen = rng.choice(base_indices, requested, replace=False)
            selected.append(select_rows(base, chosen))
            base_count = requested
        mixed = concatenate_parts(selected)
        destination = output_dir / f"policy_snapshot_mixture_period_{period}.npz"
        np.savez_compressed(destination, **mixed)
        row = {
            "period": int(period),
            "recent_groups": int(len(recent_indices)),
            "base_groups": int(base_count),
            "total_groups": int(len(mixed["features"])),
            "realized_recent_fraction": float(
                len(recent_indices) / len(mixed["features"])
            ),
            "path": str(destination),
        }
        summary.append(row)
        print("POLICY_SNAPSHOT_MIX " + json.dumps(row), flush=True)

    payload = {
        "config": vars(args),
        "periods": summary,
        "total_groups": int(sum(row["total_groups"] for row in summary)),
    }
    (output_dir / "mixture_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
