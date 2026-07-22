#!/usr/bin/env python3
"""Screen experimental per-side features beyond a fold-local linear base.

The paired JSONL must be produced by `eval-pair-features
--experimental-candidates`. Validation is grouped by starting matchup and
rows are weighted so every decisive game contributes equally.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fit36 import FILE_STRIDE, fit, weighted_bce
from tune import FEATURE_NAMES, split_by_state_index


CANDIDATE_NAMES = [
    "ENTRY_HP_LOSS",
    "ENTRY_KO_COUNT",
    "BEST_COVERAGE",
    "SECOND_COVERAGE",
    "WINCON_VITALITY",
    "UNIQUE_ANSWER_FRAGILITY",
    "SAFE_SWITCH_COUNT",
    "TOXIC_PRESSURE",
    "SUBSTITUTE_HP",
    "WISH_RECOVERY",
    "ACTIVE_HP",
    "BENCH_HP",
    "HP_SQUARED_SUM",
    "LOW_HP_COUNT",
    "HEALTHY_COUNT",
]

# BENCH_HP and WISH_RECOVERY were appended to the stable schema at zero weight
# so their candidate can be loaded by selfplay. Keep them out of the fold-local
# base here; otherwise screening their duplicate candidate columns would test
# a residual after the feature had already been fitted.
STABLE_BASE_FEATURES = 36


def load(pairs):
    features, candidates, outcomes, states, games = [], [], [], [], []
    for file_index, (trajectory_path, pair_path) in enumerate(pairs):
        with Path(trajectory_path).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        with Path(pair_path).open(encoding="utf-8") as stream:
            vectors = [json.loads(line) for line in stream if line.strip()]
        if len(records) != len(vectors):
            raise SystemExit(
                f"{trajectory_path}: {len(records)} rows vs "
                f"{len(vectors)} candidate rows"
            )
        truncated_games = {
            record["game_id"] for record in records if record["truncated"]
        }
        for record, vector in zip(records, vectors):
            if record["game_id"] in truncated_games:
                continue
            side_one = np.asarray(vector["side_one"], dtype=np.float64)
            side_two = np.asarray(vector["side_two"], dtype=np.float64)
            candidate_one = np.asarray(vector["candidate_one"], dtype=np.float64)
            candidate_two = np.asarray(vector["candidate_two"], dtype=np.float64)
            if len(side_one) != len(FEATURE_NAMES):
                raise SystemExit(
                    f"{pair_path}: {len(side_one)} base features, "
                    f"expected {len(FEATURE_NAMES)}; re-extract it"
                )
            if len(candidate_one) != len(CANDIDATE_NAMES):
                raise SystemExit(
                    f"{pair_path}: {len(candidate_one)} candidates, "
                    f"expected {len(CANDIDATE_NAMES)}; re-extract it"
                )
            features.append(side_one - side_two)
            candidates.append(candidate_one - candidate_two)
            outcomes.append(float(record["outcome"]))
            states.append(file_index * FILE_STRIDE + record["state_index"])
            games.append(file_index * FILE_STRIDE + record["game_id"])

    games = np.asarray(games, dtype=np.int64)
    _, game_ids = np.unique(games, return_inverse=True)
    counts = np.bincount(game_ids)
    weights = 1.0 / counts[game_ids]
    return (
        np.asarray(features),
        np.asarray(candidates),
        np.asarray(outcomes),
        np.asarray(states, dtype=np.int64),
        weights,
        len(counts),
    )


def fit_residual(base_margin, values, target, weights, l2):
    if values.ndim == 1:
        values = values[:, None]
    means = np.average(values, axis=0, weights=weights)
    scales = np.sqrt(np.average((values - means) ** 2, axis=0, weights=weights))
    scales = np.maximum(scales, 1e-9)
    normalized = values / scales
    coefficients = np.zeros(values.shape[1])
    normalized_weights = weights / weights.sum()
    for _ in range(100):
        margin = np.clip(base_margin + normalized @ coefficients, -40, 40)
        probability = 1.0 / (1.0 + np.exp(-margin))
        curvature = normalized_weights * probability * (1.0 - probability)
        gradient = (
            normalized.T @ ((probability - target) * normalized_weights)
            + l2 * coefficients
        )
        hessian = (
            (normalized.T * curvature) @ normalized
            + l2 * np.eye(values.shape[1])
        )
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return coefficients / scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="TRAJECTORY=CANDIDATE_PAIR",
    )
    parser.add_argument("--l1", type=float, default=0.3)
    parser.add_argument("--residual-l2", type=float, default=0.02)
    parser.add_argument(
        "--joint",
        nargs="+",
        default=["BENCH_HP", "WISH_RECOVERY"],
        choices=CANDIDATE_NAMES,
    )
    args = parser.parse_args()

    pairs = [spec.split("=", 1) for spec in args.pair]
    x, candidates, y, states, weights, games = load(pairs)
    base_x = x.copy()
    base_x[:, STABLE_BASE_FEATURES:] = 0.0
    print(f"loaded {len(y)} positions from {games} decisive games")

    deltas = np.zeros((5, len(CANDIDATE_NAMES)))
    coefficients = np.zeros_like(deltas)
    joint_indices = [CANDIDATE_NAMES.index(name) for name in args.joint]
    joint_deltas, joint_coefficients = [], []
    for seed in range(5):
        train, validation = split_by_state_index(states, 0.2, seed)
        base_weights = fit(base_x[train], y[train], weights[train], 80.0, args.l1)
        train_margin = base_x[train] @ base_weights / 80.0
        validation_margin = base_x[validation] @ base_weights / 80.0
        baseline = weighted_bce(
            validation_margin, y[validation], weights[validation]
        )
        for feature in range(len(CANDIDATE_NAMES)):
            coefficient = fit_residual(
                train_margin,
                candidates[train, feature],
                y[train],
                weights[train],
                args.residual_l2,
            )[0]
            coefficients[seed, feature] = coefficient
            deltas[seed, feature] = weighted_bce(
                validation_margin + coefficient * candidates[validation, feature],
                y[validation],
                weights[validation],
            ) - baseline
        joint = fit_residual(
            train_margin,
            candidates[train][:, joint_indices],
            y[train],
            weights[train],
            args.residual_l2,
        )
        joint_coefficients.append(joint)
        joint_deltas.append(
            weighted_bce(
                validation_margin + candidates[validation][:, joint_indices] @ joint,
                y[validation],
                weights[validation],
            )
            - baseline
        )

    print("\nindividual residuals (negative held-out BCE delta is better)")
    for feature in np.argsort(deltas.mean(axis=0)):
        fold_text = " ".join(f"{value:+.5f}" for value in deltas[:, feature])
        mean_eval_weight = 80.0 * coefficients[:, feature].mean()
        print(
            f"{CANDIDATE_NAMES[feature]:28s} {deltas[:, feature].mean():+.6f}  "
            f"w={mean_eval_weight:+.2f}  {fold_text}"
        )
    print(
        f"\njoint {','.join(args.joint)}: {np.mean(joint_deltas):+.6f}  "
        f"eval weights="
        + ",".join(
            f"{value:+.2f}" for value in 80.0 * np.mean(joint_coefficients, axis=0)
        )
        + "  folds="
        + " ".join(f"{value:+.5f}" for value in joint_deltas)
    )


if __name__ == "__main__":
    main()
