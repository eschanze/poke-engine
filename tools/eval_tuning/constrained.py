#!/usr/bin/env python3
"""Fit semantically constrained linear eval weights with grouped validation."""

import argparse
from pathlib import Path

import numpy as np

from context_mlp import load_data, weighted_bce
from sweep import fit as fit_unconstrained
from tune import FEATURE_NAMES, bce_grad, split_by_state_index


def load_handcrafted_weights():
    """Load the fixed pre-tuning baseline used to reproduce this experiment."""
    path = Path(__file__).resolve().parents[2] / "data" / "weights" / "eval-handcrafted-36.weights"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, value = line.split()
            values[name] = float(value)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)


HANDCRAFTED_WEIGHTS = load_handcrafted_weights()


BOUNDS = {
    "POKEMON_ALIVE": (10, 120),
    "POKEMON_HP": (10, 160),
    "POKEMON_ITEM": (0, 30),
    "POKEMON_FROZEN": (-120, 0),
    "POKEMON_ASLEEP": (-80, 0),
    "POKEMON_PARALYZED": (-80, 0),
    "POKEMON_TOXIC": (-100, 0),
    "POKEMON_POISONED": (-60, 0),
    "POKEMON_BURNED": (-60, 0),
    "POISON_HEAL": (0, 80),
    "STATUS_ABILITY_BONUS": (0, 80),
    "POKEMON_ATTACK_BOOST": (0, 50),
    "POKEMON_DEFENSE_BOOST": (0, 40),
    "POKEMON_SPECIAL_ATTACK_BOOST": (0, 50),
    "POKEMON_SPECIAL_DEFENSE_BOOST": (0, 40),
    "POKEMON_SPEED_BOOST": (0, 50),
    "LEECH_SEED": (-60, 0),
    "SUBSTITUTE": (0, 120),
    "CONFUSION": (-50, 0),
    "REFLECT": (0, 40),
    "LIGHT_SCREEN": (0, 40),
    "AURORA_VEIL": (0, 60),
    "SAFE_GUARD": (0, 15),
    "TAILWIND": (0, 30),
    "HEALING_WISH": (0, 60),
    "STEALTH_ROCK": (-30, 0),
    "SPIKES": (-30, 0),
    "TOXIC_SPIKES": (-30, 0),
    "STICKY_WEB": (-40, 0),
    "USED_TERA": (-100, 0),
    "REVENGE_COVERAGE": (0, 80),
    "THREAT_BREADTH": (0, 100),
    "WINCON": (0, 120),
    "UNANSWERED": (0, 100),
    "ACTIVE_DUEL": (0, 80),
    "PIVOT_PRESSURE": (0, 80),
    "BENCH_HP": (0, 40),
    "WISH_RECOVERY": (0, 100),
}

LOWER = np.asarray([BOUNDS[name][0] for name in FEATURE_NAMES], dtype=np.float64)
UPPER = np.asarray([BOUNDS[name][1] for name in FEATURE_NAMES], dtype=np.float64)


def fit(x, y, k, l1, steps=3000, lr=0.5):
    weights = np.clip(HANDCRAFTED_WEIGHTS.copy(), LOWER, UPPER)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        gradient = bce_grad(weights, x, y, k)
        first = beta1 * first + (1 - beta1) * gradient
        second = beta2 * second + (1 - beta2) * gradient * gradient
        first_hat = first / (1 - beta1**step)
        second_hat = second / (1 - beta2**step)
        weights -= lr * first_hat / (np.sqrt(second_hat) + epsilon)
        if l1 > 0:
            weights = np.sign(weights) * np.maximum(np.abs(weights) - lr * l1, 0.0)
        weights = np.clip(weights, LOWER, UPPER)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=float, default=80.0)
    parser.add_argument("--l1", type=float, default=0.3)
    args = parser.parse_args()

    side_one, side_two, outcomes, state_indices, row_weights = load_data(
        args.data, args.pair_data
    )
    features = (side_one - side_two).astype(np.float64)
    print("\nGrouped five-fold BCE delta from handcrafted baseline")
    constrained_deltas = []
    unconstrained_deltas = []
    for seed in range(5):
        train, validation = split_by_state_index(state_indices, 0.2, seed)
        constrained = fit(features[train], outcomes[train], args.k, args.l1)
        unconstrained = fit_unconstrained(
            features[train], outcomes[train], HANDCRAFTED_WEIGHTS,
            args.k, 0.0, l1=args.l1,
        )
        baseline_margin = features[validation] @ HANDCRAFTED_WEIGHTS / args.k
        constrained_margin = features[validation] @ constrained / args.k
        unconstrained_margin = features[validation] @ unconstrained / args.k
        baseline = weighted_bce(
            baseline_margin, outcomes[validation], row_weights[validation]
        )
        constrained_delta = weighted_bce(
            constrained_margin, outcomes[validation], row_weights[validation]
        ) - baseline
        unconstrained_delta = weighted_bce(
            unconstrained_margin, outcomes[validation], row_weights[validation]
        ) - baseline
        constrained_deltas.append(constrained_delta)
        unconstrained_deltas.append(unconstrained_delta)
        print(
            f"seed {seed}: constrained {constrained_delta:+.5f}  "
            f"unconstrained {unconstrained_delta:+.5f}"
        )
    print(
        f"mean: constrained {np.mean(constrained_deltas):+.5f}  "
        f"unconstrained {np.mean(unconstrained_deltas):+.5f}"
    )

    fitted = fit(features, outcomes, args.k, args.l1)
    with Path(args.out).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# constrained L1 fit: l1={args.l1} K={args.k}\n")
        for name, value in zip(FEATURE_NAMES, fitted):
            stream.write(f"{name} {value:.9g}\n")
    print(f"\nwrote {args.out}")
    print(f"{'feature':<30} {'default':>9} {'fitted':>9} {'bound':>17}")
    for name, before, after in zip(FEATURE_NAMES, HANDCRAFTED_WEIGHTS, fitted):
        print(f"{name:<30} {before:>9.2f} {after:>9.2f} {str(BOUNDS[name]):>17}")


if __name__ == "__main__":
    main()
