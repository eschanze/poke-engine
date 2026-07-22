#!/usr/bin/env python3
"""Does distilling teacher root values into the LINEAR eval weights help?

Fits the 36 linear weights two ways on the same grouped folds — against the
noisy game outcome, and against the teacher's search root value — and scores
both arms on the SAME held-out target: the real game outcome. Training-target
BCE is not comparable across arms (soft root-value labels have lower variance),
so only outcome-scored held-out BCE decides the winner. A retuned linear vector
has zero inference cost, so any held-out gain here is pure profit — no
throughput tax to overcome, unlike the tree/MLP residuals.

Input is a relabel-root-value output file carrying both "outcome" and
"root_value" per position. Reuses the constrained/unconstrained fitters and the
grouped-by-starting-state splits that produced the adopted eval.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from constrained import HANDCRAFTED_WEIGHTS, fit as fit_constrained
from context_mlp import weighted_bce
from sweep import fit as fit_unconstrained
from tune import DEFAULT_WEIGHTS, FEATURE_NAMES, FEATURE_SCHEMA, split_by_state_index


def load(path, include_truncated):
    """Return per-position difference features, outcome, root_value, starting
    state index, inverse-game-count row weight, and a truncated-game mask."""
    features, outcomes, root_values, states, game_ids, truncated = [], [], [], [], [], []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("feature_schema") != FEATURE_SCHEMA:
                sys.exit(f"{path}:{line_no}: feature schema mismatch; regenerate the dump")
            if "root_value" not in rec:
                sys.exit(f"{path}:{line_no}: no root_value; run relabel-root-value first")
            if rec["truncated"] and not include_truncated:
                continue
            features.append(rec["features"])
            outcomes.append(rec["outcome"])
            root_values.append(rec["root_value"])
            states.append(rec["state_index"])
            game_ids.append(rec["game_id"])
            truncated.append(bool(rec["truncated"]))
    if not features:
        sys.exit("no usable positions loaded")
    game_ids = np.asarray(game_ids, dtype=np.int64)
    _, inverse = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(inverse)
    row_weights = (1.0 / counts[inverse]).astype(np.float64)
    truncated = np.asarray(truncated, dtype=bool)
    print(
        f"loaded {len(features)} positions from {len(counts)} games "
        f"({int(truncated.sum())} truncated positions "
        f"{'included' if include_truncated else 'dropped'})"
    )
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(root_values, dtype=np.float64),
        np.asarray(states, dtype=np.int64),
        row_weights,
        truncated,
    )


def fit_arm(x, target, k, l1, constrained):
    if constrained:
        return fit_constrained(x, target, k, l1)
    return fit_unconstrained(x, target, DEFAULT_WEIGHTS, k, 0.0, l1=l1)


def fit_scale(margin, outcome, weight):
    """Single global temperature s>0 minimizing outcome-BCE of sigmoid(s*margin).

    root_value labels are compressed toward 0.5, so weights fit to them are
    under-confident for sharp 0/1 outcomes; this nuisance scalar separates
    'did the weight DIRECTION improve' from 'the target's range differs'. 1-D
    convex in s, solved with a few damped Newton steps from s=1."""
    s = 1.0
    for _ in range(50):
        z = s * margin
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        g = np.sum(weight * margin * (p - outcome))
        h = np.sum(weight * margin * margin * p * (1.0 - p))
        if h <= 1e-12:
            break
        step = g / h
        s -= np.clip(step, -0.5, 0.5)
        s = max(s, 1e-6)
        if abs(step) < 1e-9:
            break
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="relabel-root-value JSONL (outcome + root_value)")
    ap.add_argument("--k", type=float, default=80.0)
    ap.add_argument("--l1", type=float, default=0.3)
    ap.add_argument("--out", help="write the root_value-distilled constrained weights here")
    ap.add_argument(
        "--augment-truncated",
        action="store_true",
        help="add truncated-game positions (root_value labels) to each training fold; "
        "held-out scoring still uses only decisive-game outcomes",
    )
    args = ap.parse_args()

    x, outcome, root_value, states, row_w, truncated = load(args.data, args.augment_truncated)
    decisive = ~truncated

    # Fixed references (no fitting), scored on outcomes over the whole decisive set.
    hand_margin = x[decisive] @ HANDCRAFTED_WEIGHTS / args.k
    adopted_margin = x[decisive] @ DEFAULT_WEIGHTS / args.k
    hand_bce = weighted_bce(hand_margin, outcome[decisive], row_w[decisive])
    adopted_bce = weighted_bce(adopted_margin, outcome[decisive], row_w[decisive])
    print(f"\nfixed references (whole decisive set, outcome-scored):")
    print(f"  handcrafted baseline      BCE {hand_bce:.5f}  (delta 0 by definition)")
    print(f"  adopted DEFAULT_WEIGHTS   BCE {adopted_bce:.5f}  delta {adopted_bce - hand_bce:+.5f}")

    arms = [
        ("outcome  / constrained  ", "outcome", True),
        ("rootval  / constrained  ", "root_value", True),
        ("outcome  / unconstrained", "outcome", False),
        ("rootval  / unconstrained", "root_value", False),
    ]
    targets = {"outcome": outcome, "root_value": root_value}

    print(
        f"\nGrouped five-fold held-out BCE delta vs handcrafted "
        f"(scored on OUTCOMES; negative is better)"
    )
    print(f"{'arm':<26} {'mean':>9}  {'scale':>5}   per-fold")
    results = {}
    for label, target_name, constrained in arms:
        deltas = []
        scales = []
        for seed in range(5):
            train_dec, val_dec = split_by_state_index(states[decisive], 0.2, seed)
            # Map decisive-only masks back into full-array index space.
            dec_idx = np.flatnonzero(decisive)
            train_idx = dec_idx[train_dec]
            val_idx = dec_idx[val_dec]
            if args.augment_truncated:
                # Truncated positions never enter validation; only augment training.
                trunc_idx = np.flatnonzero(truncated)
                fit_idx = np.concatenate([train_idx, trunc_idx])
            else:
                fit_idx = train_idx
            weights = fit_arm(x[fit_idx], targets[target_name][fit_idx], args.k, args.l1, constrained)
            # Recalibrate one global temperature on TRAINING outcomes (decisive
            # rows only), applied symmetrically to every arm; ~1 for outcome
            # arms, corrects the compressed range of the root_value arms.
            train_margin = x[train_idx] @ weights / args.k
            scale = fit_scale(train_margin, outcome[train_idx], row_w[train_idx])
            scales.append(scale)
            base = x[val_idx] @ HANDCRAFTED_WEIGHTS / args.k
            cand = scale * (x[val_idx] @ weights) / args.k
            delta = weighted_bce(cand, outcome[val_idx], row_w[val_idx]) - weighted_bce(
                base, outcome[val_idx], row_w[val_idx]
            )
            deltas.append(delta)
        results[label] = deltas
        print(
            f"{label:<26} {np.mean(deltas):>+9.5f}  {np.mean(scales):>5.2f}   "
            + " ".join(f"{d:+.4f}" for d in deltas)
        )

    if args.out:
        weights = fit_arm(x[decisive], root_value[decisive], args.k, args.l1, True)
        full_margin = x[decisive] @ weights / args.k
        scale = fit_scale(full_margin, outcome[decisive], row_w[decisive])
        weights = weights * scale  # bake temperature in so K stays 80 at load
        with Path(args.out).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"# root_value-distilled constrained L1 fit: l1={args.l1} K={args.k} "
                f"outcome_scale={scale:.4f}\n"
            )
            stream.write(f"# data: {args.data}\n")
            for name, value in zip(FEATURE_NAMES, weights):
                stream.write(f"{name} {value:.9g}\n")
        print(f"\nwrote {args.out} (outcome recalibration scale {scale:.3f})")


if __name__ == "__main__":
    main()
