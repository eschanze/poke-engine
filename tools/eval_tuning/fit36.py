#!/usr/bin/env python3
"""Constrained 36-feature refit on merged re-extracted dumps.

Feature vectors are recomputed from the serialized states by the
eval_pair_features binary (current 36-feature engine schema), so dumps
written under older feature schemas remain usable. Rows are weighted so
every decisive game contributes equal total weight, and validation folds
are grouped by (globally remapped) starting-state index.

Usage:
  python fit36.py --pair suite=<traj.jsonl>:<pair.jsonl> [more pairs...] \
      --val-tag suite --out fitted.weights
"""

import argparse
import json
from pathlib import Path

import numpy as np

from constrained import HANDCRAFTED_WEIGHTS, LOWER, UPPER, BOUNDS
from tune import DEFAULT_WEIGHTS, FEATURE_NAMES, split_by_state_index

FILE_STRIDE = 10_000_000


def load_merged(pairs):
    """pairs: list of (tag, traj_path, pair_path). Returns arrays + tag mask."""
    feats, outcomes, states, games, tags = [], [], [], [], []
    for file_idx, (tag, traj_path, pair_path) in enumerate(pairs):
        with Path(traj_path).open("r", encoding="utf-8") as stream:
            recs = [json.loads(l) for l in stream if l.strip()]
        with Path(pair_path).open("r", encoding="utf-8") as stream:
            vecs = [json.loads(l) for l in stream if l.strip()]
        if len(recs) != len(vecs):
            raise SystemExit(f"{traj_path}: {len(recs)} rows vs {len(vecs)} pair rows")
        truncated = {r["game_id"] for r in recs if r["truncated"]}
        kept = 0
        for rec, vec in zip(recs, vecs):
            if rec["game_id"] in truncated:
                continue
            one = np.asarray(vec["side_one"], dtype=np.float64)
            two = np.asarray(vec["side_two"], dtype=np.float64)
            if len(one) != len(FEATURE_NAMES):
                raise SystemExit(f"{pair_path}: dim {len(one)} != {len(FEATURE_NAMES)}")
            feats.append(one - two)
            outcomes.append(float(rec["outcome"]))
            states.append(file_idx * FILE_STRIDE + rec["state_index"])
            games.append(file_idx * FILE_STRIDE + rec["game_id"])
            tags.append(tag)
            kept += 1
        print(f"{tag}: {traj_path} -> {kept} usable rows")
    feats = np.asarray(feats)
    outcomes = np.asarray(outcomes)
    states = np.asarray(states, dtype=np.int64)
    games = np.asarray(games, dtype=np.int64)
    lookup = {}
    game_ids = np.asarray([lookup.setdefault(g, len(lookup)) for g in games])
    counts = np.bincount(game_ids)
    weights = 1.0 / counts[game_ids]
    print(f"total: {len(feats)} rows, {len(lookup)} decisive games")
    return feats, outcomes, states, weights, np.asarray(tags)


def weighted_bce(margin, target, weight):
    z = np.clip(margin, -60, 60)
    loss = np.logaddexp(0.0, z) - target * z
    return float(np.sum(loss * weight) / np.sum(weight))


def fit(x, y, wt, k, l1, steps=3000, lr=0.5):
    """Adam + proximal L1 + bound projection, per-game-weighted BCE."""
    w = np.clip(HANDCRAFTED_WEIGHTS.copy(), LOWER, UPPER)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    wt_norm = wt / np.sum(wt)
    for step in range(1, steps + 1):
        z = np.clip(x @ w / k, -60, 60)
        p = 1.0 / (1.0 + np.exp(-z))
        g = x.T @ ((p - y) * wt_norm) / k
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + eps)
        if l1 > 0:
            w = np.sign(w) * np.maximum(np.abs(w) - lr * l1, 0.0)
        w = np.clip(w, LOWER, UPPER)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    metavar="TAG=TRAJ:PAIR", help="tag=trajectory.jsonl:pairfeatures.jsonl")
    ap.add_argument("--val-tag", default="suite",
                    help="tag whose rows are scored in validation reports")
    ap.add_argument("--k", type=float, default=80.0)
    ap.add_argument("--l1", type=float, default=0.3)
    ap.add_argument("--out")
    args = ap.parse_args()

    parsed = []
    for spec in args.pair:
        tag, rest = spec.split("=", 1)
        traj, pair = rest.split(":", 1)
        parsed.append((tag, traj, pair))
    x, y, states, wt, tags = load_merged(parsed)
    val_mask_tag = tags == args.val_tag

    print(f"\nGrouped 5-fold val BCE on '{args.val_tag}' rows "
          f"(handcrafted / adopted / refit)")
    deltas = []
    for seed in range(5):
        train, val = split_by_state_index(states, 0.2, seed)
        val &= val_mask_tag
        w = fit(x[train], y[train], wt[train], args.k, args.l1)
        scores = {}
        for name, vec in (("hand", HANDCRAFTED_WEIGHTS),
                          ("adopted", DEFAULT_WEIGHTS), ("refit", w)):
            scores[name] = weighted_bce(x[val] @ vec / args.k, y[val], wt[val])
        deltas.append(scores["refit"] - scores["adopted"])
        print(f"seed {seed}: {scores['hand']:.5f} / {scores['adopted']:.5f} / "
              f"{scores['refit']:.5f}  (refit-adopted {deltas[-1]:+.5f})")
    print(f"mean refit-adopted: {np.mean(deltas):+.5f}")

    if args.out:
        w = fit(x, y, wt, args.k, args.l1)
        with Path(args.out).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"# fit36: l1={args.l1} K={args.k} "
                         f"data={','.join(t for t, _, _ in parsed)}\n")
            for name, value in zip(FEATURE_NAMES, w):
                stream.write(f"{name} {value:.9g}\n")
        print(f"\nwrote {args.out}")
        print(f"{'feature':<30} {'adopted':>9} {'refit':>9}")
        for name, before, after in zip(FEATURE_NAMES, DEFAULT_WEIGHTS, w):
            marker = "  *" if abs(after - before) > 5 else ""
            print(f"{name:<30} {before:>9.2f} {after:>9.2f}{marker}")


if __name__ == "__main__":
    main()
