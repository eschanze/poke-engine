#!/usr/bin/env python3
"""L2-strength x split-seed cross-validation: does any regularization level
beat the current weights on held-out starting states?

Usage: python sweep.py [--allow-legacy-schema] <dump.jsonl> [more.jsonl ...]
"""
import argparse

import numpy as np

from tune import (
    DEFAULT_WEIGHTS,
    bce_grad,
    bce_loss,
    load_positions,
    split_by_state_index,
)


def fit(x, y, w0, k, l2, lr=0.5, steps=3000):
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        g = bce_grad(w, x, y, k) + l2 * (w - w0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + eps)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="+", help="JSONL trajectory dump(s)")
    ap.add_argument("--allow-legacy-schema", action="store_true")
    args = ap.parse_args()
    k = 80.0
    x, y, si = load_positions(
        args.data,
        include_truncated=False,
        allow_legacy_schema=args.allow_legacy_schema,
    )

    print(f"\n{'l2':>8} | mean delta val BCE (tuned - baseline) over 5 seeds | per-seed")
    for l2 in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        deltas = []
        for seed in range(5):
            tr, va = split_by_state_index(si, 0.2, seed)
            w = fit(x[tr], y[tr], DEFAULT_WEIGHTS, k, l2)
            base = bce_loss(DEFAULT_WEIGHTS, x[va], y[va], k)
            tuned = bce_loss(w, x[va], y[va], k)
            deltas.append(tuned - base)
        print(f"{l2:>8} | {np.mean(deltas):+.5f} | " + " ".join(f"{d:+.4f}" for d in deltas))


if __name__ == "__main__":
    main()
