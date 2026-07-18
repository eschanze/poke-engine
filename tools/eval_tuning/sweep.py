#!/usr/bin/env python3
"""Regularization-strength x split-seed cross-validation: does any
regularization level beat the current weights on held-out starting states?

--mode l2 (default) sweeps the L2 pull toward the current weights.
--mode l1 sweeps the L1 pull toward zero (feature elimination), optionally
on top of a fixed --l2; also reports how many weights die to exactly 0.

Usage: python sweep.py [--mode l1] [--l2 X] [--allow-legacy-schema] <dump.jsonl> ...
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


def fit(x, y, w0, k, l2, l1=0.0, lr=0.5, steps=3000):
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        g = bce_grad(w, x, y, k) + l2 * (w - w0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + eps)
        if l1 > 0:
            w = np.sign(w) * np.maximum(np.abs(w) - lr * l1, 0.0)
    return w


L2_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
L1_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="+", help="JSONL trajectory dump(s)")
    ap.add_argument("--mode", choices=["l2", "l1"], default="l2")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="fixed L2 strength while sweeping L1 (l1 mode only)")
    ap.add_argument("--allow-legacy-schema", action="store_true")
    args = ap.parse_args()
    k = 80.0
    x, y, si = load_positions(
        args.data,
        include_truncated=False,
        allow_legacy_schema=args.allow_legacy_schema,
    )

    sweep_l1 = args.mode == "l1"
    grid = L1_GRID if sweep_l1 else L2_GRID
    label = "l1" if sweep_l1 else "l2"
    zeros_col = " | zeroed" if sweep_l1 else ""
    print(f"\n{label:>8} | mean delta val BCE (tuned - baseline) over 5 seeds | per-seed{zeros_col}")
    for strength in grid:
        deltas = []
        zeroed = []
        for seed in range(5):
            tr, va = split_by_state_index(si, 0.2, seed)
            if sweep_l1:
                w = fit(x[tr], y[tr], DEFAULT_WEIGHTS, k, args.l2, l1=strength)
            else:
                w = fit(x[tr], y[tr], DEFAULT_WEIGHTS, k, strength)
            base = bce_loss(DEFAULT_WEIGHTS, x[va], y[va], k)
            tuned = bce_loss(w, x[va], y[va], k)
            deltas.append(tuned - base)
            zeroed.append(int(np.sum(w == 0.0)))
        line = f"{strength:>8} | {np.mean(deltas):+.5f} | " + " ".join(f"{d:+.4f}" for d in deltas)
        if sweep_l1:
            line += f" | {'/'.join(str(z) for z in zeroed)}"
        print(line)


if __name__ == "__main__":
    main()
