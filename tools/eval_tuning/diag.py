#!/usr/bin/env python3
"""Diagnostics on a trajectory dump: baseline accuracy and calibration of
the current eval as an outcome predictor, and a 1-parameter global scale
(temperature) fit — the alpha that would calibrate sigmoid(alpha*score/K).

Usage: python diag.py [--allow-legacy-schema] <dump.jsonl> [more.jsonl ...]
"""
import argparse

import numpy as np

from tune import DEFAULT_WEIGHTS, load_positions, split_by_state_index


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
    score = x @ DEFAULT_WEIGHTS
    z = score / k
    p = 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    print(f"\npositions: {len(y)}  outcome mean: {y.mean():.3f}")
    print(f"baseline BCE {np.mean(np.logaddexp(0, z) - y * z):.4f}  (coin {np.log(2):.4f})")
    print(f"baseline accuracy (p>0.5 vs outcome): {np.mean((p > 0.5) == (y > 0.5)):.3f}")

    print("\ncalibration (pred bucket -> actual win rate, n):")
    for lo in np.arange(0, 1.0, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum():
            print(f"  [{lo:.1f},{lo + 0.1:.1f}) -> {y[m].mean():.3f}  n={m.sum()}")

    print("\nglobal scale alpha (5 seeds, fit on train, eval on val):")
    alphas = np.linspace(0.05, 3.0, 296)
    for seed in range(5):
        tr, va = split_by_state_index(si, 0.2, seed)
        ztr, zva = score[tr] / k, score[va] / k
        losses = [np.mean(np.logaddexp(0, a * ztr) - y[tr] * a * ztr) for a in alphas]
        a = alphas[int(np.argmin(losses))]
        base = np.mean(np.logaddexp(0, zva) - y[va] * zva)
        scaled = np.mean(np.logaddexp(0, a * zva) - y[va] * a * zva)
        print(f"  seed {seed}: alpha={a:.2f}  val BCE {base:.4f} -> {scaled:.4f} ({scaled - base:+.4f})")


if __name__ == "__main__":
    main()
