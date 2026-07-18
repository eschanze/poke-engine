#!/usr/bin/env python3
"""Score weight vectors on held-out dumps: per-game-weighted BCE + accuracy.

Every vector is evaluated on the same rows; nothing is trained, so dumps
from matchups no vector has seen give an unbiased comparison.

Usage:
  python score36.py --pair tag=<traj.jsonl>:<pair.jsonl> [...] \
      --weights adopted=DEFAULT --weights refit=data/refit36-suite.weights
"""

import argparse
from pathlib import Path

import numpy as np

from fit36 import load_merged, weighted_bce
from tune import DEFAULT_WEIGHTS, FEATURE_NAMES
from constrained import HANDCRAFTED_WEIGHTS


def load_vector(spec):
    if spec == "DEFAULT":
        return np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    if spec == "HANDCRAFTED":
        return np.asarray(HANDCRAFTED_WEIGHTS, dtype=np.float64)
    values = {}
    for line in Path(spec).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, value = line.split()
            values[name] = float(value)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    metavar="TAG=TRAJ:PAIR")
    ap.add_argument("--weights", action="append", required=True,
                    metavar="NAME=FILE|DEFAULT|HANDCRAFTED")
    ap.add_argument("--k", type=float, default=80.0)
    args = ap.parse_args()

    parsed = []
    for spec in args.pair:
        tag, rest = spec.split("=", 1)
        traj, pair = rest.split(":", 1)
        parsed.append((tag, traj, pair))
    x, y, _states, wt, _tags = load_merged(parsed)

    print(f"\n{'vector':<24} {'BCE':>8} {'accuracy':>9}")
    for spec in args.weights:
        name, path = spec.split("=", 1)
        w = load_vector(path)
        margin = x @ w / args.k
        bce = weighted_bce(margin, y, wt)
        correct = (margin > 0) == (y > 0.5)
        accuracy = float(np.sum(correct * wt) / np.sum(wt))
        print(f"{name:<24} {bce:>8.5f} {accuracy:>8.1%}")


if __name__ == "__main__":
    main()
