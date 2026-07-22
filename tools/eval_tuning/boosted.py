#!/usr/bin/env python3
"""Train a small antisymmetric boosted-tree correction to the linear eval.

The tree ensemble learns a correction in logit units on top of
``dot(base_weights, features) / K``.  At inference the correction is
symmetrized as ``0.5 * (trees(f) - trees(-f))`` so swapping sides always
negates the score.

Validation is grouped by starting state.  Rows are weighted so every game
has equal total weight even when it contributes a different number of
positions.

Example:
  python boosted.py --data data/traj500-sub12.jsonl --out boosted.model
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from sweep import fit as fit_linear
from tune import DEFAULT_WEIGHTS, FEATURE_NAMES, FEATURE_SCHEMA, split_by_state_index


def load_base_weights(path):
    if path is None:
        return DEFAULT_WEIGHTS.astype(np.float32)
    values = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2 or parts[0] not in FEATURE_NAMES:
                sys.exit(f"{path}:{line_no}: expected FEATURE_NAME value")
            if parts[0] in values:
                sys.exit(f"{path}:{line_no}: duplicate weight {parts[0]}")
            values[parts[0]] = float(parts[1])
    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        sys.exit(f"{path}: missing weights: {', '.join(missing)}")
    result = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    if not np.all(np.isfinite(result)):
        sys.exit(f"{path}: weights must be finite")
    return result


def load_positions(paths, target="outcome"):
    features = []
    outcomes = []
    state_indices = []
    game_keys = []
    truncated_games = set()
    records = []
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    sys.exit(f"{path}:{line_no}: bad JSON: {exc}")
                if rec.get("feature_schema") != FEATURE_SCHEMA:
                    sys.exit(
                        f"{path}:{line_no}: feature schema "
                        f"{rec.get('feature_schema')!r}, expected {FEATURE_SCHEMA!r}"
                    )
                key = (str(path.resolve()), rec["game_id"])
                if rec["truncated"]:
                    truncated_games.add(key)
                records.append((key, rec))

    usable = [(key, rec) for key, rec in records if key not in truncated_games]
    if not usable:
        sys.exit("no usable non-truncated positions loaded")
    if any(target not in rec for _, rec in usable):
        sys.exit(f"--target {target} needs relabeled data (see relabel-root-value)")
    for key, rec in usable:
        feats = rec["features"]
        if len(feats) != len(FEATURE_NAMES):
            sys.exit(f"game {key}: got {len(feats)} features, expected {len(FEATURE_NAMES)}")
        features.append(feats)
        outcomes.append(rec[target])
        state_indices.append(rec["state_index"])
        game_keys.append(key)

    game_lookup = {}
    game_ids = np.asarray(
        [game_lookup.setdefault(key, len(game_lookup)) for key in game_keys],
        dtype=np.int64,
    )
    print(
        f"loaded {len(features)} positions from {len(game_lookup)} decisive games "
        f"({len(truncated_games)} truncated games dropped)"
    )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(outcomes, dtype=np.float32),
        np.asarray(state_indices, dtype=np.int64),
        game_ids,
    )


def game_balanced_weights(game_ids):
    counts = np.bincount(game_ids)
    return (1.0 / counts[game_ids]).astype(np.float32)


def weighted_bce(margin, outcome, weight):
    losses = np.logaddexp(0.0, margin) - outcome * margin
    return float(np.sum(losses * weight) / np.sum(weight))


def matrix(x, y, weight, base_margin):
    return xgb.DMatrix(
        x,
        label=y,
        weight=weight,
        base_margin=base_margin,
        feature_names=FEATURE_NAMES,
    )


def augmented_matrix(x, y, weight, base_margin):
    return matrix(
        np.concatenate((x, -x)),
        np.concatenate((y, 1.0 - y)),
        np.concatenate((0.5 * weight, 0.5 * weight)),
        np.concatenate((base_margin, -base_margin)),
    )


def predict_margin(model, x, base_margin, rounds=None):
    kwargs = {"output_margin": True}
    if rounds is not None:
        kwargs["iteration_range"] = (0, rounds)
    plus = model.predict(matrix(x, np.zeros(len(x)), np.ones(len(x)), base_margin), **kwargs)
    minus = model.predict(matrix(-x, np.zeros(len(x)), np.ones(len(x)), -base_margin), **kwargs)
    return 0.5 * (plus - minus)


def train_fold(x, y, weights, base_margin, train_mask, val_mask, params, max_rounds):
    dtrain = augmented_matrix(
        x[train_mask], y[train_mask], weights[train_mask], base_margin[train_mask]
    )
    dval = augmented_matrix(
        x[val_mask], y[val_mask], weights[val_mask], base_margin[val_mask]
    )
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dval, "validation")],
        early_stopping_rounds=24,
        verbose_eval=False,
    )
    rounds = model.best_iteration + 1
    margin = predict_margin(model, x[val_mask], base_margin[val_mask], rounds)
    return weighted_bce(margin, y[val_mask], weights[val_mask]), rounds


def flatten_tree(tree):
    flat = []

    def visit(node):
        index = len(flat)
        flat.append(None)
        if "leaf" in node:
            flat[index] = ("leaf", float(node["leaf"]))
            return index
        children = {child["nodeid"]: child for child in node["children"]}
        yes = visit(children[node["yes"]])
        no = visit(children[node["no"]])
        split = node["split"]
        feature = FEATURE_NAMES.index(split) if split in FEATURE_NAMES else int(split[1:])
        flat[index] = ("split", feature, float(node["split_condition"]), yes, no)
        return index

    visit(tree)
    return flat


def tree_sum(flat_trees, row):
    total = 0.0
    for tree in flat_trees:
        index = 0
        while tree[index][0] == "split":
            _, feature, threshold, yes, no = tree[index]
            index = yes if row[feature] < threshold else no
        total += tree[index][1]
    return total


def export_model(path, model, base_weights, k, correction_clip, params, rounds, x):
    flat_trees = [flatten_tree(json.loads(raw)) for raw in model.get_dump(dump_format="json")]
    base = x @ base_weights / k
    raw = model.predict(matrix(x, np.zeros(len(x)), np.ones(len(x)), base), output_margin=True)
    check_rows = np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int)
    for row_index in check_rows:
        expected = float(raw[row_index] - base[row_index])
        actual = tree_sum(flat_trees, x[row_index])
        if not np.isclose(expected, actual, atol=2e-5):
            raise RuntimeError(
                f"export check failed at row {row_index}: xgboost={expected}, exported={actual}"
            )

    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write("poke-engine-tree-eval-v1\n")
        f.write(f"feature_schema {FEATURE_SCHEMA}\n")
        f.write(f"k {k:.9g}\n")
        f.write(f"correction_clip {correction_clip:.9g}\n")
        f.write(f"# params {json.dumps(params, sort_keys=True)}\n")
        f.write(f"# rounds {rounds}\n")
        for name, value in zip(FEATURE_NAMES, base_weights):
            f.write(f"base {name} {value:.9g}\n")
        f.write(f"trees {len(flat_trees)}\n")
        for tree in flat_trees:
            f.write(f"tree {len(tree)}\n")
            for node in tree:
                if node[0] == "leaf":
                    f.write(f"leaf {node[1]:.9g}\n")
                else:
                    _, feature, threshold, yes, no = node
                    f.write(f"split {feature} {threshold:.9g} {yes} {no}\n")
    print(f"wrote {len(flat_trees)} trees to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-weights", help="linear weights corrected by the trees (default: built-in)")
    ap.add_argument("--fit-l1", type=float,
                    help="fit an L1 linear base independently inside every CV fold")
    ap.add_argument("--k", type=float, default=80.0)
    ap.add_argument("--max-rounds", type=int, default=256)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=20.0)
    ap.add_argument("--correction-clip", type=float, default=1.0,
                    help="absolute tree correction limit in logit units at engine inference")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--min-child-weights", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--target", choices=["outcome", "root_value"], default="outcome",
                    help="training label: game outcome or relabeled teacher root value")
    args = ap.parse_args()
    if args.k <= 0 or not np.isfinite(args.k):
        ap.error("--k must be finite and positive")
    if args.max_rounds <= 0:
        ap.error("--max-rounds must be positive")
    if args.correction_clip <= 0 or not np.isfinite(args.correction_clip):
        ap.error("--correction-clip must be finite and positive")
    if args.fit_l1 is not None and (args.fit_l1 < 0 or not np.isfinite(args.fit_l1)):
        ap.error("--fit-l1 must be finite and non-negative")
    if args.fit_l1 is not None and args.base_weights:
        ap.error("--fit-l1 and --base-weights are mutually exclusive")

    x, y, state_indices, game_ids = load_positions(args.data, args.target)
    weights = game_balanced_weights(game_ids)
    base_weights = load_base_weights(args.base_weights)
    fold_masks = [split_by_state_index(state_indices, 0.2, seed) for seed in range(5)]
    if args.fit_l1 is not None:
        print(f"fitting fold-local L1={args.fit_l1:g} linear baselines")
        fold_base_margins = []
        for train_mask, _ in fold_masks:
            fold_weights = fit_linear(
                x[train_mask].astype(np.float64), y[train_mask].astype(np.float64),
                DEFAULT_WEIGHTS, args.k, 0.0, l1=args.fit_l1,
            )
            fold_base_margins.append(x @ fold_weights.astype(np.float32) / args.k)
        base_weights = fit_linear(
            x.astype(np.float64), y.astype(np.float64), DEFAULT_WEIGHTS,
            args.k, 0.0, l1=args.fit_l1,
        ).astype(np.float32)
    base_margin = x @ base_weights / args.k

    common = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": args.eta,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": args.reg_lambda,
        "alpha": 0.1,
        "tree_method": "hist",
        "seed": 0,
        "nthread": 1,
    }
    candidates = []
    print("\nGrouped five-fold sweep (tree BCE - linear BCE; negative is better)")
    for depth in args.depths:
        for min_child in args.min_child_weights:
            params = {**common, "max_depth": depth, "min_child_weight": min_child}
            deltas = []
            rounds = []
            for seed in range(5):
                train_mask, val_mask = fold_masks[seed]
                this_base_margin = (
                    fold_base_margins[seed] if args.fit_l1 is not None else base_margin
                )
                baseline = weighted_bce(
                    this_base_margin[val_mask], y[val_mask], weights[val_mask]
                )
                loss, best_rounds = train_fold(
                    x, y, weights, this_base_margin, train_mask, val_mask,
                    params, args.max_rounds
                )
                deltas.append(loss - baseline)
                rounds.append(best_rounds)
            mean_delta = statistics.mean(deltas)
            candidates.append((mean_delta, depth, min_child, deltas, rounds))
            print(
                f"depth={depth} child={min_child:>4g}: {mean_delta:+.5f}  "
                + " ".join(f"{delta:+.4f}" for delta in deltas)
                + "  rounds=" + "/".join(str(r) for r in rounds)
            )

    best = min(candidates)
    _, depth, min_child, deltas, fold_rounds = best
    final_rounds = max(1, int(round(statistics.median(fold_rounds))))
    params = {**common, "max_depth": depth, "min_child_weight": min_child}
    print(
        f"\nselected depth={depth}, min_child_weight={min_child:g}, "
        f"rounds={final_rounds}, mean delta={statistics.mean(deltas):+.5f}"
    )
    full = augmented_matrix(x, y, weights, base_margin)
    model = xgb.train(params, full, num_boost_round=final_rounds, verbose_eval=False)
    importance = sorted(model.get_score(importance_type="gain").items(), key=lambda pair: -pair[1])
    print("top split features: " + ", ".join(f"{name}={gain:.2f}" for name, gain in importance[:12]))
    export_model(
        args.out, model, base_weights, args.k, args.correction_clip,
        params, final_rounds, x,
    )


if __name__ == "__main__":
    main()
