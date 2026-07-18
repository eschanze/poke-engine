#!/usr/bin/env python3
"""Train a tiny context-gated antisymmetric MLP on per-side eval features."""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from sweep import fit as fit_linear
from tune import DEFAULT_WEIGHTS, FEATURE_NAMES, FEATURE_SCHEMA, split_by_state_index


def load_data(data_path, pair_path):
    with Path(data_path).open("r", encoding="utf-8") as data_stream:
        records = [json.loads(line) for line in data_stream if line.strip()]
    with Path(pair_path).open("r", encoding="utf-8") as pair_stream:
        pairs = [json.loads(line) for line in pair_stream if line.strip()]
    if len(records) != len(pairs):
        sys.exit(f"trajectory/pair row mismatch: {len(records)} vs {len(pairs)}")
    truncated_games = {
        rec["game_id"] for rec in records if rec["truncated"]
    }
    rows = [
        (rec, pair) for rec, pair in zip(records, pairs)
        if rec["game_id"] not in truncated_games
    ]
    if any(rec.get("feature_schema") != FEATURE_SCHEMA for rec, _ in rows):
        sys.exit("trajectory feature schema mismatch")
    side_one = np.asarray([pair["side_one"] for _, pair in rows], dtype=np.float32)
    side_two = np.asarray([pair["side_two"] for _, pair in rows], dtype=np.float32)
    difference = side_one - side_two
    dumped = np.asarray([rec["features"] for rec, _ in rows], dtype=np.float32)
    max_drift = float(np.max(np.abs(difference - dumped)))
    if max_drift > 2e-5:
        print(
            f"warning: recomputed features differ from cached trajectory features "
            f"(max {max_drift:.4g}); training on current-engine recomputation",
            file=sys.stderr,
        )
    outcomes = np.asarray([rec["outcome"] for rec, _ in rows], dtype=np.float32)
    states = np.asarray([rec["state_index"] for rec, _ in rows], dtype=np.int64)
    game_ids_raw = [rec["game_id"] for rec, _ in rows]
    game_lookup = {}
    game_ids = np.asarray(
        [game_lookup.setdefault(game, len(game_lookup)) for game in game_ids_raw], dtype=np.int64
    )
    counts = np.bincount(game_ids)
    weights = (1.0 / counts[game_ids]).astype(np.float32)
    print(f"loaded {len(rows)} positions from {len(game_lookup)} decisive games")
    return side_one, side_two, outcomes, states, weights


def weighted_bce(margin, target, weight):
    loss = np.logaddexp(0.0, margin) - target * margin
    return float(np.sum(loss * weight) / np.sum(weight))


class ContextMLP(nn.Module):
    """Odd difference pathway gated by swap-invariant total context."""

    def __init__(self, features, hidden, clip):
        super().__init__()
        self.difference = nn.Linear(features, hidden, bias=False)
        self.context = nn.Linear(features, hidden, bias=True)
        self.output = nn.Linear(hidden, 1, bias=False)
        self.clip = clip
        nn.init.xavier_uniform_(self.difference.weight)
        nn.init.xavier_uniform_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, difference, context):
        odd = torch.tanh(self.difference(difference))
        gate = torch.sigmoid(self.context(context))
        raw = self.output(odd * gate).squeeze(1)
        return self.clip * torch.tanh(raw / self.clip)


def scaling(side_one, side_two, mask):
    difference = side_one - side_two
    total = side_one + side_two
    diff_scale = np.maximum(np.sqrt(np.mean(difference[mask] ** 2, axis=0)), 1e-3)
    total_mean = np.mean(total[mask], axis=0)
    total_scale = np.maximum(np.std(total[mask], axis=0), 1e-3)
    return diff_scale.astype(np.float32), total_mean.astype(np.float32), total_scale.astype(np.float32)


def train_model(
    side_one, side_two, y, weights, base_margin, scales, train_mask, val_mask,
    hidden, weight_decay, clip, max_epochs, seed, fixed_epochs=None,
):
    torch.manual_seed(seed)
    model = ContextMLP(side_one.shape[1], hidden, clip)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    diff_scale, total_mean, total_scale = scales
    diff = torch.from_numpy((side_one - side_two) / diff_scale)
    total = torch.from_numpy((side_one + side_two - total_mean) / total_scale)
    target = torch.from_numpy(y)
    row_weight = torch.from_numpy(weights)
    base = torch.from_numpy(base_margin.astype(np.float32))
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    val_idx = torch.from_numpy(np.flatnonzero(val_mask)) if np.any(val_mask) else None
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stop_epoch = fixed_epochs or max_epochs
    for epoch in range(1, stop_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        margin = base[train_idx] + model(diff[train_idx], total[train_idx])
        losses = functional.binary_cross_entropy_with_logits(
            margin, target[train_idx], reduction="none"
        )
        loss = torch.sum(losses * row_weight[train_idx]) / torch.sum(row_weight[train_idx])
        loss.backward()
        optimizer.step()
        if fixed_epochs is not None or epoch % 5:
            continue
        model.eval()
        with torch.no_grad():
            margin = base[val_idx] + model(diff[val_idx], total[val_idx])
            losses = functional.binary_cross_entropy_with_logits(
                margin, target[val_idx], reduction="none"
            )
            val_loss = float(torch.sum(losses * row_weight[val_idx]) / torch.sum(row_weight[val_idx]))
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= 200:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch if fixed_epochs is None else fixed_epochs


def predict(model, side_one, side_two, scales, base_margin):
    diff_scale, total_mean, total_scale = scales
    difference = torch.from_numpy((side_one - side_two) / diff_scale)
    total = torch.from_numpy((side_one + side_two - total_mean) / total_scale)
    with torch.no_grad():
        return base_margin + model(difference, total).numpy()


def write_values(stream, label, values):
    stream.write(label + " " + " ".join(f"{value:.9g}" for value in values) + "\n")


def export_model(path, model, base_weights, scales, k, clip, metadata):
    diff_scale, total_mean, total_scale = scales
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("poke-engine-context-mlp-v1\n")
        stream.write(f"feature_schema {FEATURE_SCHEMA}\n")
        stream.write(f"k {k:.9g}\ncorrection_clip {clip:.9g}\n")
        stream.write(f"# {json.dumps(metadata, sort_keys=True)}\n")
        for name, value in zip(FEATURE_NAMES, base_weights):
            stream.write(f"base {name} {value:.9g}\n")
        write_values(stream, "diff_scale", diff_scale)
        write_values(stream, "total_mean", total_mean)
        write_values(stream, "total_scale", total_scale)
        hidden = model.output.weight.shape[1]
        stream.write(f"hidden {hidden}\n")
        for row in model.difference.weight.detach().numpy():
            write_values(stream, "difference", row)
        for row, bias in zip(model.context.weight.detach().numpy(), model.context.bias.detach().numpy()):
            write_values(stream, "context", np.append(row, bias))
        write_values(stream, "output", model.output.weight.detach().numpy()[0])
    print(f"wrote {hidden}-unit contextual MLP to {path}")


def export_base_weights(path, weights, metadata):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# contextual MLP linear base: {json.dumps(metadata, sort_keys=True)}\n")
        for name, value in zip(FEATURE_NAMES, weights):
            stream.write(f"{name} {value:.9g}\n")
    print(f"wrote linear base weights to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--pair-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-out", help="also export the fitted linear base as a weights file")
    ap.add_argument("--k", type=float, default=80.0)
    ap.add_argument("--l1", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--weight-decay", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--correction-clip", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=int, default=2000)
    args = ap.parse_args()

    side_one, side_two, y, state_indices, weights = load_data(args.data, args.pair_data)
    difference = side_one - side_two
    folds = [split_by_state_index(state_indices, 0.2, seed) for seed in range(5)]
    fold_base = []
    fold_scales = []
    for train_mask, _ in folds:
        fitted = fit_linear(
            difference[train_mask].astype(np.float64), y[train_mask].astype(np.float64),
            DEFAULT_WEIGHTS, args.k, 0.0, l1=args.l1,
        )
        fold_base.append(difference @ fitted.astype(np.float32) / args.k)
        fold_scales.append(scaling(side_one, side_two, train_mask))

    candidates = []
    print("\nGrouped five-fold sweep (context MLP BCE - fold-local linear BCE)")
    for hidden in args.hidden:
        for weight_decay in args.weight_decay:
            deltas, epochs = [], []
            for fold, (train_mask, val_mask) in enumerate(folds):
                model, epoch = train_model(
                    side_one, side_two, y, weights, fold_base[fold], fold_scales[fold],
                    train_mask, val_mask, hidden, weight_decay, args.correction_clip,
                    args.max_epochs, 3000 + fold,
                )
                margin = predict(
                    model, side_one[val_mask], side_two[val_mask], fold_scales[fold],
                    fold_base[fold][val_mask],
                )
                baseline = weighted_bce(fold_base[fold][val_mask], y[val_mask], weights[val_mask])
                deltas.append(weighted_bce(margin, y[val_mask], weights[val_mask]) - baseline)
                epochs.append(epoch)
            mean_delta = statistics.mean(deltas)
            candidates.append((mean_delta, hidden, weight_decay, deltas, epochs))
            print(
                f"hidden={hidden:>2} wd={weight_decay:<5g}: {mean_delta:+.5f}  "
                + " ".join(f"{delta:+.4f}" for delta in deltas)
                + "  epochs=" + "/".join(str(epoch) for epoch in epochs)
            )

    mean_delta, hidden, weight_decay, deltas, epochs = min(candidates)
    final_epochs = max(1, int(round(statistics.median(epochs))))
    base_weights = fit_linear(
        difference.astype(np.float64), y.astype(np.float64), DEFAULT_WEIGHTS,
        args.k, 0.0, l1=args.l1,
    ).astype(np.float32)
    base_margin = difference @ base_weights / args.k
    all_mask = np.ones(len(y), dtype=bool)
    scales = scaling(side_one, side_two, all_mask)
    model, _ = train_model(
        side_one, side_two, y, weights, base_margin, scales, all_mask,
        np.zeros(len(y), dtype=bool), hidden, weight_decay, args.correction_clip,
        args.max_epochs, 2026, fixed_epochs=final_epochs,
    )
    print(
        f"\nselected hidden={hidden}, weight_decay={weight_decay:g}, "
        f"epochs={final_epochs}, mean delta={mean_delta:+.5f}"
    )
    metadata = {"hidden": hidden, "weight_decay": weight_decay, "epochs": final_epochs,
                "mean_cv_delta": mean_delta}
    export_model(
        args.out, model, base_weights, scales, args.k, args.correction_clip,
        metadata,
    )
    if args.base_out:
        export_base_weights(args.base_out, base_weights, metadata)


if __name__ == "__main__":
    main()
