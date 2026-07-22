#!/usr/bin/env python3
"""Train a tiny odd-symmetric MLP correction to the linear evaluator.

The network has no biases and uses odd activations, so r(-f) == -r(f)
without evaluating both orientations. It learns only a bounded residual on
top of a fold-local L1 linear fit and exports a dependency-free text model.
"""

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


def load_positions(paths, target="outcome"):
    records = []
    truncated = set()
    for path_text in paths:
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    sys.exit(f"{path}:{line_no}: bad JSON: {exc}")
                if rec.get("feature_schema") != FEATURE_SCHEMA:
                    sys.exit(f"{path}:{line_no}: incompatible feature schema")
                key = (str(path.resolve()), rec["game_id"])
                if rec["truncated"]:
                    truncated.add(key)
                records.append((key, rec))

    usable = [(key, rec) for key, rec in records if key not in truncated]
    game_lookup = {}
    game_ids = np.asarray(
        [game_lookup.setdefault(key, len(game_lookup)) for key, _ in usable], dtype=np.int64
    )
    if any(target not in rec for _, rec in usable):
        sys.exit(f"--target {target} needs relabeled data (see relabel-root-value)")
    x = np.asarray([rec["features"] for _, rec in usable], dtype=np.float32)
    y = np.asarray([rec[target] for _, rec in usable], dtype=np.float32)
    state_indices = np.asarray([rec["state_index"] for _, rec in usable], dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        sys.exit("trajectory feature width does not match the current evaluator")
    counts = np.bincount(game_ids)
    weights = (1.0 / counts[game_ids]).astype(np.float32)
    print(
        f"loaded {len(x)} positions from {len(game_lookup)} decisive games "
        f"({len(truncated)} truncated games dropped)"
    )
    return x, y, state_indices, weights


def weighted_bce(margin, target, weight):
    losses = np.logaddexp(0.0, margin) - target * margin
    return float(np.sum(losses * weight) / np.sum(weight))


class OddResidualMLP(nn.Module):
    def __init__(self, inputs, hidden, correction_clip):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 1, bias=False)
        self.correction_clip = correction_clip
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        raw = self.fc2(torch.tanh(self.fc1(x))).squeeze(1)
        return self.correction_clip * torch.tanh(raw / self.correction_clip)


def train_model(
    x,
    y,
    weights,
    base_margin,
    scale,
    train_mask,
    val_mask,
    hidden,
    weight_decay,
    correction_clip,
    max_epochs,
    seed,
    fixed_epochs=None,
):
    torch.manual_seed(seed)
    model = OddResidualMLP(x.shape[1], hidden, correction_clip)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    tx = torch.from_numpy(x / scale)
    ty = torch.from_numpy(y)
    tw = torch.from_numpy(weights)
    tb = torch.from_numpy(base_margin.astype(np.float32))
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    val_idx = torch.from_numpy(np.flatnonzero(val_mask)) if np.any(val_mask) else None
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 200
    stop_epoch = fixed_epochs or max_epochs

    for epoch in range(1, stop_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        margin = tb[train_idx] + model(tx[train_idx])
        losses = functional.binary_cross_entropy_with_logits(
            margin, ty[train_idx], reduction="none"
        )
        loss = torch.sum(losses * tw[train_idx]) / torch.sum(tw[train_idx])
        loss.backward()
        optimizer.step()

        if fixed_epochs is not None or epoch % 5 != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_margin = tb[val_idx] + model(tx[val_idx])
            val_losses = functional.binary_cross_entropy_with_logits(
                val_margin, ty[val_idx], reduction="none"
            )
            val_loss = float(torch.sum(val_losses * tw[val_idx]) / torch.sum(tw[val_idx]))
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if fixed_epochs is None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch if fixed_epochs is None else fixed_epochs


def predict(model, x, scale, base_margin):
    with torch.no_grad():
        residual = model(torch.from_numpy(x / scale)).numpy()
    return base_margin + residual


def export_model(path, model, base_weights, scale, k, correction_clip, metadata):
    w1 = model.fc1.weight.detach().numpy()
    w2 = model.fc2.weight.detach().numpy()[0]
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("poke-engine-mlp-eval-v1\n")
        stream.write(f"feature_schema {FEATURE_SCHEMA}\n")
        stream.write(f"k {k:.9g}\n")
        stream.write(f"correction_clip {correction_clip:.9g}\n")
        stream.write(f"# {json.dumps(metadata, sort_keys=True)}\n")
        for name, value in zip(FEATURE_NAMES, base_weights):
            stream.write(f"base {name} {value:.9g}\n")
        for name, value in zip(FEATURE_NAMES, scale):
            stream.write(f"scale {name} {value:.9g}\n")
        stream.write(f"hidden {len(w2)}\n")
        for row in w1:
            stream.write("w1 " + " ".join(f"{value:.9g}" for value in row) + "\n")
        stream.write("w2 " + " ".join(f"{value:.9g}" for value in w2) + "\n")
    print(f"wrote {len(w2)}-unit MLP to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=80.0)
    ap.add_argument("--l1", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--weight-decay", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--correction-clip", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--target", choices=["outcome", "root_value"], default="outcome",
                    help="training label: game outcome or relabeled teacher root value")
    args = ap.parse_args()

    x, y, state_indices, weights = load_positions(args.data, args.target)
    folds = [split_by_state_index(state_indices, 0.2, seed) for seed in range(5)]
    fold_bases = []
    fold_scales = []
    print(f"fitting fold-local L1={args.l1:g} linear baselines")
    for train_mask, _ in folds:
        fitted = fit_linear(
            x[train_mask].astype(np.float64), y[train_mask].astype(np.float64),
            DEFAULT_WEIGHTS, args.k, 0.0, l1=args.l1,
        )
        fold_bases.append(x @ fitted.astype(np.float32) / args.k)
        fold_scales.append(np.maximum(np.sqrt(np.mean(x[train_mask] ** 2, axis=0)), 1e-3))

    candidates = []
    print("\nGrouped five-fold sweep (MLP BCE - fold-local linear BCE)")
    for hidden in args.hidden:
        for weight_decay in args.weight_decay:
            deltas = []
            epochs = []
            for fold, (train_mask, val_mask) in enumerate(folds):
                model, best_epoch = train_model(
                    x, y, weights, fold_bases[fold], fold_scales[fold],
                    train_mask, val_mask, hidden, weight_decay,
                    args.correction_clip, args.max_epochs, seed=1000 + fold,
                )
                margin = predict(model, x[val_mask], fold_scales[fold], fold_bases[fold][val_mask])
                baseline = weighted_bce(
                    fold_bases[fold][val_mask], y[val_mask], weights[val_mask]
                )
                deltas.append(weighted_bce(margin, y[val_mask], weights[val_mask]) - baseline)
                epochs.append(best_epoch)
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
        x.astype(np.float64), y.astype(np.float64), DEFAULT_WEIGHTS,
        args.k, 0.0, l1=args.l1,
    ).astype(np.float32)
    base_margin = x @ base_weights / args.k
    scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=0)), 1e-3)
    all_mask = np.ones(len(x), dtype=bool)
    no_val = np.zeros(len(x), dtype=bool)
    model, _ = train_model(
        x, y, weights, base_margin, scale, all_mask, no_val,
        hidden, weight_decay, args.correction_clip, args.max_epochs,
        seed=2026, fixed_epochs=final_epochs,
    )
    print(
        f"\nselected hidden={hidden}, weight_decay={weight_decay:g}, "
        f"epochs={final_epochs}, mean delta={mean_delta:+.5f}"
    )
    export_model(
        args.out, model, base_weights, scale, args.k, args.correction_clip,
        {"hidden": hidden, "weight_decay": weight_decay, "epochs": final_epochs,
         "mean_cv_delta": mean_delta},
    )


if __name__ == "__main__":
    main()
