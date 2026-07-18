#!/usr/bin/env python3
"""Texel-style tuning of the hand-crafted eval weights (numpy only).

Fits the 40 eval weights by logistic regression on game outcomes:
minimize BCE of sigmoid(dot(w, features) / K) against the final result of
the game each position came from. K is fixed at 80 to match the MCTS
rollout squash sigmoid(0.0125 * (eval - root_eval)) — do not fit it.

Input: JSONL trajectory dumps produced by the selfplay binary's
--dump-trajectories flag (one visited position per line: eval features,
game outcome for side one, truncation flag, originating state index).

Output: a weights file loadable by selfplay --a-eval-weights / --b-eval-weights.

Usage:
    python tune.py --data traj.jsonl --out tuned_weights.txt

See EVAL_TUNING_PLAN.md at the repo root for the method and caveats.
"""

import argparse
import json
import sys

import numpy as np

# Must match EVAL_FEATURE_NAMES / DEFAULT_EVAL_WEIGHTS in src/genx/evaluate.rs.
# Trajectory records carry a hash of these names in positional order, so
# schema drift is rejected before fitting.
FEATURE_NAMES = [
    "POKEMON_ALIVE",
    "POKEMON_HP",
    "POKEMON_ITEM",
    "POKEMON_FROZEN",
    "POKEMON_ASLEEP",
    "POKEMON_PARALYZED",
    "POKEMON_TOXIC",
    "POKEMON_POISONED",
    "POKEMON_BURNED",
    "POISON_HEAL",
    "STATUS_ABILITY_BONUS",
    "POKEMON_ATTACK_BOOST",
    "POKEMON_DEFENSE_BOOST",
    "POKEMON_SPECIAL_ATTACK_BOOST",
    "POKEMON_SPECIAL_DEFENSE_BOOST",
    "POKEMON_SPEED_BOOST",
    "LEECH_SEED",
    "SUBSTITUTE",
    "CONFUSION",
    "REFLECT",
    "LIGHT_SCREEN",
    "AURORA_VEIL",
    "SAFE_GUARD",
    "TAILWIND",
    "HEALING_WISH",
    "STEALTH_ROCK",
    "SPIKES",
    "TOXIC_SPIKES",
    "STICKY_WEB",
    "USED_TERA",
    "EFFECTIVE_HEALTH",
    "TWO_HIT_KO_PRESSURE",
    "REVENGE_COVERAGE",
    "WALLBREAK_PRESSURE",
    "THREAT_BREADTH",
    "ANSWER_SCARCITY",
    "WINCON",
    "UNANSWERED",
    "ACTIVE_DUEL",
    "PIVOT_PRESSURE",
]

DEFAULT_WEIGHTS = np.array(
    [
        62.7669384,   # POKEMON_ALIVE
        41.0806585,   # POKEMON_HP
        0.0,          # POKEMON_ITEM
        -90.2561015,  # POKEMON_FROZEN
        -0.408866373, # POKEMON_ASLEEP
        -25.6112665,  # POKEMON_PARALYZED
        -75.4539139,  # POKEMON_TOXIC
        -24.6924447,  # POKEMON_POISONED
        0.0,          # POKEMON_BURNED
        52.9131556,   # POISON_HEAL
        0.0,          # STATUS_ABILITY_BONUS
        1.34682091,   # POKEMON_ATTACK_BOOST
        4.54921779,   # POKEMON_DEFENSE_BOOST
        0.0,          # POKEMON_SPECIAL_ATTACK_BOOST
        24.9293773,   # POKEMON_SPECIAL_DEFENSE_BOOST
        23.6278393,   # POKEMON_SPEED_BOOST
        -32.9769423,  # LEECH_SEED
        104.108085,   # SUBSTITUTE
        -10.196138,   # CONFUSION
        0.0,          # REFLECT
        0.0,          # LIGHT_SCREEN
        0.0,          # AURORA_VEIL
        0.0,          # SAFE_GUARD
        0.0,          # TAILWIND
        60.0,         # HEALING_WISH
        -2.97986335,  # STEALTH_ROCK
        -17.3453197,  # SPIKES
        -30.0,        # TOXIC_SPIKES
        0.0,          # STICKY_WEB
        -48.5478855,  # USED_TERA
        0.0,          # EFFECTIVE_HEALTH
        0.0,          # TWO_HIT_KO_PRESSURE
        3.1326059,    # REVENGE_COVERAGE
        0.0,          # WALLBREAK_PRESSURE
        40.5900329,   # THREAT_BREADTH
        0.0,          # ANSWER_SCARCITY
        14.4660979,   # WINCON
        20.1621173,   # UNANSWERED
        4.86702255,   # ACTIVE_DUEL
        3.7430539,    # PIVOT_PRESSURE
    ],
    dtype=np.float64,
)


def feature_schema(names):
    """Match eval_feature_schema() in src/genx/evaluate.rs."""
    value = 0xCBF29CE484222325
    for name in names:
        for byte in name.encode("ascii"):
            value ^= byte
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        value ^= 0xFF
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


FEATURE_SCHEMA = feature_schema(FEATURE_NAMES)


def load_positions(paths, include_truncated, allow_legacy_schema=False):
    features, outcomes, state_indices = [], [], []
    games = set()
    truncated_games = set()
    warned_legacy = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{line_no + 1}: bad JSON: {e}")
                schema = rec.get("feature_schema")
                if schema is None and allow_legacy_schema:
                    if path not in warned_legacy:
                        print(
                            f"warning: {path} has no feature schema; "
                            "assuming the original 30-feature layout",
                            file=sys.stderr,
                        )
                        warned_legacy.add(path)
                    schema = FEATURE_SCHEMA
                if schema != FEATURE_SCHEMA:
                    sys.exit(
                        f"{path}:{line_no + 1}: feature schema {schema!r}, "
                        f"expected {FEATURE_SCHEMA!r}; regenerate the dump"
                    )
                key = (path, rec["game_id"])
                if rec["truncated"]:
                    truncated_games.add(key)
                    if not include_truncated:
                        continue
                games.add(key)
                feats = rec["features"]
                if len(feats) != len(FEATURE_NAMES):
                    sys.exit(
                        f"{path}:{line_no + 1}: {len(feats)} features, "
                        f"expected {len(FEATURE_NAMES)} — regenerate the dump "
                        f"or update FEATURE_NAMES"
                    )
                features.append(feats)
                outcomes.append(rec["outcome"])
                state_indices.append(rec["state_index"])
    if not features:
        sys.exit("no usable positions loaded")
    print(
        f"loaded {len(features)} positions from {len(games)} games "
        f"({len(truncated_games)} truncated games "
        f"{'included' if include_truncated else 'dropped'})"
    )
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(outcomes, dtype=np.float64),
        np.asarray(state_indices, dtype=np.int64),
    )


def split_by_state_index(state_indices, val_frac, seed):
    """Positions from one starting state share teams and often labels;
    keep every state's positions on one side of the split."""
    unique = np.unique(state_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac))) if val_frac > 0 else 0
    val_states = set(unique[:n_val].tolist())
    is_val = np.isin(state_indices, list(val_states))
    return ~is_val, is_val


def bce_loss(w, x, y, k):
    z = x @ w / k
    # stable: log(1 + e^z) - y*z
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def bce_grad(w, x, y, k):
    z = x @ w / k
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return x.T @ (p - y) / (len(y) * k)


def tune(x, y, w0, k, l2, l1, lr, steps, x_val, y_val, log_every):
    """Full-batch Adam on BCE + 0.5*l2*||w - w0||^2 + l1*||w||_1.

    The L2 term pulls toward the current weights (a prior); the L1 term pulls
    toward zero and is applied as a proximal soft-threshold after each Adam
    step (threshold lr*l1 — approximate under Adam's per-coordinate scaling,
    standard decoupled-prox practice), so useless features die to exactly 0.
    Returns best-val weights.
    """
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    have_val = len(y_val) > 0
    best_w = w.copy()
    best_val = bce_loss(w, x_val, y_val, k) if have_val else np.inf
    for step in range(1, steps + 1):
        g = bce_grad(w, x, y, k) + l2 * (w - w0)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if l1 > 0:
            w = np.sign(w) * np.maximum(np.abs(w) - lr * l1, 0.0)
        if step % log_every == 0 or step == steps:
            train = bce_loss(w, x, y, k)
            if have_val:
                val = bce_loss(w, x_val, y_val, k)
                if val < best_val:
                    best_val = val
                    best_w = w.copy()
                print(f"step {step:6d}  train BCE {train:.5f}  val BCE {val:.5f}")
            else:
                print(f"step {step:6d}  train BCE {train:.5f}")
    if not have_val:
        return w
    return best_w


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", required=True, help="JSONL trajectory dump(s)")
    ap.add_argument("--out", required=True, help="output weights file")
    ap.add_argument("--k", type=float, default=80.0, help="eval-to-probability scale; keep at 80 to match the search (see plan)")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 pull toward the current weights (prior strength)")
    ap.add_argument("--l1", type=float, default=0.0, help="L1 pull toward zero (feature elimination; proximal)")
    ap.add_argument("--lr", type=float, default=0.5, help="Adam learning rate, in eval units per step")
    ap.add_argument("--steps", type=int, default=4000, help="optimization steps (full batch)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of starting states held out for validation")
    ap.add_argument("--seed", type=int, default=0, help="split shuffle seed")
    ap.add_argument("--include-truncated", action="store_true", help="keep positions from games that hit the decision cap")
    ap.add_argument("--allow-legacy-schema", action="store_true", help="accept pre-schema dumps as the original 30-feature layout")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    if not np.isfinite(args.k) or args.k <= 0:
        ap.error("--k must be finite and greater than zero")
    if not np.isfinite(args.l2) or args.l2 < 0:
        ap.error("--l2 must be finite and non-negative")
    if not np.isfinite(args.l1) or args.l1 < 0:
        ap.error("--l1 must be finite and non-negative")
    if not np.isfinite(args.lr) or args.lr <= 0:
        ap.error("--lr must be finite and greater than zero")
    if args.steps <= 0:
        ap.error("--steps must be greater than zero")
    if not 0 <= args.val_frac < 1:
        ap.error("--val-frac must be in [0, 1)")
    if args.log_every <= 0:
        ap.error("--log-every must be greater than zero")

    x, y, state_indices = load_positions(
        args.data, args.include_truncated, args.allow_legacy_schema
    )
    train_mask, val_mask = split_by_state_index(state_indices, args.val_frac, args.seed)
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    print(
        f"train: {len(y_train)} positions / {len(np.unique(state_indices[train_mask]))} states, "
        f"val: {len(y_val)} positions / {len(np.unique(state_indices[val_mask]))} states"
    )
    if len(y_train) == 0:
        sys.exit("empty training split")

    w0 = DEFAULT_WEIGHTS
    print(f"\nbaseline (current weights): train BCE {bce_loss(w0, x_train, y_train, args.k):.5f}", end="")
    if len(y_val):
        print(f"  val BCE {bce_loss(w0, x_val, y_val, args.k):.5f}", end="")
    print(f"  (coin flip: {np.log(2):.5f})\n")

    w = tune(
        x_train, y_train, w0, args.k, args.l2, args.l1, args.lr, args.steps,
        x_val, y_val, args.log_every,
    )
    if not np.all(np.isfinite(w)):
        sys.exit("tuning produced non-finite weights; adjust the optimizer settings")

    print(f"\n{'weight':<30} {'current':>10} {'tuned':>10} {'delta':>10}")
    for name, before, after in zip(FEATURE_NAMES, w0, w):
        print(f"{name:<30} {before:>10.2f} {after:>10.2f} {after - before:>+10.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# tuned by tools/eval_tuning/tune.py — K={args.k} l2={args.l2} "
                f"l1={args.l1} steps={args.steps} lr={args.lr} seed={args.seed}\n")
        f.write(f"# data: {' '.join(args.data)}\n")
        for name, value in zip(FEATURE_NAMES, w):
            f.write(f"{name} {value:.4f}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
