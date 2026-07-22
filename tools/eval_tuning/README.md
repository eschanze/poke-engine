# Eval tuning (texel-style)

Outcome-tunes the hand-crafted eval weights in `src/genx/evaluate.rs`.
It uses self-play trajectories, logistic regression over the named features,
and an Elo gate before a candidate is adopted. The linear workflow requires
Python 3 + numpy; optional tree/MLP experiments require xgboost or PyTorch.

## Workflow

```powershell
# 1. Collect: sharded suite selfplay with trajectory dumps (from repo root)
python tools\run_suite.py --time-ms 250 --dump-prefix data\suite-shards\traj

# 2. Decorrelate: N evenly spaced positions per game; also remaps the
# per-shard game_id/state_index to be globally unique before merging
python tools\eval_tuning\subsample.py 12 sub12.jsonl data\suite-shards\traj-*.jsonl

# 3. Tune: logistic regression on outcomes (BCE at fixed K=80).
# --l2 pulls toward the current weights (prior); --l1 pulls toward zero
# and kills useless features exactly (veto-list friendly).
python tools\eval_tuning\tune.py --data sub12.jsonl --l2 1e-4 --out tuned.txt

# Optional distillation labels: re-search every subsampled position with a
# fixed-iteration teacher (default eval, single-threaded searches in
# parallel) and add a "root_value" field next to the game outcome. Training
# scripts consume it via --target root_value; the outcome field is kept, so
# the same file serves both label sets.
cargo run --release --no-default-features --features terastallization --bin relabel-root-value -- --input sub12.jsonl --output sub12-rv.jsonl --iterations 1000000 --threads 8

# Optional nonlinear residual: grouped-CV shallow boosted trees on top of
# the built-in or a fold-fitted L1 linear evaluator. The exported text model
# is loadable directly by selfplay.
python tools\eval_tuning\boosted.py --data sub12.jsonl --fit-l1 0.3 --out boosted.model

# Optional context model: extract each side separately, then fit an
# antisymmetric gated MLP. Its exported text model is loadable by selfplay.
cargo run --release --no-default-features --features terastallization --bin eval-pair-features -- --input sub12.jsonl --output sub12-pair.jsonl
python tools\eval_tuning\context_mlp.py --data sub12.jsonl --pair-data sub12-pair.jsonl --out context.model

# Experimental semantic features can be emitted without changing the
# production evaluator, then screened beyond a fold-local constrained base.
cargo run --release --no-default-features --features terastallization --bin eval-pair-features -- --input sub12.jsonl --output candidates.jsonl --experimental-candidates
python tools\eval_tuning\candidate_features.py --pair sub12.jsonl=candidates.jsonl

# Reproduce the adopted 500-game constrained fit. Bounds enforce sensible
# signs/ranges; the fixed old baseline is data/weights/eval-handcrafted-36.weights.
python tools\eval_tuning\constrained.py --data sub12.jsonl --pair-data sub12-pair.jsonl --out constrained.weights

# 4. Gate: tuned linear eval vs the historical default at iteration parity
# (Elo decides, not BCE). Clamp options take an explicit true/false value.
C:\Users\escha\.cargo\bin\cargo.exe run --release --no-default-features --features terastallization --bin selfplay -- -f data/datasets/battle-factory/no-ubers-states.txt --rounds 2 --a-eval-weights tuned.txt --a-eval-clamp false

# Or gate the boosted residual at equal wall-clock time:
python tools\run_suite.py --time-ms 250 --a-eval-trees boosted.model
```

Diagnostics:

- `sweep.py <dump.jsonl>` — regularization-strength x split-seed
  cross-validation: does any level beat the current weights on held-out
  starting states? `--mode l1` sweeps L1 instead (optionally on a fixed
  `--l2`) and reports how many weights die to exactly zero.
- `diag.py <dump.jsonl>` — baseline accuracy, calibration table, and a
  1-parameter global scale (temperature) fit.

New dumps carry a feature-schema hash and are rejected if their positional
layout does not match the scripts. For a known pre-schema dump from the
original 30-feature implementation, pass `--allow-legacy-schema` explicitly.

`tools/eval_tuning/data/` and suite outputs are gitignored (dumps are ~65 MB).
Dumps include the serialized
state per position, so new features can be re-extracted from old games.

Keep `FEATURE_NAMES` / `DEFAULT_WEIGHTS` in `tune.py` in sync with
`EVAL_FEATURE_NAMES` / `DEFAULT_EVAL_WEIGHTS` in `src/genx/evaluate.rs`;
trajectory schema hash and the Rust weight-file parser are strict, so drift
fails loudly at load time.

## Phase A result (2026-07-07)

The hand-guessed constants survived tuning: the 30-weight fit does not
generalize past the 102-matchup data ceiling, and the one robust finding
(all weights x 0.68, i.e. outcome-calibrated K≈117 vs the search's K=80)
gated Elo-neutral. Nothing adopted. See WORKLOG for the full numbers.

## 500-game metagame result (2026-07-18)

Shallow boosted trees, a tiny odd residual MLP, and a context-gated MLP were
tested with grouped validation. The nonlinear models added little beyond a
regularized linear refit while costing roughly 9–16% search throughput. They
remain available as experimental `--eval-trees` / `--eval-mlp` models.

## Distillation result (2026-07-19)

Relabeling the 504-game sub12 set with 1M-iteration teacher root values
(`relabel-root-value`) made the nested nonlinear gain over a fold-local L1
linear base fold-consistent for the first time: trees reached -0.0047 BCE
(depth 4, 790 trees) and the odd MLP -0.0043 (hidden 64), versus noise-level
gains on outcome labels. Label noise, not the 36-feature class, was the
nonlinear bottleneck at this data scale — but the gain is still small and the
required capacity expensive, so nothing was gated or adopted. See the diary
for full numbers.

Distilling the same root values into the *linear* weights does **not** help
(`distill_linear.py`, 2026-07-21): fit on root_value scores ~0.028 BCE worse
on held-out outcomes than fitting on outcomes directly, robust across L1. The
value search adds over the leaf eval is nonlinear, so only the nonlinear
residual can use it; for the linear class, unbiased outcomes remain the better
target. Keep tuning linear weights on outcomes. Note: root_value labels are
compressed toward 0.5, so any outcome-scored comparison must refit a global
temperature first (`distill_linear.py` does this automatically).

The winner was a projected L1 logistic fit of the 40 existing features (schema since trimmed to 36).
Semantic sign/range constraints removed pathological small-data coefficients
and improved mean grouped five-fold BCE by **0.1155** versus the handcrafted
evaluator. On 240 games from held-out starting positions at equal 250 ms, it
scored **61.7%** (134/78/28), **+82.6 Elo** with a 95% interval of
**[+38.5, +129.5]**. It is now `DEFAULT_EVAL_WEIGHTS`, with the per-mon clamp
disabled. Reproduce the old evaluator with `data/weights/eval-handcrafted-36.weights`
and `--eval-clamp true`.

## Structured candidate result (2026-07-21)

Ten matchup/leaf candidates and five HP-distribution candidates were
re-extracted from 945 decisive games. Exact entry-hazard burden, matchup
concentration, answer fragility, and raw switching mobility did not survive
outcome validation. `BENCH_HP` and `WISH_RECOVERY` did: together they improved
starting-matchup-grouped five-fold BCE by **0.00030** beyond a fold-local
constrained base, with the same positive coefficient signs when trained on
either independent ~500-game suite and scored on the other. Fixed candidate
weights also improved the adopted evaluator on both full suites (-0.00013 and
-0.00039 BCE).

The equal-100-ms smoke gate on 96 untouched games was neutral: 45/46/5,
49.5%, **-3.6 Elo [-74.2, +66.7]**. The features remain in the schema but are
zero by default; `data/weights/eval-bench-wish-38.weights` is the gated candidate.
This follows the project rule that offline gains do not bypass an Elo gate.
