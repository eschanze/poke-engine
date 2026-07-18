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

# Optional nonlinear residual: grouped-CV shallow boosted trees on top of
# the built-in or a fold-fitted L1 linear evaluator. The exported text model
# is loadable directly by selfplay.
python tools\eval_tuning\boosted.py --data sub12.jsonl --fit-l1 0.3 --out boosted.model

# Optional context model: extract each side separately, then fit an
# antisymmetric gated MLP. Its exported text model is loadable by selfplay.
cargo run --release --no-default-features --features terastallization --bin eval-pair-features -- sub12.jsonl sub12-pair.jsonl
python tools\eval_tuning\context_mlp.py --data sub12.jsonl --pair-data sub12-pair.jsonl --out context.model

# Reproduce the adopted 500-game constrained fit. Bounds enforce sensible
# signs/ranges; the fixed old baseline is data/eval-handcrafted-36.weights.
python tools\eval_tuning\constrained.py --data sub12.jsonl --pair-data sub12-pair.jsonl --out constrained.weights

# 4. Gate: tuned linear eval vs the historical default at iteration parity
# (Elo decides, not BCE). Clamp options take an explicit true/false value.
C:\Users\escha\.cargo\bin\cargo.exe run --release --no-default-features --features terastallization --bin selfplay -- -f gen9-battle-factory-no-ubers-states.txt --rounds 2 --a-eval-weights tuned.txt --a-eval-clamp false

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

The winner was a projected L1 logistic fit of the 40 existing features (schema since trimmed to 36).
Semantic sign/range constraints removed pathological small-data coefficients
and improved mean grouped five-fold BCE by **0.1155** versus the handcrafted
evaluator. On 240 games from held-out starting positions at equal 250 ms, it
scored **61.7%** (134/78/28), **+82.6 Elo** with a 95% interval of
**[+38.5, +129.5]**. It is now `DEFAULT_EVAL_WEIGHTS`, with the per-mon clamp
disabled. Reproduce the old evaluator with `data/eval-handcrafted-36.weights`
and `--eval-clamp true`.
