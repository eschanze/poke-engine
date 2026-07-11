# Eval tuning (texel-style)

Outcome-tunes the hand-crafted eval weights in `src/genx/evaluate.rs`.
It uses self-play trajectories, logistic regression over the named features,
and an Elo gate before a candidate is adopted. Requires Python 3 + numpy only.

## Workflow

```powershell
# 1. Collect: mirror selfplay with trajectory dump (from repo root)
C:\Users\escha\.cargo\bin\cargo.exe run --release --no-default-features --features terastallization --bin selfplay -- -f gen9randombattle.txt --rounds 3 --dump-trajectories tools\eval_tuning\data\dump.jsonl

# 2. Tune: logistic regression on outcomes (BCE at fixed K=80)
python tools\eval_tuning\tune.py --data tools\eval_tuning\data\dump.jsonl --out tuned.txt

# 3. Gate: tuned linear eval vs the historical default at iteration parity
# (Elo decides, not BCE). Clamp options take an explicit true/false value.
C:\Users\escha\.cargo\bin\cargo.exe run --release --no-default-features --features terastallization --bin selfplay -- -f gen9randombattle.txt --rounds 2 --a-eval-weights tuned.txt --a-eval-clamp false
```

Diagnostics:

- `sweep.py <dump.jsonl>` — L2-strength x split-seed cross-validation:
  does any regularization level beat the current weights on held-out
  starting states?
- `diag.py <dump.jsonl>` — baseline accuracy, calibration table, and a
  1-parameter global scale (temperature) fit.

New dumps carry a feature-schema hash and are rejected if their positional
layout does not match the scripts. For a known pre-schema dump from the
original 30-feature implementation, pass `--allow-legacy-schema` explicitly.

`data/` is gitignored (dumps are ~65 MB). Dumps include the serialized
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
