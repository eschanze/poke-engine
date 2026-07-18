# Eval tuning (texel-style)

Outcome-tunes the hand-crafted eval weights in `src/genx/evaluate.rs`.
It uses self-play trajectories, logistic regression over the named features,
and an Elo gate before a candidate is adopted. Requires Python 3 + numpy only.

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

# 4. Gate: tuned linear eval vs the historical default at iteration parity
# (Elo decides, not BCE). Clamp options take an explicit true/false value.
C:\Users\escha\.cargo\bin\cargo.exe run --release --no-default-features --features terastallization --bin selfplay -- -f gen9randombattle.txt --rounds 2 --a-eval-weights tuned.txt --a-eval-clamp false
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
