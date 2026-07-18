#!/usr/bin/env python3
"""Subsample trajectory JSONL dumps: N evenly spaced positions per game.

Adjacent positions in one game are highly correlated, so tuning on every
visited position mostly duplicates rows. Evenly spaced picks (first and
last position always included) cover opening/mid/endgame while cutting the
per-game count to N.

Sharded dumps restart game_id and state_index at 0, so merging them naively
collides: tune.py counts games by (path, game_id) and splits train/val by
state_index alone. Both ids are remapped here to be globally unique across
the input files.

Usage: python subsample.py N out.jsonl in1.jsonl [in2.jsonl ...]
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: subsample.py N out.jsonl in1.jsonl [in2.jsonl ...]")
    n = int(sys.argv[1])
    if n < 2:
        sys.exit("N must be at least 2 (first and last position are always kept)")
    out_path = Path(sys.argv[2])
    in_paths = [Path(p) for p in sys.argv[3:]]

    games = {}  # (file_idx, original game_id) -> list of records
    for file_idx, path in enumerate(sorted(in_paths)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (file_idx, rec["game_id"])
                rec["game_id"] = file_idx * 10000 + rec["game_id"]
                rec["state_index"] = file_idx * 1000 + rec["state_index"]
                games.setdefault(key, []).append(rec)

    total_in = sum(len(v) for v in games.values())
    kept = []
    for key in sorted(games):
        recs = sorted(games[key], key=lambda r: r["turn"])
        if len(recs) <= n:
            kept.extend(recs)
            continue
        idxs = sorted({round(i * (len(recs) - 1) / (n - 1)) for i in range(n)})
        kept.extend(recs[i] for i in idxs)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print(
        f"{len(games)} games, {total_in} positions in -> {len(kept)} out "
        f"({len(kept) / len(games):.1f}/game), wrote {out_path}"
    )


if __name__ == "__main__":
    main()
