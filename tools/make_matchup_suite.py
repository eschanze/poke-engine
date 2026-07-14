"""Build a large, duplicate-free matchup suite from an existing states file.

The input file (e.g. data/gen9-battle-factory-no-ubers-states.txt) holds N
serialized states = 2N teams. This script re-pairs those teams into new
matchups using circular offsets: for each chosen offset r, team i plays team
(i + r) mod 2N. Properties of this design:

- every unordered pair is distinct across offsets (the circular distance of a
  pair identifies its offset, given offsets in 2..2N/2-1);
- offset >= 2 never reproduces the original pairings, which sit at distance 1
  (team 2k vs team 2k+1);
- no team plays itself;
- perfectly balanced: with K offsets each team appears in exactly 2K games,
  K times as side one and K times as side two.

Output: one combined states file plus round-robin shards for running several
single-threaded selfplay processes in parallel.
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/gen9-battle-factory-no-ubers-states.txt",
        help="source states file (side1/side2/weather/terrain/trickroom/preview per line)",
    )
    parser.add_argument(
        "--output",
        default="data/gen9-battle-factory-matchups-2000.txt",
        help="combined output states file",
    )
    parser.add_argument(
        "--shard-dir",
        default="data/suite-shards",
        help="directory for shard files (round-robin split of the output)",
    )
    parser.add_argument("--shards", type=int, default=12, help="number of shard files")
    parser.add_argument(
        "--offsets", type=int, default=10, help="number of circular offsets (games per team = 2x this)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for offset choice")
    args = parser.parse_args()

    lines = [ln.strip() for ln in Path(args.input).read_text().splitlines() if ln.strip()]
    teams: list[str] = []
    tails: set[str] = set()
    original_pairs: set[frozenset[int]] = set()
    for ln in lines:
        parts = ln.split("/")
        if len(parts) != 6:
            raise SystemExit(f"unexpected state format ({len(parts)} '/' fields): {ln[:80]}...")
        side_one, side_two = parts[0], parts[1]
        tails.add("/".join(parts[2:]))
        original_pairs.add(frozenset((len(teams), len(teams) + 1)))
        teams.append(side_one)
        teams.append(side_two)

    if len(tails) != 1:
        raise SystemExit(f"expected identical field tails across states, found {len(tails)}")
    tail = tails.pop()

    n = len(teams)
    distinct = len(set(teams))
    if distinct != n:
        raise SystemExit(f"expected all teams distinct, found {distinct} distinct of {n}")

    max_offset = n // 2 - 1
    if args.offsets > max_offset - 1:
        raise SystemExit(f"at most {max_offset - 1} offsets available for {n} teams")
    rng = random.Random(args.seed)
    offsets = sorted(rng.sample(range(2, max_offset + 1), args.offsets))

    out_lines: list[str] = []
    seen: set[frozenset[int]] = set(original_pairs)
    for r in offsets:
        for i in range(n):
            pair = frozenset((i, (i + r) % n))
            if pair in seen:
                raise SystemExit(f"duplicate pair generated at offset {r}, index {i}")
            seen.add(pair)
            out_lines.append(f"{teams[i]}/{teams[(i + r) % n]}/{tail}")

    # shuffle so shards and any --limit prefix are unbiased samples of the suite
    rng.shuffle(out_lines)

    Path(args.output).write_text("\n".join(out_lines) + "\n")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[str]] = [[] for _ in range(args.shards)]
    for idx, line in enumerate(out_lines):
        shards[idx % args.shards].append(line)
    for s, chunk in enumerate(shards):
        (shard_dir / f"shard-{s:02d}.txt").write_text("\n".join(chunk) + "\n")

    per_team = 2 * args.offsets
    print(f"teams: {n} (all distinct)")
    print(f"offsets: {offsets}")
    print(f"matchups: {len(out_lines)} (unique, originals excluded), {per_team} games per team")
    print(f"wrote {args.output} and {args.shards} shards in {args.shard_dir}/")


if __name__ == "__main__":
    main()
