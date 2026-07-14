"""Run a sharded selfplay suite in parallel and aggregate the results.

Launches one selfplay process per shard file (see make_matchup_suite.py) so a
12-core machine can play 12 games concurrently with single-threaded searches.
Aggregates every shard summary into one combined W/L/D, winrate, and Elo diff.

Example (overnight full suite, new-vs-old eval at 250 ms):

    python tools/run_suite.py --time-ms 250 \
        --b-eval-weights data/eval-old-30.weights \
        --dump-prefix suite-trajectories

Each shard writes its log to <shard>.log next to the shard file; pass
--dump-prefix to also dump per-shard trajectory JSONL for tuning.
"""

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

SUMMARY_RE = re.compile(
    r"games=(\d+) W/L/D=(\d+)/(\d+)/(\d+) points=([0-9.]+) winrate=([0-9.]+)%"
)


def elo(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", default="data/suite-shards")
    parser.add_argument("--selfplay", default="target/release/selfplay.exe")
    parser.add_argument("--time-ms", type=int, default=250)
    parser.add_argument("--search-threads", type=int, default=1,
                        help="MCTS threads per search (keep 1 when running many shards)")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="states per shard, 0 = all")
    parser.add_argument("--a-eval-weights", default=None)
    parser.add_argument("--b-eval-weights", default=None)
    parser.add_argument("--a-bench-scale", type=float, default=None)
    parser.add_argument("--b-bench-scale", type=float, default=None)
    parser.add_argument("--dump-prefix", default=None,
                        help="dump per-shard trajectory JSONL as <prefix>-<shard>.jsonl")
    args = parser.parse_args()

    shards = sorted(Path(args.shard_dir).glob("shard-*.txt"))
    if not shards:
        sys.exit(f"no shard files in {args.shard_dir}")

    selfplay = Path(args.selfplay).resolve()
    if not selfplay.exists():
        sys.exit(f"selfplay binary not found: {selfplay}")

    procs = []
    start = time.time()
    for shard in shards:
        # selfplay resolves relative -f paths against its own source dir, so
        # pass absolute paths
        cmd = [
            str(selfplay),
            "-f", str(shard.resolve()),
            "-l", str(args.limit),
            "--rounds", str(args.rounds),
            "--max-turns", str(args.max_turns),
            "--a-iterations", "0", "--a-time-ms", str(args.time_ms),
            "--a-threads", str(args.search_threads),
            "--b-iterations", "0", "--b-time-ms", str(args.time_ms),
            "--b-threads", str(args.search_threads),
        ]
        if args.a_eval_weights:
            cmd += ["--a-eval-weights", str(Path(args.a_eval_weights).resolve())]
        if args.b_eval_weights:
            cmd += ["--b-eval-weights", str(Path(args.b_eval_weights).resolve())]
        if args.a_bench_scale is not None:
            cmd += ["--a-bench-scale", str(args.a_bench_scale)]
        if args.b_bench_scale is not None:
            cmd += ["--b-bench-scale", str(args.b_bench_scale)]
        if args.dump_prefix:
            cmd += ["--dump-trajectories", f"{args.dump_prefix}-{shard.stem}.jsonl"]
        log = open(shard.with_suffix(".log"), "w")
        procs.append((shard, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"launched {shard.name} (pid {procs[-1][1].pid})")

    total = {"games": 0, "w": 0, "l": 0, "d": 0, "points": 0.0}
    failed = False
    for shard, proc, log in procs:
        code = proc.wait()
        log.close()
        text = shard.with_suffix(".log").read_text()
        match = SUMMARY_RE.search(text)
        if code != 0 or match is None:
            print(f"{shard.name}: FAILED (exit {code}), see {shard.with_suffix('.log')}")
            failed = True
            continue
        games, w, l, d, points, winrate = match.groups()
        print(f"{shard.name}: games={games} W/L/D={w}/{l}/{d} winrate={winrate}%")
        total["games"] += int(games)
        total["w"] += int(w)
        total["l"] += int(l)
        total["d"] += int(d)
        total["points"] += float(points)

    if total["games"] == 0:
        sys.exit("no results")
    p = total["points"] / total["games"]
    # normal-approx CI on the mean game score; paired games are not fully
    # independent, so treat the interval as slightly optimistic
    se = math.sqrt(max(p * (1 - p), 1e-12) / total["games"])
    lo, hi = elo(p - 1.96 * se), elo(p + 1.96 * se)
    hours = (time.time() - start) / 3600.0
    print("\n=== Combined ===")
    print(f"games={total['games']} W/L/D={total['w']}/{total['l']}/{total['d']} "
          f"score={100 * p:.1f}%")
    print(f"elo diff (A vs B): {elo(p):+.1f} [{lo:+.1f}, {hi:+.1f}] (95% CI)")
    print(f"wall time: {hours:.2f} h")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
