# Data and self-play tools

These scripts turn Pokémon Showdown teams into serialized poke-engine battle
states, construct matchup schedules, and run sharded self-play suites. Run the
commands below from the repository root.

## Pokémon Showdown teams to battle states

The conversion has two stages. The first stage uses Pokémon Showdown's own
parser, so a local, built Pokémon Showdown checkout and Node.js are required.
By default the checkout is expected at `../pokemon-showdown`; use `--showdown`
to select another location.

Convert one or more normal Import/Export files:

```powershell
node tools/showdown_import_to_packed.js `
  --output packed.txt `
  team-a.txt team-b.txt
```

An input may also be a Showdown backup-style collection whose teams are
separated by `=== team name ===` headings:

```powershell
node tools/showdown_import_to_packed.js `
  --output packed.txt `
  data/datasets/gen9ou/sample-teams.txt
```

Each team must contain six Pokémon with four moves each. A missing Tera type
uses the species' primary type, matching Pokémon Showdown. The output contains
one canonical packed team per line.

Next, turn the packed teams into clean turn-one battle states. Adjacent pairing
is the default: lines 1–2 become one state, lines 3–4 another, and so on.

```powershell
python tools/convert_battle_factory_teams.py `
  --input packed.txt `
  --output states.txt `
  --dex data/datasets/gen9ou/sample-dex.json
```

The packed format does not include base stats, types, weights, or move PP. If
the selected Dex snapshot does not cover every set, build a tailored snapshot
from the Pokémon Showdown checkout while converting:

```powershell
python tools/convert_battle_factory_teams.py `
  --input packed.txt `
  --output states.txt `
  --dex dex.json `
  --build-dex `
  --pokedex-ts ..\pokemon-showdown\data\pokedex.ts `
  --moves-ts ..\pokemon-showdown\data\moves.ts
```

The resulting states lead with the first listed Pokémon, start with full HP
and PP, and have a clear field.

## Round-robin schedules

Pass `--pairing round-robin` to emit every unordered pair of input teams:

```powershell
python tools/convert_battle_factory_teams.py `
  --input packed.txt `
  --output round-robin-states.txt `
  --dex dex.json `
  --pairing round-robin
```

For `N` teams this writes `N × (N - 1) / 2` states. Side assignment is
balanced around the schedule; with an odd number of teams, every team appears
equally often as side one and side two.

The self-play harness plays two games per state per round, swapping the A/B
search configurations between sides:

```powershell
target\release\selfplay.exe `
  -f (Resolve-Path round-robin-states.txt) `
  --rounds 1 `
  --a-iterations 0 --a-time-ms 250 `
  --b-iterations 0 --b-time-ms 250
```

`selfplay` reports aggregate A/B configuration results; it does not currently
produce per-team standings.

## Larger sampled matchup suites

`make_matchup_suite.py` is different from a full round robin. It extracts the
teams from an existing states file and uses seeded circular offsets to create a
large, duplicate-free sample of new pairings, excluding the original pairs.
It also writes round-robin-distributed shard files for parallel self-play:

```powershell
python tools/make_matchup_suite.py `
  --input states.txt `
  --output sampled-matchups.txt `
  --shard-dir data/suite-shards `
  --shards 12 --offsets 10
```

Run those shards concurrently with `run_suite.py`; pass `--dump-prefix` when
trajectory JSONL is needed for evaluation tuning:

```powershell
python tools/run_suite.py `
  --shard-dir data/suite-shards `
  --time-ms 250 `
  --dump-prefix data/suite-shards/traj
```

See [eval_tuning/README.md](eval_tuning/README.md) for subsampling, fitting,
validation, and Elo-gating of those trajectories.
