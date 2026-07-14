#!/usr/bin/env python3
"""Convert compact Pokémon Showdown Battle Factory teams to poke-engine states.

The compact team format does not contain species types, base stats, weights, or
move PP.  Build a small local Dex snapshot once from Pokémon Showdown's public
``pokedex.ts`` and ``moves.ts`` files, then rerun this tool offline.

Example:
    python tools/convert_battle_factory_teams.py --build-dex \
        --pokedex-ts C:\\path\\to\\pokedex.ts --moves-ts C:\\path\\to\\moves.ts
    python tools/convert_battle_factory_teams.py

The resulting states pair adjacent source rows (1 vs 2, 3 vs 4, ...).  They
are immediately usable turn-one states: each side leads with its first listed
Pokémon, all Pokémon have full HP and PP, and the field is clear.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "gen9-battle-factory-no-ubers.txt"
DEFAULT_OUTPUT = ROOT / "data" / "gen9-battle-factory-no-ubers-states.txt"
DEFAULT_DEX = ROOT / "data" / "gen9-battle-factory-no-ubers-dex.json"

NATURE_EFFECTS = {
    "ADAMANT": ("ATK", "SPA"),
    "BOLD": ("DEF", "ATK"),
    "CALM": ("SPD", "ATK"),
    "CAREFUL": ("SPD", "SPA"),
    "HASTY": ("SPE", "DEF"),
    "IMPISH": ("DEF", "SPA"),
    "JOLLY": ("SPE", "SPA"),
    "MODEST": ("SPA", "ATK"),
    "NAIVE": ("SPE", "SPD"),
    "QUIET": ("SPA", "SPE"),
    "RELAXED": ("DEF", "SPE"),
    "SASSY": ("SPD", "SPE"),
    "TIMID": ("SPE", "ATK"),
}

# These items are legal in the input but not represented by poke-engine's
# Gen 9 Items enum.  Its deserializer maps unknown items to UNKNOWNITEM too;
# write that explicit value so the output is canonical when reserialized.
ITEM_SUBSTITUTIONS = {
    "KEEBERRY": "UNKNOWNITEM",
    "STICKYBARB": "UNKNOWNITEM",
}


def to_id(value: str) -> str:
    """Match Pokémon Showdown's ID convention and poke-engine enum names."""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


@dataclass(frozen=True)
class PackedPokemon:
    species: str
    item: str
    ability: str
    moves: tuple[str, str, str, str]
    nature: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int]
    level: int
    tera_type: str


def parse_six_values(value: str, default: int, description: str) -> tuple[int, int, int, int, int, int]:
    parts = value.split(",") if value else [""] * 6
    if len(parts) != 6:
        raise ValueError(f"{description} must have six comma-separated values, got {value!r}")
    return tuple(int(part) if part else default for part in parts)  # type: ignore[return-value]


def parse_pokemon(packed: str, line_number: int, slot: int) -> PackedPokemon:
    fields = packed.split("|")
    if len(fields) != 12:
        raise ValueError(
            f"line {line_number}, Pokémon {slot}: expected 12 packed fields, got {len(fields)}"
        )

    nickname, species, item, ability, moves, nature, evs, _gender, ivs, _shiny, level, extra = fields
    move_ids = tuple(to_id(move) for move in moves.split(",") if move)
    if len(move_ids) != 4:
        raise ValueError(
            f"line {line_number}, Pokémon {slot}: expected exactly four moves, got {moves!r}"
        )

    extra_fields = extra.split(",")
    tera_type = extra_fields[5] if len(extra_fields) > 5 else ""
    if not tera_type:
        raise ValueError(f"line {line_number}, Pokémon {slot}: missing Tera type")

    return PackedPokemon(
        species=to_id(species or nickname),
        item=to_id(item) if item else "NONE",
        ability=to_id(ability) if ability else "NONE",
        moves=move_ids,  # type: ignore[arg-type]
        nature=to_id(nature) if nature else "SERIOUS",
        evs=parse_six_values(evs, 0, "EVs"),
        ivs=parse_six_values(ivs, 31, "IVs"),
        level=int(level) if level else 100,
        tera_type=to_id(tera_type),
    )


def parse_teams(path: Path) -> list[list[PackedPokemon]]:
    teams: list[list[PackedPokemon]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        packed_team = line.split("]")
        if len(packed_team) != 6:
            raise ValueError(f"line {line_number}: expected six Pokémon, got {len(packed_team)}")
        teams.append(
            [parse_pokemon(packed, line_number, slot) for slot, packed in enumerate(packed_team, 1)]
        )

    if len(teams) % 2:
        raise ValueError(f"expected an even number of teams, got {len(teams)}")
    return teams


def ts_objects(source: str) -> Iterable[tuple[str, str]]:
    """Yield top-level key/object pairs from a Showdown TS data table."""
    lines = source.splitlines(keepends=True)
    start = re.compile(r"^\t([a-z0-9]+): \{")
    line_index = 0
    while line_index < len(lines):
        match = start.match(lines[line_index])
        if not match:
            line_index += 1
            continue

        key = match.group(1).upper()
        object_lines = [lines[line_index]]
        depth = lines[line_index].count("{") - lines[line_index].count("}")
        line_index += 1
        while depth > 0 and line_index < len(lines):
            line = lines[line_index]
            object_lines.append(line)
            depth += line.count("{") - line.count("}")
            line_index += 1
        if depth:
            raise ValueError(f"unterminated Pokémon Showdown data entry {key}")
        yield key, "".join(object_lines)


def build_dex(teams: list[list[PackedPokemon]], pokedex_path: Path, moves_path: Path) -> dict[str, object]:
    required_species = {pokemon.species for team in teams for pokemon in team}
    required_moves = {move for team in teams for pokemon in team for move in pokemon.moves}

    species: dict[str, dict[str, object]] = {}
    for key, body in ts_objects(pokedex_path.read_text(encoding="utf-8")):
        if key not in required_species:
            continue
        type_match = re.search(r"types:\s*\[([^\]]+)\]", body)
        weight_match = re.search(r"weightkg:\s*([0-9]+(?:\.[0-9]+)?)", body)
        if not type_match or not weight_match:
            raise ValueError(f"incomplete Pokédex data for {key}")
        types = re.findall(r'"([^"]+)"', type_match.group(1))
        stats = []
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            stat_match = re.search(rf"\b{stat}:\s*(\d+)", body)
            if not stat_match:
                raise ValueError(f"missing {stat} base stat for {key}")
            stats.append(int(stat_match.group(1)))
        species[key] = {
            "types": [to_id(pokemon_type) for pokemon_type in types],
            "base_stats": stats,
            "weight_kg": float(weight_match.group(1)),
        }

    moves: dict[str, int] = {}
    for key, body in ts_objects(moves_path.read_text(encoding="utf-8")):
        if key not in required_moves:
            continue
        pp_match = re.search(r"\bpp:\s*(\d+)", body)
        if not pp_match:
            raise ValueError(f"missing PP data for {key}")
        moves[key] = int(pp_match.group(1))

    missing_species = sorted(required_species - species.keys())
    missing_moves = sorted(required_moves - moves.keys())
    if missing_species or missing_moves:
        errors = []
        if missing_species:
            errors.append(f"species: {', '.join(missing_species)}")
        if missing_moves:
            errors.append(f"moves: {', '.join(missing_moves)}")
        raise ValueError(f"missing Showdown data ({'; '.join(errors)})")

    return {
        "schema_version": 1,
        "source": "Pokemon Showdown data/pokedex.ts and data/moves.ts",
        "species": species,
        "moves": moves,
    }


def calculate_stats(pokemon: PackedPokemon, species: dict[str, object]) -> tuple[int, int, int, int, int, int]:
    base_stats = species["base_stats"]
    if not isinstance(base_stats, list) or len(base_stats) != 6:
        raise ValueError(f"invalid base stats for {pokemon.species}")

    raw = [
        ((2 * int(base) + iv + ev // 4) * pokemon.level) // 100
        for base, iv, ev in zip(base_stats, pokemon.ivs, pokemon.evs)
    ]
    hp = raw[0] + pokemon.level + 10
    stats = {"ATK": raw[1] + 5, "DEF": raw[2] + 5, "SPA": raw[3] + 5, "SPD": raw[4] + 5, "SPE": raw[5] + 5}

    try:
        boosted, reduced = NATURE_EFFECTS[pokemon.nature]
    except KeyError as error:
        raise ValueError(f"unsupported nature {pokemon.nature}") from error
    stats[boosted] = stats[boosted] * 110 // 100
    stats[reduced] = stats[reduced] * 90 // 100
    return hp, stats["ATK"], stats["DEF"], stats["SPA"], stats["SPD"], stats["SPE"]


def format_number(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def serialize_pokemon(
    pokemon: PackedPokemon, dex_species: dict[str, dict[str, object]], dex_moves: dict[str, int], substituted_items: set[str]
) -> str:
    try:
        species = dex_species[pokemon.species]
    except KeyError as error:
        raise ValueError(f"missing Dex data for {pokemon.species}") from error

    raw_types = species["types"]
    if not isinstance(raw_types, list) or not raw_types:
        raise ValueError(f"invalid type data for {pokemon.species}")
    types = [str(pokemon_type) for pokemon_type in raw_types]
    if len(types) == 1:
        types.append("TYPELESS")
    elif len(types) != 2:
        raise ValueError(f"invalid type count for {pokemon.species}")

    item = ITEM_SUBSTITUTIONS.get(pokemon.item, pokemon.item)
    if item != pokemon.item:
        substituted_items.add(pokemon.item)

    stats = calculate_stats(pokemon, species)
    try:
        move_strings = [f"{move};false;{dex_moves[move] * 8 // 5}" for move in pokemon.moves]
    except KeyError as error:
        raise ValueError(f"missing PP data for {error.args[0]}") from error

    evs = ";".join(str(ev) for ev in pokemon.evs)
    return ",".join(
        [
            pokemon.species,
            str(pokemon.level),
            types[0],
            types[1],
            types[0],
            types[1],
            str(stats[0]),
            str(stats[0]),
            pokemon.ability,
            pokemon.ability,
            item,
            pokemon.nature,
            evs,
            *(str(stat) for stat in stats[1:]),
            "NONE",
            "0",
            "0",
            format_number(species["weight_kg"]),
            *move_strings,
            "false",
            pokemon.tera_type,
        ]
    )


def serialize_side(team: list[PackedPokemon], dex_species: dict[str, dict[str, object]], dex_moves: dict[str, int], substituted_items: set[str]) -> str:
    pokemon = [serialize_pokemon(member, dex_species, dex_moves, substituted_items) for member in team]
    fields = pokemon + [
        "0",  # active Pokémon: first listed team member
        ";".join("0" for _ in range(19)),  # side conditions
        "",  # volatile statuses
        ";".join("0" for _ in range(6)),  # volatile durations
        "0",  # substitute health
        "0", "0", "0", "0", "0", "0", "0",  # stat and accuracy/evasion boosts
        "0", "0",  # wish
        "0", "0",  # future sight
        "false", "NONE", "false", "false", "false", "move:none", "false",
    ]
    if len(fields) != 29:
        raise AssertionError(f"expected 29 Side fields, generated {len(fields)}")
    return "=".join(fields)


def serialize_states(teams: list[list[PackedPokemon]], dex: dict[str, object]) -> tuple[list[str], set[str]]:
    try:
        dex_species = dex["species"]
        dex_moves = dex["moves"]
    except KeyError as error:
        raise ValueError(f"invalid Dex snapshot: missing {error.args[0]}") from error
    if not isinstance(dex_species, dict) or not isinstance(dex_moves, dict):
        raise ValueError("invalid Dex snapshot")

    substituted_items: set[str] = set()
    states = []
    for first, second in zip(teams[::2], teams[1::2]):
        side_one = serialize_side(first, dex_species, dex_moves, substituted_items)  # type: ignore[arg-type]
        side_two = serialize_side(second, dex_species, dex_moves, substituted_items)  # type: ignore[arg-type]
        states.append(f"{side_one}/{side_two}/NONE;-1/NONE;0/false;0/false")
    return states, substituted_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"compact team input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"serialized state output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dex", type=Path, default=DEFAULT_DEX, help=f"local Dex snapshot (default: {DEFAULT_DEX})")
    parser.add_argument("--build-dex", action="store_true", help="create/update --dex from the supplied Showdown TypeScript files")
    parser.add_argument("--pokedex-ts", type=Path, help="Pokémon Showdown data/pokedex.ts; required with --build-dex")
    parser.add_argument("--moves-ts", type=Path, help="Pokémon Showdown data/moves.ts; required with --build-dex")
    args = parser.parse_args()

    teams = parse_teams(args.input)
    if args.build_dex:
        if not args.pokedex_ts or not args.moves_ts:
            parser.error("--build-dex requires both --pokedex-ts and --moves-ts")
        dex = build_dex(teams, args.pokedex_ts, args.moves_ts)
        args.dex.parent.mkdir(parents=True, exist_ok=True)
        args.dex.write_text(json.dumps(dex, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        dex = json.loads(args.dex.read_text(encoding="utf-8"))

    states, substituted_items = serialize_states(teams, dex)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(states) + "\n", encoding="utf-8")
    print(f"Converted {len(teams)} teams into {len(states)} paired battle states: {args.output}")
    if substituted_items:
        print(
            "Mapped unsupported engine items to UNKNOWNITEM: " + ", ".join(sorted(substituted_items)),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
