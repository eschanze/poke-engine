//! Allocation-free static matchup kernel used by the evaluator.

use super::abilities::Abilities;
use super::damage_calc::{calculate_damage_for_matchup, type_effectiveness_modifier, DamageRolls};
use super::items::Items;
use super::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::choices::{Choice, Choices, MoveCategory, MultiHitMove};
use crate::state::{Pokemon, PokemonIndex, PokemonStatus, PokemonType, Side, State};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DuelResult {
    Win,
    Loss,
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PairResult {
    pub damage: i16,
    pub hits: Option<i16>,
    pub priority: i8,
    pub speed: i16,
    pub move_slot: u8,
}

impl Default for PairResult {
    fn default() -> Self {
        Self {
            damage: 0,
            hits: None,
            priority: 0,
            speed: 0,
            move_slot: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatchupKernel {
    pub one_to_two: [PairResult; 36],
    pub two_to_one: [PairResult; 36],
    pub alive_one: [bool; 6],
    pub alive_two: [bool; 6],
    pub count_one: usize,
    pub count_two: usize,
}

const INDICES: [PokemonIndex; 6] = [
    PokemonIndex::P0,
    PokemonIndex::P1,
    PokemonIndex::P2,
    PokemonIndex::P3,
    PokemonIndex::P4,
    PokemonIndex::P5,
];

// 64K was the best point in the benchmark sweep: 128K lost locality and was
// slower, while smaller tables paid for more damage recalculations.
const CACHE_SIZE: usize = 65536;

#[derive(Clone, Copy)]
struct CacheEntry {
    key: u64,
    value: PairResult,
}

thread_local! {
    // Worker-local, fixed-capacity and lock-free. The one allocation happens
    // when a search thread first evaluates a state, never on subsequent calls.
    static PAIR_CACHE: RefCell<Box<[CacheEntry]>> = RefCell::new(
        vec![CacheEntry { key: 0, value: PairResult::default() }; CACHE_SIZE].into_boxed_slice()
    );
}

#[inline]
fn mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100000001b3);
}

fn participant_fingerprint(state: &State, side: &Side, pokemon: &Pokemon, active: bool) -> u64 {
    crate::prof_scope!(crate::prof::sec::MATCHUP_PAIR_KEY);
    let mut h = 0xcbf29ce484222325u64;
    mix(&mut h, pokemon.id as u64);
    mix(&mut h, pokemon.level as u64);
    mix(&mut h, pokemon.types.0 as u64);
    mix(&mut h, pokemon.types.1 as u64);
    mix(&mut h, pokemon.maxhp as u64);
    mix(&mut h, pokemon.ability as u64);
    mix(&mut h, pokemon.item as u64);
    mix(&mut h, pokemon.attack as u64);
    mix(&mut h, pokemon.defense as u64);
    mix(&mut h, pokemon.special_attack as u64);
    mix(&mut h, pokemon.special_defense as u64);
    mix(&mut h, pokemon.speed as u64);
    mix(&mut h, pokemon.status as u64);
    mix(&mut h, pokemon.weight_kg.to_bits() as u64);
    mix(&mut h, pokemon.terastallized as u64);
    mix(&mut h, pokemon.tera_type as u64);
    for mv in pokemon.moves.into_iter() {
        mix(&mut h, mv.id as u64);
        mix(&mut h, mv.pp as u64);
        mix(&mut h, mv.disabled as u64);
    }
    mix(&mut h, active as u64);
    if active {
        mix(&mut h, side.attack_boost as u64);
        mix(&mut h, side.defense_boost as u64);
        mix(&mut h, side.special_attack_boost as u64);
        mix(&mut h, side.special_defense_boost as u64);
        mix(&mut h, side.speed_boost as u64);
        mix(&mut h, side.volatile_statuses.0 as u64);
        mix(&mut h, (side.volatile_statuses.0 >> 64) as u64);
    }
    mix(&mut h, side.side_conditions.reflect as u64);
    mix(&mut h, side.side_conditions.light_screen as u64);
    mix(&mut h, side.side_conditions.aurora_veil as u64);
    mix(&mut h, side.side_conditions.tailwind as u64);
    mix(
        &mut h,
        side.pokemon.into_iter().filter(|p| p.hp == 0).count() as u64,
    );
    mix(&mut h, state.weather.weather_type as u64);
    mix(&mut h, state.weather.turns_remaining as u64);
    mix(&mut h, state.terrain.terrain_type as u64);
    mix(&mut h, state.terrain.turns_remaining as u64);
    h
}

#[derive(Clone, Copy, Default)]
struct HpSensitivity {
    attacker: bool,
    defender: bool,
}

fn hp_sensitivity(pokemon: &Pokemon) -> HpSensitivity {
    let mut result = HpSensitivity {
        attacker: matches!(
            pokemon.ability,
            Abilities::BLAZE
                | Abilities::TORRENT
                | Abilities::OVERGROW
                | Abilities::SWARM
                | Abilities::GALEWINGS
        ),
        defender: false,
    };
    for mv in pokemon.moves.into_iter() {
        result.attacker |= matches!(
            mv.id,
            Choices::REVERSAL | Choices::FLAIL | Choices::WATERSPOUT | Choices::ERUPTION
        );
        result.defender |= matches!(
            mv.id,
            Choices::HARDPRESS | Choices::BRINE | Choices::SUPERFANG | Choices::RUINATION
        );
    }
    result
}

#[inline]
fn hp_sensitive_defender(pokemon: &Pokemon) -> bool {
    matches!(
        pokemon.ability,
        Abilities::MULTISCALE | Abilities::SHADOWSHIELD
    )
}

fn pair_key(
    attacker_fingerprint: u64,
    defender_fingerprint: u64,
    attacker: &Pokemon,
    defender: &Pokemon,
    sensitivity: HpSensitivity,
) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    mix(&mut h, attacker_fingerprint);
    mix(&mut h, defender_fingerprint);
    if sensitivity.attacker {
        mix(&mut h, attacker.hp as u64);
    }
    if sensitivity.defender || hp_sensitive_defender(defender) {
        mix(&mut h, defender.hp as u64);
    }
    if h == 0 {
        1
    } else {
        h
    }
}

#[inline]
fn at(attacker: usize, defender: usize) -> usize {
    attacker * 6 + defender
}

fn compact_choice(source: &Choice) -> Choice {
    let mut choice = Choice::default();
    choice.move_id = source.move_id;
    choice.move_index = source.move_index;
    choice.move_type = source.move_type;
    choice.accuracy = source.accuracy;
    choice.category = source.category;
    choice.base_power = source.base_power;
    choice.priority = source.priority;
    choice.flags = source.flags.clone();
    choice.target = source.target.clone();
    choice.first_move = true;
    choice
}

fn normalize_choice(
    state: &State,
    attacking_side: &Side,
    attacker: &Pokemon,
    defending_side: &Side,
    defender: &Pokemon,
    source: &Choice,
    attacker_active: bool,
) -> Choice {
    let mut c = compact_choice(source);
    let hp_ratio = attacker.hp as f32 / attacker.maxhp.max(1) as f32;
    match c.move_id {
        Choices::REVERSAL => {
            c.base_power = if hp_ratio >= 0.688 {
                20.0
            } else if hp_ratio >= 0.354 {
                40.0
            } else if hp_ratio >= 0.208 {
                80.0
            } else if hp_ratio >= 0.104 {
                100.0
            } else if hp_ratio >= 0.042 {
                150.0
            } else {
                200.0
            }
        }
        Choices::HARDPRESS => {
            c.base_power = 100.0 * defender.hp as f32 / defender.maxhp.max(1) as f32
        }
        Choices::WATERSPOUT | Choices::ERUPTION => c.base_power = 150.0 * hp_ratio,
        Choices::FLAIL => {
            c.base_power = if hp_ratio >= 0.688 {
                20.0
            } else if hp_ratio >= 0.354 {
                40.0
            } else if hp_ratio >= 0.208 {
                80.0
            } else if hp_ratio >= 0.104 {
                100.0
            } else if hp_ratio >= 0.042 {
                150.0
            } else {
                200.0
            }
        }
        Choices::STOREDPOWER => {
            let boosts = if attacker_active {
                attacking_side.attack_boost.max(0)
                    + attacking_side.defense_boost.max(0)
                    + attacking_side.special_attack_boost.max(0)
                    + attacking_side.special_defense_boost.max(0)
                    + attacking_side.speed_boost.max(0)
            } else {
                0
            };
            c.base_power = 20.0 + boosts as f32 * 20.0;
        }
        Choices::LASTRESPECTS => {
            c.base_power = 50.0
                + 50.0
                    * attacking_side
                        .pokemon
                        .into_iter()
                        .filter(|p| p.hp == 0)
                        .count() as f32
        }
        Choices::LOWKICK | Choices::GRASSKNOT => {
            c.base_power = if defender.weight_kg >= 200.0 {
                120.0
            } else if defender.weight_kg >= 100.0 {
                100.0
            } else if defender.weight_kg >= 50.0 {
                80.0
            } else if defender.weight_kg >= 25.0 {
                60.0
            } else if defender.weight_kg >= 10.0 {
                40.0
            } else {
                20.0
            }
        }
        Choices::HEAVYSLAM | Choices::HEATCRASH => {
            let ratio = attacker.weight_kg / defender.weight_kg.max(0.1);
            c.base_power = if ratio >= 5.0 {
                120.0
            } else if ratio >= 4.0 {
                100.0
            } else if ratio >= 3.0 {
                80.0
            } else if ratio >= 2.0 {
                60.0
            } else {
                40.0
            };
        }
        Choices::ACROBATICS if attacker.item == Items::NONE => c.base_power *= 2.0,
        Choices::KNOCKOFF if defender.item != Items::NONE => c.base_power *= 1.5,
        Choices::FACADE if attacker.status != PokemonStatus::NONE => c.base_power *= 2.0,
        Choices::HEX if defender.status != PokemonStatus::NONE => c.base_power *= 2.0,
        Choices::BRINE if defender.hp * 2 <= defender.maxhp => c.base_power *= 2.0,
        Choices::WEATHERBALL if state.weather.weather_type != Weather::NONE => {
            c.base_power *= 2.0;
            c.move_type = match state.weather.weather_type {
                Weather::SUN | Weather::HARSHSUN => PokemonType::FIRE,
                Weather::RAIN | Weather::HEAVYRAIN => PokemonType::WATER,
                Weather::SAND => PokemonType::ROCK,
                Weather::HAIL | Weather::SNOW => PokemonType::ICE,
                _ => c.move_type,
            };
        }
        Choices::TERABLAST if attacker.terastallized => {
            c.move_type = attacker.tera_type;
            if attacker.attack > attacker.special_attack {
                c.category = MoveCategory::Physical;
            }
        }
        Choices::IVYCUDGEL => {
            c.move_type = match attacker.item {
                Items::WELLSPRINGMASK => PokemonType::WATER,
                Items::HEARTHFLAMEMASK => PokemonType::FIRE,
                Items::CORNERSTONEMASK => PokemonType::ROCK,
                _ => c.move_type,
            }
        }
        Choices::RAGINGBULL => {
            if defending_side.side_conditions.reflect > 0
                || defending_side.side_conditions.aurora_veil > 0
            {
                c.base_power *= 2.0;
            }
        }
        Choices::NIGHTSHADE | Choices::SEISMICTOSS => c.base_power = 0.0,
        Choices::SUPERFANG | Choices::RUINATION => c.base_power = 0.0,
        _ => {}
    }

    if attacker.ability == Abilities::PIXILATE && c.move_type == PokemonType::NORMAL {
        c.move_type = PokemonType::FAIRY;
        c.base_power *= 1.2;
    } else if attacker.ability == Abilities::REFRIGERATE && c.move_type == PokemonType::NORMAL {
        c.move_type = PokemonType::ICE;
        c.base_power *= 1.2;
    } else if attacker.ability == Abilities::AERILATE && c.move_type == PokemonType::NORMAL {
        c.move_type = PokemonType::FLYING;
        c.base_power *= 1.2;
    } else if attacker.ability == Abilities::GALVANIZE && c.move_type == PokemonType::NORMAL {
        c.move_type = PokemonType::ELECTRIC;
        c.base_power *= 1.2;
    } else if attacker.ability == Abilities::LIQUIDVOICE && c.flags.sound {
        c.move_type = PokemonType::WATER;
    }

    match attacker.ability {
        Abilities::TECHNICIAN if c.base_power <= 60.0 => c.base_power *= 1.5,
        Abilities::IRONFIST if c.flags.punch => c.base_power *= 1.2,
        Abilities::STRONGJAW if c.flags.bite => c.base_power *= 1.5,
        Abilities::SHARPNESS if c.flags.slicing => c.base_power *= 1.5,
        Abilities::MEGALAUNCHER if c.flags.pulse => c.base_power *= 1.5,
        Abilities::TOUGHCLAWS if c.flags.contact => c.base_power *= 1.3,
        Abilities::TINTEDLENS if type_effectiveness_modifier(&c.move_type, defender) < 1.0 => {
            c.base_power *= 2.0
        }
        Abilities::HUGEPOWER | Abilities::PUREPOWER if c.category == MoveCategory::Physical => {
            c.base_power *= 2.0
        }
        Abilities::GUTS
            if c.category == MoveCategory::Physical && attacker.status != PokemonStatus::NONE =>
        {
            c.base_power *= 3.0
        }
        Abilities::ADAPTABILITY
            if c.move_type == attacker.types.0 || c.move_type == attacker.types.1 =>
        {
            c.base_power *= 4.0 / 3.0
        }
        Abilities::SUPREMEOVERLORD => {
            c.base_power *= 1.0
                + 0.1
                    * attacking_side
                        .pokemon
                        .into_iter()
                        .filter(|p| p.hp == 0)
                        .count() as f32
        }
        Abilities::SWORDOFRUIN if c.category == MoveCategory::Physical => c.base_power /= 0.75,
        Abilities::BEADSOFRUIN if c.category == MoveCategory::Special => c.base_power /= 0.75,
        Abilities::BLAZE if hp_ratio <= 1.0 / 3.0 && c.move_type == PokemonType::FIRE => {
            c.base_power *= 1.5
        }
        Abilities::TORRENT if hp_ratio <= 1.0 / 3.0 && c.move_type == PokemonType::WATER => {
            c.base_power *= 1.5
        }
        Abilities::OVERGROW if hp_ratio <= 1.0 / 3.0 && c.move_type == PokemonType::GRASS => {
            c.base_power *= 1.5
        }
        Abilities::SWARM if hp_ratio <= 1.0 / 3.0 && c.move_type == PokemonType::BUG => {
            c.base_power *= 1.5
        }
        _ => {}
    }
    match defender.ability {
        Abilities::FURCOAT if c.category == MoveCategory::Physical => c.base_power *= 0.5,
        Abilities::ICESCALES if c.category == MoveCategory::Special => c.base_power *= 0.5,
        Abilities::THICKFAT
            if c.move_type == PokemonType::FIRE || c.move_type == PokemonType::ICE =>
        {
            c.base_power *= 0.5
        }
        Abilities::HEATPROOF if c.move_type == PokemonType::FIRE => c.base_power *= 0.5,
        Abilities::FLUFFY if c.flags.contact => c.base_power *= 0.5,
        Abilities::FLUFFY if c.move_type == PokemonType::FIRE => c.base_power *= 2.0,
        Abilities::MULTISCALE | Abilities::SHADOWSHIELD if defender.hp == defender.maxhp => {
            c.base_power *= 0.5
        }
        Abilities::FILTER | Abilities::SOLIDROCK | Abilities::PRISMARMOR
            if type_effectiveness_modifier(&c.move_type, defender) > 1.0 =>
        {
            c.base_power *= 0.75
        }
        Abilities::TABLETSOFRUIN
            if c.category == MoveCategory::Physical
                && attacker.ability != Abilities::TABLETSOFRUIN =>
        {
            c.base_power *= 0.75
        }
        Abilities::VESSELOFRUIN
            if c.category == MoveCategory::Special
                && attacker.ability != Abilities::VESSELOFRUIN =>
        {
            c.base_power *= 0.75
        }
        _ => {}
    }
    match attacker.item {
        Items::CHOICEBAND if c.category == MoveCategory::Physical => c.base_power *= 1.5,
        Items::CHOICESPECS if c.category == MoveCategory::Special => c.base_power *= 1.5,
        Items::LIFEORB if c.category != MoveCategory::Status => c.base_power *= 1.3,
        Items::EXPERTBELT if type_effectiveness_modifier(&c.move_type, defender) > 1.0 => {
            c.base_power *= 1.2
        }
        Items::MUSCLEBAND if c.category == MoveCategory::Physical => c.base_power *= 1.1,
        Items::WISEGLASSES if c.category == MoveCategory::Special => c.base_power *= 1.1,
        _ => {}
    }
    if defender.item == Items::EVIOLITE
        || (defender.item == Items::ASSAULTVEST && c.category == MoveCategory::Special)
    {
        c.base_power /= 1.5;
    }
    c
}

fn immune(attacker: &Pokemon, defender: &Pokemon, c: &Choice) -> bool {
    if attacker.ability != Abilities::MOLDBREAKER {
        match defender.ability {
            Abilities::LEVITATE
                if c.move_type == PokemonType::GROUND && c.move_id != Choices::THOUSANDARROWS =>
            {
                return true
            }
            Abilities::FLASHFIRE if c.move_type == PokemonType::FIRE => return true,
            Abilities::VOLTABSORB | Abilities::LIGHTNINGROD | Abilities::MOTORDRIVE
                if c.move_type == PokemonType::ELECTRIC =>
            {
                return true
            }
            Abilities::WATERABSORB | Abilities::STORMDRAIN | Abilities::DRYSKIN
                if c.move_type == PokemonType::WATER =>
            {
                return true
            }
            Abilities::SAPSIPPER if c.move_type == PokemonType::GRASS => return true,
            Abilities::EARTHEATER if c.move_type == PokemonType::GROUND => return true,
            Abilities::WONDERGUARD
                if type_effectiveness_modifier(&c.move_type, defender) <= 1.0 =>
            {
                return true
            }
            _ => {}
        }
    }
    false
}

fn hit_multiplier(attacker: &Pokemon, c: &Choice) -> i16 {
    match c.multi_hit() {
        MultiHitMove::None => 1,
        MultiHitMove::DoubleHit => 2,
        MultiHitMove::TripleHit | MultiHitMove::TripleAxel => 3,
        MultiHitMove::TwoToFiveHits => {
            if attacker.ability == Abilities::SKILLLINK {
                5
            } else if attacker.item == Items::LOADEDDICE {
                4
            } else {
                3
            }
        }
        MultiHitMove::PopulationBomb => {
            if attacker.item == Items::WIDELENS {
                9
            } else {
                6
            }
        }
    }
}

fn effective_priority(state: &State, pokemon: &Pokemon, c: &Choice) -> i8 {
    let mut priority = c.priority;
    if c.move_id == Choices::GRASSYGLIDE && state.terrain.terrain_type == Terrain::GRASSYTERRAIN {
        priority += 1;
    }
    if pokemon.ability == Abilities::GALEWINGS
        && c.move_type == PokemonType::FLYING
        && pokemon.hp == pokemon.maxhp
    {
        priority += 1;
    }
    priority
}

fn effective_speed(state: &State, side: &Side, pokemon: &Pokemon, active: bool) -> i16 {
    let mut speed = pokemon.speed as f32;
    if active {
        let b = side.speed_boost;
        speed = if b < 0 {
            (pokemon.speed * 2 / (2 - b as i16)) as f32
        } else {
            (pokemon.speed * (2 + b as i16) / 2) as f32
        };
    }
    match state.weather.weather_type {
        Weather::SUN | Weather::HARSHSUN if pokemon.ability == Abilities::CHLOROPHYLL => {
            speed *= 2.0
        }
        Weather::RAIN | Weather::HEAVYRAIN if pokemon.ability == Abilities::SWIFTSWIM => {
            speed *= 2.0
        }
        Weather::SAND if pokemon.ability == Abilities::SANDRUSH => speed *= 2.0,
        Weather::HAIL | Weather::SNOW if pokemon.ability == Abilities::SLUSHRUSH => speed *= 2.0,
        _ => {}
    }
    if pokemon.ability == Abilities::SURGESURFER
        && state.terrain.terrain_type == Terrain::ELECTRICTERRAIN
    {
        speed *= 2.0;
    }
    if pokemon.ability == Abilities::QUICKFEET && pokemon.status != PokemonStatus::NONE {
        speed *= 1.5;
    }
    if active
        && side
            .volatile_statuses
            .contains(&PokemonVolatileStatus::UNBURDEN)
    {
        speed *= 2.0;
    }
    if active
        && side
            .volatile_statuses
            .contains(&PokemonVolatileStatus::SLOWSTART)
    {
        speed *= 0.5;
    }
    if active
        && (side
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PROTOSYNTHESISSPE)
            || side
                .volatile_statuses
                .contains(&PokemonVolatileStatus::QUARKDRIVESPE))
    {
        speed *= 1.5;
    }
    if side.side_conditions.tailwind > 0 {
        speed *= 2.0;
    }
    match pokemon.item {
        Items::IRONBALL => speed *= 0.5,
        Items::CHOICESCARF => speed *= 1.5,
        _ => {}
    }
    if pokemon.status == PokemonStatus::PARALYZE && pokemon.ability != Abilities::QUICKFEET {
        #[cfg(any(feature = "gen3", feature = "gen4", feature = "gen5", feature = "gen6"))]
        {
            speed *= 0.25;
        }
        #[cfg(any(feature = "gen7", feature = "gen8", feature = "gen9"))]
        {
            speed *= 0.5;
        }
    }
    speed as i16
}

fn ordered_pair_uncached(
    state: &State,
    a_side: &Side,
    attacker: &Pokemon,
    a_active: bool,
    d_side: &Side,
    defender: &Pokemon,
    d_active: bool,
) -> PairResult {
    crate::prof_scope!(crate::prof::sec::MATCHUP_PAIR_COMPUTE);
    let mut best = PairResult {
        speed: effective_speed(state, a_side, attacker, a_active),
        ..PairResult::default()
    };
    for (slot, mv) in attacker.moves.into_iter().enumerate() {
        if mv.disabled
            || mv.pp <= 0
            || mv.choice.category == MoveCategory::Status
            || mv.choice.category == MoveCategory::Switch
        {
            continue;
        }
        let c = normalize_choice(
            state, a_side, attacker, d_side, defender, &mv.choice, a_active,
        );
        let priority = effective_priority(state, attacker, &c);
        let damage = if immune(attacker, defender, &c) {
            0
        } else if matches!(c.move_id, Choices::NIGHTSHADE | Choices::SEISMICTOSS) {
            attacker.level as i16
        } else if matches!(c.move_id, Choices::SUPERFANG | Choices::RUINATION) {
            defender.hp / 2
        } else {
            {
                crate::prof_scope!(crate::prof::sec::MATCHUP_DAMAGE);
                calculate_damage_for_matchup(
                    state,
                    a_side,
                    attacker,
                    a_active,
                    d_side,
                    defender,
                    d_active,
                    &c,
                    DamageRolls::Average,
                )
                .map(|x| x.0.max(0) * hit_multiplier(attacker, &c))
                .unwrap_or(0)
            }
        };
        if damage > best.damage
            || (damage == best.damage
                && (priority > best.priority
                    || (priority == best.priority && slot < best.move_slot as usize)))
        {
            best.damage = damage;
            best.priority = priority;
            best.move_slot = slot as u8;
        }
    }
    best
}

#[inline]
fn with_hits(mut result: PairResult, defender_hp: i16) -> PairResult {
    result.hits = if result.damage > 0 {
        Some((defender_hp + result.damage - 1) / result.damage.max(1))
    } else {
        None
    };
    result
}

fn ordered_pair_keyed(
    cache: &mut [CacheEntry],
    state: &State,
    a_side: &Side,
    attacker: &Pokemon,
    a_active: bool,
    d_side: &Side,
    defender: &Pokemon,
    d_active: bool,
    attacker_fingerprint: u64,
    defender_fingerprint: u64,
    sensitivity: HpSensitivity,
) -> PairResult {
    let key = {
        crate::prof_scope!(crate::prof::sec::MATCHUP_PAIR_KEY);
        pair_key(
            attacker_fingerprint,
            defender_fingerprint,
            attacker,
            defender,
            sensitivity,
        )
    };
    let slot = key as usize & (CACHE_SIZE - 1);
    let cached = {
        crate::prof_scope!(crate::prof::sec::MATCHUP_CACHE_LOOKUP);
        let entry = cache[slot];
        (entry.key == key).then_some(entry.value)
    };
    if let Some(value) = cached {
        return with_hits(value, defender.hp);
    }
    let mut value = ordered_pair_uncached(
        state, a_side, attacker, a_active, d_side, defender, d_active,
    );
    value.hits = None;
    {
        crate::prof_scope!(crate::prof::sec::MATCHUP_CACHE_LOOKUP);
        cache[slot] = CacheEntry { key, value };
    }
    with_hits(value, defender.hp)
}

#[cfg(test)]
fn ordered_pair(
    state: &State,
    a_side: &Side,
    attacker: &Pokemon,
    a_active: bool,
    d_side: &Side,
    defender: &Pokemon,
    d_active: bool,
) -> PairResult {
    PAIR_CACHE.with(|cache| {
        ordered_pair_keyed(
            &mut *cache.borrow_mut(),
            state,
            a_side,
            attacker,
            a_active,
            d_side,
            defender,
            d_active,
            participant_fingerprint(state, a_side, attacker, a_active),
            participant_fingerprint(state, d_side, defender, d_active),
            hp_sensitivity(attacker),
        )
    })
}

pub(crate) fn moves_before(state: &State, first: PairResult, second: PairResult) -> Option<bool> {
    if first.priority != second.priority {
        return Some(first.priority > second.priority);
    }
    if first.speed == second.speed {
        return None;
    }
    Some(if state.trick_room.active {
        first.speed < second.speed
    } else {
        first.speed > second.speed
    })
}

pub(crate) fn duel(state: &State, first: PairResult, second: PairResult) -> DuelResult {
    match (first.hits, second.hits) {
        (Some(a), Some(b)) if a < b => DuelResult::Win,
        (Some(a), Some(b)) if a > b => DuelResult::Loss,
        (Some(_), Some(_)) => match moves_before(state, first, second) {
            Some(true) => DuelResult::Win,
            Some(false) => DuelResult::Loss,
            None => DuelResult::Draw,
        },
        (Some(_), None) => DuelResult::Win,
        (None, Some(_)) => DuelResult::Loss,
        (None, None) => DuelResult::Draw,
    }
}

impl MatchupKernel {
    pub fn new(state: &State) -> Self {
        let mut k = Self {
            one_to_two: [PairResult::default(); 36],
            two_to_one: [PairResult::default(); 36],
            alive_one: [false; 6],
            alive_two: [false; 6],
            count_one: 0,
            count_two: 0,
        };
        for i in 0..6 {
            k.alive_one[i] = state.side_one.pokemon[INDICES[i]].hp > 0;
            k.alive_two[i] = state.side_two.pokemon[INDICES[i]].hp > 0;
            k.count_one += k.alive_one[i] as usize;
            k.count_two += k.alive_two[i] as usize;
        }
        let mut fingerprints_one = [0u64; 6];
        let mut fingerprints_two = [0u64; 6];
        let mut sensitivity_one = [HpSensitivity::default(); 6];
        let mut sensitivity_two = [HpSensitivity::default(); 6];
        for i in 0..6 {
            if k.alive_one[i] {
                let pokemon = &state.side_one.pokemon[INDICES[i]];
                fingerprints_one[i] = participant_fingerprint(
                    state,
                    &state.side_one,
                    pokemon,
                    INDICES[i] == state.side_one.active_index,
                );
                sensitivity_one[i] = hp_sensitivity(pokemon);
            }
            if k.alive_two[i] {
                let pokemon = &state.side_two.pokemon[INDICES[i]];
                fingerprints_two[i] = participant_fingerprint(
                    state,
                    &state.side_two,
                    pokemon,
                    INDICES[i] == state.side_two.active_index,
                );
                sensitivity_two[i] = hp_sensitivity(pokemon);
            }
        }
        PAIR_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            for i in 0..6 {
                for j in 0..6 {
                    if k.alive_one[i] && k.alive_two[j] {
                        let one = &state.side_one.pokemon[INDICES[i]];
                        let two = &state.side_two.pokemon[INDICES[j]];
                        k.one_to_two[at(i, j)] = ordered_pair_keyed(
                            &mut cache[..],
                            state,
                            &state.side_one,
                            one,
                            INDICES[i] == state.side_one.active_index,
                            &state.side_two,
                            two,
                            INDICES[j] == state.side_two.active_index,
                            fingerprints_one[i],
                            fingerprints_two[j],
                            sensitivity_one[i],
                        );
                        k.two_to_one[at(j, i)] = ordered_pair_keyed(
                            &mut cache[..],
                            state,
                            &state.side_two,
                            two,
                            INDICES[j] == state.side_two.active_index,
                            &state.side_one,
                            one,
                            INDICES[i] == state.side_one.active_index,
                            fingerprints_two[j],
                            fingerprints_one[i],
                            sensitivity_two[j],
                        );
                    }
                }
            }
        });
        k
    }
    #[inline]
    pub fn one(&self, a: usize, d: usize) -> PairResult {
        self.one_to_two[at(a, d)]
    }
    #[inline]
    pub fn two(&self, a: usize, d: usize) -> PairResult {
        self.two_to_one[at(a, d)]
    }
    pub fn duel_one(&self, state: &State, one: usize, two: usize) -> DuelResult {
        duel(state, self.one(one, two), self.two(two, one))
    }
}

fn entry_hp(side: &Side, pokemon: &Pokemon) -> i16 {
    if pokemon.item == Items::HEAVYDUTYBOOTS || pokemon.ability == Abilities::MAGICGUARD {
        return pokemon.hp;
    }
    let mut hp = pokemon.hp;
    if side.side_conditions.stealth_rock > 0 {
        hp -= (pokemon.maxhp as f32 * type_effectiveness_modifier(&PokemonType::ROCK, pokemon)
            / 8.0) as i16;
    }
    if hp > 0 && side.side_conditions.spikes > 0 && pokemon.is_grounded() {
        hp -= pokemon.maxhp * side.side_conditions.spikes as i16 / 8;
    }
    hp
}

pub(crate) fn is_counter(
    state: &State,
    kernel: &MatchupKernel,
    threat_on_one: bool,
    threat: usize,
    candidate: usize,
) -> bool {
    let (a_side, b_side) = if threat_on_one {
        (&state.side_one, &state.side_two)
    } else {
        (&state.side_two, &state.side_one)
    };
    let a = &a_side.pokemon[INDICES[threat]];
    let b = &b_side.pokemon[INDICES[candidate]];
    let mut remaining_hp = entry_hp(b_side, b);
    if remaining_hp <= 0 {
        return false;
    }
    let mut attack = if threat_on_one {
        kernel.one(threat, candidate)
    } else {
        kernel.two(threat, candidate)
    };
    remaining_hp -= attack.damage;
    if remaining_hp <= 0 {
        return false;
    }
    let mut reply = if threat_on_one {
        kernel.two(candidate, threat)
    } else {
        kernel.one(candidate, threat)
    };
    attack.hits = if attack.damage > 0 {
        Some((remaining_hp + attack.damage - 1) / attack.damage)
    } else {
        None
    };
    reply.hits = if reply.damage > 0 {
        Some((a.hp + reply.damage - 1) / reply.damage)
    } else {
        None
    };
    duel(state, reply, attack) == DuelResult::Win
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choices::MOVES;
    use crate::state::PokemonMoveIndex;

    fn tackle_state() -> State {
        let mut state = State::default();
        let tackle = MOVES.get(&Choices::TACKLE).unwrap();
        state.side_one.pokemon[PokemonIndex::P0].moves.m0.choice = compact_choice(tackle);
        state.side_one.pokemon[PokemonIndex::P0].moves.m0.id = Choices::TACKLE;
        state.side_two.pokemon[PokemonIndex::P0].moves.m0.choice = compact_choice(tackle);
        state.side_two.pokemon[PokemonIndex::P0].moves.m0.id = Choices::TACKLE;
        for i in 1..6 {
            state.side_one.pokemon[INDICES[i]].hp = 0;
            state.side_two.pokemon[INDICES[i]].hp = 0;
        }
        state
    }

    #[test]
    fn active_damage_matches_battle_damage_api() {
        let state = tackle_state();
        let choice = &state.side_one.get_active_immutable().moves[&PokemonMoveIndex::M0].choice;
        let direct = super::super::damage_calc::calculate_damage(
            &state,
            &crate::state::SideReference::SideOne,
            choice,
            DamageRolls::Average,
        )
        .unwrap()
        .0;
        assert_eq!(direct, MatchupKernel::new(&state).one(0, 0).damage);
    }

    #[test]
    fn bench_does_not_inherit_active_attack_boost() {
        let mut state = tackle_state();
        state.side_one.pokemon[PokemonIndex::P1] = state.side_one.pokemon[PokemonIndex::P0].clone();
        state.side_one.pokemon[PokemonIndex::P1].hp = 100;
        state.side_one.attack_boost = 6;
        let kernel = MatchupKernel::new(&state);
        assert!(kernel.one(0, 0).damage > kernel.one(1, 0).damage);
    }

    #[test]
    fn true_speed_tie_draws() {
        let mut state = tackle_state();
        state.side_two.get_active().speed = state.side_one.get_active_immutable().speed;
        let kernel = MatchupKernel::new(&state);
        assert_eq!(DuelResult::Draw, kernel.duel_one(&state, 0, 0));
    }

    #[test]
    fn exact_one_hp_ko_threshold_changes_hits() {
        let mut state = tackle_state();
        let damage = MatchupKernel::new(&state).one(0, 0).damage;
        state.side_two.get_active().hp = damage;
        assert_eq!(Some(1), MatchupKernel::new(&state).one(0, 0).hits);
        state.side_two.get_active().hp = damage + 1;
        assert_eq!(Some(2), MatchupKernel::new(&state).one(0, 0).hits);
    }

    #[test]
    fn cached_and_uncached_pairs_match() {
        let mut state = tackle_state();
        for hp in [1, 31, 32, 33, 100] {
            state.side_two.get_active().hp = hp;
            for boost in [-2, 0, 3] {
                state.side_one.attack_boost = boost;
                let a = state.side_one.get_active_immutable();
                let d = state.side_two.get_active_immutable();
                let cached =
                    ordered_pair(&state, &state.side_one, a, true, &state.side_two, d, true);
                let uncached = with_hits(
                    ordered_pair_uncached(
                        &state,
                        &state.side_one,
                        a,
                        true,
                        &state.side_two,
                        d,
                        true,
                    ),
                    d.hp,
                );
                assert_eq!(cached, uncached);
            }
        }
    }

    #[test]
    fn cache_key_tracks_attacker_hp_for_eruption() {
        let mut state = tackle_state();
        let eruption = MOVES.get(&Choices::ERUPTION).unwrap();
        state.side_one.get_active().moves.m0.choice = compact_choice(eruption);
        state.side_one.get_active().moves.m0.id = Choices::ERUPTION;
        state.side_one.get_active().hp = state.side_one.get_active().maxhp;
        let full_hp_damage = MatchupKernel::new(&state).one(0, 0).damage;
        state.side_one.get_active().hp /= 2;
        let half_hp_damage = MatchupKernel::new(&state).one(0, 0).damage;
        assert!(full_hp_damage > half_hp_damage);
    }

    #[test]
    fn cache_key_tracks_target_hp_for_brine() {
        let mut state = tackle_state();
        let brine = MOVES.get(&Choices::BRINE).unwrap();
        state.side_one.get_active().moves.m0.choice = compact_choice(brine);
        state.side_one.get_active().moves.m0.id = Choices::BRINE;
        state.side_two.get_active().hp = state.side_two.get_active().maxhp;
        let full_hp_damage = MatchupKernel::new(&state).one(0, 0).damage;
        state.side_two.get_active().hp = state.side_two.get_active().maxhp / 2;
        let half_hp_damage = MatchupKernel::new(&state).one(0, 0).damage;
        assert!(half_hp_damage > full_hp_damage);
    }

    #[test]
    fn cache_key_tracks_full_hp_defensive_abilities() {
        let mut state = tackle_state();
        state.side_two.get_active().ability = Abilities::MULTISCALE;
        state.side_two.get_active().hp = state.side_two.get_active().maxhp;
        let full_hp_damage = MatchupKernel::new(&state).one(0, 0).damage;
        state.side_two.get_active().hp -= 1;
        let damaged_damage = MatchupKernel::new(&state).one(0, 0).damage;
        assert!(damaged_damage > full_hp_damage);
    }

    #[test]
    fn entry_hazards_honor_boots_grounding_and_magic_guard() {
        let mut state = tackle_state();
        let p = state.side_two.get_active().clone();
        state.side_two.side_conditions.stealth_rock = 1;
        state.side_two.side_conditions.spikes = 1;
        assert!(entry_hp(&state.side_two, &p) < p.hp);
        let mut protected = p.clone();
        protected.item = Items::HEAVYDUTYBOOTS;
        assert_eq!(protected.hp, entry_hp(&state.side_two, &protected));
        protected.item = Items::NONE;
        protected.ability = Abilities::MAGICGUARD;
        assert_eq!(protected.hp, entry_hp(&state.side_two, &protected));
        protected.ability = Abilities::LEVITATE;
        assert_eq!(
            protected.hp - protected.maxhp / 8,
            entry_hp(&state.side_two, &protected)
        );
    }
}
