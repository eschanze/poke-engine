use super::abilities::Abilities;
use super::items::Items;
use super::state::PokemonVolatileStatus;
use crate::choices::MoveCategory;
use crate::state::{Pokemon, PokemonStatus, Side, State};

// The eval is a dot product between a weight vector and a feature vector
// computed from the state (side one minus side two), so the weights can be
// texel-tuned on game outcomes (see EVAL_TUNING_PLAN.md). One optional
// nonlinearity: when EvalConfig::mon_clamp is on, each mon's hp/status/item
// subtotal is clamped at 0 before the alive bonus is added. The historical
// clamp remains the production default; tuning experiments can disable it
// per search when they need evaluate() to be exactly linear in the weights.

pub const NUM_EVAL_FEATURES: usize = 30;

// Feature indices. Grouped: per-mon clampable subtotal (HP..STATUS_ABILITY_BONUS
// plus ITEM), active-only terms, side conditions, hazards, tera.
pub mod feat {
    pub const ALIVE: usize = 0;
    pub const HP: usize = 1;
    pub const ITEM: usize = 2;
    pub const FROZEN: usize = 3;
    pub const ASLEEP: usize = 4;
    pub const PARALYZED: usize = 5;
    pub const TOXIC: usize = 6;
    pub const POISONED: usize = 7;
    pub const BURNED: usize = 8;
    pub const POISON_HEAL: usize = 9;
    pub const STATUS_ABILITY_BONUS: usize = 10;
    pub const ATTACK_BOOST: usize = 11;
    pub const DEFENSE_BOOST: usize = 12;
    pub const SPECIAL_ATTACK_BOOST: usize = 13;
    pub const SPECIAL_DEFENSE_BOOST: usize = 14;
    pub const SPEED_BOOST: usize = 15;
    pub const LEECH_SEED: usize = 16;
    pub const SUBSTITUTE: usize = 17;
    pub const CONFUSION: usize = 18;
    pub const REFLECT: usize = 19;
    pub const LIGHT_SCREEN: usize = 20;
    pub const AURORA_VEIL: usize = 21;
    pub const SAFE_GUARD: usize = 22;
    pub const TAILWIND: usize = 23;
    pub const HEALING_WISH: usize = 24;
    pub const STEALTH_ROCK: usize = 25;
    pub const SPIKES: usize = 26;
    pub const TOXIC_SPIKES: usize = 27;
    pub const STICKY_WEB: usize = 28;
    pub const USED_TERA: usize = 29;
}

pub const EVAL_FEATURE_NAMES: [&str; NUM_EVAL_FEATURES] = [
    "POKEMON_ALIVE",
    "POKEMON_HP",
    "POKEMON_ITEM",
    "POKEMON_FROZEN",
    "POKEMON_ASLEEP",
    "POKEMON_PARALYZED",
    "POKEMON_TOXIC",
    "POKEMON_POISONED",
    "POKEMON_BURNED",
    "POISON_HEAL",
    "STATUS_ABILITY_BONUS",
    "POKEMON_ATTACK_BOOST",
    "POKEMON_DEFENSE_BOOST",
    "POKEMON_SPECIAL_ATTACK_BOOST",
    "POKEMON_SPECIAL_DEFENSE_BOOST",
    "POKEMON_SPEED_BOOST",
    "LEECH_SEED",
    "SUBSTITUTE",
    "CONFUSION",
    "REFLECT",
    "LIGHT_SCREEN",
    "AURORA_VEIL",
    "SAFE_GUARD",
    "TAILWIND",
    "HEALING_WISH",
    "STEALTH_ROCK",
    "SPIKES",
    "TOXIC_SPIKES",
    "STICKY_WEB",
    "USED_TERA",
];

// The historical hand-picked constants, now the seed weights for tuning.
// BURNED's feature is a physical-move-count multiplier and the five boost
// weights multiply the fixed 13-entry boost table below; everything else
// multiplies a count or a 0/1 flag.
pub const DEFAULT_EVAL_WEIGHTS: [f32; NUM_EVAL_FEATURES] = [
    30.0,  // POKEMON_ALIVE
    100.0, // POKEMON_HP
    10.0,  // POKEMON_ITEM
    -40.0, // POKEMON_FROZEN
    -25.0, // POKEMON_ASLEEP
    -25.0, // POKEMON_PARALYZED
    -30.0, // POKEMON_TOXIC
    -10.0, // POKEMON_POISONED
    -25.0, // POKEMON_BURNED
    15.0,  // POISON_HEAL
    10.0,  // STATUS_ABILITY_BONUS
    30.0,  // POKEMON_ATTACK_BOOST
    15.0,  // POKEMON_DEFENSE_BOOST
    30.0,  // POKEMON_SPECIAL_ATTACK_BOOST
    15.0,  // POKEMON_SPECIAL_DEFENSE_BOOST
    30.0,  // POKEMON_SPEED_BOOST
    -30.0, // LEECH_SEED
    40.0,  // SUBSTITUTE
    -20.0, // CONFUSION
    20.0,  // REFLECT
    20.0,  // LIGHT_SCREEN
    40.0,  // AURORA_VEIL
    5.0,   // SAFE_GUARD
    7.0,   // TAILWIND
    30.0,  // HEALING_WISH
    -10.0, // STEALTH_ROCK
    -7.0,  // SPIKES
    -7.0,  // TOXIC_SPIKES
    -25.0, // STICKY_WEB
    -75.0, // USED_TERA
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalConfig {
    weights: &'static [f32; NUM_EVAL_FEATURES],
    mon_clamp: bool,
}

impl EvalConfig {
    pub const fn new(weights: &'static [f32; NUM_EVAL_FEATURES], mon_clamp: bool) -> EvalConfig {
        EvalConfig { weights, mon_clamp }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig::new(&DEFAULT_EVAL_WEIGHTS, true)
    }
}

/// Stable identifier for the positional feature vector written to trajectory
/// dumps. It changes automatically if a feature name or its position changes.
pub fn eval_feature_schema() -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for name in EVAL_FEATURE_NAMES {
        for byte in name.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

/// Parse a weights file: one `NAME value` pair per line, `#` starts a
/// comment. All NUM_EVAL_FEATURES names must be present exactly once.
pub fn parse_eval_weights(text: &str) -> Result<[f32; NUM_EVAL_FEATURES], String> {
    let mut weights = [0.0f32; NUM_EVAL_FEATURES];
    let mut seen = [false; NUM_EVAL_FEATURES];
    for (line_no, raw_line) in text.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = parts.next().unwrap();
        let value = parts
            .next()
            .ok_or_else(|| format!("line {}: missing value for {}", line_no + 1, name))?;
        if parts.next().is_some() {
            return Err(format!("line {}: trailing tokens", line_no + 1));
        }
        let idx = EVAL_FEATURE_NAMES
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| format!("line {}: unknown weight name {}", line_no + 1, name))?;
        if seen[idx] {
            return Err(format!("line {}: duplicate weight {}", line_no + 1, name));
        }
        let parsed = value
            .parse::<f32>()
            .map_err(|e| format!("line {}: bad value for {}: {}", line_no + 1, name, e))?;
        if !parsed.is_finite() {
            return Err(format!(
                "line {}: non-finite value for {}",
                line_no + 1,
                name
            ));
        }
        weights[idx] = parsed;
        seen[idx] = true;
    }
    if let Some(missing) = seen.iter().position(|s| !s) {
        return Err(format!("missing weight {}", EVAL_FEATURE_NAMES[missing]));
    }
    Ok(weights)
}

const POKEMON_BOOST_MULTIPLIER_6: f32 = 3.3;
const POKEMON_BOOST_MULTIPLIER_5: f32 = 3.15;
const POKEMON_BOOST_MULTIPLIER_4: f32 = 3.0;
const POKEMON_BOOST_MULTIPLIER_3: f32 = 2.5;
const POKEMON_BOOST_MULTIPLIER_2: f32 = 2.0;
const POKEMON_BOOST_MULTIPLIER_1: f32 = 1.0;
const POKEMON_BOOST_MULTIPLIER_0: f32 = 0.0;
const POKEMON_BOOST_MULTIPLIER_NEG_1: f32 = -1.0;
const POKEMON_BOOST_MULTIPLIER_NEG_2: f32 = -2.0;
const POKEMON_BOOST_MULTIPLIER_NEG_3: f32 = -2.5;
const POKEMON_BOOST_MULTIPLIER_NEG_4: f32 = -3.0;
const POKEMON_BOOST_MULTIPLIER_NEG_5: f32 = -3.15;
const POKEMON_BOOST_MULTIPLIER_NEG_6: f32 = -3.3;

fn get_boost_multiplier(boost: i8) -> f32 {
    match boost {
        6 => POKEMON_BOOST_MULTIPLIER_6,
        5 => POKEMON_BOOST_MULTIPLIER_5,
        4 => POKEMON_BOOST_MULTIPLIER_4,
        3 => POKEMON_BOOST_MULTIPLIER_3,
        2 => POKEMON_BOOST_MULTIPLIER_2,
        1 => POKEMON_BOOST_MULTIPLIER_1,
        0 => POKEMON_BOOST_MULTIPLIER_0,
        -1 => POKEMON_BOOST_MULTIPLIER_NEG_1,
        -2 => POKEMON_BOOST_MULTIPLIER_NEG_2,
        -3 => POKEMON_BOOST_MULTIPLIER_NEG_3,
        -4 => POKEMON_BOOST_MULTIPLIER_NEG_4,
        -5 => POKEMON_BOOST_MULTIPLIER_NEG_5,
        -6 => POKEMON_BOOST_MULTIPLIER_NEG_6,
        _ => panic!("Invalid boost value: {}", boost),
    }
}

/// burn is not as punishing in certain situations; the feature value is the
/// multiplier applied to the POKEMON_BURNED weight
fn burn_multiplier(pokemon: &Pokemon) -> f32 {
    // guts, marvel scale, quick feet will result in a positive evaluation
    match pokemon.ability {
        Abilities::GUTS | Abilities::MARVELSCALE | Abilities::QUICKFEET => return -2.0,
        _ => {}
    }

    let mut multiplier = 0.0;
    for mv in pokemon.moves.into_iter() {
        if mv.choice.category == MoveCategory::Physical {
            multiplier += 1.0;
        }
    }

    // don't make burn as punishing for special attackers
    if pokemon.special_attack > pokemon.attack {
        multiplier /= 2.0;
    }

    multiplier
}

/// which feature a poisoned/toxic'd mon contributes to, given its ability
fn poison_feature(pokemon: &Pokemon, base_feature: usize) -> usize {
    match pokemon.ability {
        Abilities::POISONHEAL => feat::POISON_HEAL,
        Abilities::GUTS
        | Abilities::MARVELSCALE
        | Abilities::QUICKFEET
        | Abilities::TOXICBOOST
        | Abilities::MAGICGUARD => feat::STATUS_ABILITY_BONUS,
        _ => base_feature,
    }
}

// The single source of truth for what the eval looks at. Both the score
// computation and the feature-vector extraction walk the state through this
// trait, so they cannot drift apart.
trait EvalSink {
    fn set_sign(&mut self, sign: f32);
    fn hp(&mut self, hp: f32, maxhp: f32);
    /// feature inside the per-mon clampable subtotal (hp, status, item)
    fn mon(&mut self, idx: usize, value: f32);
    /// close the current mon: clamp the subtotal, credit the alive bonus
    fn finish_mon(&mut self);
    /// feature outside the clamp
    fn global(&mut self, idx: usize, value: f32);
    /// preserve legacy arithmetic that accumulated hazards per Pokemon before
    /// adding that subtotal to the side's score
    fn start_global_group(&mut self);
    fn finish_global_group(&mut self);
}

struct ScoreSink<'a> {
    weights: &'a [f32; NUM_EVAL_FEATURES],
    clamp: bool,
    sign: f32,
    mon_subtotal: f32,
    global_subtotal: Option<f32>,
    total: f32,
}

impl EvalSink for ScoreSink<'_> {
    fn set_sign(&mut self, sign: f32) {
        self.sign = sign;
    }
    fn hp(&mut self, hp: f32, maxhp: f32) {
        // Keep the historical multiplication/division order exactly.
        self.mon_subtotal += self.weights[feat::HP] * hp / maxhp;
    }
    fn mon(&mut self, idx: usize, value: f32) {
        self.mon_subtotal += self.weights[idx] * value;
    }
    fn finish_mon(&mut self) {
        let mut subtotal = self.mon_subtotal;
        self.mon_subtotal = 0.0;
        // without this a low hp pokemon could get a negative score and
        // incentivize the other side to keep it alive
        if self.clamp && subtotal < 0.0 {
            subtotal = 0.0;
        }
        subtotal += self.weights[feat::ALIVE];
        self.total += self.sign * subtotal;
    }
    fn global(&mut self, idx: usize, value: f32) {
        let contribution = value * self.weights[idx];
        if let Some(subtotal) = self.global_subtotal.as_mut() {
            *subtotal += contribution;
        } else {
            self.total += self.sign * contribution;
        }
    }
    fn start_global_group(&mut self) {
        debug_assert!(self.global_subtotal.is_none());
        self.global_subtotal = Some(0.0);
    }
    fn finish_global_group(&mut self) {
        self.total += self.sign * self.global_subtotal.take().unwrap();
    }
}

struct FeatureSink {
    sign: f32,
    features: [f32; NUM_EVAL_FEATURES],
}

impl EvalSink for FeatureSink {
    fn set_sign(&mut self, sign: f32) {
        self.sign = sign;
    }
    fn hp(&mut self, hp: f32, maxhp: f32) {
        self.features[feat::HP] += self.sign * hp / maxhp;
    }
    fn mon(&mut self, idx: usize, value: f32) {
        self.features[idx] += self.sign * value;
    }
    fn finish_mon(&mut self) {
        self.features[feat::ALIVE] += self.sign;
    }
    fn global(&mut self, idx: usize, value: f32) {
        self.features[idx] += self.sign * value;
    }
    fn start_global_group(&mut self) {}
    fn finish_global_group(&mut self) {}
}

fn walk_mon<S: EvalSink>(pokemon: &Pokemon, sink: &mut S) {
    sink.hp(pokemon.hp as f32, pokemon.maxhp as f32);

    match pokemon.status {
        PokemonStatus::BURN => sink.mon(feat::BURNED, burn_multiplier(pokemon)),
        PokemonStatus::FREEZE => sink.mon(feat::FROZEN, 1.0),
        PokemonStatus::SLEEP => sink.mon(feat::ASLEEP, 1.0),
        PokemonStatus::PARALYZE => sink.mon(feat::PARALYZED, 1.0),
        PokemonStatus::TOXIC => sink.mon(poison_feature(pokemon, feat::TOXIC), 1.0),
        PokemonStatus::POISON => sink.mon(poison_feature(pokemon, feat::POISONED), 1.0),
        PokemonStatus::NONE => {}
    }

    if pokemon.item != Items::NONE {
        sink.mon(feat::ITEM, 1.0);
    }

    sink.finish_mon();
}

fn walk_hazards<S: EvalSink>(pokemon: &Pokemon, side: &Side, sink: &mut S) {
    sink.start_global_group();
    let pkmn_is_grounded = pokemon.is_grounded();
    if pokemon.item != Items::HEAVYDUTYBOOTS {
        if pokemon.ability != Abilities::MAGICGUARD {
            sink.global(feat::STEALTH_ROCK, side.side_conditions.stealth_rock as f32);
            if pkmn_is_grounded {
                sink.global(feat::SPIKES, side.side_conditions.spikes as f32);
                sink.global(feat::TOXIC_SPIKES, side.side_conditions.toxic_spikes as f32);
            }
        }
        if pkmn_is_grounded {
            sink.global(feat::STICKY_WEB, side.side_conditions.sticky_web as f32);
        }
    }
    sink.finish_global_group();
}

fn walk_side_pokemon<S: EvalSink>(side: &Side, sink: &mut S, sign: f32) {
    sink.set_sign(sign);

    let mut used_tera = false;
    let mut iter = side.pokemon.into_iter();
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            walk_mon(pkmn, sink);
            walk_hazards(pkmn, side, sink);
            if iter.pokemon_index == side.active_index {
                if side
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::LEECHSEED)
                {
                    sink.global(feat::LEECH_SEED, 1.0);
                }
                if side
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::SUBSTITUTE)
                {
                    sink.global(feat::SUBSTITUTE, 1.0);
                }
                if side
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::CONFUSION)
                {
                    sink.global(feat::CONFUSION, 1.0);
                }

                sink.global(feat::ATTACK_BOOST, get_boost_multiplier(side.attack_boost));
                sink.global(
                    feat::DEFENSE_BOOST,
                    get_boost_multiplier(side.defense_boost),
                );
                sink.global(
                    feat::SPECIAL_ATTACK_BOOST,
                    get_boost_multiplier(side.special_attack_boost),
                );
                sink.global(
                    feat::SPECIAL_DEFENSE_BOOST,
                    get_boost_multiplier(side.special_defense_boost),
                );
                sink.global(feat::SPEED_BOOST, get_boost_multiplier(side.speed_boost));
            }
        }
        if pkmn.terastallized {
            used_tera = true;
        }
    }
    if used_tera {
        sink.global(feat::USED_TERA, 1.0);
    }
}

fn walk_side_conditions<S: EvalSink>(side: &Side, sink: &mut S, sign: f32) {
    sink.set_sign(sign);
    sink.global(feat::REFLECT, side.side_conditions.reflect as f32);
    sink.global(feat::LIGHT_SCREEN, side.side_conditions.light_screen as f32);
    sink.global(feat::AURORA_VEIL, side.side_conditions.aurora_veil as f32);
    sink.global(feat::SAFE_GUARD, side.side_conditions.safeguard as f32);
    sink.global(feat::TAILWIND, side.side_conditions.tailwind as f32);
    sink.global(feat::HEALING_WISH, side.side_conditions.healing_wish as f32);
}

fn eval_walk<S: EvalSink>(state: &State, sink: &mut S) {
    walk_side_pokemon(&state.side_one, sink, 1.0);
    walk_side_pokemon(&state.side_two, sink, -1.0);
    walk_side_conditions(&state.side_one, sink, 1.0);
    walk_side_conditions(&state.side_two, sink, -1.0);
}

/// side-one-minus-side-two feature vector; `dot(weights, features)` equals
/// `evaluate_with_config()` when the per-mon clamp is off (the clamp is not linear)
pub fn eval_features(state: &State) -> [f32; NUM_EVAL_FEATURES] {
    let mut sink = FeatureSink {
        sign: 1.0,
        features: [0.0; NUM_EVAL_FEATURES],
    };
    eval_walk(state, &mut sink);
    sink.features
}

pub fn evaluate_with_config(state: &State, config: EvalConfig) -> f32 {
    let mut sink = ScoreSink {
        weights: config.weights,
        clamp: config.mon_clamp,
        sign: 1.0,
        mon_subtotal: 0.0,
        global_subtotal: None,
        total: 0.0,
    };
    eval_walk(state, &mut sink);
    sink.total
}

pub fn evaluate(state: &State) -> f32 {
    evaluate_with_config(state, EvalConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::PokemonIndex;

    // The pre-decomposition evaluate(), kept verbatim as the reference the
    // weighted walk must reproduce (constants inlined from the old file).
    mod reference {
        use super::super::*;

        const POKEMON_ALIVE: f32 = 30.0;
        const POKEMON_HP: f32 = 100.0;
        const USED_TERA: f32 = -75.0;

        const POKEMON_ATTACK_BOOST: f32 = 30.0;
        const POKEMON_DEFENSE_BOOST: f32 = 15.0;
        const POKEMON_SPECIAL_ATTACK_BOOST: f32 = 30.0;
        const POKEMON_SPECIAL_DEFENSE_BOOST: f32 = 15.0;
        const POKEMON_SPEED_BOOST: f32 = 30.0;

        const POKEMON_FROZEN: f32 = -40.0;
        const POKEMON_ASLEEP: f32 = -25.0;
        const POKEMON_PARALYZED: f32 = -25.0;
        const POKEMON_TOXIC: f32 = -30.0;
        const POKEMON_POISONED: f32 = -10.0;
        const POKEMON_BURNED: f32 = -25.0;

        const LEECH_SEED: f32 = -30.0;
        const SUBSTITUTE: f32 = 40.0;
        const CONFUSION: f32 = -20.0;

        const REFLECT: f32 = 20.0;
        const LIGHT_SCREEN: f32 = 20.0;
        const AURORA_VEIL: f32 = 40.0;
        const SAFE_GUARD: f32 = 5.0;
        const TAILWIND: f32 = 7.0;
        const HEALING_WISH: f32 = 30.0;

        const STEALTH_ROCK: f32 = -10.0;
        const SPIKES: f32 = -7.0;
        const TOXIC_SPIKES: f32 = -7.0;
        const STICKY_WEB: f32 = -25.0;

        fn evaluate_poison(pokemon: &Pokemon, base_score: f32) -> f32 {
            match pokemon.ability {
                Abilities::POISONHEAL => 15.0,
                Abilities::GUTS
                | Abilities::MARVELSCALE
                | Abilities::QUICKFEET
                | Abilities::TOXICBOOST
                | Abilities::MAGICGUARD => 10.0,
                _ => base_score,
            }
        }

        fn evaluate_burned(pokemon: &Pokemon) -> f32 {
            match pokemon.ability {
                Abilities::GUTS | Abilities::MARVELSCALE | Abilities::QUICKFEET => {
                    return -2.0 * POKEMON_BURNED
                }
                _ => {}
            }

            let mut multiplier = 0.0;
            for mv in pokemon.moves.into_iter() {
                if mv.choice.category == MoveCategory::Physical {
                    multiplier += 1.0;
                }
            }

            if pokemon.special_attack > pokemon.attack {
                multiplier /= 2.0;
            }

            multiplier * POKEMON_BURNED
        }

        fn evaluate_hazards(pokemon: &Pokemon, side: &Side) -> f32 {
            let mut score = 0.0;
            let pkmn_is_grounded = pokemon.is_grounded();
            if pokemon.item != Items::HEAVYDUTYBOOTS {
                if pokemon.ability != Abilities::MAGICGUARD {
                    score += side.side_conditions.stealth_rock as f32 * STEALTH_ROCK;
                    if pkmn_is_grounded {
                        score += side.side_conditions.spikes as f32 * SPIKES;
                        score += side.side_conditions.toxic_spikes as f32 * TOXIC_SPIKES;
                    }
                }
                if pkmn_is_grounded {
                    score += side.side_conditions.sticky_web as f32 * STICKY_WEB;
                }
            }

            score
        }

        fn evaluate_pokemon(pokemon: &Pokemon) -> f32 {
            let mut score = 0.0;
            score += POKEMON_HP * pokemon.hp as f32 / pokemon.maxhp as f32;

            match pokemon.status {
                PokemonStatus::BURN => score += evaluate_burned(pokemon),
                PokemonStatus::FREEZE => score += POKEMON_FROZEN,
                PokemonStatus::SLEEP => score += POKEMON_ASLEEP,
                PokemonStatus::PARALYZE => score += POKEMON_PARALYZED,
                PokemonStatus::TOXIC => score += evaluate_poison(pokemon, POKEMON_TOXIC),
                PokemonStatus::POISON => score += evaluate_poison(pokemon, POKEMON_POISONED),
                PokemonStatus::NONE => {}
            }

            if pokemon.item != Items::NONE {
                score += 10.0;
            }

            if score < 0.0 {
                score = 0.0;
            }

            score += POKEMON_ALIVE;

            score
        }

        pub fn evaluate(state: &State) -> f32 {
            let mut score = 0.0;

            let mut iter = state.side_one.pokemon.into_iter();
            let mut s1_used_tera = false;
            while let Some(pkmn) = iter.next() {
                if pkmn.hp > 0 {
                    score += evaluate_pokemon(pkmn);
                    score += evaluate_hazards(pkmn, &state.side_one);
                    if iter.pokemon_index == state.side_one.active_index {
                        if state
                            .side_one
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::LEECHSEED)
                        {
                            score += LEECH_SEED;
                        }
                        if state
                            .side_one
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::SUBSTITUTE)
                        {
                            score += SUBSTITUTE;
                        }
                        if state
                            .side_one
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::CONFUSION)
                        {
                            score += CONFUSION;
                        }

                        score += get_boost_multiplier(state.side_one.attack_boost)
                            * POKEMON_ATTACK_BOOST;
                        score += get_boost_multiplier(state.side_one.defense_boost)
                            * POKEMON_DEFENSE_BOOST;
                        score += get_boost_multiplier(state.side_one.special_attack_boost)
                            * POKEMON_SPECIAL_ATTACK_BOOST;
                        score += get_boost_multiplier(state.side_one.special_defense_boost)
                            * POKEMON_SPECIAL_DEFENSE_BOOST;
                        score +=
                            get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
                    }
                }
                if pkmn.terastallized {
                    s1_used_tera = true;
                }
            }
            if s1_used_tera {
                score += USED_TERA;
            }
            let mut iter = state.side_two.pokemon.into_iter();
            let mut s2_used_tera = false;
            while let Some(pkmn) = iter.next() {
                if pkmn.hp > 0 {
                    score -= evaluate_pokemon(pkmn);
                    score -= evaluate_hazards(pkmn, &state.side_two);

                    if iter.pokemon_index == state.side_two.active_index {
                        if state
                            .side_two
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::LEECHSEED)
                        {
                            score -= LEECH_SEED;
                        }
                        if state
                            .side_two
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::SUBSTITUTE)
                        {
                            score -= SUBSTITUTE;
                        }
                        if state
                            .side_two
                            .volatile_statuses
                            .contains(&PokemonVolatileStatus::CONFUSION)
                        {
                            score -= CONFUSION;
                        }

                        score -= get_boost_multiplier(state.side_two.attack_boost)
                            * POKEMON_ATTACK_BOOST;
                        score -= get_boost_multiplier(state.side_two.defense_boost)
                            * POKEMON_DEFENSE_BOOST;
                        score -= get_boost_multiplier(state.side_two.special_attack_boost)
                            * POKEMON_SPECIAL_ATTACK_BOOST;
                        score -= get_boost_multiplier(state.side_two.special_defense_boost)
                            * POKEMON_SPECIAL_DEFENSE_BOOST;
                        score -=
                            get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
                    }
                }
                if pkmn.terastallized {
                    s2_used_tera = true;
                }
            }
            if s2_used_tera {
                score -= USED_TERA;
            }

            score += state.side_one.side_conditions.reflect as f32 * REFLECT;
            score += state.side_one.side_conditions.light_screen as f32 * LIGHT_SCREEN;
            score += state.side_one.side_conditions.aurora_veil as f32 * AURORA_VEIL;
            score += state.side_one.side_conditions.safeguard as f32 * SAFE_GUARD;
            score += state.side_one.side_conditions.tailwind as f32 * TAILWIND;
            score += state.side_one.side_conditions.healing_wish as f32 * HEALING_WISH;

            score -= state.side_two.side_conditions.reflect as f32 * REFLECT;
            score -= state.side_two.side_conditions.light_screen as f32 * LIGHT_SCREEN;
            score -= state.side_two.side_conditions.aurora_veil as f32 * AURORA_VEIL;
            score -= state.side_two.side_conditions.safeguard as f32 * SAFE_GUARD;
            score -= state.side_two.side_conditions.tailwind as f32 * TAILWIND;
            score -= state.side_two.side_conditions.healing_wish as f32 * HEALING_WISH;

            score
        }
    }

    fn bundled_states() -> Vec<State> {
        include_str!("../../data/gen9randombattle.txt")
            .lines()
            .filter(|line| !line.is_empty())
            .map(State::deserialize)
            .collect()
    }

    // deterministic mutations exercising every feature: statuses, low hp,
    // hazards, screens, boosts, volatiles, tera, fainted mons
    fn mutated_variants(base: &State) -> Vec<State> {
        let mut variants = vec![base.clone()];

        let mut s = base.clone();
        s.side_one.side_conditions.stealth_rock = 1;
        s.side_one.side_conditions.spikes = 2;
        s.side_one.side_conditions.toxic_spikes = 1;
        s.side_two.side_conditions.sticky_web = 1;
        s.side_two.side_conditions.stealth_rock = 1;
        s.side_two.side_conditions.reflect = 1;
        s.side_two.side_conditions.light_screen = 1;
        s.side_one.side_conditions.aurora_veil = 1;
        s.side_one.side_conditions.safeguard = 1;
        s.side_one.side_conditions.tailwind = 2;
        s.side_two.side_conditions.healing_wish = 1;
        variants.push(s);

        let mut s = base.clone();
        s.side_one.pokemon.pkmn[0].status = PokemonStatus::BURN;
        s.side_one.pokemon.pkmn[0].hp = 1;
        s.side_one.pokemon.pkmn[1].status = PokemonStatus::TOXIC;
        s.side_one.pokemon.pkmn[1].hp = 1;
        s.side_one.pokemon.pkmn[2].status = PokemonStatus::FREEZE;
        s.side_two.pokemon.pkmn[0].status = PokemonStatus::PARALYZE;
        s.side_two.pokemon.pkmn[1].status = PokemonStatus::POISON;
        s.side_two.pokemon.pkmn[2].status = PokemonStatus::SLEEP;
        s.side_two.pokemon.pkmn[3].hp = 0;
        s.side_one.pokemon.pkmn[5].hp = 0;
        variants.push(s);

        let mut s = base.clone();
        s.side_one.attack_boost = 2;
        s.side_one.speed_boost = -1;
        s.side_one.special_defense_boost = 1;
        s.side_two.defense_boost = -3;
        s.side_two.special_attack_boost = 6;
        s.side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        s.side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::SUBSTITUTE);
        s.side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::CONFUSION);
        s.side_one.pokemon.pkmn[0].terastallized = true;
        s.side_two.pokemon.pkmn[1].item = Items::NONE;
        variants.push(s);

        variants
    }

    fn dot(weights: &[f32; NUM_EVAL_FEATURES], features: &[f32; NUM_EVAL_FEATURES]) -> f32 {
        weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    fn score_with_clamp(state: &State, clamp: bool) -> f32 {
        let mut sink = ScoreSink {
            weights: &DEFAULT_EVAL_WEIGHTS,
            clamp,
            sign: 1.0,
            mon_subtotal: 0.0,
            global_subtotal: None,
            total: 0.0,
        };
        eval_walk(state, &mut sink);
        sink.total
    }

    #[test]
    fn weighted_walk_matches_reference_eval() {
        // Clamp on with default weights is the historical evaluator.
        for state in bundled_states().iter() {
            for variant in mutated_variants(state) {
                let expected = reference::evaluate(&variant);
                let actual = score_with_clamp(&variant, true);
                assert_eq!(expected, actual, "state={}", variant.serialize());
            }
        }
    }

    #[test]
    fn features_dot_weights_matches_unclamped_score() {
        for state in bundled_states().iter() {
            for variant in mutated_variants(state) {
                let expected = score_with_clamp(&variant, false);
                let actual = dot(&DEFAULT_EVAL_WEIGHTS, &eval_features(&variant));
                assert!(
                    expected == actual || (expected - actual).abs() < 0.05,
                    "mismatch: unclamped={} dot={} state={}",
                    expected,
                    actual,
                    variant.serialize()
                );
            }
        }
    }

    #[test]
    fn clamp_binds_on_negative_mon_subtotal() {
        let mut state = bundled_states().remove(0);
        // hp ~0, toxic, no item, generic ability: subtotal is clearly negative
        let mon = &mut state.side_one.pokemon.pkmn[1];
        mon.hp = 1;
        mon.status = PokemonStatus::TOXIC;
        mon.ability = Abilities::NONE;
        mon.item = Items::NONE;
        assert_ne!(state.side_one.active_index, PokemonIndex::P1);

        let clamped = score_with_clamp(&state, true);
        let unclamped = score_with_clamp(&state, false);
        assert!(
            clamped > unclamped,
            "clamp should raise the score: clamped={} unclamped={}",
            clamped,
            unclamped
        );
    }

    #[test]
    fn parse_eval_weights_roundtrip() {
        let mut text = String::from("# tuned weights\n\n");
        for (name, weight) in EVAL_FEATURE_NAMES.iter().zip(DEFAULT_EVAL_WEIGHTS.iter()) {
            text.push_str(&format!("{} {}\n", name, weight));
        }
        let parsed = parse_eval_weights(&text).unwrap();
        assert_eq!(parsed, DEFAULT_EVAL_WEIGHTS);

        assert!(parse_eval_weights("POKEMON_ALIVE 30.0").is_err()); // missing names
        assert!(parse_eval_weights(&format!("{}\nNOT_A_WEIGHT 1.0\n", text)).is_err());
        assert!(parse_eval_weights(&format!("{}POKEMON_ALIVE 30.0\n", text)).is_err());
        // duplicate
        assert!(
            parse_eval_weights(&text.replace("POKEMON_ALIVE 30", "POKEMON_ALIVE NaN")).is_err()
        );
        assert!(
            parse_eval_weights(&text.replace("POKEMON_ALIVE 30", "POKEMON_ALIVE inf")).is_err()
        );
    }

    #[test]
    fn default_config_preserves_historical_clamp() {
        let mut state = bundled_states().remove(0);
        let mon = &mut state.side_one.pokemon.pkmn[1];
        mon.hp = 1;
        mon.status = PokemonStatus::TOXIC;
        mon.ability = Abilities::NONE;
        mon.item = Items::NONE;

        assert_eq!(evaluate(&state), score_with_clamp(&state, true));
    }
}
