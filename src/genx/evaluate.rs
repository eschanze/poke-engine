use super::abilities::Abilities;
use super::items::Items;
use super::matchup::{answers_after_entry, moves_before, DuelResult, MatchupKernel};
use super::state::PokemonVolatileStatus;
use crate::choices::MoveCategory;
use crate::state::{Pokemon, PokemonStatus, Side, State};
use std::convert::TryInto;

// The eval is a dot product between a weight vector and a feature vector
// computed from the state (side one minus side two), so the weights can be
// texel-tuned on game outcomes (see EVAL_TUNING_PLAN.md). One optional
// nonlinearity: when EvalConfig::mon_clamp is on, each mon's hp/status/item
// subtotal is clamped at 0 before the alive bonus is added. The tuned
// production evaluator is exactly linear; the clamp remains available for
// reproducing the historical evaluator and for experiments.

pub const NUM_EVAL_FEATURES: usize = 36;

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
    pub const REVENGE_COVERAGE: usize = 30;
    pub const THREAT_BREADTH: usize = 31;
    pub const WINCON: usize = 32;
    pub const UNANSWERED: usize = 33;
    pub const ACTIVE_DUEL: usize = 34;
    pub const PIVOT_PRESSURE: usize = 35;
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
    "REVENGE_COVERAGE",
    "THREAT_BREADTH",
    "WINCON",
    "UNANSWERED",
    "ACTIVE_DUEL",
    "PIVOT_PRESSURE",
];

// Outcome-tuned production weights, fit with semantic sign/range constraints
// on 480 decisive games and gated +82.6 Elo [38.5, 129.5] over 240 held-out
// games at equal 250 ms. The historical hand-picked vector remains in
// data/eval-handcrafted-40.weights for regression gates.
// BURNED's feature is a physical-move-count multiplier and the five boost
// weights multiply the fixed 13-entry boost table below; everything else
// multiplies a count or a 0/1 flag.
pub const DEFAULT_EVAL_WEIGHTS: [f32; NUM_EVAL_FEATURES] = [
    62.7669384,   // POKEMON_ALIVE
    41.0806585,   // POKEMON_HP
    0.0,          // POKEMON_ITEM
    -90.2561015,  // POKEMON_FROZEN
    -0.408866373, // POKEMON_ASLEEP
    -25.6112665,  // POKEMON_PARALYZED
    -75.4539139,  // POKEMON_TOXIC
    -24.6924447,  // POKEMON_POISONED
    0.0,          // POKEMON_BURNED
    52.9131556,   // POISON_HEAL
    0.0,          // STATUS_ABILITY_BONUS
    1.34682091,   // POKEMON_ATTACK_BOOST
    4.54921779,   // POKEMON_DEFENSE_BOOST
    0.0,          // POKEMON_SPECIAL_ATTACK_BOOST
    24.9293773,   // POKEMON_SPECIAL_DEFENSE_BOOST
    23.6278393,   // POKEMON_SPEED_BOOST
    -32.9769423,  // LEECH_SEED
    104.108085,   // SUBSTITUTE
    -10.196138,   // CONFUSION
    0.0,          // REFLECT
    0.0,          // LIGHT_SCREEN
    0.0,          // AURORA_VEIL
    0.0,          // SAFE_GUARD
    0.0,          // TAILWIND
    60.0,         // HEALING_WISH
    -2.97986335,  // STEALTH_ROCK
    -17.3453197,  // SPIKES
    -30.0,        // TOXIC_SPIKES
    0.0,          // STICKY_WEB
    -48.5478855,  // USED_TERA
    3.1326059,    // REVENGE_COVERAGE
    40.5900329,   // THREAT_BREADTH
    14.4660979,   // WINCON
    20.1621173,   // UNANSWERED
    4.86702255,   // ACTIVE_DUEL
    3.7430539,    // PIVOT_PRESSURE
];

/// How much a matchup credited to a benched Pokemon is worth relative to the
/// same matchup for the active Pokemon. Bench threats must pay an entry (a
/// switch and usually a free hit) before their matchups become real, so they
/// are discounted; 1.0 reproduces the original undiscounted aggregation.
pub const DEFAULT_BENCH_SCALE: f32 = 0.5;

#[derive(Clone, Debug, PartialEq)]
enum EvalTreeNode {
    Split {
        feature: u8,
        threshold: f32,
        yes: u16,
        no: u16,
    },
    Leaf(f32),
}

/// A compact gradient-boosted correction on top of a linear evaluator.
/// Tree outputs are logits; evaluating both side orientations and taking
/// half their difference guarantees score(swap(state)) == -score(state).
#[derive(Clone, Debug, PartialEq)]
pub struct EvalTreeModel {
    base_weights: [f32; NUM_EVAL_FEATURES],
    k: f32,
    correction_clip: f32,
    trees: Vec<Vec<EvalTreeNode>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalContextModel {
    base_weights: [f32; NUM_EVAL_FEATURES],
    k: f32,
    correction_clip: f32,
    diff_scale: [f32; NUM_EVAL_FEATURES],
    total_mean: [f32; NUM_EVAL_FEATURES],
    total_scale: [f32; NUM_EVAL_FEATURES],
    difference_weights: Vec<[f32; NUM_EVAL_FEATURES]>,
    context_weights: Vec<[f32; NUM_EVAL_FEATURES]>,
    context_bias: Vec<f32>,
    output_weights: Vec<f32>,
}

impl EvalContextModel {
    fn evaluate(&self, pair: &[[f32; NUM_EVAL_FEATURES]; 2]) -> f32 {
        let mut difference = [0.0; NUM_EVAL_FEATURES];
        let mut context = [0.0; NUM_EVAL_FEATURES];
        let mut base = 0.0;
        for i in 0..NUM_EVAL_FEATURES {
            let diff = pair[0][i] - pair[1][i];
            base += self.base_weights[i] * diff;
            difference[i] = diff / self.diff_scale[i];
            context[i] = (pair[0][i] + pair[1][i] - self.total_mean[i]) / self.total_scale[i];
        }
        let mut raw = 0.0;
        for hidden in 0..self.output_weights.len() {
            let mut diff_sum = 0.0;
            let mut context_sum = self.context_bias[hidden];
            for feature in 0..NUM_EVAL_FEATURES {
                diff_sum += self.difference_weights[hidden][feature] * difference[feature];
                context_sum += self.context_weights[hidden][feature] * context[feature];
            }
            let odd = diff_sum.tanh();
            let gate = 1.0 / (1.0 + (-context_sum).exp());
            raw += self.output_weights[hidden] * odd * gate;
        }
        let correction = self.correction_clip * (raw / self.correction_clip).tanh();
        base + self.k * correction
    }
}

impl EvalTreeModel {
    fn tree_sum(&self, features: &[f32; NUM_EVAL_FEATURES], sign: f32) -> f32 {
        let mut total = 0.0;
        for tree in &self.trees {
            let mut index = 0usize;
            loop {
                match tree[index] {
                    EvalTreeNode::Leaf(value) => {
                        total += value;
                        break;
                    }
                    EvalTreeNode::Split {
                        feature,
                        threshold,
                        yes,
                        no,
                    } => {
                        index = if sign * features[feature as usize] < threshold {
                            yes as usize
                        } else {
                            no as usize
                        };
                    }
                }
            }
        }
        total
    }

    fn evaluate(&self, features: &[f32; NUM_EVAL_FEATURES]) -> f32 {
        let base: f32 = self
            .base_weights
            .iter()
            .zip(features)
            .map(|(weight, feature)| weight * feature)
            .sum();
        let correction = (0.5 * (self.tree_sum(features, 1.0) - self.tree_sum(features, -1.0)))
            .clamp(-self.correction_clip, self.correction_clip);
        base + self.k * correction
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalConfig {
    weights: &'static [f32; NUM_EVAL_FEATURES],
    mon_clamp: bool,
    bench_scale: f32,
    tree_model: Option<&'static EvalTreeModel>,
    context_model: Option<&'static EvalContextModel>,
}

impl EvalConfig {
    pub const fn new(weights: &'static [f32; NUM_EVAL_FEATURES], mon_clamp: bool) -> EvalConfig {
        EvalConfig {
            weights,
            mon_clamp,
            bench_scale: DEFAULT_BENCH_SCALE,
            tree_model: None,
            context_model: None,
        }
    }
    pub const fn from_tree_model(model: &'static EvalTreeModel) -> EvalConfig {
        EvalConfig {
            weights: &model.base_weights,
            mon_clamp: false,
            bench_scale: DEFAULT_BENCH_SCALE,
            tree_model: Some(model),
            context_model: None,
        }
    }
    pub const fn from_context_model(model: &'static EvalContextModel) -> EvalConfig {
        EvalConfig {
            weights: &model.base_weights,
            mon_clamp: false,
            bench_scale: DEFAULT_BENCH_SCALE,
            tree_model: None,
            context_model: Some(model),
        }
    }
    pub const fn with_bench_scale(mut self, bench_scale: f32) -> EvalConfig {
        self.bench_scale = bench_scale;
        self
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        EvalConfig::new(&DEFAULT_EVAL_WEIGHTS, false)
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

fn parse_finite_f32(value: &str, context: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|e| format!("bad {} value {}: {}", context, value, e))?;
    if !parsed.is_finite() {
        return Err(format!("{} must be finite", context));
    }
    Ok(parsed)
}

/// Parse the compact text format exported by tools/eval_tuning/boosted.py.
pub fn parse_eval_tree_model(text: &str) -> Result<EvalTreeModel, String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let mut next = || -> Result<&str, String> {
        let line = lines
            .get(cursor)
            .copied()
            .ok_or_else(|| "unexpected end of tree model".to_string())?;
        cursor += 1;
        Ok(line)
    };

    if next()? != "poke-engine-tree-eval-v1" {
        return Err("unsupported tree model format".to_string());
    }
    let schema_line = next()?;
    let schema = schema_line
        .strip_prefix("feature_schema ")
        .ok_or_else(|| "missing feature_schema".to_string())?;
    let expected_schema = eval_feature_schema();
    if schema != expected_schema {
        return Err(format!(
            "tree model feature schema {}, expected {}",
            schema, expected_schema
        ));
    }
    let k_line = next()?;
    let k = parse_finite_f32(
        k_line
            .strip_prefix("k ")
            .ok_or_else(|| "missing k".to_string())?,
        "k",
    )?;
    if k <= 0.0 {
        return Err("k must be positive".to_string());
    }
    let clip_line = next()?;
    let correction_clip = parse_finite_f32(
        clip_line
            .strip_prefix("correction_clip ")
            .ok_or_else(|| "missing correction_clip".to_string())?,
        "correction_clip",
    )?;
    if correction_clip <= 0.0 {
        return Err("correction_clip must be positive".to_string());
    }

    let mut base_weights = [0.0; NUM_EVAL_FEATURES];
    let mut seen = [false; NUM_EVAL_FEATURES];
    for _ in 0..NUM_EVAL_FEATURES {
        let line = next()?;
        let mut parts = line.split_whitespace();
        if parts.next() != Some("base") {
            return Err("expected base weight".to_string());
        }
        let name = parts
            .next()
            .ok_or_else(|| "missing base weight name".to_string())?;
        let value = parts
            .next()
            .ok_or_else(|| "missing base weight value".to_string())?;
        if parts.next().is_some() {
            return Err("trailing base weight tokens".to_string());
        }
        let feature = EVAL_FEATURE_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .ok_or_else(|| format!("unknown base weight {}", name))?;
        if seen[feature] {
            return Err(format!("duplicate base weight {}", name));
        }
        base_weights[feature] = parse_finite_f32(value, name)?;
        seen[feature] = true;
    }

    let trees_line = next()?;
    let tree_count = trees_line
        .strip_prefix("trees ")
        .ok_or_else(|| "missing trees count".to_string())?
        .parse::<usize>()
        .map_err(|e| format!("bad trees count: {}", e))?;
    let mut trees = Vec::with_capacity(tree_count);
    for tree_index in 0..tree_count {
        let tree_line = next()?;
        let node_count = tree_line
            .strip_prefix("tree ")
            .ok_or_else(|| format!("missing tree {} header", tree_index))?
            .parse::<usize>()
            .map_err(|e| format!("bad tree {} node count: {}", tree_index, e))?;
        if node_count == 0 || node_count > u16::MAX as usize {
            return Err(format!(
                "tree {} has invalid node count {}",
                tree_index, node_count
            ));
        }
        let mut nodes = Vec::with_capacity(node_count);
        for node_index in 0..node_count {
            let line = next()?;
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("leaf") => {
                    let value = parts
                        .next()
                        .ok_or_else(|| "missing leaf value".to_string())?;
                    if parts.next().is_some() {
                        return Err("trailing leaf tokens".to_string());
                    }
                    nodes.push(EvalTreeNode::Leaf(parse_finite_f32(value, "leaf")?));
                }
                Some("split") => {
                    let feature = parts
                        .next()
                        .ok_or_else(|| "missing split feature".to_string())?
                        .parse::<usize>()
                        .map_err(|e| format!("bad split feature: {}", e))?;
                    let threshold = parse_finite_f32(
                        parts
                            .next()
                            .ok_or_else(|| "missing split threshold".to_string())?,
                        "split threshold",
                    )?;
                    let yes = parts
                        .next()
                        .ok_or_else(|| "missing yes child".to_string())?
                        .parse::<usize>()
                        .map_err(|e| format!("bad yes child: {}", e))?;
                    let no = parts
                        .next()
                        .ok_or_else(|| "missing no child".to_string())?
                        .parse::<usize>()
                        .map_err(|e| format!("bad no child: {}", e))?;
                    if parts.next().is_some() {
                        return Err("trailing split tokens".to_string());
                    }
                    if feature >= NUM_EVAL_FEATURES {
                        return Err(format!("split feature {} out of range", feature));
                    }
                    if yes <= node_index
                        || no <= node_index
                        || yes >= node_count
                        || no >= node_count
                    {
                        return Err(format!(
                            "tree {} node {} has invalid children",
                            tree_index, node_index
                        ));
                    }
                    nodes.push(EvalTreeNode::Split {
                        feature: feature as u8,
                        threshold,
                        yes: yes as u16,
                        no: no as u16,
                    });
                }
                token => return Err(format!("unknown tree node token {:?}", token)),
            }
        }
        trees.push(nodes);
    }
    drop(next);
    if cursor != lines.len() {
        return Err("trailing tree model content".to_string());
    }
    Ok(EvalTreeModel {
        base_weights,
        k,
        correction_clip,
        trees,
    })
}

fn parse_model_vector(line: &str, label: &str, expected: usize) -> Result<Vec<f32>, String> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some(label) {
        return Err(format!("expected {}", label));
    }
    let values: Result<Vec<f32>, String> =
        parts.map(|value| parse_finite_f32(value, label)).collect();
    let values = values?;
    if values.len() != expected {
        return Err(format!(
            "{} has {} values, expected {}",
            label,
            values.len(),
            expected
        ));
    }
    Ok(values)
}

/// Parse the context-gated MLP exported by tools/eval_tuning/context_mlp.py.
pub fn parse_eval_context_model(text: &str) -> Result<EvalContextModel, String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let mut next = || -> Result<&str, String> {
        let line = lines
            .get(cursor)
            .copied()
            .ok_or_else(|| "unexpected end of context model".to_string())?;
        cursor += 1;
        Ok(line)
    };
    if next()? != "poke-engine-context-mlp-v1" {
        return Err("unsupported context model format".to_string());
    }
    let schema_line = next()?;
    let schema = schema_line
        .strip_prefix("feature_schema ")
        .ok_or_else(|| "missing feature_schema".to_string())?;
    let expected_schema = eval_feature_schema();
    if schema != expected_schema {
        return Err(format!(
            "context model feature schema {}, expected {}",
            schema, expected_schema
        ));
    }
    let k_line = next()?;
    let k = parse_finite_f32(
        k_line
            .strip_prefix("k ")
            .ok_or_else(|| "missing k".to_string())?,
        "k",
    )?;
    let clip_line = next()?;
    let correction_clip = parse_finite_f32(
        clip_line
            .strip_prefix("correction_clip ")
            .ok_or_else(|| "missing correction_clip".to_string())?,
        "correction_clip",
    )?;
    if k <= 0.0 || correction_clip <= 0.0 {
        return Err("k and correction_clip must be positive".to_string());
    }

    let mut base_weights = [0.0; NUM_EVAL_FEATURES];
    let mut seen = [false; NUM_EVAL_FEATURES];
    for _ in 0..NUM_EVAL_FEATURES {
        let line = next()?;
        let mut parts = line.split_whitespace();
        if parts.next() != Some("base") {
            return Err("expected base weight".to_string());
        }
        let name = parts
            .next()
            .ok_or_else(|| "missing base weight name".to_string())?;
        let value = parts
            .next()
            .ok_or_else(|| "missing base weight value".to_string())?;
        if parts.next().is_some() {
            return Err("trailing base weight tokens".to_string());
        }
        let feature = EVAL_FEATURE_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .ok_or_else(|| format!("unknown base weight {}", name))?;
        if seen[feature] {
            return Err(format!("duplicate base weight {}", name));
        }
        base_weights[feature] = parse_finite_f32(value, name)?;
        seen[feature] = true;
    }

    let diff_scale_vec = parse_model_vector(next()?, "diff_scale", NUM_EVAL_FEATURES)?;
    let total_mean_vec = parse_model_vector(next()?, "total_mean", NUM_EVAL_FEATURES)?;
    let total_scale_vec = parse_model_vector(next()?, "total_scale", NUM_EVAL_FEATURES)?;
    let diff_scale: [f32; NUM_EVAL_FEATURES] = diff_scale_vec.try_into().unwrap();
    let total_mean: [f32; NUM_EVAL_FEATURES] = total_mean_vec.try_into().unwrap();
    let total_scale: [f32; NUM_EVAL_FEATURES] = total_scale_vec.try_into().unwrap();
    if diff_scale
        .iter()
        .chain(total_scale.iter())
        .any(|value| *value <= 0.0)
    {
        return Err("model scales must be positive".to_string());
    }
    let hidden_line = next()?;
    let hidden = hidden_line
        .strip_prefix("hidden ")
        .ok_or_else(|| "missing hidden size".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("bad hidden size: {}", error))?;
    if hidden == 0 || hidden > 64 {
        return Err(format!("hidden size {} is out of range", hidden));
    }
    let mut difference_weights = Vec::with_capacity(hidden);
    for _ in 0..hidden {
        difference_weights.push(
            parse_model_vector(next()?, "difference", NUM_EVAL_FEATURES)?
                .try_into()
                .unwrap(),
        );
    }
    let mut context_weights = Vec::with_capacity(hidden);
    let mut context_bias = Vec::with_capacity(hidden);
    for _ in 0..hidden {
        let mut values = parse_model_vector(next()?, "context", NUM_EVAL_FEATURES + 1)?;
        context_bias.push(values.pop().unwrap());
        context_weights.push(values.try_into().unwrap());
    }
    let output_weights = parse_model_vector(next()?, "output", hidden)?;
    drop(next);
    if cursor != lines.len() {
        return Err("trailing context model content".to_string());
    }
    Ok(EvalContextModel {
        base_weights,
        k,
        correction_clip,
        diff_scale,
        total_mean,
        total_scale,
        difference_weights,
        context_weights,
        context_bias,
        output_weights,
    })
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

struct PairFeatureSink {
    side: usize,
    features: [[f32; NUM_EVAL_FEATURES]; 2],
}

impl EvalSink for PairFeatureSink {
    fn set_sign(&mut self, sign: f32) {
        self.side = if sign > 0.0 { 0 } else { 1 };
    }
    fn hp(&mut self, hp: f32, maxhp: f32) {
        self.features[self.side][feat::HP] += hp / maxhp;
    }
    fn mon(&mut self, idx: usize, value: f32) {
        self.features[self.side][idx] += value;
    }
    fn finish_mon(&mut self) {
        self.features[self.side][feat::ALIVE] += 1.0;
    }
    fn global(&mut self, idx: usize, value: f32) {
        self.features[self.side][idx] += value;
    }
    fn start_global_group(&mut self) {}
    fn finish_global_group(&mut self) {}
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

fn walk_matchups<S: EvalSink>(state: &State, sink: &mut S, bench_scale: f32) {
    crate::prof_scope!(crate::prof::sec::MATCHUP_TOTAL);
    const IDX: [crate::state::PokemonIndex; 6] = [
        crate::state::PokemonIndex::P0,
        crate::state::PokemonIndex::P1,
        crate::state::PokemonIndex::P2,
        crate::state::PokemonIndex::P3,
        crate::state::PokemonIndex::P4,
        crate::state::PokemonIndex::P5,
    ];
    let k = {
        crate::prof_scope!(crate::prof::sec::MATCHUP_KERNEL);
        MatchupKernel::new(state)
    };
    #[cfg(feature = "prof")]
    let _aggregate_profile = crate::prof::ProfScope::new(crate::prof::sec::MATCHUP_AGGREGATE);
    let active_one = state.side_one.active_index as usize;
    let active_two = state.side_two.active_index as usize;
    let mut one = [0.0f32; 6];
    let mut two = [0.0f32; 6];

    // Compute every living cross-team pair once, accumulating all matchup features
    // while its two directional results are hot in cache. Threat credit earned by
    // a benched mon is worth `bench_scale` of the active mon's credit, because a
    // bench threat still has to pay an entry before its matchups become real.
    // Revenge coverage is exempt: a revenge kill enters on a faint for free.
    let mut revenge_one = [false; 6];
    let mut revenge_two = [false; 6];
    let mut wins_one = [0usize; 6];
    let mut wins_two = [0usize; 6];
    let mut answered_one = [0usize; 6];
    let mut answered_two = [0usize; 6];

    for i in 0..6 {
        if !k.alive_one[i] {
            continue;
        }
        for j in 0..6 {
            if !k.alive_two[j] {
                continue;
            }
            let attack_one = k.one(i, j);
            let attack_two = k.two(j, i);
            let hp_one = state.side_one.pokemon[IDX[i]].hp;
            let hp_two = state.side_two.pokemon[IDX[j]].hp;

            revenge_one[i] |= attack_two.damage >= hp_one
                && moves_before(state, attack_two, attack_one) == Some(true);
            revenge_two[j] |= attack_one.damage >= hp_two
                && moves_before(state, attack_one, attack_two) == Some(true);

            match k.duel_one(state, i, j) {
                DuelResult::Win => {
                    wins_one[i] += 1;
                }
                DuelResult::Loss => {
                    wins_two[j] += 1;
                }
                DuelResult::Draw => {}
            }
            answered_one[i] += answers_after_entry(state, &k, true, i, j) as usize;
            answered_two[j] += answers_after_entry(state, &k, false, j, i) as usize;
        }
    }

    for i in 0..6 {
        if k.alive_one[i] {
            let scale = if i == active_one { 1.0 } else { bench_scale };
            two[0] += revenge_one[i] as u8 as f32;
            if k.count_two > 0 {
                one[1] += scale * wins_one[i] as f32 / k.count_two as f32;
                one[2] += scale * (wins_one[i] == k.count_two) as u8 as f32;
                one[3] += scale * (answered_one[i] == 0) as u8 as f32;
            }
        }
        if k.alive_two[i] {
            let scale = if i == active_two { 1.0 } else { bench_scale };
            one[0] += revenge_two[i] as u8 as f32;
            if k.count_one > 0 {
                two[1] += scale * wins_two[i] as f32 / k.count_one as f32;
                two[2] += scale * (wins_two[i] == k.count_one) as u8 as f32;
                two[3] += scale * (answered_two[i] == 0) as u8 as f32;
            }
        }
    }

    if k.alive_one[active_one] && k.alive_two[active_two] {
        match k.duel_one(state, active_one, active_two) {
            DuelResult::Win => {
                one[4] = 1.0;
                if answered_one[active_one] == 0 {
                    one[5] = 1.0;
                }
            }
            DuelResult::Loss => {
                two[4] = 1.0;
                if answered_two[active_two] == 0 {
                    two[5] = 1.0;
                }
            }
            DuelResult::Draw => {}
        }
    }

    sink.set_sign(1.0);
    for i in 0..6 {
        sink.global(feat::REVENGE_COVERAGE + i, one[i]);
    }
    sink.set_sign(-1.0);
    for i in 0..6 {
        sink.global(feat::REVENGE_COVERAGE + i, two[i]);
    }
}

fn eval_walk_base<S: EvalSink>(state: &State, sink: &mut S) {
    walk_side_pokemon(&state.side_one, sink, 1.0);
    walk_side_pokemon(&state.side_two, sink, -1.0);
    walk_side_conditions(&state.side_one, sink, 1.0);
    walk_side_conditions(&state.side_two, sink, -1.0);
}

fn eval_walk<S: EvalSink>(state: &State, sink: &mut S, bench_scale: f32) {
    eval_walk_base(state, sink);
    walk_matchups(state, sink, bench_scale);
}

/// side-one-minus-side-two feature vector; `dot(weights, features)` equals
/// `evaluate_with_config()` when the per-mon clamp is off (the clamp is not linear)
pub fn eval_features(state: &State) -> [f32; NUM_EVAL_FEATURES] {
    let mut sink = FeatureSink {
        sign: 1.0,
        features: [0.0; NUM_EVAL_FEATURES],
    };
    eval_walk(state, &mut sink, DEFAULT_BENCH_SCALE);
    sink.features
}

/// Per-side feature vectors before subtraction. These preserve shared context
/// (for example, both sides having high threat breadth) for nonlinear models.
/// `eval_pair_features(state)[0] - eval_pair_features(state)[1]` is exactly
/// `eval_features(state)` up to floating-point accumulation order.
pub fn eval_pair_features_with_bench_scale(
    state: &State,
    bench_scale: f32,
) -> [[f32; NUM_EVAL_FEATURES]; 2] {
    let mut sink = PairFeatureSink {
        side: 0,
        features: [[0.0; NUM_EVAL_FEATURES]; 2],
    };
    eval_walk(state, &mut sink, bench_scale);
    sink.features
}

pub fn eval_pair_features(state: &State) -> [[f32; NUM_EVAL_FEATURES]; 2] {
    eval_pair_features_with_bench_scale(state, DEFAULT_BENCH_SCALE)
}

pub fn evaluate_with_config(state: &State, config: EvalConfig) -> f32 {
    if let Some(model) = config.context_model {
        let pair = eval_pair_features_with_bench_scale(state, config.bench_scale);
        return model.evaluate(&pair);
    }
    if let Some(model) = config.tree_model {
        let mut sink = FeatureSink {
            sign: 1.0,
            features: [0.0; NUM_EVAL_FEATURES],
        };
        eval_walk(state, &mut sink, config.bench_scale);
        return model.evaluate(&sink.features);
    }
    let mut sink = ScoreSink {
        weights: config.weights,
        clamp: config.mon_clamp,
        sign: 1.0,
        mon_subtotal: 0.0,
        global_subtotal: None,
        total: 0.0,
    };
    eval_walk_base(state, &mut sink);
    if config.weights[feat::REVENGE_COVERAGE..]
        .iter()
        .any(|weight| *weight != 0.0)
    {
        walk_matchups(state, &mut sink, config.bench_scale);
    }
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
        include_str!("../../data/gen9-battle-factory-no-ubers-states.txt")
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
        eval_walk(state, &mut sink, DEFAULT_BENCH_SCALE);
        sink.total
    }

    #[test]
    fn weighted_walk_matches_reference_eval() {
        // The first 30 terms of the preserved handcrafted vector reproduce
        // the evaluator that predated feature decomposition.
        let mut historical =
            parse_eval_weights(include_str!("../../data/eval-handcrafted-36.weights")).unwrap();
        historical[30..].fill(0.0);
        for state in bundled_states().iter() {
            for variant in mutated_variants(state) {
                let expected = reference::evaluate(&variant);
                let mut sink = ScoreSink {
                    weights: &historical,
                    clamp: true,
                    sign: 1.0,
                    mon_subtotal: 0.0,
                    global_subtotal: None,
                    total: 0.0,
                };
                eval_walk(&variant, &mut sink, DEFAULT_BENCH_SCALE);
                let actual = sink.total;
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
    fn pair_features_reconstruct_difference_features() {
        for state in bundled_states() {
            let difference = eval_features(&state);
            let pair = eval_pair_features(&state);
            for i in 0..NUM_EVAL_FEATURES {
                let reconstructed = pair[0][i] - pair[1][i];
                assert!(
                    (difference[i] - reconstructed).abs() < 1e-5,
                    "feature {} differs: {} vs {}",
                    EVAL_FEATURE_NAMES[i],
                    difference[i],
                    reconstructed
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
        let alive_line = format!("POKEMON_ALIVE {}", DEFAULT_EVAL_WEIGHTS[feat::ALIVE]);
        assert!(parse_eval_weights(&text.replace(&alive_line, "POKEMON_ALIVE NaN")).is_err());
        assert!(parse_eval_weights(&text.replace(&alive_line, "POKEMON_ALIVE inf")).is_err());
    }

    #[test]
    fn parse_tree_model_and_preserve_side_symmetry() {
        let mut text = format!(
            "poke-engine-tree-eval-v1\nfeature_schema {}\nk 80\ncorrection_clip 1\n",
            eval_feature_schema()
        );
        for (name, weight) in EVAL_FEATURE_NAMES.iter().zip(DEFAULT_EVAL_WEIGHTS.iter()) {
            text.push_str(&format!("base {} {}\n", name, weight));
        }
        text.push_str("trees 1\ntree 3\nsplit 0 0 1 2\nleaf 0.25\nleaf -0.25\n");
        let model = parse_eval_tree_model(&text).unwrap();
        let mut features = [0.0; NUM_EVAL_FEATURES];
        features[feat::ALIVE] = 1.0;
        let one = model.evaluate(&features);
        for feature in &mut features {
            *feature = -*feature;
        }
        let two = model.evaluate(&features);
        assert_eq!(one, -two);
        assert!(parse_eval_tree_model(&text.replace(&eval_feature_schema(), "wrong")).is_err());
    }

    #[test]
    fn matchup_features_are_side_swap_symmetric() {
        for state in bundled_states() {
            let original = eval_features(&state);
            let mut swapped = state.clone();
            std::mem::swap(&mut swapped.side_one, &mut swapped.side_two);
            let mirrored = eval_features(&swapped);
            for i in feat::REVENGE_COVERAGE..NUM_EVAL_FEATURES {
                assert!(
                    (original[i] + mirrored[i]).abs() < 1e-5,
                    "feature {} is not symmetric: {} vs {}",
                    EVAL_FEATURE_NAMES[i],
                    original[i],
                    mirrored[i]
                );
            }
        }
    }

    #[test]
    fn default_config_uses_unclamped_tuned_weights() {
        let mut state = bundled_states().remove(0);
        let mon = &mut state.side_one.pokemon.pkmn[1];
        mon.hp = 1;
        mon.status = PokemonStatus::TOXIC;
        mon.ability = Abilities::NONE;
        mon.item = Items::NONE;

        assert_eq!(evaluate(&state), score_with_clamp(&state, false));
    }
}
