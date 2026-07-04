use clap::Parser;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::StateInstructions;
use poke_engine::mcts::{perform_mcts, MctsSideResult, DEFAULT_EXPLORATION_CONSTANT};
use poke_engine::mcts_threaded::perform_mcts_shared_tree;
use poke_engine::state::State;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::process::exit;
use std::time::Duration;

// Self-play A/B harness. See SELFPLAY_PLAN.md for design rationale.
//
// Plays paired games (colors swapped) from each input state. Each side picks
// its move with its own MCTS configuration; chance outcomes are sampled from
// the weighted instruction list. Reports A's win rate, Elo diff, and LOS.

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    file_name: String,

    /// number of states to use from the file. 0 means all
    #[clap(short = 'l', long, default_value_t = 0)]
    limit: usize,

    /// swapped pairs per state (each round = 2 games)
    #[clap(long, default_value_t = 1)]
    rounds: usize,

    /// max decision points per game before calling it a draw
    #[clap(long, default_value_t = 500)]
    max_turns: usize,

    #[clap(short = 'v', long, default_value_t = false)]
    verbose: bool,

    #[clap(long, default_value_t = 20000)]
    a_iterations: u32,
    #[clap(long, default_value_t = 0)]
    a_time_ms: u64,
    #[clap(long, default_value_t = 1)]
    a_threads: usize,
    /// UCB1 exploration constant for A
    #[clap(long, default_value_t = DEFAULT_EXPLORATION_CONSTANT)]
    a_c: f32,

    #[clap(long, default_value_t = 20000)]
    b_iterations: u32,
    #[clap(long, default_value_t = 0)]
    b_time_ms: u64,
    #[clap(long, default_value_t = 1)]
    b_threads: usize,
    /// UCB1 exploration constant for B
    #[clap(long, default_value_t = DEFAULT_EXPLORATION_CONSTANT)]
    b_c: f32,
}

#[derive(Clone, Copy)]
struct EngineConfig {
    iterations: u32,
    time_ms: u64,
    threads: usize,
    exploration_constant: f32,
}

impl EngineConfig {
    fn describe(&self) -> String {
        format!(
            "iterations={} time_ms={} threads={} c={}",
            self.iterations, self.time_ms, self.threads, self.exploration_constant
        )
    }
}

enum SideRole {
    SideOne,
    SideTwo,
}

/// run this side's search and return its chosen move: most visits,
/// tie-broken by average score (robust child)
fn pick_move(state: &mut State, config: &EngineConfig, role: &SideRole) -> MoveChoice {
    let (s1_options, s2_options) = state.root_get_all_options();

    // forced decisions don't need a search
    let own_options = match role {
        SideRole::SideOne => &s1_options,
        SideRole::SideTwo => &s2_options,
    };
    match own_options.len() {
        0 => return MoveChoice::None,
        1 => return own_options[0],
        _ => {}
    }

    let max_time = Duration::from_millis(config.time_ms);
    let result = if config.threads > 1 {
        perform_mcts_shared_tree(
            state,
            s1_options,
            s2_options,
            max_time,
            config.iterations,
            config.threads,
            config.exploration_constant,
        )
    } else {
        perform_mcts(
            state,
            s1_options,
            s2_options,
            max_time,
            config.iterations,
            config.exploration_constant,
        )
    };

    let side_result = match role {
        SideRole::SideOne => &result.s1,
        SideRole::SideTwo => &result.s2,
    };
    best_by_visits(side_result)
}

fn best_by_visits(side_result: &[MctsSideResult]) -> MoveChoice {
    let mut best = &side_result[0];
    for candidate in side_result.iter().skip(1) {
        if candidate.visits > best.visits
            || (candidate.visits == best.visits && candidate.average_score() > best.average_score())
        {
            best = candidate;
        }
    }
    best.move_choice
}

fn sample_outcome<'a>(
    instructions: &'a [StateInstructions],
    rng: &mut SmallRng,
) -> &'a StateInstructions {
    let total_weight: f32 = instructions.iter().map(|i| i.percentage.max(0.0)).sum();
    if instructions.len() <= 1 || total_weight <= 0.0 {
        return &instructions[0];
    }
    let mut threshold = rng.random_range(0.0..total_weight);
    for instruction in instructions.iter() {
        threshold -= instruction.percentage.max(0.0);
        if threshold <= 0.0 {
            return instruction;
        }
    }
    &instructions[instructions.len() - 1]
}

struct GameResult {
    /// 1.0 side one won, 0.0 side two won, 0.5 draw (decision cap reached)
    s1_score: f64,
    decisions: usize,
    capped: bool,
}

fn play_game(
    initial_state: &State,
    s1_config: &EngineConfig,
    s2_config: &EngineConfig,
    max_turns: usize,
    verbose: bool,
    rng: &mut SmallRng,
) -> GameResult {
    let mut state = initial_state.clone();
    let mut decisions = 0;
    while decisions < max_turns {
        let over = state.battle_is_over();
        if over != 0.0 {
            return GameResult {
                s1_score: if over > 0.0 { 1.0 } else { 0.0 },
                decisions,
                capped: false,
            };
        }

        let s1_choice = pick_move(&mut state, s1_config, &SideRole::SideOne);
        let s2_choice = pick_move(&mut state, s2_config, &SideRole::SideTwo);
        if verbose {
            println!(
                "    decision {}: s1={} s2={}",
                decisions,
                s1_choice.to_string(&state.side_one),
                s2_choice.to_string(&state.side_two),
            );
        }

        // chance node: sample one outcome of the joint move
        let instructions =
            generate_instructions_from_move_pair(&mut state, &s1_choice, &s2_choice, false);
        if instructions.is_empty() {
            // no legal continuation; treat as a draw rather than crash
            return GameResult {
                s1_score: 0.5,
                decisions,
                capped: true,
            };
        }
        let outcome = sample_outcome(&instructions, rng);
        state.apply_instructions(&outcome.instruction_list);
        decisions += 1;
    }
    GameResult {
        s1_score: 0.5,
        decisions,
        capped: true,
    }
}

// Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7. good enough for LOS
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn main() {
    let args = Args::parse();
    if args.file_name.is_empty() {
        eprintln!("File name is required");
        exit(1);
    }

    let file_path = {
        let this_file = std::path::Path::new(file!());
        let this_dir = this_file.parent().unwrap();
        this_dir.join(&args.file_name)
    };
    let contents = std::fs::read_to_string(file_path).expect("Failed to read the file");
    let lines = contents
        .lines()
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>();

    let mut states = Vec::with_capacity(lines.len());
    for line in lines {
        states.push(State::deserialize(&line))
    }
    let state_limit = if args.limit == 0 {
        states.len()
    } else {
        args.limit.min(states.len())
    };

    let config_a = EngineConfig {
        iterations: args.a_iterations,
        time_ms: args.a_time_ms,
        threads: args.a_threads,
        exploration_constant: args.a_c,
    };
    let config_b = EngineConfig {
        iterations: args.b_iterations,
        time_ms: args.b_time_ms,
        threads: args.b_threads,
        exploration_constant: args.b_c,
    };
    println!("A: {}", config_a.describe());
    println!("B: {}", config_b.describe());
    println!(
        "states={} rounds={} games={}",
        state_limit,
        args.rounds,
        state_limit * args.rounds * 2
    );

    let mut rng = SmallRng::from_os_rng();
    let mut a_scores: Vec<f64> = Vec::new();
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let start_time = std::time::Instant::now();

    for (state_index, state) in states.iter().take(state_limit).enumerate() {
        for round in 0..args.rounds {
            // two games per round with colors swapped
            for a_is_side_one in [true, false] {
                let (s1_config, s2_config) = if a_is_side_one {
                    (&config_a, &config_b)
                } else {
                    (&config_b, &config_a)
                };
                let result = play_game(
                    state,
                    s1_config,
                    s2_config,
                    args.max_turns,
                    args.verbose,
                    &mut rng,
                );
                let a_score = if a_is_side_one {
                    result.s1_score
                } else {
                    1.0 - result.s1_score
                };
                if a_score > 0.5 {
                    wins += 1;
                } else if a_score < 0.5 {
                    losses += 1;
                } else {
                    draws += 1;
                }
                a_scores.push(a_score);

                let games = a_scores.len();
                let points: f64 = a_scores.iter().sum();
                println!(
                    "state={} round={} A={} score(A)={:.1} decisions={}{} | cum: {:.1}/{} ({:.1}%)",
                    state_index,
                    round,
                    if a_is_side_one { "s1" } else { "s2" },
                    a_score,
                    result.decisions,
                    if result.capped { " (capped)" } else { "" },
                    points,
                    games,
                    100.0 * points / games as f64,
                );
            }
        }
    }

    let games = a_scores.len();
    if games == 0 {
        println!("no games played");
        exit(0);
    }
    let points: f64 = a_scores.iter().sum();
    let p = points / games as f64;

    println!("\n=== Summary ===");
    println!("A: {}", config_a.describe());
    println!("B: {}", config_b.describe());
    println!(
        "games={} W/L/D={}/{}/{} points={:.1} winrate={:.1}%",
        games,
        wins,
        losses,
        draws,
        points,
        100.0 * p
    );

    let clamp = |x: f64| x.clamp(0.001, 0.999);
    let elo = |p: f64| -400.0 * (1.0 / clamp(p) - 1.0).log10();
    if games > 1 {
        let variance = a_scores.iter().map(|s| (s - p).powi(2)).sum::<f64>() / (games as f64 - 1.0);
        let standard_error = (variance / games as f64).sqrt();
        println!(
            "elo diff (A vs B): {:+.1} [{:+.1}, {:+.1}] (95% CI)",
            elo(p),
            elo(p - 1.96 * standard_error),
            elo(p + 1.96 * standard_error),
        );
    }
    if wins + losses > 0 {
        let los = 0.5
            * (1.0 + erf((wins as f32 - losses as f32) / (2.0 * (wins + losses) as f32).sqrt()))
                as f64;
        println!("LOS (A stronger than B): {:.1}%", 100.0 * los);
    }
    println!("Took: {:.1} seconds", start_time.elapsed().as_secs_f64());
    exit(0);
}
