use clap::Parser;
use poke_engine::mcts::{perform_mcts_with_eval, DEFAULT_EXPLORATION_CONSTANT};
use poke_engine::mcts_threaded::perform_mcts_shared_tree_with_eval;
use poke_engine::state::State;
use std::process::exit;

#[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
mod eval_api {
    pub use poke_engine::engine::evaluate::{
        parse_eval_context_model, parse_eval_tree_model, EvalConfig, EvalContextModel,
        EvalTreeModel,
    };
    pub fn config_from_tree(model: &'static EvalTreeModel) -> EvalConfig {
        EvalConfig::from_tree_model(model)
    }
    pub fn config_from_context(model: &'static EvalContextModel) -> EvalConfig {
        EvalConfig::from_context_model(model)
    }
}

#[cfg(any(feature = "gen1", feature = "gen2", feature = "gen3"))]
mod eval_api {
    pub use poke_engine::engine::evaluate::EvalConfig;
    pub struct EvalTreeModel;
    pub struct EvalContextModel;
    pub fn parse_eval_tree_model(_text: &str) -> Result<EvalTreeModel, String> {
        Err("eval tree models are only supported for gen4+ builds".to_string())
    }
    pub fn config_from_tree(_model: &'static EvalTreeModel) -> EvalConfig {
        EvalConfig
    }
    pub fn parse_eval_context_model(_text: &str) -> Result<EvalContextModel, String> {
        Err("eval context models are only supported for gen4+ builds".to_string())
    }
    pub fn config_from_context(_model: &'static EvalContextModel) -> EvalConfig {
        EvalConfig
    }
}

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    file_name: String,

    #[clap(short = 'i', long, default_value_t = 250000)]
    iterations: u32,

    #[clap(short = 't', long, default_value_t = 0)]
    time_ms: u64,

    #[clap(short = 'n', long, default_value_t = 1)]
    threads: usize,

    #[clap(short = 'l', long, default_value_t = 0)]
    limit: usize,

    /// optional boosted-tree evaluator (gen4+ only)
    #[clap(long)]
    eval_trees: Option<String>,

    /// optional contextual MLP evaluator (gen4+ only)
    #[clap(long)]
    eval_mlp: Option<String>,
}

fn main() {
    let args = Args::parse();
    if args.file_name.is_empty() {
        eprintln!("File name is required");
        exit(1);
    }
    if args.eval_trees.is_some() && args.eval_mlp.is_some() {
        eprintln!("--eval-trees and --eval-mlp are mutually exclusive");
        exit(1);
    }
    let eval_config = if let Some(ref path) = args.eval_mlp {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read eval MLP model {}: {}", path, e));
        let model = eval_api::parse_eval_context_model(&text)
            .unwrap_or_else(|e| panic!("bad eval MLP model {}: {}", path, e));
        eval_api::config_from_context(&*Box::leak(Box::new(model)))
    } else {
        match args.eval_trees {
            Some(ref path) => {
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("failed to read eval tree model {}: {}", path, e));
                let model = eval_api::parse_eval_tree_model(&text)
                    .unwrap_or_else(|e| panic!("bad eval tree model {}: {}", path, e));
                eval_api::config_from_tree(&*Box::leak(Box::new(model)))
            }
            None => eval_api::EvalConfig::default(),
        }
    };

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

    let start_time = std::time::Instant::now();
    let state_limit = if args.limit == 0 {
        states.len()
    } else {
        args.limit.min(states.len())
    };
    for (i, state) in states.iter_mut().take(state_limit).enumerate() {
        let (side_one_options, side_two_options) = state.root_get_all_options();
        let max_time = std::time::Duration::from_millis(args.time_ms);

        let state_start = std::time::Instant::now();
        let result = if args.threads > 1 {
            perform_mcts_shared_tree_with_eval(
                state,
                side_one_options,
                side_two_options,
                eval_config,
                max_time,
                args.iterations,
                args.threads,
                DEFAULT_EXPLORATION_CONSTANT,
            )
        } else {
            perform_mcts_with_eval(
                state,
                side_one_options,
                side_two_options,
                eval_config,
                max_time,
                args.iterations,
                DEFAULT_EXPLORATION_CONSTANT,
            )
        };
        let state_elapsed = state_start.elapsed().as_secs_f64();
        println!(
            "{}: iterations={} elapsed={:.3}s it/s={:.0}",
            i,
            result.iteration_count,
            state_elapsed,
            result.iteration_count as f64 / state_elapsed
        );
    }

    let elapsed_time = start_time.elapsed().as_secs_f64();
    println!("Took: {} seconds", elapsed_time);

    // thread-local counters: only meaningful for single-threaded runs (-n 1)
    #[cfg(feature = "prof")]
    poke_engine::prof::report();

    exit(0);
}
