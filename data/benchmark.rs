use clap::Parser;
use poke_engine::mcts::{perform_mcts, DEFAULT_EXPLORATION_CONSTANT};
use poke_engine::mcts_threaded::perform_mcts_shared_tree;
use poke_engine::state::State;
use std::process::exit;

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
            perform_mcts_shared_tree(
                state,
                side_one_options,
                side_two_options,
                max_time,
                args.iterations,
                args.threads,
                DEFAULT_EXPLORATION_CONSTANT,
            )
        } else {
            perform_mcts(
                state,
                side_one_options,
                side_two_options,
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
