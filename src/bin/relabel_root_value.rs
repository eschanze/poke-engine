// Offline distillation labeler. Reads a trajectory JSONL dump (selfplay
// --dump-trajectories output, before or after subsample.py), runs a fresh
// fixed-iteration single-threaded MCTS on every stored state with the default
// engine configuration, and rewrites each line with the searcher's root value
// (visit-weighted mean score from side one's perspective, in [0, 1]) inserted
// before the state field. The state stays the final JSON field so downstream
// parsers (eval-pair-features) keep working.
//
// Positions are independent, so the work is distributed across worker threads
// with one single-threaded search each; output order matches input order.

use clap::Parser;
use poke_engine::mcts::{perform_mcts, DEFAULT_EXPLORATION_CONSTANT};
use poke_engine::state::State;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Parser)]
struct Args {
    /// trajectory JSONL whose lines contain a serialized "state" field
    #[clap(long)]
    input: String,
    /// output JSONL with "root_value" and "teacher_iterations" added per line
    #[clap(long)]
    output: String,
    /// fixed MCTS iterations per position (the teacher search budget)
    #[clap(long, default_value_t = 1_000_000)]
    iterations: u32,
    /// worker threads; each runs an independent single-threaded search
    #[clap(long, default_value_t = 8)]
    threads: usize,
    /// only relabel the first N non-empty lines (0 = all); for smoke runs
    #[clap(long, default_value_t = 0)]
    limit: usize,
}

/// Byte offset of the `"state"` key. Mirrors the assumption in
/// eval-pair-features: the first occurrence of `"state"` is the key (earlier
/// fields are numeric/boolean except the schema hash, which cannot contain it).
fn state_key_offset(line: &str, line_no: usize) -> usize {
    line.find("\"state\"")
        .unwrap_or_else(|| panic!("line {}: missing state field", line_no))
}

/// The serialized state string inside the line; requires state to be the
/// final field, same contract as the dump writer and eval-pair-features.
fn serialized_state(line: &str, key: usize, line_no: usize) -> &str {
    let after_key = &line[key + "\"state\"".len()..];
    let colon = after_key
        .find(':')
        .unwrap_or_else(|| panic!("line {}: malformed state field", line_no));
    let after_colon = &after_key[colon + 1..];
    let quote = after_colon.len() - after_colon.trim_start().len();
    let value = &after_colon[quote..];
    if !value.starts_with('"') {
        panic!("line {}: state must be a JSON string", line_no);
    }
    let value = &value[1..];
    value
        .strip_suffix("\"}")
        .unwrap_or_else(|| panic!("line {}: state must be the final JSON field", line_no))
}

fn root_value(state: &mut State, iterations: u32) -> f64 {
    let over = state.battle_is_over();
    if over != 0.0 {
        return if over > 0.0 { 1.0 } else { 0.0 };
    }
    let (s1_options, s2_options) = state.root_get_all_options();
    let result = perform_mcts(
        state,
        s1_options,
        s2_options,
        Duration::ZERO,
        iterations,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    let mut score = 0.0;
    let mut visits = 0u64;
    for side_result in &result.s1 {
        score += side_result.total_score;
        visits += side_result.visits;
    }
    if visits == 0 {
        panic!("search returned zero root visits");
    }
    score / visits as f64
}

fn main() {
    let args = Args::parse();
    let input = std::fs::File::open(&args.input)
        .unwrap_or_else(|error| panic!("failed to open {}: {}", args.input, error));
    let mut lines: Vec<(usize, String)> = Vec::new();
    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line_no = line_index + 1;
        let line = line.unwrap_or_else(|error| panic!("line {}: {}", line_no, error));
        if line.trim().is_empty() {
            continue;
        }
        if line.contains("\"root_value\"") {
            panic!(
                "line {}: already has root_value; relabel the original dump instead",
                line_no
            );
        }
        lines.push((line_no, line));
        if args.limit > 0 && lines.len() == args.limit {
            break;
        }
    }
    let total = lines.len();
    println!(
        "relabeling {} positions at {} iterations on {} threads",
        total, args.iterations, args.threads
    );

    let started = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<String>>> = Mutex::new(vec![None; total]);
    let progress_every = (total / 20).max(1);
    std::thread::scope(|scope| {
        for _ in 0..args.threads.max(1) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= total {
                    break;
                }
                let (line_no, line) = &lines[index];
                let key = state_key_offset(line, *line_no);
                let mut state = State::deserialize(serialized_state(line, key, *line_no));
                let value = root_value(&mut state, args.iterations);
                let relabeled = format!(
                    "{}\"root_value\":{},\"teacher_iterations\":{},{}",
                    &line[..key],
                    value,
                    args.iterations,
                    &line[key..]
                );
                results.lock().unwrap()[index] = Some(relabeled);
                let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                if finished % progress_every == 0 || finished == total {
                    let elapsed = started.elapsed().as_secs_f64();
                    println!(
                        "{}/{} positions ({:.1}/min, ~{:.0}s remaining)",
                        finished,
                        total,
                        finished as f64 / elapsed * 60.0,
                        elapsed / finished as f64 * (total - finished) as f64
                    );
                }
            });
        }
    });

    let output = std::fs::File::create(&args.output)
        .unwrap_or_else(|error| panic!("failed to create {}: {}", args.output, error));
    let mut writer = BufWriter::new(output);
    for relabeled in results.into_inner().unwrap() {
        writeln!(writer, "{}", relabeled.expect("worker skipped a position"))
            .expect("failed to write output");
    }
    writer.flush().expect("failed to flush output");
    println!(
        "wrote {} relabeled positions to {} in {:.0}s",
        total,
        args.output,
        started.elapsed().as_secs_f64()
    );
}
