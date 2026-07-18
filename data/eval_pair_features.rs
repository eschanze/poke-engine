use clap::Parser;
use poke_engine::engine::evaluate::eval_pair_features;
use poke_engine::state::State;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Parser)]
struct Args {
    /// Existing trajectory JSONL containing serialized states.
    #[clap(long)]
    input: String,
    /// Output JSONL with side_one and side_two feature vectors.
    #[clap(long)]
    output: String,
}

fn serialized_state(line: &str, line_no: usize) -> Result<&str, String> {
    let key = line
        .find("\"state\"")
        .ok_or_else(|| format!("line {}: missing state", line_no))?
        + "\"state\"".len();
    let after_key = &line[key..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| format!("line {}: malformed state field", line_no))?;
    let after_colon = &after_key[colon + 1..];
    let leading_space = after_colon.len() - after_colon.trim_start().len();
    let quote = key + colon + 1 + leading_space;
    if line.as_bytes().get(quote) != Some(&b'"') {
        return Err(format!("line {}: state must be a JSON string", line_no));
    }
    let start = quote + 1;
    let end = line
        .strip_suffix("\"}")
        .ok_or_else(|| format!("line {}: state must be the final JSON field", line_no))?
        .len();
    if end < start {
        return Err(format!("line {}: malformed state field", line_no));
    }
    Ok(&line[start..end])
}

fn write_vector(output: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    output.write_all(b"[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.write_all(b",")?;
        }
        write!(output, "{}", value)?;
    }
    output.write_all(b"]")
}

fn main() {
    let args = Args::parse();
    let input = std::fs::File::open(&args.input)
        .unwrap_or_else(|error| panic!("failed to open {}: {}", args.input, error));
    let output = std::fs::File::create(&args.output)
        .unwrap_or_else(|error| panic!("failed to create {}: {}", args.output, error));
    let mut output = BufWriter::new(output);
    let mut count = 0usize;
    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line_no = line_index + 1;
        let line = line.unwrap_or_else(|error| panic!("line {}: {}", line_no, error));
        if line.trim().is_empty() {
            continue;
        }
        let state_text =
            serialized_state(&line, line_no).unwrap_or_else(|error| panic!("{}", error));
        let state = State::deserialize(state_text);
        let pair = eval_pair_features(&state);
        output.write_all(b"{\"side_one\":").unwrap();
        write_vector(&mut output, &pair[0]).unwrap();
        output.write_all(b",\"side_two\":").unwrap();
        write_vector(&mut output, &pair[1]).unwrap();
        output.write_all(b"}\n").unwrap();
        count += 1;
    }
    output.flush().unwrap();
    println!("wrote {} paired feature rows to {}", count, args.output);
}
