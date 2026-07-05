#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::mcts::{
    perform_mcts_with_tree, MctsSideResult, ReusableTree, DEFAULT_EXPLORATION_CONSTANT,
};
use poke_engine::state::State;
use std::time::Duration;

fn first_bundled_state() -> State {
    let contents = std::fs::read_to_string("data/gen9randombattle.txt")
        .expect("bundled state file should exist");
    let line = contents
        .lines()
        .find(|line| !line.is_empty())
        .expect("state file should not be empty");
    State::deserialize(line)
}

fn best_by_visits(side_result: &[MctsSideResult]) -> MoveChoice {
    let mut best = &side_result[0];
    for candidate in side_result.iter().skip(1) {
        if candidate.visits > best.visits {
            best = candidate;
        }
    }
    best.move_choice
}

#[test]
fn test_advance_keeps_subtree_and_iteration_counts_are_per_search() {
    let mut state = first_bundled_state();
    let mut tree = ReusableTree::new();
    let (s1_options, s2_options) = state.root_get_all_options();

    let result = perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    assert_eq!(result.iteration_count, 5000);
    assert_eq!(tree.root_visits(), 5000);

    // play the most-visited pair; generate the outcomes exactly like the
    // tree's expansion did (branch_on_damage=true at the root) so the
    // sampled outcome is guaranteed to match a child
    let s1_move = best_by_visits(&result.s1);
    let s2_move = best_by_visits(&result.s2);
    let outcomes = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, true);
    let outcome = outcomes
        .iter()
        .max_by(|a, b| a.percentage.total_cmp(&b.percentage))
        .unwrap();

    assert!(tree.advance(&s1_move, &s2_move, &outcome.instruction_list));
    let warm_visits = tree.root_visits();
    assert!(
        warm_visits > 0,
        "the most likely outcome of the most-visited pair should have visits"
    );

    // continue the search from the reused subtree; internal nodes store
    // options from get_all_options, so that is what must match
    state.apply_instructions(&outcome.instruction_list);
    let (s1_options, s2_options) = state.get_all_options();
    let result = perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    assert_eq!(
        result.iteration_count, 5000,
        "iteration limit is per-search"
    );
    assert_eq!(
        tree.root_visits(),
        warm_visits + 5000,
        "warm visits carry over"
    );
}

#[test]
fn test_advance_mismatch_clears_the_tree() {
    let mut state = first_bundled_state();
    let mut tree = ReusableTree::new();
    let (s1_options, s2_options) = state.root_get_all_options();

    let result = perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    let s1_move = best_by_visits(&result.s1);
    let s2_move = best_by_visits(&result.s2);

    // an instruction list that matches no generated branch
    let bogus = [Instruction::DecrementWeatherTurnsRemaining];
    assert!(!tree.advance(&s1_move, &s2_move, &bogus));
    assert_eq!(tree.root_visits(), 0, "mismatch discards the tree");

    // a cleared tree still searches (cold start)
    let (s1_options, s2_options) = state.root_get_all_options();
    let result = perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    assert_eq!(result.iteration_count, 5000);
    assert_eq!(tree.root_visits(), 5000);
}
