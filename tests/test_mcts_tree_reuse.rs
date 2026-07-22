#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::engine::evaluate::{EvalConfig, DEFAULT_EVAL_WEIGHTS};
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::Instruction;
use poke_engine::mcts::{
    perform_mcts_ponder, perform_mcts_with_tree, perform_mcts_with_tree_and_eval, MctsSideResult,
    ReusableTree, DEFAULT_EXPLORATION_CONSTANT,
};
use poke_engine::state::{SideReference, State};
use std::time::Duration;

fn first_bundled_state() -> State {
    let contents = std::fs::read_to_string("data/datasets/battle-factory/no-ubers-states.txt")
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
fn test_advance_to_state_matches_by_resulting_state() {
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
    let outcomes = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, true);
    let outcome = outcomes
        .iter()
        .max_by(|a, b| a.percentage.total_cmp(&b.percentage))
        .unwrap();

    // the observed post-turn state, as a bot would see it
    let mut target = state.clone();
    target.apply_instructions(&outcome.instruction_list);

    let before = state.serialize();
    assert!(tree.advance_to_state(&mut state, &s1_move, &s2_move, &target));
    assert!(tree.root_visits() > 0);
    assert_eq!(state.serialize(), before, "root state must be restored");

    // a state matching no predicted outcome clears the tree
    let result = perform_mcts_with_tree(
        &mut tree,
        &mut target.clone(),
        target.get_all_options().0,
        target.get_all_options().1,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    let s1_move = best_by_visits(&result.s1);
    let s2_move = best_by_visits(&result.s2);
    let unrelated = first_bundled_state();
    assert!(!tree.advance_to_state(&mut target, &s1_move, &s2_move, &unrelated));
    assert_eq!(tree.root_visits(), 0);
}

#[test]
fn test_ponder_pins_the_committed_root_move() {
    let mut state = first_bundled_state();
    let mut tree = ReusableTree::new();
    let (s1_options, s2_options) = state.root_get_all_options();

    let result = perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options.clone(),
        s2_options.clone(),
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    let committed = best_by_visits(&result.s1);

    let ponder_result = perform_mcts_ponder(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        SideReference::SideOne,
        &committed,
        Duration::ZERO,
        5000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    assert_eq!(ponder_result.iteration_count, 5000);

    // every ponder iteration flows through the committed move; the other
    // moves gain nothing
    for (before, after) in result.s1.iter().zip(ponder_result.s1.iter()) {
        if after.move_choice == committed {
            assert_eq!(after.visits, before.visits + 5000);
        } else {
            assert_eq!(after.visits, before.visits);
        }
    }
    // the opponent's side still explores freely during the ponder
    let s2_new_visits: u64 = ponder_result
        .s2
        .iter()
        .zip(result.s2.iter())
        .map(|(after, before)| after.visits - before.visits)
        .sum();
    assert_eq!(s2_new_visits, 5000);
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

#[test]
fn test_changing_eval_config_discards_reusable_tree_statistics() {
    let mut state = first_bundled_state();
    let mut tree = ReusableTree::new();
    let (s1_options, s2_options) = state.root_get_all_options();

    perform_mcts_with_tree(
        &mut tree,
        &mut state,
        s1_options.clone(),
        s2_options.clone(),
        Duration::ZERO,
        1000,
        DEFAULT_EXPLORATION_CONSTANT,
    );
    assert_eq!(tree.root_visits(), 1000);

    let linear_config = EvalConfig::new(&DEFAULT_EVAL_WEIGHTS, true);
    let result = perform_mcts_with_tree_and_eval(
        &mut tree,
        &mut state,
        s1_options,
        s2_options,
        linear_config,
        Duration::ZERO,
        1000,
        DEFAULT_EXPLORATION_CONSTANT,
    );

    assert_eq!(result.iteration_count, 1000);
    assert_eq!(tree.root_visits(), 1000, "config change must cold-start");
}
