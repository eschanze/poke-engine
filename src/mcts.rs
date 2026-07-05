use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::{Instruction, StateInstructions};
use crate::state::State;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::time::Duration;

// UCB1 exploration constant c in `avg + c * sqrt(ln(N) / n)`.
// Tuned by self-play (2026-07-03, see WORKLOG.md): 0.5 beats the classical
// sqrt(2) by ~+87 Elo at 20k iterations and ~+108 Elo at 100ms/12-thread
// searches; 0.3 and 0.75 measured slightly worse than 0.5.
pub const DEFAULT_EXPLORATION_CONSTANT: f32 = 0.5;

// When true (default since 2026-07-04), damage-roll/crit branching happens
// at every tree depth instead of only the first two plies: damage nodes
// split into KO/no-KO (or crit/no-crit) outcomes weighted by roll
// probability. Selfplay-validated as Elo-neutral vs the 2-ply limit at both
// 20k-iteration and 100ms/12T budgets (612 + 50 games, see WORKLOG), and it
// models kill ranges honestly deep in the tree. The selfplay harness can
// flip it per side (--a-branch-all/--b-branch-all) to re-test after eval
// changes. Not part of the public API.
pub static BRANCH_ALL_DEPTHS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

// Approximate cap on tree memory, replacing the old fixed 10M iteration cap.
// The estimate under-counts allocator overhead, so keep some headroom below
// physical RAM.
pub(crate) const MCTS_MAX_TREE_BYTES: u64 = 24 * 1024 * 1024 * 1024;

// rough per-branch estimate: map entry overhead + node structs + instruction
// heap allocations. per-node constant absorbs options vecs and allocator slack
pub(crate) const MCTS_BRANCH_ENTRY_OVERHEAD: usize = 64;
pub(crate) const MCTS_NODE_OVERHEAD: usize = 256;
pub(crate) const TERMINAL_UNKNOWN: i8 = 0;
pub(crate) const TERMINAL_SIDE_ONE_WIN: i8 = 1;
pub(crate) const TERMINAL_SIDE_TWO_WIN: i8 = -1;
pub(crate) const TERMINAL_NONTERMINAL: i8 = 2;

pub(crate) fn terminal_result_from_battle_result(result: f32) -> i8 {
    if result == 0.0 {
        TERMINAL_NONTERMINAL
    } else if result == -1.0 {
        TERMINAL_SIDE_TWO_WIN
    } else {
        TERMINAL_SIDE_ONE_WIN
    }
}

pub(crate) fn terminal_score_from_cached_result(result: i8) -> Option<f32> {
    match result {
        TERMINAL_SIDE_ONE_WIN => Some(1.0),
        TERMINAL_SIDE_TWO_WIN => Some(0.0),
        _ => None,
    }
}

fn approx_branch_bytes(nodes: &[Node]) -> u64 {
    let instr_bytes: usize = nodes
        .iter()
        .map(|n| n.instructions.instruction_list.capacity() * std::mem::size_of::<Instruction>())
        .sum();
    (MCTS_BRANCH_ENTRY_OVERHEAD
        + nodes.len() * (std::mem::size_of::<Node>() + MCTS_NODE_OVERHEAD)
        + instr_bytes) as u64
}

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u64,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,

    // expanded move-pairs: (s1_idx * s2_options.len() + s2_idx, outcome
    // branch). A linear scan beats any hash map here: most nodes only ever
    // expand a handful of pairs, and the entries are hot in cache
    pub children: Vec<(u32, Box<[Node]>)>,

    // Cached battle-over state for this node. This avoids repeatedly scanning
    // teams and, for terminal children, avoids option generation below a battle
    // that is already over.
    pub terminal_result: i8,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            children: Vec::new(),
            terminal_result: TERMINAL_UNKNOWN,
        }
    }

    fn terminal_score(&mut self, state: &State) -> Option<f32> {
        if self.root {
            return None;
        }
        if self.terminal_result == TERMINAL_UNKNOWN {
            self.terminal_result = terminal_result_from_battle_result(state.battle_is_over());
        }
        terminal_score_from_cached_result(self.terminal_result)
    }

    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode], exploration_sq: f32) -> usize {
        // ln(parent_visits) is the same for every option; compute it once
        let ln_parent_visits = (self.times_visited as f32).ln();
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1_with_ln(ln_parent_visits, exploration_sq);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        rng: &mut impl Rng,
        exploration_sq: f32,
    ) -> (*mut Node, usize, usize) {
        if self.terminal_score(state).is_some() {
            return (self as *mut Node, 0, 0);
        }

        if self.s1_options.is_none() {
            crate::prof_scope!(crate::prof::sec::GET_OPTIONS);
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index =
            self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap(), exploration_sq);
        let s2_mc_index =
            self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap(), exploration_sq);
        let key = (s1_mc_index * self.s2_options.as_ref().unwrap().len() + s2_mc_index) as u32;
        let child_vec_ptr: *mut Box<[Node]> =
            match self.children.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => &mut entry.1,
                None => return (self as *mut Node, s1_mc_index, s2_mc_index),
            };
        let chosen_child = self.sample_node(child_vec_ptr, rng);
        state.apply_instructions(&(*chosen_child).instructions.instruction_list);
        (*chosen_child).selection(state, rng, exploration_sq)
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>, rng: &mut impl Rng) -> *mut Node {
        let nodes = &mut **move_vector;

        let total_weight: f32 = nodes
            .iter()
            .map(|n| n.instructions.percentage.max(0.0))
            .sum();

        let mut threshold = rng.random_range(0.0..total_weight);

        for node in nodes.iter_mut() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node as *mut Node;
            }
        }

        // fallback: return last node (handles float rounding issues that can come up)
        &mut nodes[nodes.len() - 1] as *mut Node
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        rng: &mut impl Rng,
        tree_bytes: &mut u64,
    ) -> *mut Node {
        if self.terminal_score(state).is_some() {
            return self as *mut Node;
        }

        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if both moves are none there is no need to expand. terminal (battle
        // over) non-root nodes were already handled by terminal_score above
        if s1_move == &MoveChoice::None && s2_move == &MoveChoice::None {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root
            || (*self.parent).root
            || BRANCH_ALL_DEPTHS.load(std::sync::atomic::Ordering::Relaxed);
        let mut new_instructions = {
            crate::prof_scope!(crate::prof::sec::GEN_INS);
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage)
        };
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized: parent pointers reference
        // nodes inside this heap slice, and while the children Vec may
        // reallocate (moving the Box), the slice itself never moves
        let boxed = this_pair_vec.into_boxed_slice();
        *tree_bytes += approx_branch_bytes(&boxed);
        let key = (s1_move_index * self.s2_options.as_ref().unwrap().len() + s2_move_index) as u32;
        self.children.push((key, boxed));

        // sample a node from the new branch; the rollout will be done on it
        let branch: *mut Box<[Node]> = &mut self.children.last_mut().unwrap().1;
        let new_node_ptr = self.sample_node(branch, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score as f64;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += (1.0 - score) as f64;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32) -> f32 {
        if self.root {
            if let Some(score) = terminal_score_from_cached_result(
                terminal_result_from_battle_result(state.battle_is_over()),
            ) {
                return score;
            }
        } else if let Some(score) = self.terminal_score(state) {
            return score;
        }

        let eval = evaluate(state);
        sigmoid(eval - root_eval)
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f64,
    pub visits: u64,
}

impl MoveNode {
    // exploration_sq is c^2: the formula is avg + sqrt(c^2 * ln(N) / n)
    fn ucb1_with_ln(&self, ln_parent_visits: f32, exploration_sq: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f64) as f32
            + (exploration_sq * ln_parent_visits / self.visits as f32).sqrt();
        score
    }
    pub fn ucb1(&self, parent_visits: u64) -> f32 {
        self.ucb1_with_ln(
            (parent_visits as f32).ln(),
            DEFAULT_EXPLORATION_CONSTANT * DEFAULT_EXPLORATION_CONSTANT,
        )
    }
    pub fn average_score(&self) -> f32 {
        let score = (self.total_score / self.visits as f64) as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f64,
    pub visits: u64,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = (self.total_score / self.visits as f64) as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u64,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    rng: &mut impl Rng,
    tree_bytes: &mut u64,
    exploration_sq: f32,
) {
    let (mut new_node, s1_move, s2_move) = {
        crate::prof_scope!(crate::prof::sec::SELECTION);
        unsafe { root_node.selection(state, rng, exploration_sq) }
    };
    new_node = {
        crate::prof_scope!(crate::prof::sec::EXPAND);
        unsafe { (*new_node).expand(state, s1_move, s2_move, rng, tree_bytes) }
    };
    let rollout_result = {
        crate::prof_scope!(crate::prof::sec::ROLLOUT);
        unsafe { (*new_node).rollout(state, root_eval) }
    };
    {
        crate::prof_scope!(crate::prof::sec::BACKPROP);
        unsafe { (*new_node).backpropagate(rollout_result, state) }
    }
}

#[derive(Clone, Copy)]
enum SearchLimit {
    Time(Duration),
    Iterations(u32),
    TimeOrIterations(Duration, u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    limit: SearchLimit,
    exploration_sq: f32,
) {
    // SmallRng is much cheaper than the default crypto-grade ThreadRng and
    // statistical quality is all that matters here
    let mut rng = SmallRng::from_os_rng();
    let start_time = std::time::Instant::now();
    let mut tree_bytes: u64 = 0;
    loop {
        for _ in 0..1000 {
            mcts_iteration(
                root_node,
                state,
                root_eval,
                &mut rng,
                &mut tree_bytes,
                exploration_sq,
            );
        }
        if tree_bytes >= MCTS_MAX_TREE_BYTES {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n as u64 {
                    break;
                }
            }
            SearchLimit::TimeOrIterations(max_time, n) => {
                if start_time.elapsed() >= max_time || root_node.times_visited >= n as u64 {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;

    let root_eval = evaluate(state);
    let search_limit = if max_iterations > 0 && max_time > Duration::from_millis(0) {
        SearchLimit::TimeOrIterations(max_time, max_iterations)
    } else if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &root_eval,
        search_limit,
        exploration_constant * exploration_constant,
    );

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}
