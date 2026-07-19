use crate::engine::evaluate::{evaluate_with_config, EvalConfig};
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::{Instruction, StateInstructions};
use crate::state::{SideReference, State};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::time::Duration;

// UCB1 exploration constant c in `avg + c * sqrt(ln(N) / n)`.
// Tuned by self-play on 2026-07-03: 0.5 beats the classical
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

// When true, an expanded move pair whose every positive-weight chance outcome
// is terminal gets an exact expected score cached at the pair level. Future
// visits back up that expectation directly instead of resampling terminal
// outcome children. Selfplay can flip this per side for A/B validation.
pub static TERMINAL_PAIR_CACHE: std::sync::atomic::AtomicBool =
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

fn terminal_pair_score_for_nodes(state: &mut State, nodes: &mut [Node]) -> Option<f32> {
    if !TERMINAL_PAIR_CACHE.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }

    let mut total_weight = 0.0;
    let mut weighted_score = 0.0;

    for node in nodes.iter_mut() {
        let weight = node.instructions.percentage.max(0.0);
        if weight == 0.0 {
            continue;
        }

        state.apply_instructions(&node.instructions.instruction_list);
        let terminal_result = terminal_result_from_battle_result(state.battle_is_over());
        node.terminal_result = terminal_result;
        state.reverse_instructions(&node.instructions.instruction_list);

        let score = terminal_score_from_cached_result(terminal_result)?;
        total_weight += weight;
        weighted_score += weight * score;
    }

    if total_weight > 0.0 {
        Some(weighted_score / total_weight)
    } else {
        None
    }
}

#[derive(Debug)]
pub struct ChildBranch {
    pub key: u32,
    pub nodes: Box<[Node]>,
    pub terminal_score: Option<f32>,
}

struct SelectionResult {
    node: *mut Node,
    s1_move: usize,
    s2_move: usize,
    terminal_pair_score: Option<f32>,
}

enum ExpansionResult {
    Child(*mut Node),
    TerminalPair(f32),
    NoExpansion,
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

    // expanded move-pairs. A linear scan beats any hash map here: most nodes
    // only ever expand a handful of pairs, and the entries are hot in cache
    pub children: Vec<ChildBranch>,

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

    unsafe fn selection(
        &mut self,
        state: &mut State,
        rng: &mut impl Rng,
        exploration_sq: f32,
        root_force: (Option<usize>, Option<usize>),
    ) -> SelectionResult {
        // root_force pins one side's root move while pondering a committed
        // move; it is consumed on the first (root) step so the walk below
        // pays nothing for it
        let mut force = root_force;
        let mut current: *mut Node = self;
        loop {
            let node = &mut *current;
            if node.terminal_score(state).is_some() {
                return SelectionResult {
                    node: current,
                    s1_move: 0,
                    s2_move: 0,
                    terminal_pair_score: None,
                };
            }

            if node.s1_options.is_none() {
                crate::prof_scope!(crate::prof::sec::GET_OPTIONS);
                let (s1_options, s2_options) = state.get_all_options();
                node.populate(s1_options, s2_options);
            }

            let s1_mc_index = match force.0 {
                Some(index) => index,
                None => {
                    node.maximize_ucb_for_side(node.s1_options.as_ref().unwrap(), exploration_sq)
                }
            };
            let s2_mc_index = match force.1 {
                Some(index) => index,
                None => {
                    node.maximize_ucb_for_side(node.s2_options.as_ref().unwrap(), exploration_sq)
                }
            };
            force = (None, None);
            let key = (s1_mc_index * node.s2_options.as_ref().unwrap().len() + s2_mc_index) as u32;
            let child_vec_ptr: *mut Box<[Node]> =
                match node.children.iter_mut().find(|branch| branch.key == key) {
                    Some(branch) => {
                        if let Some(score) = branch.terminal_score {
                            return SelectionResult {
                                node: current,
                                s1_move: s1_mc_index,
                                s2_move: s2_mc_index,
                                terminal_pair_score: Some(score),
                            };
                        }
                        &mut branch.nodes
                    }
                    None => {
                        return SelectionResult {
                            node: current,
                            s1_move: s1_mc_index,
                            s2_move: s2_mc_index,
                            terminal_pair_score: None,
                        };
                    }
                };
            let chosen_child = node.sample_node(child_vec_ptr, rng);
            state.apply_instructions(&(*chosen_child).instructions.instruction_list);
            current = chosen_child;
        }
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

    unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        rng: &mut impl Rng,
        tree_bytes: &mut u64,
    ) -> ExpansionResult {
        if self.terminal_score(state).is_some() {
            return ExpansionResult::NoExpansion;
        }

        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if both moves are none there is no need to expand. terminal (battle
        // over) non-root nodes were already handled by terminal_score above
        if s1_move == &MoveChoice::None && s2_move == &MoveChoice::None {
            return ExpansionResult::NoExpansion;
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
        let mut boxed = this_pair_vec.into_boxed_slice();
        let terminal_score = terminal_pair_score_for_nodes(state, &mut boxed);
        *tree_bytes += approx_branch_bytes(&boxed);
        let key = (s1_move_index * self.s2_options.as_ref().unwrap().len() + s2_move_index) as u32;
        self.children.push(ChildBranch {
            key,
            nodes: boxed,
            terminal_score,
        });

        if let Some(score) = terminal_score {
            return ExpansionResult::TerminalPair(score);
        }

        // sample a node from the new branch; the rollout will be done on it
        let branch: *mut Box<[Node]> = &mut self.children.last_mut().unwrap().nodes;
        let new_node_ptr = self.sample_node(branch, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        ExpansionResult::Child(new_node_ptr)
    }

    unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
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

    unsafe fn backpropagate_pair(
        &mut self,
        score: f32,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) {
        self.times_visited += 1;

        let s1_movenode = &mut self.s1_options.as_mut().unwrap()[s1_move_index];
        s1_movenode.total_score += score as f64;
        s1_movenode.visits += 1;

        let s2_movenode = &mut self.s2_options.as_mut().unwrap()[s2_move_index];
        s2_movenode.total_score += (1.0 - score) as f64;
        s2_movenode.visits += 1;

        if self.root {
            return;
        }

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32, eval_config: EvalConfig) -> f32 {
        if self.root {
            if let Some(score) = terminal_score_from_cached_result(
                terminal_result_from_battle_result(state.battle_is_over()),
            ) {
                return score;
            }
        } else if let Some(score) = self.terminal_score(state) {
            return score;
        }

        let eval = evaluate_with_config(state, eval_config);
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
    root_force: (Option<usize>, Option<usize>),
    eval_config: EvalConfig,
) {
    let selected = {
        crate::prof_scope!(crate::prof::sec::SELECTION);
        unsafe { root_node.selection(state, rng, exploration_sq, root_force) }
    };

    if let Some(score) = selected.terminal_pair_score {
        crate::prof_scope!(crate::prof::sec::BACKPROP);
        unsafe {
            (*selected.node).backpropagate_pair(score, state, selected.s1_move, selected.s2_move)
        }
        return;
    }

    let expanded = {
        crate::prof_scope!(crate::prof::sec::EXPAND);
        unsafe {
            (*selected.node).expand(state, selected.s1_move, selected.s2_move, rng, tree_bytes)
        }
    };

    match expanded {
        ExpansionResult::Child(new_node) => {
            let rollout_result = {
                crate::prof_scope!(crate::prof::sec::ROLLOUT);
                unsafe { (*new_node).rollout(state, root_eval, eval_config) }
            };
            {
                crate::prof_scope!(crate::prof::sec::BACKPROP);
                unsafe { (*new_node).backpropagate(rollout_result, state) }
            }
        }
        ExpansionResult::TerminalPair(score) => {
            crate::prof_scope!(crate::prof::sec::BACKPROP);
            unsafe {
                (*selected.node).backpropagate_pair(
                    score,
                    state,
                    selected.s1_move,
                    selected.s2_move,
                )
            }
        }
        ExpansionResult::NoExpansion => {
            let rollout_result = {
                crate::prof_scope!(crate::prof::sec::ROLLOUT);
                unsafe { (*selected.node).rollout(state, root_eval, eval_config) }
            };
            {
                crate::prof_scope!(crate::prof::sec::BACKPROP);
                unsafe { (*selected.node).backpropagate(rollout_result, state) }
            }
        }
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
    start_visits: u64,
    root_force: (Option<usize>, Option<usize>),
    eval_config: EvalConfig,
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
                root_force,
                eval_config,
            );
        }
        if tree_bytes >= MCTS_MAX_TREE_BYTES {
            break;
        }
        // iteration limits count iterations of *this* search: a tree reused
        // across turns starts with visits from previous searches
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited - start_visits >= n as u64 {
                    break;
                }
            }
            SearchLimit::TimeOrIterations(max_time, n) => {
                if start_time.elapsed() >= max_time
                    || root_node.times_visited - start_visits >= n as u64
                {
                    break;
                }
            }
        }
    }
}

fn search_limit(max_time: Duration, max_iterations: u32) -> SearchLimit {
    if max_iterations > 0 && max_time > Duration::from_millis(0) {
        SearchLimit::TimeOrIterations(max_time, max_iterations)
    } else if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    }
}

fn collect_result(root_node: &Node, iteration_count: u64) -> MctsResult {
    MctsResult {
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
        iteration_count,
    }
}

// The current root and the allocation that keeps it alive. A re-rooted node
// stays inside the boxed slice it was created in: its children hold parent
// pointers to that address, so the whole slice is kept (siblings get their
// subtrees pruned but occupy their Node-sized slots until the next advance)
enum TreeStorage {
    Fresh(Box<Node>),
    Branch {
        nodes: Box<[Node]>,
        root_index: usize,
    },
}

impl TreeStorage {
    fn root_ref(&self) -> &Node {
        match self {
            TreeStorage::Fresh(node) => node,
            TreeStorage::Branch { nodes, root_index } => &nodes[*root_index],
        }
    }

    fn root_mut(&mut self) -> &mut Node {
        match self {
            TreeStorage::Fresh(node) => node,
            TreeStorage::Branch { nodes, root_index } => &mut nodes[*root_index],
        }
    }
}

// A tree's rollout values are all centered on the eval anchor it was born
// with (see `ReusableTree`), so the anchor goes stale as the game drifts.
// Past this eval distance the sigmoid loses too much resolution
// (sigmoid(150 * 0.0125) ~= 0.87) and the tree is discarded instead of
// reused.
const MAX_ANCHOR_DRIFT: f32 = 150.0;

/// A single-threaded MCTS tree that can be kept across turns. Search with
/// [`perform_mcts_with_tree`]; after the move pair is actually played, call
/// [`ReusableTree::advance`] with the pair and the sampled outcome's
/// instruction list to promote the matching subtree (statistics included) to
/// the root of the next search. If the outcome doesn't exactly match an
/// expanded child, the tree is discarded and the next search starts cold.
///
/// Rollout scores are `sigmoid(eval - anchor)`, so scores from different
/// anchors are not comparable. A tree therefore keeps the anchor of the
/// search that created it for its whole lifetime (mixing anchors across
/// reused statistics measured -37 Elo, see WORKLOG 2026-07-05), and is
/// discarded when the current position's eval drifts more than
/// `MAX_ANCHOR_DRIFT` from the anchor.
///
/// Caveat: the memory budget only counts bytes allocated within one search,
/// not the carried-over tree.
pub struct ReusableTree {
    storage: Option<TreeStorage>,
    anchor_eval: f32,
    eval_config: EvalConfig,
}

impl ReusableTree {
    pub fn new() -> Self {
        ReusableTree {
            storage: None,
            anchor_eval: 0.0,
            eval_config: EvalConfig::default(),
        }
    }

    /// drop any stored tree; the next search starts cold
    pub fn reset(&mut self) {
        self.storage = None;
    }

    /// visits accumulated on the current root (0 for a cold tree)
    pub fn root_visits(&self) -> u64 {
        self.storage
            .as_ref()
            .map_or(0, |storage| storage.root_ref().times_visited)
    }

    /// Re-root to the child reached by (`s1_move`, `s2_move`) whose
    /// instruction list equals `outcome` (identical instructions imply an
    /// identical resulting state). Returns whether the subtree was kept; on
    /// any mismatch the tree is cleared.
    pub fn advance(
        &mut self,
        s1_move: &MoveChoice,
        s2_move: &MoveChoice,
        outcome: &[Instruction],
    ) -> bool {
        self.advance_with(s1_move, s2_move, |nodes| {
            // duplicate instruction lists (same resulting state) are
            // possible; prefer the most-visited match
            let mut chosen: Option<usize> = None;
            for (index, node) in nodes.iter().enumerate() {
                if node.instructions.instruction_list == outcome
                    && chosen.map_or(true, |c| node.times_visited > nodes[c].times_visited)
                {
                    chosen = Some(index);
                }
            }
            chosen
        })
    }

    /// Like [`ReusableTree::advance`], but matches the child by its
    /// *resulting state* instead of its instruction list: each candidate's
    /// instructions are applied to `root_state` (the state this tree's root
    /// was searched from; restored before returning) and compared against
    /// `target` by serialization. Use when the transition was observed
    /// externally (e.g. from a battle server) rather than sampled from
    /// engine-generated instructions.
    pub fn advance_to_state(
        &mut self,
        root_state: &mut State,
        s1_move: &MoveChoice,
        s2_move: &MoveChoice,
        target: &State,
    ) -> bool {
        let target_serialized = target.serialize();
        self.advance_with(s1_move, s2_move, |nodes| {
            let mut chosen: Option<usize> = None;
            for (index, node) in nodes.iter().enumerate() {
                root_state.apply_instructions(&node.instructions.instruction_list);
                let matches = root_state.serialize() == target_serialized;
                root_state.reverse_instructions(&node.instructions.instruction_list);
                if matches && chosen.map_or(true, |c| node.times_visited > nodes[c].times_visited) {
                    chosen = Some(index);
                }
            }
            chosen
        })
    }

    /// shared re-rooting: locate the (`s1_move`, `s2_move`) branch, let
    /// `pick` choose the outcome node, prune its siblings and make it the
    /// root. Any failure clears the tree.
    fn advance_with<F>(&mut self, s1_move: &MoveChoice, s2_move: &MoveChoice, pick: F) -> bool
    where
        F: FnOnce(&[Node]) -> Option<usize>,
    {
        // take ownership so every early return drops the stale tree
        let Some(mut storage) = self.storage.take() else {
            return false;
        };

        let mut nodes = {
            let root = storage.root_mut();
            let (Some(s1_options), Some(s2_options)) = (&root.s1_options, &root.s2_options) else {
                return false;
            };
            let Some(s1_index) = s1_options.iter().position(|m| &m.move_choice == s1_move) else {
                return false;
            };
            let Some(s2_index) = s2_options.iter().position(|m| &m.move_choice == s2_move) else {
                return false;
            };
            let key = (s1_index * s2_options.len() + s2_index) as u32;
            let Some(entry_index) = root.children.iter().position(|branch| branch.key == key)
            else {
                return false;
            };
            root.children.swap_remove(entry_index).nodes
        };

        let Some(root_index) = pick(&nodes) else {
            return false;
        };

        // free everything except the extracted branch before touching parent
        // pointers: the new root's parent (the old root) is about to die
        drop(storage);

        for (index, node) in nodes.iter_mut().enumerate() {
            if index != root_index {
                // unreachable siblings share the slice allocation; free
                // their subtrees and options
                node.children = Vec::new();
                node.s1_options = None;
                node.s2_options = None;
            }
        }
        let new_root = &mut nodes[root_index];
        new_root.root = true;
        new_root.parent = std::ptr::null_mut();
        new_root.instructions = StateInstructions::default();
        self.storage = Some(TreeStorage::Branch { nodes, root_index });
        true
    }
}

impl Default for ReusableTree {
    fn default() -> Self {
        Self::new()
    }
}

fn options_match(
    node: &Node,
    side_one_options: &[MoveChoice],
    side_two_options: &[MoveChoice],
) -> bool {
    match (&node.s1_options, &node.s2_options) {
        (Some(s1), Some(s2)) => {
            s1.len() == side_one_options.len()
                && s2.len() == side_two_options.len()
                && s1
                    .iter()
                    .zip(side_one_options)
                    .all(|(a, b)| &a.move_choice == b)
                && s2
                    .iter()
                    .zip(side_two_options)
                    .all(|(a, b)| &a.move_choice == b)
        }
        _ => false,
    }
}

// reuse `tree`'s root when it offers exactly the options the caller sees in
// the same order (mismatches happen when root-level option filtering, e.g.
// trapping or locked moves, differs from the in-tree get_all_options view)
// and its eval anchor is still close enough to the current position for its
// stored scores to stay meaningful; otherwise install a fresh root
fn ensure_root(
    tree: &mut ReusableTree,
    current_eval: f32,
    eval_config: EvalConfig,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
) {
    let warm = match tree.storage.as_ref() {
        Some(storage) => {
            options_match(storage.root_ref(), &side_one_options, &side_two_options)
                && (current_eval - tree.anchor_eval).abs() <= MAX_ANCHOR_DRIFT
                && tree.eval_config == eval_config
        }
        None => false,
    };
    if !warm {
        let mut root_node = Box::new(Node::new());
        root_node.root = true;
        unsafe {
            root_node.populate(side_one_options, side_two_options);
        }
        tree.storage = Some(TreeStorage::Fresh(root_node));
        tree.anchor_eval = current_eval;
        tree.eval_config = eval_config;
    }
}

/// Like [`perform_mcts`], but continues from `tree` when its root matches the
/// caller's option lists (as after a successful [`ReusableTree::advance`]);
/// otherwise the tree is replaced with a fresh root. Iteration limits and the
/// returned `iteration_count` count only this call's iterations.
pub fn perform_mcts_with_tree_and_eval(
    tree: &mut ReusableTree,
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    eval_config: EvalConfig,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    ensure_root(
        tree,
        evaluate_with_config(state, eval_config),
        eval_config,
        side_one_options,
        side_two_options,
    );

    let root_node = tree.storage.as_mut().unwrap().root_mut();
    let start_visits = root_node.times_visited;
    let root_eval = tree.anchor_eval;
    run_mcts_loop(
        root_node,
        state,
        &root_eval,
        search_limit(max_time, max_iterations),
        exploration_constant * exploration_constant,
        start_visits,
        (None, None),
        eval_config,
    );

    collect_result(root_node, root_node.times_visited - start_visits)
}

pub fn perform_mcts_with_tree(
    tree: &mut ReusableTree,
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    perform_mcts_with_tree_and_eval(
        tree,
        state,
        side_one_options,
        side_two_options,
        EvalConfig::default(),
        max_time,
        max_iterations,
        exploration_constant,
    )
}

/// Ponder: search with `ponder_side`'s root move pinned to `committed_move`.
/// Use during the opponent's think time after committing to a move — every
/// iteration flows into subtrees that can survive the coming
/// [`ReusableTree::advance`], and the other side's root statistics double as
/// an opponent prediction. Below the root the search is unrestricted. If
/// `committed_move` is not among the root's options the search runs
/// unrestricted (equivalent to [`perform_mcts_with_tree`]).
pub fn perform_mcts_ponder_with_eval(
    tree: &mut ReusableTree,
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    ponder_side: SideReference,
    committed_move: &MoveChoice,
    eval_config: EvalConfig,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    ensure_root(
        tree,
        evaluate_with_config(state, eval_config),
        eval_config,
        side_one_options,
        side_two_options,
    );

    let root_node = tree.storage.as_mut().unwrap().root_mut();
    let committed_index = |options: &Option<Vec<MoveNode>>| {
        options
            .as_ref()
            .unwrap()
            .iter()
            .position(|m| &m.move_choice == committed_move)
    };
    let root_force = match ponder_side {
        SideReference::SideOne => (committed_index(&root_node.s1_options), None),
        SideReference::SideTwo => (None, committed_index(&root_node.s2_options)),
    };

    let start_visits = root_node.times_visited;
    let root_eval = tree.anchor_eval;
    run_mcts_loop(
        root_node,
        state,
        &root_eval,
        search_limit(max_time, max_iterations),
        exploration_constant * exploration_constant,
        start_visits,
        root_force,
        eval_config,
    );

    collect_result(root_node, root_node.times_visited - start_visits)
}

pub fn perform_mcts_ponder(
    tree: &mut ReusableTree,
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    ponder_side: SideReference,
    committed_move: &MoveChoice,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    perform_mcts_ponder_with_eval(
        tree,
        state,
        side_one_options,
        side_two_options,
        ponder_side,
        committed_move,
        EvalConfig::default(),
        max_time,
        max_iterations,
        exploration_constant,
    )
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    perform_mcts_with_eval(
        state,
        side_one_options,
        side_two_options,
        EvalConfig::default(),
        max_time,
        max_iterations,
        exploration_constant,
    )
}

pub fn perform_mcts_with_eval(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    eval_config: EvalConfig,
    max_time: Duration,
    max_iterations: u32,
    exploration_constant: f32,
) -> MctsResult {
    let mut tree = ReusableTree::new();
    perform_mcts_with_tree_and_eval(
        &mut tree,
        state,
        side_one_options,
        side_two_options,
        eval_config,
        max_time,
        max_iterations,
        exploration_constant,
    )
}
