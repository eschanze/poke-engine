from dataclasses import dataclass
from enum import StrEnum

from .poke_engine import *


class Weather(StrEnum):
    NONE = "none"
    SUN = "sun"
    RAIN = "rain"
    SAND = "sand"
    HAIL = "hail"
    SNOW = "snow"
    HARSH_SUN = "harshsun"
    HEAVY_RAIN = "heavyrain"


class Terrain(StrEnum):
    NONE = "none"
    GRASSY = "grassyterrain"
    ELECTRIC = "electricterrain"
    MISTY = "mistyterrain"
    PSYCHIC = "psychicterrain"


class PokemonIndex(StrEnum):
    P0 = "0"
    P1 = "1"
    P2 = "2"
    P3 = "3"
    P4 = "4"
    P5 = "5"


@dataclass
class IterativeDeepeningResult:
    """
    Result of an Iterative Deepening Expectiminimax Search

    :param side_one: The moves for side_one
    :type side_one: list[str]
    :param side_two: The moves for side_two
    :type side_two: list[str]
    :param matrix: A vector representing the payoff matrix of the search.
        Pruned branches are represented by None
    :type matrix: int
    :param depth_searched: The depth that was searched to
    :type depth_searched: int
    """

    side_one: list[str]
    side_two: list[str]
    matrix: list[float]
    depth_searched: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=rust_result.s1,
            side_two=rust_result.s2,
            matrix=rust_result.matrix,
            depth_searched=rust_result.depth_searched,
        )

    def get_safest_move(self) -> str:
        """
        Get the safest move for side_one
        The safest move is the move that minimizes the loss for the turn

        :return: The safest move
        :rtype: str
        """
        safest_value = float("-inf")
        safest_s1_index = 0
        vec_index = 0
        for i in range(len(self.side_one)):
            worst_case_this_row = float("inf")
            for _ in range(len(self.side_two)):
                score = self.matrix[vec_index]
                if score < worst_case_this_row:
                    worst_case_this_row = score

            if worst_case_this_row > safest_value:
                safest_s1_index = i
                safest_value = worst_case_this_row

        return self.side_one[safest_s1_index]


@dataclass
class MctsSideResult:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    """

    move_choice: str
    total_score: float
    visits: int


@dataclass
class MctsResult:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    """

    side_one: list[MctsSideResult]
    side_two: list[MctsSideResult]
    total_visits: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
        )


class MctsSearcher:
    """
    A persistent monte-carlo-tree-searcher that keeps its search tree
    across turns, so each search starts from the statistics accumulated on
    the previous turn instead of from scratch.

    Usage per turn:
        result = searcher.search(state, duration_ms=100)
        # ... commit/send your move, then while waiting for the turn:
        searcher.ponder(state, side="s1", committed_move=my_move, duration_ms=5000)
        # ... turn resolves and you observe the new state:
        searcher.advance(s1_move, s2_move, new_state)

    `advance` matches the observed new state against the tree's predicted
    outcomes; if nothing matches exactly (e.g. newly revealed opponent
    information changed the state), it returns False and the next search
    simply starts cold. `search` and `ponder` only reuse the tree when
    given the exact state a previous `advance` moved to.
    """

    def __init__(self):
        self._searcher = _MctsSearcher()

    def search(
        self, state: State, duration_ms: int = 1000, iterations: int = 0
    ) -> MctsResult:
        """
        Search the position, reusing the tree kept from previous turns
        when possible. Single-threaded.

        :param state: the state to search through
        :type state: State
        :param duration_ms: time in milliseconds to run the search. ignored if iterations > 0
        :type duration_ms: int
        :param iterations: exact number of monte-carlo iterations to run
        :type iterations: int
        :return: the result of the search
        :rtype: MctsResult
        """
        return MctsResult._from_rust(
            self._searcher.search(state, duration_ms, iterations)
        )

    def ponder(
        self,
        state: State,
        side: str,
        committed_move: str,
        duration_ms: int = 1000,
        iterations: int = 0,
    ) -> MctsResult:
        """
        Keep searching after committing to a move (i.e. during the
        opponent's think time). Your side's root move is pinned to
        `committed_move` so every iteration deepens lines that survive the
        coming `advance`; the opponent's side of the result doubles as a
        prediction of their response.

        :param state: the current (pre-turn) state, same as the last search
        :type state: State
        :param side: which side you are: "s1" or "s2"
        :type side: str
        :param committed_move: the move you committed to
        :type committed_move: str
        :return: the result of the search
        :rtype: MctsResult
        """
        return MctsResult._from_rust(
            self._searcher.ponder(state, side, committed_move, duration_ms, iterations)
        )

    def advance(self, side_one_move: str, side_two_move: str, new_state: State) -> bool:
        """
        Tell the searcher what happened: the moves both sides made and the
        state the battle is now in. Keeps the matching subtree for the next
        search when the observed state matches a predicted outcome exactly.

        :param side_one_move: the move side one made
        :type side_one_move: str
        :param side_two_move: the move side two made
        :type side_two_move: str
        :param new_state: the state the battle is now in
        :type new_state: State
        :return: whether the subtree was kept (False means the next search starts cold)
        :rtype: bool
        """
        return self._searcher.advance(side_one_move, side_two_move, new_state)

    def reset(self):
        """Drop the tree, e.g. when starting a new battle."""
        self._searcher.reset()

    def root_visits(self) -> int:
        """Visits already accumulated on the tree root (0 = cold)."""
        return self._searcher.root_visits()


def monte_carlo_tree_search(
    state: State, duration_ms: int = 1000, iterations: int = 0, threads: int = 1
) -> MctsResult:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search. ignored if iterations > 0
    :type duration_ms: int
    :param iterations: exact number of monte-carlo iterations to run
    :type iterations: int
    :param threads: number of threads to use for the search
    :type threads: int
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(mcts(state, duration_ms, iterations, threads))


def iterative_deepening_expectiminimax(
    state: State, duration_ms: int = 1000
) -> IterativeDeepeningResult:
    """
    Perform an iterative-deepening expectiminimax search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: IterativeDeepeningResult
    """
    return IterativeDeepeningResult._from_rust(id(state, duration_ms))
