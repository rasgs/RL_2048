"""Tests for expectimax search (src/agents/search.py)."""

import numpy as np

from src.agents.search import _leaf_value, select_action
from src.game.game_2048 import Action, Game2048


class ConstantValueAgent:
    """Fake value function: every board scores the same, all actions tied.

    Used to isolate search's own logic (branching, expectation over tile
    spawns, depth handling) from any real learned heuristic - with a flat
    leaf evaluator, search can only distinguish actions by whether they are
    valid/lead to a loss, not by board quality.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)


class MaxTileValueAgent:
    """Fake value function: score is just the board's max tile (log2 units).

    Lets tests construct boards where the "obviously correct" move is the
    one that reaches (or avoids losing) the highest max tile, independent of
    any trained weights.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return np.full(4, float(np.max(state)), dtype=np.float64)


class EmptyCellValueAgent:
    """Fake value function: score is the board's empty-cell count.

    Lets tests construct boards where the "obviously correct" move is the
    one that leaves more cells open (a real trained agent's value function
    is expected to favor this too, since fewer empty cells means less
    room to maneuver), independent of any trained weights.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return np.full(4, float(np.sum(state == 0)), dtype=np.float64)


def test_select_action_prefers_move_that_merges_over_plain_slide():
    """UP is a valid pure slide (no merge, empty-cell count unchanged);
    LEFT/RIGHT both merge a pair and leave one more cell open. Search with
    an empty-cell-count leaf evaluator must prefer the merge.
    """
    game = Game2048(size=4)
    game.board = np.array(
        [
            [2, 2, 0, 0],
            [4, 8, 16, 4],
            [8, 4, 8, 16],
            [4, 16, 4, 8],
        ],
        dtype=np.int32,
    )
    game.score = 0
    game.max_tile = 16
    assert game.get_valid_actions() == [Action.UP, Action.RIGHT, Action.LEFT]

    up_clone = game.clone()
    up_clone._move(Action.UP)
    assert int(np.sum(up_clone.board == 0)) == 2, "UP is a pure slide, no merge"

    left_clone = game.clone()
    left_clone._move(Action.LEFT)
    assert int(np.sum(left_clone.board == 0)) == 3, "LEFT merges the 2+2 pair"

    action = select_action(EmptyCellValueAgent(), game, depth=1)
    assert action in (Action.LEFT, Action.RIGHT)


def test_select_action_prefers_reaching_higher_max_tile():
    """Two rows, each one merge away from a different max tile: moving to
    merge the larger pair should score higher under a max-tile-based leaf
    evaluator, and search (depth=1, no meaningful chance branching needed to
    see the immediate merge) should pick that action.
    """
    game = Game2048(size=4)
    # RIGHT merges the trailing 8+8 -> 16 (this row) while also merging
    # 2+2 -> 4 in another row; UP/DOWN/LEFT do not produce the 16 merge.
    game.board = np.array(
        [
            [0, 0, 8, 8],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    game.score = 0
    game.max_tile = 8

    action = select_action(MaxTileValueAgent(), game, depth=1)

    right_clone = game.clone()
    right_clone._move(Action.RIGHT)
    assert right_clone.max_tile == 16
    assert action == Action.RIGHT


def test_select_action_returns_zero_when_no_valid_actions():
    """Game already over (no empty cells, no merges anywhere): search must
    not crash and falls back to action 0.
    """
    game = Game2048(size=4)
    game.board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ],
        dtype=np.int32,
    )
    game.score = 0
    game.max_tile = 4
    assert game.get_valid_actions() == []

    assert select_action(ConstantValueAgent(), game, depth=2) == 0


def test_select_action_respects_root_valid_actions_override():
    """When valid_actions is passed explicitly, only those are considered
    for the root move, even if other actions would also be valid.
    """
    game = Game2048(size=4)
    game.board = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    game.score = 0
    game.max_tile = 2

    action = select_action(MaxTileValueAgent(), game, depth=1, valid_actions=[Action.UP])
    assert action == Action.UP


class NegativeValueAgent:
    """Fake value function: every board scores a large negative constant.

    Models the failure mode documented in Hung Guei's thesis on 2048 RL
    (arXiv:2212.11087, Appendix B): TD training can leave some board values
    negative, and an unrectified expectimax then treats those as worse than
    a true terminal state (which the search's own game-over leaves always
    evaluate as 0, since there is no q_values call left to make). This can
    make search actively avoid the correct move in favor of one that ends
    the game, simply because 0 > a large negative number.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return np.full(4, -1000.0, dtype=np.float64)


def test_leaf_value_rectifies_negative_scores_to_zero():
    """The raw value function scores every board -1000; the rectified leaf
    evaluator search actually uses must clamp that to 0, per the thesis's
    documented fix for this exact instability.
    """
    game = Game2048(size=4)
    assert _leaf_value(NegativeValueAgent(), game) == 0.0


def test_select_action_does_not_avoid_the_only_move_due_to_negative_values():
    """With every leaf scoring the same large negative constant before
    rectification, all actions would tie at -1000 and search would be free
    to pick any of them (never actually "avoiding" a move here) - but this
    confirms rectified search still runs to completion and returns a valid
    action even when the underlying value function is entirely negative,
    rather than e.g. crashing or returning something outside valid_actions.
    """
    game = Game2048(size=4)
    game.board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    game.score = 0
    game.max_tile = 2

    action = select_action(NegativeValueAgent(), game, depth=2)
    assert action in game.get_valid_actions()
