"""Expectimax search over a learned value function, for action selection at
inference time.

None of the agents in this project look ahead - they all choose actions by a
single epsilon-greedy argmax over Q(s, a) learned during training. 2048 is a
long-horizon, stochastic game where the *board layout* strongly determines
which random tile placements are recoverable; strong non-neural-network 2048
agents in the literature (e.g. n-tuple network + TD) pair a learned value
function with expectimax search at decision time rather than acting on it
directly. This module adds that search as a pure inference-time wrapper: it
does not change training, learn(), or checkpoints, and reuses whatever value
function an already-trained agent (e.g. LinearQAgent) provides.

The search alternates two node types down to `depth` plies:
- Max node (player move): try each of the 4 actions, keep the one with the
  highest expected value.
- Chance node (tile spawn): 2048 always spawns exactly one new tile on an
  empty cell after a valid move, 2 with probability 0.9 and 4 with
  probability 0.1 (matching Game2048._add_random_tile). Rather than sample,
  this evaluates the exact expectation: every empty cell gets both possible
  values, weighted by their true spawn probability.

At the leaf (depth 0, or no empty cells left to branch on), the board is
scored by the wrapped agent's own value function - `max(agent.q_values(state))`
- so search augments the trained heuristic instead of replacing it.

Leaf values are rectified (clamped to >= 0) before being propagated up the
tree. Hung Guei's thesis on 2048 RL (arXiv:2212.11087, Appendix B) documents
this exact failure mode in the strongest published n-tuple-network 2048
agents: TD training can leave some afterstate values negative, and because
expectimax picks the max/min of whatever values reach a node, one badly
negative leaf can make an objectively bad move (e.g. one that ends the game,
evaluated at 0) look better than the objectively good move whose only
reachable leaves happen to be negative - the exact behavior confirmed
empirically in this project (see the shaped-reward LinearQAgent checkpoint,
where negative weights on "good" features made search score reliably worse
than acting greedily). Clamping leaf values to 0 - the thesis's own
mitigation - stops a negative value from ever outscoring a true terminal
state, without changing anything about training.
"""

from __future__ import annotations

from typing import Iterable, Optional, Protocol

import numpy as np

from ..game.game_2048 import Game2048


class ValueFunction(Protocol):
    """Anything exposing q_values(state) -> per-action values, e.g. LinearQAgent."""

    def q_values(self, state: np.ndarray) -> np.ndarray: ...


def _to_log2_observation(board: np.ndarray) -> np.ndarray:
    """Convert a raw tile-value board to the log2 observation agents are trained on."""
    obs = np.zeros_like(board)
    mask = board > 0
    obs[mask] = np.log2(board[mask]).astype(board.dtype)
    return obs


def _leaf_value(agent: ValueFunction, game: Game2048) -> float:
    """
    Score a board with the wrapped agent's own value function, rectified.

    Clamping to >= 0 prevents a badly-signed value estimate from
    outscoring a true terminal state (always evaluated at 0 here, since
    game-over boards have no valid actions and fall through to this same
    leaf evaluation) - see the module docstring for why this matters.
    """
    value = float(np.max(agent.q_values(_to_log2_observation(game.board))))
    return max(value, 0.0)


def _chance_value(agent: ValueFunction, game: Game2048, depth: int) -> float:
    """Expected value over every possible tile spawn on the current board."""
    empty_cells = np.argwhere(game.board == 0)
    if len(empty_cells) == 0 or depth == 0:
        return _leaf_value(agent, game)

    total = 0.0
    # Exact expectation, not sampling: every empty cell, both possible tile
    # values, weighted by their true spawn probability - matches
    # Game2048._add_random_tile exactly, and a fixed depth makes this
    # tractable without needing a Monte Carlo estimate.
    weight_per_cell = 1.0 / len(empty_cells)
    for row, col in empty_cells:
        for value, prob in ((2, 0.9), (4, 0.1)):
            child = game.clone()
            child.board[row, col] = value
            child.max_tile = max(child.max_tile, value)
            total += weight_per_cell * prob * _max_value(agent, child, depth - 1)
    return total


def _max_value(agent: ValueFunction, game: Game2048, depth: int) -> float:
    """Best expected value over the player's move choices from this board."""
    if depth == 0:
        return _leaf_value(agent, game)

    valid_actions = game.get_valid_actions()
    if not valid_actions:
        return _leaf_value(agent, game)

    best = float("-inf")
    for action in valid_actions:
        child = game.clone()
        child._move(action)
        best = max(best, _chance_value(agent, child, depth))
    return best


def select_action(
    agent: ValueFunction,
    game: Game2048,
    depth: int = 2,
    valid_actions: Optional[Iterable[int]] = None,
) -> int:
    """
    Choose an action via expectimax search, using `agent` as the leaf evaluator.

    Args:
        agent: A trained agent exposing q_values(state) -> np.ndarray, e.g.
            LinearQAgent. Its weights are used as-is; this does not train.
        game: Current game state. Not mutated - all simulation happens on
            clones.
        depth: Number of player-move plies to search. Each ply also resolves
            one chance node (tile spawn), so depth=2 means "look two moves
            ahead, accounting for every possible tile spawn along the way".
            Cost grows quickly with depth (each ply branches over up to 4
            actions times up to 2*num_empty_cells spawns), so pick a depth
            the caller has measured to be fast enough for its use case.
        valid_actions: Optional subset of actions to consider for the root
            move only (matching the agents' select_action signature); deeper
            plies always use the true valid actions for the state reached.

    Returns:
        The chosen action index. Falls back to 0 if no action is valid at
        the root (the game is already over).
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    actions = list(valid_actions) if valid_actions is not None else game.get_valid_actions()
    if not actions:
        return 0

    best_action = actions[0]
    best_value = float("-inf")
    for action in actions:
        child = game.clone()
        child._move(action)
        value = _chance_value(agent, child, depth - 1)
        if value > best_value:
            best_value = value
            best_action = action

    return int(best_action)
