"""Core 2048 game logic."""

from enum import IntEnum
from typing import Optional, Tuple

import numpy as np


class Action(IntEnum):
    """Available actions in the game."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Game2048:
    """
    2048 game implementation.

    The game is played on a 4x4 grid. Tiles with powers of 2 can be merged
    by moving them in one of four directions (up, down, left, right).
    """

    def __init__(self, size: int = 4, seed: Optional[int] = None):
        """
        Initialize the game.

        Args:
            size: Board size (default 4 for 4x4 grid)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self._add_random_tile()
        self._add_random_tile()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the game to initial state.

        Args:
            seed: Optional new random seed

        Returns:
            Initial board state
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        """
        Execute one action.

        Args:
            action: Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            Tuple of (new_board, reward, done)
            - new_board: Updated board state
            - reward: Score gained from this move
            - done: Whether game is over
        """
        if not self.is_valid_action(action):
            # Invalid move - no change
            return self.board.copy(), 0, self.is_game_over()

        old_score = self.score
        self._move(action)
        reward = self.score - old_score

        # Add new tile only if move was valid
        self._add_random_tile()

        done = self.is_game_over()
        return self.board.copy(), reward, done

    def is_valid_action(self, action: int) -> bool:
        """
        Check if an action would change the board.

        Args:
            action: Action to check

        Returns:
            True if action is valid (would change board)
        """
        temp_board = self.board.copy()
        temp_score = self.score
        self._move(action)
        is_valid = not np.array_equal(temp_board, self.board)
        self.board = temp_board
        self.score = temp_score
        return is_valid

    def get_valid_actions(self) -> list[int]:
        """
        Get list of valid actions.

        Returns:
            List of valid action indices
        """
        return [action for action in range(4) if self.is_valid_action(action)]

    def is_game_over(self) -> bool:
        """
        Check if game is over (no valid moves).

        Returns:
            True if no valid moves remain
        """
        # Check if any cell is empty
        if np.any(self.board == 0):
            return False

        # Check if any adjacent cells can be merged
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i, j]
                # Check right
                if j < self.size - 1 and self.board[i, j + 1] == current:
                    return False
                # Check down
                if i < self.size - 1 and self.board[i + 1, j] == current:
                    return False

        return True

    def has_won(self) -> bool:
        """
        Check if 2048 tile has been reached.

        Returns:
            True if 2048 tile exists
        """
        return self.max_tile >= 2048

    def _add_random_tile(self) -> bool:
        """
        Add a random tile (2 or 4) to an empty cell.

        Returns:
            True if tile was added, False if board is full
        """
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            return False

        # Pick random empty cell
        idx = self.rng.randint(len(empty_cells))
        i, j = empty_cells[idx]

        # 90% chance of 2, 10% chance of 4
        value = 2 if self.rng.random() < 0.9 else 4
        self.board[i, j] = value
        self.max_tile = max(self.max_tile, value)
        return True

    def _move(self, action: int):
        """
        Execute a move in the given direction.

        Args:
            action: Direction to move (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        if action == Action.UP:
            self._move_up()
        elif action == Action.RIGHT:
            self._move_right()
        elif action == Action.DOWN:
            self._move_down()
        elif action == Action.LEFT:
            self._move_left()

    def _move_left(self):
        """Move and merge tiles to the left."""
        for i in range(self.size):
            self.board[i] = self._merge_line(self.board[i])

    def _move_right(self):
        """Move and merge tiles to the right."""
        for i in range(self.size):
            self.board[i] = self._merge_line(self.board[i][::-1])[::-1]

    def _move_up(self):
        """Move and merge tiles upward."""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T

    def _move_down(self):
        """Move and merge tiles downward."""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T

    def _merge_line(self, line: np.ndarray) -> np.ndarray:
        """
        Merge a single line (row) to the left.

        Args:
            line: Row to merge

        Returns:
            Merged row
        """
        # Remove zeros
        non_zero = line[line != 0]

        # Merge adjacent equal values
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                self.max_tile = max(self.max_tile, merged_value)
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        # Pad with zeros
        result = np.zeros(self.size, dtype=np.int32)
        result[: len(merged)] = merged
        return result

    def get_state(self) -> dict:
        """
        Get complete game state.

        Returns:
            Dictionary with board, score, max_tile, done, won
        """
        return {
            "board": self.board.copy(),
            "score": self.score,
            "max_tile": self.max_tile,
            "done": self.is_game_over(),
            "won": self.has_won(),
        }

    def clone(self) -> "Game2048":
        """
        Create a deep copy of the game state.

        Returns:
            New Game2048 instance with same state
        """
        game = Game2048(size=self.size)
        game.board = self.board.copy()
        game.score = self.score
        game.max_tile = self.max_tile
        game.rng = np.random.RandomState()
        game.rng.set_state(self.rng.get_state())
        return game

    def __str__(self) -> str:
        """String representation of the board."""
        lines = [f"Score: {self.score}  Max Tile: {self.max_tile}"]
        lines.append("-" * (self.size * 7 + 1))

        for row in self.board:
            line = "|"
            for cell in row:
                if cell == 0:
                    line += "      |"
                else:
                    line += f" {cell:4d} |"
            lines.append(line)
            lines.append("-" * (self.size * 7 + 1))

        return "\n".join(lines)
