"""Tests for Game2048 core logic."""

import numpy as np

from src.game import Action, Game2048


def test_game_initialization():
    """Test game initializes correctly."""
    game = Game2048(seed=42)
    assert game.size == 4
    assert game.score == 0
    assert np.sum(game.board > 0) == 2  # Should have 2 initial tiles


def test_game_reset():
    """Test game reset functionality."""
    game = Game2048(seed=42)

    # Make some moves
    game.step(Action.UP)
    game.step(Action.LEFT)

    # Reset
    game.reset(seed=42)
    assert game.score == 0
    assert np.sum(game.board > 0) == 2


def test_merge_line():
    """Test line merging logic."""
    game = Game2048()

    # Test basic merge
    line = np.array([2, 2, 0, 0])
    result = game._merge_line(line)
    assert np.array_equal(result, [4, 0, 0, 0])

    # Test multiple merges
    line = np.array([2, 2, 4, 4])
    result = game._merge_line(line)
    assert np.array_equal(result, [4, 8, 0, 0])

    # Test no merge
    line = np.array([2, 4, 8, 16])
    result = game._merge_line(line)
    assert np.array_equal(result, [2, 4, 8, 16])

    # Test consecutive same values (only first pair merges)
    line = np.array([2, 2, 2, 0])
    result = game._merge_line(line)
    assert np.array_equal(result, [4, 2, 0, 0])


def test_move_left():
    """Test left move."""
    game = Game2048()
    game.board = np.array([[2, 0, 0, 2], [0, 2, 2, 0], [4, 4, 0, 0], [0, 0, 0, 0]])

    old_score = game.score
    game._move_left()

    expected = np.array([[4, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]])

    assert np.array_equal(game.board, expected)
    assert game.score == old_score + 4 + 4 + 8  # Merged values


def test_valid_actions():
    """Test valid action detection."""
    game = Game2048()

    # Create a board where only left move is valid (all tiles on right edge)
    game.board = np.array([[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16]])

    valid = game.get_valid_actions()
    assert Action.LEFT in valid  # Can move left
    assert Action.UP not in valid  # Can't move up (tiles can't merge)
    assert Action.DOWN not in valid  # Can't move down (tiles can't merge)
    assert Action.RIGHT not in valid  # Can't move right (already on right edge)


def test_game_over():
    """Test game over detection."""
    game = Game2048()

    # Not game over initially
    assert not game.is_game_over()

    # Create a full board with no valid moves
    game.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])

    assert game.is_game_over()


def test_has_won():
    """Test win condition."""
    game = Game2048()
    assert not game.has_won()

    game.board[0, 0] = 2048
    game.max_tile = 2048
    assert game.has_won()


def test_step():
    """Test step function."""
    game = Game2048(seed=42)

    board, reward, done = game.step(Action.UP)

    # Board should change (unless no valid move)
    assert board.shape == (4, 4)
    assert isinstance(reward, (int, np.integer))
    assert isinstance(done, bool)


def test_clone():
    """Test game cloning."""
    game1 = Game2048(seed=42)
    game1.step(Action.UP)

    game2 = game1.clone()

    assert np.array_equal(game1.board, game2.board)
    assert game1.score == game2.score
    assert game1.max_tile == game2.max_tile

    # Verify they're independent
    game1.step(Action.LEFT)
    assert not np.array_equal(game1.board, game2.board)
