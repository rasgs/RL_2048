"""Tests for Gym2048Env's shaped reward modes."""

import numpy as np
import pytest

from src.env import Gym2048Env


def test_shaped_reward_merge_and_score_increase():
    """A move that merges tiles, reaches a new max, and doesn't end the game."""
    env = Gym2048Env(reward_mode="shaped", invalid_move_penalty=50.0, max_steps=10000)
    env.reset(seed=1)

    env.game.board = np.array(
        [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
    )
    env.game.score = 0
    env.game.max_tile = 2

    _, reward, terminated, _, info = env.step(3)  # LEFT: merges 2+2 -> 4

    # new_max(4) > old_max(2) -> +100; open cells stay at 14 (one merge frees
    # a cell, one new tile fills a cell) so new_open < old_open is False -> -50;
    # new_score(4) > old_score(0) -> +2; game not over -> no penalty.
    assert reward == 100.0 - 50.0 + 2.0
    assert not terminated
    assert info["max_tile"] == 4


def test_shaped_open_reward_flips_open_cell_sign():
    """shaped_open_reward rewards MORE open cells; shaped rewards FEWER."""
    env = Gym2048Env(reward_mode="shaped")
    # 15 empty cells now (one tile placed) vs old_open=10 -> open cells increased.
    env.game.board = np.zeros((4, 4), dtype=np.int32)
    env.game.board[0, 0] = 2

    faithful_reward = env._compute_shaped_reward(old_max=2, old_open=10, old_score=0, done=False)

    env.reward_mode = "shaped_open_reward"
    flipped_reward = env._compute_shaped_reward(old_max=2, old_open=10, old_score=0, done=False)

    # "shaped" penalizes open cells increasing (-50); "shaped_open_reward"
    # rewards it (+25) - the other terms (score, max tile, done) are identical
    # in both calls, so the difference isolates exactly the sign flip.
    assert faithful_reward == flipped_reward - 75.0


def test_shaped_reward_new_max_tile_bonus():
    """Reaching a new max tile should apply the +100 bonus."""
    env = Gym2048Env(reward_mode="shaped")
    env.game.board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]], dtype=np.int32
    )
    env.game.score = 500
    env.game.max_tile = 128

    reward_with_new_max = env._compute_shaped_reward(
        old_max=64, old_open=0, old_score=500, done=False
    )
    reward_without_new_max = env._compute_shaped_reward(
        old_max=128, old_open=0, old_score=500, done=False
    )

    assert reward_with_new_max - reward_without_new_max == 100.0


def test_shaped_reward_game_over_penalty():
    """Ending the game should apply the large -1000 penalty on top of other terms."""
    env = Gym2048Env(reward_mode="shaped")
    env.game.board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]], dtype=np.int32
    )
    env.game.score = 500
    env.game.max_tile = 128

    reward = env._compute_shaped_reward(old_max=64, old_open=1, old_score=490, done=True)

    # new_max(128) > old_max(64) -> +100; open cells (0) < old_open(1) -> +25;
    # new_score(500) > old_score(490) -> +2; done -> -1000.
    assert reward == 100.0 + 25.0 + 2.0 - 1000.0


def test_unknown_reward_mode_raises():
    """An unrecognized reward mode should raise, same as before this change."""
    env = Gym2048Env(reward_mode="not_a_real_mode")
    with pytest.raises(ValueError):
        env._compute_reward(score_gained=0.0)
