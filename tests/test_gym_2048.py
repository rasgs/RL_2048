"""Tests for Gym2048Env's shaped reward mode."""

import numpy as np
import pytest

from src.env import Gym2048Env


def test_shaped_reward_merge_and_new_max_tile():
    """A move that merges tiles and reaches a new max, without ending the game."""
    env = Gym2048Env(reward_mode="shaped", invalid_move_penalty=50.0, max_steps=10000)
    env.reset(seed=1)

    env.game.board = np.array(
        [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
    )
    env.game.score = 0
    env.game.max_tile = 2

    _, reward, terminated, _, info = env.step(3)  # LEFT: merges 2+2 -> 4

    # new_max(4) > old_max(2) -> +1.0; open cells stay at 14 (one merge frees
    # a cell, one new tile fills a cell) so new_open > old_open is False -> -0.5;
    # game not over -> no penalty.
    assert reward == 1.0 - 0.5
    assert not terminated
    assert info["max_tile"] == 4


def test_shaped_reward_open_cell_term_rewards_more_empty_cells():
    """More open cells after the move than before should apply the +0.5 bonus."""
    env = Gym2048Env(reward_mode="shaped")

    # Board has 15 empty cells (one tile placed on an empty 4x4).
    env.game.board = np.zeros((4, 4), dtype=np.int32)
    env.game.board[0, 0] = 2
    env.game.max_tile = 2

    reward_more_open = env._compute_shaped_reward(old_max=2, old_open=10, done=False)
    reward_fewer_open = env._compute_shaped_reward(old_max=2, old_open=20, done=False)

    assert reward_more_open - reward_fewer_open == 1.0


def test_shaped_reward_new_max_tile_bonus():
    """Reaching a new max tile should apply the +1.0 bonus."""
    env = Gym2048Env(reward_mode="shaped")
    env.game.board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]], dtype=np.int32
    )
    env.game.score = 500
    env.game.max_tile = 128

    reward_with_new_max = env._compute_shaped_reward(old_max=64, old_open=0, done=False)
    reward_without_new_max = env._compute_shaped_reward(old_max=128, old_open=0, done=False)

    assert reward_with_new_max - reward_without_new_max == 1.0


def test_shaped_reward_game_over_penalty():
    """Ending the game should apply the -5.0 penalty on top of other terms."""
    env = Gym2048Env(reward_mode="shaped")
    env.game.board = np.array(
        [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]], dtype=np.int32
    )
    env.game.score = 500
    env.game.max_tile = 128

    reward = env._compute_shaped_reward(old_max=64, old_open=1, done=True)

    # new_max(128) > old_max(64) -> +1.0; open cells (0) < old_open(1) -> -0.5;
    # done -> -5.0.
    assert reward == 1.0 - 0.5 - 5.0


def test_unknown_reward_mode_raises():
    """An unrecognized reward mode should raise, same as before this change."""
    env = Gym2048Env(reward_mode="not_a_real_mode")
    with pytest.raises(ValueError):
        env._compute_reward(score_gained=0.0)


@pytest.mark.parametrize("reward_mode", ["score", "log_score", "max_tile", "shaped"])
def test_step_reward_is_always_a_plain_python_float(reward_mode):
    """np.log2(score_gained) (log_score mode) and score_gained itself
    (score mode) are numpy scalars unless explicitly cast - and
    numpy.float64 IS a subclass of the builtin float, so callers that only
    isinstance-check for float won't notice. This matters concretely: an
    n-tuple agent storing a numpy-typed reward in its lookup tables produces
    a checkpoint that fails to load under torch.load's weights_only=True
    default (PyTorch 2.6+), since that unpickler rejects numpy scalar types
    outright. Every reward mode's step() reward must be type(reward) is
    float, not just isinstance(reward, float).
    """
    env = Gym2048Env(reward_mode=reward_mode, max_steps=50)
    env.reset(seed=0)

    for _ in range(20):
        valid_actions = env.game.get_valid_actions()
        if not valid_actions:
            break
        _, reward, terminated, truncated, _ = env.step(valid_actions[0])
        assert type(reward) is float, f"expected plain float, got {type(reward)}"
        if terminated or truncated:
            break
