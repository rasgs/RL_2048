"""Tests for LinearQAgent."""

import numpy as np

from src.agents import LinearQAgent


def test_directional_monotonicity_distinguishes_mirror_images():
    """Boards anchored toward opposite edges must produce different features.

    This is the core bug LinearQAgent fixes: FeatureQAgent's
    max(forward, backward) monotonicity scores a row decreasing left-to-right
    identically to its mirror image decreasing right-to-left, even though the
    correct move differs (LEFT vs RIGHT). Fixed-direction scoring must tell
    them apart.
    """
    agent = LinearQAgent(seed=1)

    anchored_left = np.array(
        [[4, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
    )
    anchored_right = np.array(
        [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32
    )

    phi_left = agent.extract_features(anchored_left)
    phi_right = agent.extract_features(anchored_right)

    assert not np.array_equal(phi_left, phi_right), "Mirror-image boards must differ"

    # mono_left (index 2) is high for the left-anchored board, mono_right
    # (index 3) is high for the right-anchored board - the reverse of each other.
    assert phi_left[2] == phi_right[3]
    assert phi_left[3] == phi_right[2]
    assert phi_left[2] != phi_left[3]


def test_extract_features_full_board():
    """Hand-computed feature vector for a board with merges and asymmetry."""
    agent = LinearQAgent(seed=456)

    board = np.array([[1, 1, 2, 2], [3, 2, 2, 0], [3, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    phi = agent.extract_features(board)

    (
        empty_cells,
        max_tile_log2,
        mono_left,
        mono_right,
        mono_up,
        mono_down,
        smoothness,
        merge_count,
        max_row,
        max_col,
    ) = phi

    # Raw values (8 empty, max tile log2=3, mono scores 4/4/2/4, smoothness 5,
    # merge count 5, max tile at row 1 col 0) normalized by their fixed bounds.
    assert np.isclose(empty_cells, 8 / 16.0)
    assert np.isclose(max_tile_log2, 3 / 12.0)
    assert np.isclose(mono_left, 4 / 12.0)
    assert np.isclose(mono_right, 4 / 12.0)
    assert np.isclose(mono_up, 2 / 12.0)
    assert np.isclose(mono_down, 4 / 12.0)
    assert np.isclose(smoothness, 5 / 48.0)
    assert np.isclose(merge_count, 5 / 8.0)
    # First occurrence of the max tile (3, i.e. value 8) is at row 1, col 0
    assert np.isclose(max_row, 1 / 3.0)
    assert np.isclose(max_col, 0.0)


def test_agent_initialization():
    """Test agent initializes with expected defaults."""
    agent = LinearQAgent(seed=123)

    assert agent.action_size == 4
    assert agent.gamma == 0.95
    assert agent.epsilon == 1.0
    assert agent.steps == 0
    assert agent.episodes == 0
    assert len(agent) == 0
    assert agent.weights.shape == (4, 10)
    assert np.all(agent.weights == 0.0)


def test_select_action_respects_valid_actions():
    """Greedy selection should only choose among valid actions."""
    agent = LinearQAgent(epsilon_start=0.0, seed=7)
    agent.weights[1] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    agent.weights[3] = np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    state = np.zeros((4, 4), dtype=np.int32)  # 16 empty cells -> feature[0] = 16

    action = agent.select_action(state, valid_actions=[1, 3], use_epsilon=False)

    # Q(1) = 1.0*16 = 16, Q(3) = 2.0*16 = 32 -> action 3 wins
    assert action == 3


def test_learn_one_step_td_updates_weights_toward_target():
    """With n_step=1, a single transition should immediately move weights toward the target."""
    agent = LinearQAgent(learning_rate=0.1, gamma=0.9, epsilon_start=0.0, n_step=1, seed=1)

    state = np.zeros((4, 4), dtype=np.int32)
    next_state = np.zeros((4, 4), dtype=np.int32)

    td_error, updated_q = agent.learn(
        state,
        action=0,
        reward=2.0,
        next_state=next_state,
        done=False,
        next_valid_actions=[1, 2],
    )

    # Both states are the all-empty board (feature vector all zeros except
    # empty_cells=1.0 normalized); weights start at zero, so current_q and
    # next_best_q are both 0 before the update -> target = reward, td_error = reward.
    assert np.isclose(td_error, 2.0)
    assert agent.steps == 1

    # Normalized update: weights += (lr * td_error / ||phi||^2) * phi
    phi = agent.extract_features(state)
    feature_norm_sq = float(phi @ phi)
    expected_weights = (0.1 * 2.0 / feature_norm_sq) * phi
    assert np.allclose(agent.weights[0], expected_weights)
    assert np.isclose(updated_q, float(expected_weights @ phi))


def test_learn_buffers_until_n_step_reached():
    """With n_step=3, no update should fire until 3 real rewards have accumulated."""
    agent = LinearQAgent(learning_rate=0.1, gamma=0.9, epsilon_start=0.0, n_step=3, seed=1)

    states = [np.zeros((4, 4), dtype=np.int32) for _ in range(4)]
    rewards = [1.0, 2.0, 3.0]

    for i in range(2):
        td_error, updated_q = agent.learn(
            states[i],
            action=0,
            reward=rewards[i],
            next_state=states[i + 1],
            done=False,
            next_valid_actions=[0, 1, 2, 3],
        )
        assert td_error == 0.0
        assert updated_q == 0.0
        assert agent.steps == 0

    td_error, updated_q = agent.learn(
        states[2],
        action=0,
        reward=rewards[2],
        next_state=states[3],
        done=False,
        next_valid_actions=[0, 1, 2, 3],
    )

    # 3-step return: r0 + gamma*r1 + gamma^2*r2 (weights are still all zero,
    # so the bootstrap term Q(states[3]) is also 0).
    expected_target = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
    assert np.isclose(td_error, expected_target)
    assert agent.steps == 1
    assert len(agent._trajectory) == 2  # popped the oldest, 2 remain buffered


def test_learn_monte_carlo_flushes_full_episode_on_done():
    """With n_step=None, no updates fire until the episode ends, then all flush at once."""
    agent = LinearQAgent(learning_rate=0.1, gamma=0.9, epsilon_start=0.0, n_step=None, seed=1)

    # Distinct states so each buffered entry's feature vector differs and
    # updates don't interfere with each other's Q(state) readback below.
    states = [
        np.zeros((4, 4), dtype=np.int32),
        np.full((4, 4), 2, dtype=np.int32),
        np.full((4, 4), 5, dtype=np.int32),
        np.full((4, 4), 8, dtype=np.int32),
    ]
    rewards = [1.0, 2.0, 3.0]

    for i in range(2):
        td_error, updated_q = agent.learn(
            states[i], action=0, reward=rewards[i], next_state=states[i + 1], done=False
        )
        assert td_error == 0.0
        assert agent.steps == 0

    agent.learn(states[2], action=0, reward=rewards[2], next_state=states[3], done=True)

    assert agent.steps == 3
    assert len(agent._trajectory) == 0

    # Each entry's target is the true discounted return from that point on,
    # with no bootstrapping at all (the episode truly ended).
    phi0 = agent.extract_features(states[0])
    assert np.isclose(float(agent.weights[0] @ phi0), 0.1 * (1.0 + 0.9 * 2.0 + 0.9**2 * 3.0))


def test_save_and_load_round_trip(tmp_path):
    """Saving and loading should preserve the weight vector, n_step, and counters."""
    agent = LinearQAgent(seed=9, n_step=5)
    state = np.arange(16, dtype=np.int32).reshape(4, 4) % 5
    agent.learn(state, action=3, reward=5.0, next_state=state, done=True)
    agent.epsilon = 0.33
    agent.episodes = 12

    save_path = tmp_path / "linear_q.pt"
    agent.save(str(save_path))

    loaded_agent = LinearQAgent(seed=10)
    loaded_agent.load(str(save_path))

    assert loaded_agent.epsilon == 0.33
    assert loaded_agent.episodes == 12
    assert loaded_agent.steps == 1
    assert loaded_agent.n_step == 5
    assert len(loaded_agent._trajectory) == 0
    assert np.allclose(loaded_agent.weights, agent.weights)
