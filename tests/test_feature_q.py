"""Tests for FeatureQAgent."""

import numpy as np

from src.agents import FeatureQAgent


def test_feature_extraction_simple():
    """Test feature extraction on a simple board."""
    agent = FeatureQAgent(seed=123)

    # Simple board: one 2 tile, one 4 tile, rest empty
    # Board in log2 representation: 0=empty, 1=2, 2=4
    board = np.array([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

    empty_bucket, max_tile_log2, mono_bucket, smoothness_bucket, merge_bucket = (
        agent._extract_features(board)
    )

    # 14 empty cells -> bucket 4 (7+)
    assert empty_bucket == 4
    # Max tile is the 4 (log2 = 2)
    assert max_tile_log2 == 2
    # Only one non-empty adjacent pair, (1,2): line score 1; no other row/column
    # has 2+ non-empty tiles -> total monotonicity score = 1 -> bucket 0
    assert mono_bucket == 0
    # Same single pair contributes |1-2| = 1 to smoothness -> bucket 0
    assert smoothness_bucket == 0
    # No adjacent equal non-zero pairs
    assert merge_bucket == 0


def test_feature_extraction_full_board():
    """Test feature extraction on a board with merge and ordering structure."""
    agent = FeatureQAgent(seed=456)

    # Row 0: [2,2,4,4] -> log2: [1,1,2,2]
    # Row 1: [8,4,4,0] -> log2: [3,2,2,0]
    # Row 2: [8,0,0,0] -> log2: [3,0,0,0]
    # Row 3: [0,0,0,0] -> log2: [0,0,0,0]
    board = np.array([[1, 1, 2, 2], [3, 2, 2, 0], [3, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

    empty_bucket, max_tile_log2, mono_bucket, smoothness_bucket, merge_bucket = (
        agent._extract_features(board)
    )

    # 8 empty cells -> bucket 4 (7+)
    assert empty_bucket == 4
    # Max tile is the 8 (log2 = 3)
    assert max_tile_log2 == 3
    # Row line-scores [3,2,0,0] + column line-scores [2,1,1,0] = 9 -> bucket 1 (>=5, <10)
    assert mono_bucket == 1
    # Smoothness sum over non-empty adjacent pairs = 5 -> bucket 1 (>=4, <8)
    assert smoothness_bucket == 1
    # 5 adjacent equal non-zero pairs, capped at 3+
    assert merge_bucket == 3


def test_feature_tuple_length_and_ranges():
    """Feature tuples should always have 5 entries within their documented ranges."""
    agent = FeatureQAgent(seed=321)

    boards = [
        np.zeros((4, 4), dtype=np.int32),
        np.array([[1, 1, 2, 2], [3, 2, 2, 0], [3, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        np.array([[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 12], [1, 2, 3, 4]], dtype=np.int32),
    ]

    for board in boards:
        empty_bucket, max_tile_log2, mono_bucket, smoothness_bucket, merge_bucket = (
            agent._extract_features(board)
        )
        assert 0 <= empty_bucket <= 4
        assert 0 <= max_tile_log2 <= 12
        assert 0 <= mono_bucket <= 4
        assert 0 <= smoothness_bucket <= 4
        assert 0 <= merge_bucket <= 3


def test_agent_initialization():
    """Test agent initializes with expected defaults."""
    agent = FeatureQAgent(seed=123)

    assert agent.action_size == 4
    assert agent.learning_rate == 0.1
    assert agent.gamma == 0.99
    assert agent.epsilon == 1.0
    assert agent.steps == 0
    assert agent.episodes == 0
    assert len(agent) == 0


def test_select_action_respects_valid_actions():
    """Greedy selection should only choose among valid actions."""
    agent = FeatureQAgent(epsilon_start=0.0, seed=7)
    state = np.zeros((4, 4), dtype=np.int32)
    agent._ensure_state(state)[:] = np.array([0.0, 5.0, 1.0, 9.0], dtype=np.float32)

    action = agent.select_action(state, valid_actions=[0, 2], use_epsilon=False)

    assert action == 2


def test_learn_updates_q_values():
    """With n_step=1, a single transition should immediately update the Q-table."""
    agent = FeatureQAgent(learning_rate=0.5, gamma=0.9, epsilon_start=0.0, n_step=1, seed=1)

    state = np.zeros((4, 4), dtype=np.int32)
    next_state = np.ones((4, 4), dtype=np.int32)
    agent._ensure_state(next_state)[:] = np.array([1.0, 4.0, 2.0, 0.5], dtype=np.float32)

    td_error, updated_q = agent.learn(
        state,
        action=0,
        reward=2.0,
        next_state=next_state,
        done=False,
        next_valid_actions=[1, 2],
    )

    expected_target = 2.0 + 0.9 * 4.0
    expected_q = 0.5 * expected_target

    assert np.isclose(td_error, expected_target)
    assert np.isclose(updated_q, expected_q)
    assert agent.steps == 1


def test_learn_buffers_until_n_step_reached():
    """With n_step=3, no update should fire until 3 real rewards have accumulated."""
    agent = FeatureQAgent(learning_rate=0.1, gamma=0.9, epsilon_start=0.0, n_step=3, seed=1)

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

    # 3-step return: r0 + gamma*r1 + gamma^2*r2 (Q-table starts empty, so the
    # bootstrap term Q(states[3]) is also 0).
    expected_target = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
    assert np.isclose(td_error, expected_target)
    assert agent.steps == 1
    assert len(agent._trajectory) == 2


def test_learn_monte_carlo_flushes_full_episode_on_done():
    """With n_step=None (the default), updates only fire once the episode ends."""
    agent = FeatureQAgent(learning_rate=0.1, gamma=0.9, epsilon_start=0.0, seed=1)

    states = [
        np.zeros((4, 4), dtype=np.int32),
        np.full((4, 4), 2, dtype=np.int32),
        np.full((4, 4), 5, dtype=np.int32),
        np.full((4, 4), 8, dtype=np.int32),
    ]
    rewards = [1.0, 2.0, 3.0]

    for i in range(2):
        td_error, _ = agent.learn(
            states[i], action=0, reward=rewards[i], next_state=states[i + 1], done=False
        )
        assert td_error == 0.0
        assert agent.steps == 0

    agent.learn(states[2], action=0, reward=rewards[2], next_state=states[3], done=True)

    assert agent.steps == 3
    assert len(agent._trajectory) == 0

    key0 = agent._extract_features(states[0])
    expected_q0 = 0.1 * (1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert np.isclose(float(agent.q_table[key0][0]), expected_q0)


def test_feature_state_space_reduction():
    """Boards with the same bucketed heuristics alias; boards that differ don't."""
    agent = FeatureQAgent(seed=42)

    # Two boards with the same tiles in different positions: same empty count,
    # max tile, monotonicity, smoothness, and merge count -> intended aliasing.
    board1 = np.array(
        [
            [3, 2, 0, 0],  # 8, 4, empty, empty
            [1, 0, 0, 0],  # 2, empty, empty, empty
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    board2 = np.array(
        [
            [3, 1, 0, 0],  # 8, 2, empty, empty (different positions)
            [2, 0, 0, 0],  # 4, empty, empty, empty
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    features1 = agent._extract_features(board1)
    features2 = agent._extract_features(board2)

    assert features1 == features2, "Boards sharing all bucketed heuristics should alias"

    # A board with a higher max tile is a strategically different state and
    # must map to a different key (differs in max_tile_log2: 4 vs 3).
    board3 = np.array(
        [
            [3, 2, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 4],
        ],
        dtype=np.int32,
    )

    features3 = agent._extract_features(board3)
    assert features1 != features3, "A higher max tile should produce a different feature key"


def test_save_and_load_round_trip(tmp_path):
    """Saving and loading should preserve the Q-table and counters."""
    agent = FeatureQAgent(seed=9)
    state = np.arange(16, dtype=np.int32).reshape(4, 4) % 5  # Small values to stay in valid range
    agent.learn(state, action=3, reward=5.0, next_state=state, done=True)
    agent.epsilon = 0.33
    agent.episodes = 12

    save_path = tmp_path / "feature_q.pt"
    agent.save(str(save_path))

    loaded_agent = FeatureQAgent(seed=10)
    loaded_agent.load(str(save_path))

    assert loaded_agent.epsilon == 0.33
    assert loaded_agent.episodes == 12
    assert loaded_agent.steps == 1
    assert np.allclose(
        loaded_agent._ensure_state(state),
        agent._ensure_state(state),
    )
