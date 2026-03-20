"""Tests for FeatureQAgent."""

import numpy as np

from src.agents import FeatureQAgent


def test_feature_extraction_simple():
    """Test feature extraction on a simple board."""
    agent = FeatureQAgent(seed=123)

    # Simple board: one 2 tile, one 4 tile, rest empty
    # Board in log2 representation: 0=empty, 1=2, 2=4
    board = np.array([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

    features = agent._extract_features(board)

    # Expected: 1 tile of value 2, 1 tile of value 4, 14 empty, max in quadrant 0
    assert features[0] == 1  # count of 2^1 (tile 2)
    assert features[1] == 1  # count of 2^2 (tile 4)
    assert features[2:12] == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # no higher tiles
    assert features[12] == 14  # empty cells
    assert features[13] == 0  # max tile (4) in top-left quadrant


def test_feature_extraction_full_board():
    """Test feature extraction on a more complex board."""
    agent = FeatureQAgent(seed=456)

    # Board with various tiles
    # 8=128 (log2=7), 64=6, 32=5, 16=4, 8=3, 4=2, 2=1
    board = np.array([[7, 6, 5, 4], [3, 2, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

    features = agent._extract_features(board)

    # Check tile counts
    assert features[0] == 1  # one 2
    assert features[1] == 1  # one 4
    assert features[2] == 1  # one 8
    assert features[3] == 1  # one 16
    assert features[4] == 1  # one 32
    assert features[5] == 1  # one 64
    assert features[6] == 1  # one 128
    assert features[12] == 9  # 9 empty cells
    assert features[13] == 0  # max tile (128) in top-left quadrant


def test_feature_extraction_max_quadrant():
    """Test that max tile quadrant is correctly identified."""
    agent = FeatureQAgent(seed=789)

    # Test each quadrant
    boards = [
        # Max in top-left (quadrant 0)
        np.array([[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        # Max in top-right (quadrant 1)
        np.array([[0, 0, 0, 5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        # Max in bottom-left (quadrant 2)
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [5, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        # Max in bottom-right (quadrant 3)
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]], dtype=np.int32),
    ]

    for quadrant, board in enumerate(boards):
        features = agent._extract_features(board)
        assert features[13] == quadrant, f"Expected quadrant {quadrant}, got {features[13]}"


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
    """A single Q-learning step should move the chosen action value toward the target."""
    agent = FeatureQAgent(learning_rate=0.5, gamma=0.9, epsilon_start=0.0, seed=1)

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


def test_feature_state_space_reduction():
    """Verify that different boards with same features map to same state."""
    agent = FeatureQAgent(seed=42)

    # Two different boards with same tile distribution and max in same quadrant
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

    # Same tile counts, empty cells, and max quadrant
    assert features1 == features2, "Different boards with same features should map to same state"


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
