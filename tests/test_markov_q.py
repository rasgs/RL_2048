"""Tests for MarkovQAgent."""

import numpy as np

from src.agents import MarkovQAgent
from src.utils.checkpoint import ModelCheckpoint


def test_agent_initialization():
    """Test agent initializes with expected defaults."""
    agent = MarkovQAgent(seed=123)

    assert agent.action_size == 4
    assert agent.learning_rate == 0.1
    assert agent.gamma == 0.99
    assert agent.epsilon == 1.0
    assert agent.steps == 0
    assert agent.episodes == 0
    assert len(agent) == 0


def test_select_action_respects_valid_actions():
    """Greedy selection should only choose among valid actions."""
    agent = MarkovQAgent(epsilon_start=0.0, seed=7)
    state = np.zeros((4, 4), dtype=np.int32)
    agent._ensure_state(state)[:] = np.array([0.0, 5.0, 1.0, 9.0], dtype=np.float32)

    action = agent.select_action(state, valid_actions=[0, 2], use_epsilon=False)

    assert action == 2


def test_select_action_explores_with_epsilon():
    """Exploration should sample from valid actions."""
    agent = MarkovQAgent(epsilon_start=1.0, epsilon_end=1.0, seed=5)
    state = np.zeros((4, 4), dtype=np.int32)

    actions = {
        agent.select_action(state, valid_actions=[1, 3], use_epsilon=True) for _ in range(20)
    }

    assert actions == {1, 3}


def test_learn_updates_q_values():
    """A single Q-learning step should move the chosen action value toward the target."""
    agent = MarkovQAgent(learning_rate=0.5, gamma=0.9, epsilon_start=0.0, seed=1)
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


def test_update_epsilon_reaches_floor():
    """Epsilon should decay linearly to the configured floor."""
    agent = MarkovQAgent(epsilon_start=1.0, epsilon_end=0.2, epsilon_decay=4, seed=2)

    values = []
    for _ in range(5):
        agent.update_epsilon()
        values.append(agent.epsilon)

    assert values[0] == 0.8
    assert values[-1] == 0.2
    assert agent.episodes == 5


def test_save_and_load_round_trip(tmp_path):
    """Saving and loading should preserve the Q-table and counters."""
    agent = MarkovQAgent(seed=9)
    state = np.arange(16, dtype=np.int32).reshape(4, 4)
    agent.learn(state, action=3, reward=5.0, next_state=state, done=True)
    agent.epsilon = 0.33
    agent.episodes = 12

    save_path = tmp_path / "markov_q.pt"
    agent.save(str(save_path))

    loaded_agent = MarkovQAgent(seed=10)
    loaded_agent.load(str(save_path))

    assert loaded_agent.epsilon == 0.33
    assert loaded_agent.episodes == 12
    assert loaded_agent.steps == 1
    assert np.allclose(
        loaded_agent._ensure_state(state),
        agent._ensure_state(state),
    )


def test_model_checkpoint_can_save_tabular_agent(tmp_path):
    """ModelCheckpoint should work with non-torch agents exposing state_dict."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        filename_prefix="markov_q",
        verbose=False,
    )
    agent = MarkovQAgent(seed=4)
    state = np.zeros((4, 4), dtype=np.int32)
    agent.learn(state, action=1, reward=3.0, next_state=state, done=True)

    checkpoint.save(agent, epoch=3, metric=10.0)

    restored = MarkovQAgent(seed=8)
    loaded = checkpoint.load(restored, load_best=True)

    assert loaded["epoch"] == 3
    assert restored.steps == 1
    assert np.allclose(restored._ensure_state(state), agent._ensure_state(state))
