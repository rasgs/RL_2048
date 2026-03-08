"""Tests for ReplayBuffer."""

import numpy as np
import torch

from src.agents.replay_buffer import ReplayBuffer


def test_buffer_initialization():
    """Test buffer initializes correctly."""
    buffer = ReplayBuffer(capacity=100)
    assert len(buffer) == 0
    assert buffer.capacity == 100
    assert not buffer.is_ready(batch_size=32)


def test_push_transition():
    """Test adding transitions to buffer."""
    buffer = ReplayBuffer(capacity=100)

    state = np.zeros((4, 4))
    action = 0
    reward = 10.0
    next_state = np.ones((4, 4))
    done = False

    buffer.push(state, action, reward, next_state, done)

    assert len(buffer) == 1


def test_buffer_capacity():
    """Test buffer respects capacity limit."""
    buffer = ReplayBuffer(capacity=5)

    # Add more transitions than capacity
    for i in range(10):
        state = np.full((4, 4), i)
        buffer.push(state, 0, 0.0, state, False)

    # Should only keep last 5
    assert len(buffer) == 5


def test_is_ready():
    """Test is_ready method."""
    buffer = ReplayBuffer(capacity=100)

    assert not buffer.is_ready(batch_size=32)

    # Add 32 transitions
    for i in range(32):
        state = np.zeros((4, 4))
        buffer.push(state, 0, 0.0, state, False)

    assert buffer.is_ready(batch_size=32)
    assert not buffer.is_ready(batch_size=64)


def test_sample():
    """Test sampling from buffer."""
    buffer = ReplayBuffer(capacity=100)

    # Add some transitions
    for i in range(50):
        state = np.full((4, 4), i)
        action = i % 4
        reward = float(i)
        next_state = np.full((4, 4), i + 1)
        done = i % 10 == 0
        buffer.push(state, action, reward, next_state, done)

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=32)

    # Check shapes
    assert states.shape == (32, 4, 4)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert next_states.shape == (32, 4, 4)
    assert dones.shape == (32,)

    # Check types
    assert isinstance(states, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(next_states, torch.Tensor)
    assert isinstance(dones, torch.Tensor)

    # Check dtypes
    assert states.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert dones.dtype == torch.float32


def test_sample_randomness():
    """Test that sampling is random."""
    buffer = ReplayBuffer(capacity=100)

    # Add transitions with unique states
    for i in range(100):
        state = np.full((4, 4), i)
        buffer.push(state, 0, 0.0, state, False)

    # Sample twice
    states1, _, _, _, _ = buffer.sample(batch_size=10)
    states2, _, _, _, _ = buffer.sample(batch_size=10)

    # Samples should be different (with high probability)
    # Check first element of first state
    assert not torch.allclose(states1[0], states2[0])
