"""Experience replay buffer for DQN training."""

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    This buffer stores transitions (state, action, reward, next_state, done)
    and allows random sampling for training. This breaks temporal correlation
    in the data and improves learning stability.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        # Store transition as tuple
        # We store numpy arrays to save memory (convert to tensors during sampling)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # Randomly sample transitions
        batch = random.sample(self.buffer, batch_size)

        # Unzip the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        # states: (batch_size, 4, 4)
        states_tensor = torch.FloatTensor(np.array(states))

        # actions: (batch_size,)
        actions_tensor = torch.LongTensor(actions)

        # rewards: (batch_size,)
        rewards_tensor = torch.FloatTensor(rewards)

        # next_states: (batch_size, 4, 4)
        next_states_tensor = torch.FloatTensor(np.array(next_states))

        # dones: (batch_size,)
        dones_tensor = torch.FloatTensor(dones)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            batch_size: Minimum number of samples needed

        Returns:
            True if buffer has enough samples
        """
        return len(self.buffer) >= batch_size
