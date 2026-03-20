"""Feature-based Markov Q-Learning Agent for 2048.

This agent uses a minimal feature representation to make the state space tractable:
- Tile counts (12 values: count of each tile from 2 to 4096)
- Empty cells (0-16)
- Max tile quadrant (0-3: which corner quadrant has the max tile)

This dramatically reduces the state space from ~10^18 to ~10^6-10^8 states,
enabling convergence through repeated state visits.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

# Feature tuple: (count_2, count_4, ..., count_4096, empty_cells, max_quadrant)
StateKey = Tuple[int, ...]


class FeatureQAgent:
    """Q-learning agent using minimal feature representation."""

    def __init__(
        self,
        action_size: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        decay_type: str = "exponential",
        seed: Optional[int] = None,
    ):
        """
        Initialize the agent.

        Args:
            action_size: Number of actions.
            learning_rate: Q-learning update step size.
            gamma: Discount factor.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Decay rate (exponential: multiply per step,
                linear: episodes to reach end).
            decay_type: "exponential" or "linear" epsilon decay.
            seed: Optional RNG seed.
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type
        self.rng = np.random.RandomState(seed)

        self.q_table: Dict[StateKey, np.ndarray] = {}
        self.steps = 0
        self.episodes = 0

    def _extract_features(self, state: np.ndarray) -> StateKey:
        """
        Extract minimal features from board state.

        Features:
        1. Tile counts (12 values): count of each tile from 2^1 to 2^12
        2. Empty cells (1 value): number of empty cells
        3. Max tile quadrant (1 value): which quadrant (0-3) contains max tile

        Args:
            state: 4x4 board with log2 values (0=empty, 1=2, 2=4, ..., 11=2048, 12=4096)

        Returns:
            Tuple of 14 integers representing the features
        """
        # Count tiles by value (log2 representation)
        # 0 = empty (skip), 1 = tile 2, 2 = tile 4, ..., 12 = tile 4096
        tile_counts = []
        for tile_value in range(1, 13):  # 1 to 12 (representing 2 to 4096)
            count = int(np.sum(state == tile_value))
            tile_counts.append(count)

        # Empty cells
        empty_cells = int(np.sum(state == 0))

        # Max tile quadrant
        # Divide board into 4 quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        max_value = np.max(state)
        max_pos = np.argwhere(state == max_value)[0]  # Get first occurrence
        row, col = max_pos

        # Determine quadrant (0-3)
        if row < 2 and col < 2:
            max_quadrant = 0  # Top-left
        elif row < 2 and col >= 2:
            max_quadrant = 1  # Top-right
        elif row >= 2 and col < 2:
            max_quadrant = 2  # Bottom-left
        else:
            max_quadrant = 3  # Bottom-right

        # Combine into feature tuple
        features = tuple(tile_counts + [empty_cells, max_quadrant])
        return features

    def _ensure_state(self, state: np.ndarray) -> np.ndarray:
        """Return the Q-values for a state, initializing if needed."""
        state_key = self._extract_features(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state_key]

    def _normalize_actions(self, valid_actions: Optional[Iterable[int]]) -> list[int]:
        """Normalize an optional action iterable into a concrete action list."""
        if valid_actions is None:
            return list(range(self.action_size))

        actions = [int(action) for action in valid_actions]
        if not actions:
            return list(range(self.action_size))
        return actions

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[Iterable[int]] = None,
        use_epsilon: bool = True,
    ) -> int:
        """
        Select an action with epsilon-greedy exploration.

        Args:
            state: Current observation.
            valid_actions: Optional subset of actions to consider.
            use_epsilon: Whether to apply exploration.

        Returns:
            Action index.
        """
        actions = self._normalize_actions(valid_actions)

        if use_epsilon and self.rng.random_sample() < self.epsilon:
            return int(self.rng.choice(actions))

        q_values = self._ensure_state(state)
        valid_q_values = q_values[actions]
        best_value = float(np.max(valid_q_values))
        best_actions = [action for action in actions if q_values[action] == best_value]
        return int(self.rng.choice(best_actions))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: Optional[Iterable[int]] = None,
    ) -> Tuple[float, float]:
        """
        Update the Q-table from a single transition.

        Args:
            state: Previous observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether the episode ended.
            next_valid_actions: Optional valid actions for the next state.

        Returns:
            Tuple of (absolute_td_error, updated_q_value).
        """
        q_values = self._ensure_state(state)
        current_q = float(q_values[action])

        if done:
            next_best_q = 0.0
        else:
            next_q_values = self._ensure_state(next_state)
            next_actions = self._normalize_actions(next_valid_actions)
            next_best_q = float(np.max(next_q_values[next_actions]))

        target = reward + self.gamma * next_best_q
        td_error = target - current_q
        q_values[action] += self.learning_rate * td_error

        self.steps += 1
        return abs(float(td_error)), float(q_values[action])

    def update_epsilon(self):
        """Decay epsilon (exponential or linear)."""
        self.episodes += 1

        if self.decay_type == "exponential":
            # Exponential decay: epsilon *= decay_rate each episode
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            # Linear decay: interpolate from start to end over decay episodes
            if self.episodes < self.epsilon_decay:
                progress = self.episodes / self.epsilon_decay
                self.epsilon = (
                    self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
                )
            else:
                self.epsilon = self.epsilon_end

    def state_dict(self) -> dict:
        """Return a serializable agent state."""
        return {
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "decay_type": self.decay_type,
            "steps": self.steps,
            "episodes": self.episodes,
            "q_table": {key: values.tolist() for key, values in self.q_table.items()},
        }

    def load_state_dict(self, state_dict: dict):
        """Load a serialized agent state."""
        self.action_size = state_dict["action_size"]
        self.learning_rate = state_dict["learning_rate"]
        self.gamma = state_dict["gamma"]
        self.epsilon = state_dict["epsilon"]
        self.epsilon_start = state_dict["epsilon_start"]
        self.epsilon_end = state_dict["epsilon_end"]
        self.epsilon_decay = state_dict["epsilon_decay"]
        self.decay_type = state_dict.get(
            "decay_type", "exponential"
        )  # Default for backward compatibility
        self.steps = state_dict["steps"]
        self.episodes = state_dict["episodes"]
        self.q_table = {
            tuple(key): np.asarray(values, dtype=np.float32)
            for key, values in state_dict["q_table"].items()
        }

    def save(self, path: str):
        """Save the agent state."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load the agent state from a direct save or checkpoint file."""
        payload = torch.load(path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        self.load_state_dict(state_dict)

    def __len__(self) -> int:
        """Return the number of visited states."""
        return len(self.q_table)
