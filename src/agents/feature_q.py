"""Feature-based Markov Q-Learning Agent for 2048.

This agent uses classical 2048-strategy heuristics, discretized into buckets,
to make the state space tractable while staying expressive enough to
distinguish strategically different boards:
- Empty cells (5 buckets: 0, 1, 2-3, 4-6, 7+): mobility / danger-of-loss signal
- Max tile (13 buckets, raw log2 value): what stage of the game this is
- Monotonicity (5 buckets): how consistently ordered each row/column is,
  rewarding boards shaped like [8,4,2,1] that cascade merges toward one edge
- Smoothness (5 buckets): how large the gaps between neighboring tiles are;
  complements monotonicity by catching large unmergeable gaps a monotonic
  board can still have (e.g. [2048,4,2,1] is monotonic but very rough)
- Merge potential (4 buckets, capped at 3+): count of adjacent equal pairs

Theoretical max state space: 5 x 13 x 5 x 5 x 4 = 6,500 combinations, small
enough for a dict-based Q-table to accumulate real signal per key instead of
averaging over near-random noise.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

# Feature tuple: (empty_bucket, max_tile_log2, mono_bucket, smoothness_bucket, merge_bucket)
StateKey = Tuple[int, ...]


class FeatureQAgent:
    """Q-learning agent using discretized classical 2048-strategy heuristics."""

    def __init__(
        self,
        action_size: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        decay_type: str = "exponential",
        n_step: Optional[int] = None,
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
            n_step: How many real rewards to accumulate before bootstrapping
                off Q(s', a') to form the target, instead of the classic
                1-step TD target (reward + gamma * max_a' Q(s')). None means
                full Monte Carlo: accumulate real rewards for the entire
                episode with no bootstrapping at all. 1 is plain 1-step TD.
                See LinearQAgent's module docstring for the full rationale.
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
        self.n_step = n_step
        self.rng = np.random.RandomState(seed)

        self.q_table: Dict[StateKey, np.ndarray] = {}
        self.steps = 0
        self.episodes = 0

        # Rolling trajectory buffer for the in-progress episode: each entry is
        # (state_key, action, reward). n_step=None (Monte Carlo) buffers the
        # whole episode; a finite n_step only ever needs n_step+1 entries.
        self._trajectory: deque = deque()

    def _extract_features(self, state: np.ndarray) -> StateKey:
        """
        Extract discretized heuristic features from board state.

        Features:
        1. Empty cells bucket (0-4): number of empty cells, bucketed as
           0, 1, 2-3, 4-6, 7+
        2. Max tile (0-12): raw log2 value of the largest tile on the board
        3. Monotonicity bucket (0-4): quintile bucket of how consistently
           ordered each row/column is (best of increasing/decreasing per line)
        4. Smoothness bucket (0-4): quintile bucket of the sum of absolute
           log2 differences between adjacent non-empty tiles
        5. Merge bucket (0-3): count of adjacent equal non-zero pairs,
           capped at 3+

        Args:
            state: 4x4 board with log2 values (0=empty, 1=2, 2=4, ..., 11=2048, 12=4096)

        Returns:
            Tuple of 5 integers representing the bucketed features
        """
        empty_bucket = self._bucket_empty(int(np.sum(state == 0)))
        max_tile_log2 = int(np.max(state))
        mono_bucket = self._bucket_quintile(self._monotonicity_score(state), [5, 10, 15, 20])
        smoothness_bucket = self._bucket_quintile(self._smoothness_score(state), [4, 8, 12, 16])
        merge_bucket = min(self._count_mergeable_pairs(state), 3)

        return (empty_bucket, max_tile_log2, mono_bucket, smoothness_bucket, merge_bucket)

    @staticmethod
    def _bucket_empty(count: int) -> int:
        """Bucket empty-cell count into 5 bins: 0, 1, 2-3, 4-6, 7+."""
        if count == 0:
            return 0
        if count == 1:
            return 1
        if count <= 3:
            return 2
        if count <= 6:
            return 3
        return 4

    @staticmethod
    def _bucket_quintile(value: int, thresholds: list[int]) -> int:
        """Bucket a non-negative score into 5 bins using ascending thresholds."""
        bucket = 0
        for threshold in thresholds:
            if value >= threshold:
                bucket += 1
        return bucket

    @staticmethod
    def _monotonicity_score(state: np.ndarray) -> int:
        """
        Score how consistently ordered each row/column is.

        For each row and column, take the better of the left-to-right or
        right-to-left non-increasing run length among adjacent non-empty
        pairs, then sum across all 4 rows and 4 columns. A perfectly
        monotonic board (e.g. row [8,4,2,1]) scores the maximum for every
        line.
        """

        def line_score(line: np.ndarray) -> int:
            forward = 0
            backward = 0
            for i in range(len(line) - 1):
                left, right = int(line[i]), int(line[i + 1])
                if left == 0 or right == 0:
                    continue
                if left >= right:
                    forward += 1
                if right >= left:
                    backward += 1
            return max(forward, backward)

        total = 0
        for row in range(state.shape[0]):
            total += line_score(state[row])
        for col in range(state.shape[1]):
            total += line_score(state[:, col])
        return total

    @staticmethod
    def _smoothness_score(state: np.ndarray) -> int:
        """Sum of absolute log2 differences between adjacent non-empty tiles."""
        total = 0
        rows, cols = state.shape
        for row in range(rows):
            for col in range(cols - 1):
                left, right = int(state[row, col]), int(state[row, col + 1])
                if left > 0 and right > 0:
                    total += abs(left - right)
        for col in range(cols):
            for row in range(rows - 1):
                top, bottom = int(state[row, col]), int(state[row + 1, col])
                if top > 0 and bottom > 0:
                    total += abs(top - bottom)
        return total

    @staticmethod
    def _count_mergeable_pairs(state: np.ndarray) -> int:
        """Count adjacent equal non-zero pairs (horizontal and vertical)."""
        count = 0
        rows, cols = state.shape
        for row in range(rows):
            for col in range(cols - 1):
                left, right = state[row, col], state[row, col + 1]
                if left > 0 and left == right:
                    count += 1
        for col in range(cols):
            for row in range(rows - 1):
                top, bottom = state[row, col], state[row + 1, col]
                if top > 0 and top == bottom:
                    count += 1
        return count

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
        Buffer a transition and apply any n-step/Monte-Carlo updates it completes.

        See LinearQAgent.learn's docstring for the full rationale: waiting
        for n real rewards (or the whole episode, for Monte Carlo) before
        bootstrapping off Q(s') replaces the guessed portion of the target
        with what actually happened, at the cost of updating less often.

        Args:
            state: Previous observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether the episode ended.
            next_valid_actions: Optional valid actions for the next state.

        Returns:
            Tuple of (mean absolute TD error, mean updated Q-value) across
            whatever updates this call triggered (0.0, 0.0 if none did).
        """
        state_key = self._extract_features(state)
        self._trajectory.append((state_key, action, reward))

        applied = []

        if self.n_step is not None and len(self._trajectory) >= self.n_step and not done:
            applied.append(self._apply_n_step_update(next_state, next_valid_actions))

        if done:
            applied.extend(self._flush_trajectory())

        if not applied:
            return 0.0, 0.0

        td_errors, updated_qs = zip(*applied)
        return float(np.mean(td_errors)), float(np.mean(updated_qs))

    def _apply_n_step_update(
        self,
        bootstrap_state: np.ndarray,
        bootstrap_valid_actions: Optional[Iterable[int]],
    ) -> Tuple[float, float]:
        """Pop the oldest buffered transition and update it with an n-step target."""
        state_key, action, _ = self._trajectory[0]

        discounted_return = 0.0
        for _, _, r in reversed(self._trajectory):
            discounted_return = r + self.gamma * discounted_return
        self._trajectory.popleft()

        next_q_values = self._ensure_state(bootstrap_state)
        next_actions = self._normalize_actions(bootstrap_valid_actions)
        bootstrap = float(np.max(next_q_values[next_actions]))

        target = discounted_return + (self.gamma**self.n_step) * bootstrap
        return self._update_q_table(state_key, action, target)

    def _flush_trajectory(self) -> list:
        """Apply a no-bootstrap update to every remaining buffered transition."""
        rewards = [r for _, _, r in self._trajectory]
        results = []
        for i, (state_key, action, _) in enumerate(self._trajectory):
            discounted_return = 0.0
            for r in reversed(rewards[i:]):
                discounted_return = r + self.gamma * discounted_return
            results.append(self._update_q_table(state_key, action, discounted_return))
        self._trajectory.clear()
        return results

    def _update_q_table(
        self, state_key: StateKey, action: int, target: float
    ) -> Tuple[float, float]:
        """Apply one TD update toward target for the Q-table cell at state_key."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        q_values = self.q_table[state_key]

        current_q = float(q_values[action])
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
            "n_step": self.n_step,
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
        self.n_step = state_dict.get("n_step", None)  # Default for backward compatibility
        self.steps = state_dict["steps"]
        self.episodes = state_dict["episodes"]
        self.q_table = {
            tuple(key): np.asarray(values, dtype=np.float32)
            for key, values in state_dict["q_table"].items()
        }
        self._trajectory = deque()

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
