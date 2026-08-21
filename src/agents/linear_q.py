"""Linear function-approximation Q-learning agent for 2048.

FeatureQAgent's discretized features solve state aliasing (many boards
sharing one key) but remain direction-blind: monotonicity there is computed
as max(left-to-right score, right-to-left score) per line, which makes two
mirror-image boards - one built toward the left edge, one toward the right -
score identically even though the correct move differs (LEFT vs RIGHT).
A tabular Q-table keyed on such features can never learn to prefer one
direction over its mirror.

This agent instead uses linear function approximation:

    Q(s, a) = w . phi(board_after(s, a))

phi is a small vector of classical 2048 heuristics, normalized to roughly
[0, 1] using fixed bounds, including four *fixed-direction* monotonicity
scores (left, right, up, down) so that anchoring toward one edge is
distinguishable from anchoring toward the opposite edge:
- Empty cells (mobility / danger-of-loss signal)
- Max tile (what stage of the game this is)
- Monotonicity, one score per direction (left, right, up, down): rewards
  boards ordered consistently toward that edge
- Smoothness: sum of absolute log2 differences between adjacent non-empty
  tiles (direction-agnostic, complements monotonicity)
- Merge potential: count of adjacent equal non-zero pairs
- Max tile row/column index: directly tells the model which edge the
  largest tile is anchored toward

There is a single shared weight vector w (one weight per feature), updated
via a normalized TD/gradient step (dividing by ||phi||^2, a la normalized
LMS) - the classical, interpretable analogue of tabular Q-learning's
per-state values, generalizing across all boards through shared
coefficients instead of a per-key lookup table. Plain (non-normalized)
semi-gradient TD diverges here: bootstrapping through gamma * max_a' Q(s')
compounds any overshoot into an ever-larger target, and correlated features
(e.g. mono_left/mono_right pull in opposite directions for the same board)
make a fixed step size unstable regardless of how small it's set.
Normalizing by the feature vector's squared norm decouples the step size
from phi's magnitude and keeps training stable.

The `n_step` parameter controls how many real rewards are accumulated
before bootstrapping off Q(s') to form the target, instead of the classic
1-step TD target (reward + gamma * max_a' Q(s')). Real discounted
return-to-go in 2048 is very high-variance late in a game (driven by how
many random tile draws and turns are left, not by anything visible on the
board) - bootstrapping after just 1 step means every update partly trusts
the model's own, often-inaccurate value estimate for the rest of the
episode, compounding bootstrap bias on top of that already-noisy target.
Larger n_step replaces more of the target with real, unbiased reward.
n_step=None is full Monte Carlo: no bootstrapping at all, the target is the
true, fully realized discounted return for the rest of the episode.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

FeatureVector = np.ndarray

# Feature order: (empty_cells, max_tile_log2, mono_left, mono_right, mono_up,
# mono_down, smoothness, merge_count, max_tile_row, max_tile_col)
NUM_FEATURES = 10


class LinearQAgent:
    """Q-learning agent using linear function approximation over direction-aware features."""

    def __init__(
        self,
        action_size: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        decay_type: str = "exponential",
        n_step: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the agent.

        Args:
            action_size: Number of actions.
            learning_rate: Normalized TD update step size (applied after
                dividing by the feature vector's squared norm).
            gamma: Discount factor. Defaults lower than the tabular agents'
                0.99: even with the normalized update, gamma this close to 1
                lets bootstrapped targets compound faster than the updates
                can correct, causing weight divergence (verified empirically -
                gamma=0.99 diverges to numerical overflow within ~1000
                episodes on this environment, gamma=0.95 stays bounded).
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Decay rate (exponential: multiply per step,
                linear: episodes to reach end).
            decay_type: "exponential" or "linear" epsilon decay.
            n_step: How many real rewards to accumulate before bootstrapping
                off Q(s', a') to form the target, instead of bootstrapping
                after just 1 step. Larger n replaces more of the target with
                real, unbiased reward and less with the (possibly still
                inaccurate) current Q estimate - trading lower bootstrap bias
                for higher per-update variance. None means full Monte Carlo:
                accumulate real rewards for the entire episode with no
                bootstrapping at all (the target is the true, fully realized
                discounted return). 1 is plain one-step TD.
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

        # One shared weight vector per action: Q(s, a) = weights[a] . phi(s)
        self.weights = np.zeros((action_size, NUM_FEATURES), dtype=np.float64)
        self.steps = 0
        self.episodes = 0

        # Rolling trajectory buffer for the in-progress episode: each entry is
        # (phi, action, reward). n_step=None (Monte Carlo) buffers the whole
        # episode; a finite n_step only ever needs to hold n_step+1 entries.
        self._trajectory: deque = deque()

    def extract_features(self, state: np.ndarray) -> FeatureVector:
        """
        Extract a direction-aware feature vector from a board, normalized to
        roughly [0, 1] using fixed (not data-dependent) bounds.

        Normalizing keeps every feature on a comparable scale so the dot
        product w . phi isn't dominated by whichever raw heuristic happens to
        have the largest numeric range (e.g. empty_cells up to 16 vs a 0-3
        tile position), and keeps the learned weights directly comparable to
        each other as "how much this heuristic matters" - important both for
        interpretability and for TD stability (unnormalized features let
        one large-magnitude term blow up the bootstrapped target).

        Args:
            state: 4x4 board with log2 values (0=empty, 1=2, 2=4, ..., 11=2048, 12=4096)

        Returns:
            Array of NUM_FEATURES floats, each roughly in [0, 1].
        """
        empty_cells = float(np.sum(state == 0)) / 16.0
        max_tile_log2 = float(np.max(state)) / 12.0
        mono_left = self._directional_monotonicity(state, axis="row", forward=True) / 12.0
        mono_right = self._directional_monotonicity(state, axis="row", forward=False) / 12.0
        mono_up = self._directional_monotonicity(state, axis="col", forward=True) / 12.0
        mono_down = self._directional_monotonicity(state, axis="col", forward=False) / 12.0
        smoothness = self._smoothness_score(state) / 48.0
        merge_count = self._count_mergeable_pairs(state) / 8.0
        max_row, max_col = self._max_tile_position(state)

        return np.array(
            [
                empty_cells,
                max_tile_log2,
                mono_left,
                mono_right,
                mono_up,
                mono_down,
                smoothness,
                merge_count,
                max_row / 3.0,
                max_col / 3.0,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _directional_monotonicity(state: np.ndarray, axis: str, forward: bool) -> float:
        """
        Score how consistently non-increasing each line is in one fixed direction.

        Unlike max(forward, backward) scoring, this picks a single direction so
        that a board anchored toward one edge scores differently from its
        mirror image anchored toward the opposite edge.

        Args:
            state: 4x4 board.
            axis: "row" scores each row left-to-right/right-to-left;
                "col" scores each column top-to-bottom/bottom-to-top.
            forward: If True, score left-to-right (or top-to-bottom); if
                False, score right-to-left (or bottom-to-top).
        """
        lines = (
            [state[i, :] for i in range(state.shape[0])]
            if axis == "row"
            else [state[:, j] for j in range(state.shape[1])]
        )

        total = 0
        for line in lines:
            ordered = line if forward else line[::-1]
            for i in range(len(ordered) - 1):
                left, right = int(ordered[i]), int(ordered[i + 1])
                if left == 0 or right == 0:
                    continue
                if left >= right:
                    total += 1
        return float(total)

    @staticmethod
    def _smoothness_score(state: np.ndarray) -> float:
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
        return float(total)

    @staticmethod
    def _count_mergeable_pairs(state: np.ndarray) -> float:
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
        return float(count)

    @staticmethod
    def _max_tile_position(state: np.ndarray) -> Tuple[int, int]:
        """Return the (row, col) of the first occurrence of the max tile."""
        max_pos = np.argwhere(state == np.max(state))[0]
        return int(max_pos[0]), int(max_pos[1])

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q(s, a) for every action from a state's feature vector."""
        phi = self.extract_features(state)
        return self.weights @ phi

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

        q_values = self.q_values(state)
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

        Plain 1-step TD bootstraps off Q(next_state) after every single
        transition, so every update partly trusts the model's own
        (frequently inaccurate, especially early in training) value estimate
        for the entire rest of the episode - and empirically, that estimate
        is a poor predictor here: real discounted return-to-go has very high
        variance late in a 2048 game (driven by how many random tile draws
        and turns are left, not by anything visible on the board), so
        bootstrapping after 1 step compounds bootstrap bias on top of an
        already-noisy target. Waiting for n real rewards (or the whole
        episode, for Monte Carlo) before bootstrapping replaces that guessed
        portion of the target with what actually happened, at the cost of
        updating less often per step of real experience.

        Args:
            state: Previous observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether the episode ended.
            next_valid_actions: Optional valid actions for the next state.

        Returns:
            Tuple of (mean absolute TD error, mean updated Q-value) across
            whatever updates this call triggered (0.0, 0.0 if none did -
            e.g. mid-episode while still filling the n-step buffer).
        """
        phi = self.extract_features(state)
        self._trajectory.append((phi, action, reward))

        applied = []

        # A finite n_step buffer can resolve one update as soon as it holds
        # n_step real rewards - the oldest entry's target is fully known
        # (n_step real rewards plus a bootstrap off the current next_state).
        if self.n_step is not None and len(self._trajectory) >= self.n_step and not done:
            applied.append(self._apply_n_step_update(next_state, next_valid_actions))

        if done:
            # Episode over: flush every buffered entry. For Monte Carlo
            # (n_step=None) this is the entire episode with no bootstrap at
            # all - target = true discounted return. For finite n_step, each
            # remaining entry uses whatever real rewards are left before the
            # episode ended, also with no bootstrap (there's no next state).
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
        phi, action, _ = self._trajectory[0]

        discounted_return = 0.0
        for _, _, r in reversed(self._trajectory):
            discounted_return = r + self.gamma * discounted_return
        self._trajectory.popleft()

        next_q_values = self.q_values(bootstrap_state)
        next_actions = self._normalize_actions(bootstrap_valid_actions)
        bootstrap = float(np.max(next_q_values[next_actions]))

        target = discounted_return + (self.gamma**self.n_step) * bootstrap
        return self._update_weights(phi, action, target)

    def _flush_trajectory(self) -> list:
        """Apply a no-bootstrap update to every remaining buffered transition."""
        rewards = [r for _, _, r in self._trajectory]
        results = []
        # Walk the buffer left to right; each entry's target is the real
        # discounted sum of every reward from that point to the episode end.
        for i, (phi, action, _) in enumerate(self._trajectory):
            discounted_return = 0.0
            for r in reversed(rewards[i:]):
                discounted_return = r + self.gamma * discounted_return
            results.append(self._update_weights(phi, action, discounted_return))
        self._trajectory.clear()
        return results

    def _update_weights(
        self, phi: FeatureVector, action: int, target: float
    ) -> Tuple[float, float]:
        """Apply one normalized TD/gradient step toward target for (phi, action)."""
        current_q = float(self.weights[action] @ phi)
        td_error = target - current_q

        # Normalized update (a la normalized LMS): dividing by ||phi||^2
        # decouples the step size from the feature vector's magnitude.
        # Plain semi-gradient TD (weights += lr * td_error * phi) diverges
        # here - bootstrapping through gamma * max_a' Q(s') compounds any
        # overshoot into an ever-larger target, and correlated features
        # (e.g. mono_left/mono_right pull in opposite directions) make plain
        # TD's fixed step size unstable regardless of how small it's set.
        feature_norm_sq = float(phi @ phi) + 1e-8
        self.weights[action] += (self.learning_rate * td_error / feature_norm_sq) * phi

        self.steps += 1
        updated_q = float(self.weights[action] @ phi)
        return abs(float(td_error)), updated_q

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
            "weights": self.weights.tolist(),
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
        self.weights = np.asarray(state_dict["weights"], dtype=np.float64)
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
        """Return the number of training steps taken.

        There is no Q-table for a linear agent - every update refines the
        same shared weight vector, so "steps taken" is the meaningful
        analogue of "states visited" for training-progress logging.
        """
        return self.steps
