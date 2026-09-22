"""N-tuple network Q-learning agent for 2048.

LinearQAgent's 10 hand-crafted scalar features (empty cells, monotonicity,
etc.) force every board into a linear combination of a handful of global
summary statistics - there's no way for it to represent "this exact local
pattern of tiles in these 6 cells is worth X" independently of any other
pattern, since everything is squeezed through the same 10 numbers first.
N-tuple networks, the function approximator behind every published
strong non-neural 2048 agent (Szubert & Jaskowski 2014; Jaskowski's
temporal-coherence agent; Yeh et al.'s multistage agent; Matsuzaki's
8x6-tuple network; and Hung Guei's PhD thesis on the topic, arXiv:2212.11087,
which this implementation follows), sidestep that limit entirely: each
*n*-tuple is a small, fixed set of board cells, and its value is a direct
lookup into a table indexed by the raw tile values sitting in exactly those
cells. Summing a handful of these lookups gives the state value - still a
sum of table entries (tabular, fully interpretable, no gradients through
hidden layers), but with enough independent parameters (up to c^n per
tuple, c=16 possible tile values here) to capture local tile-pattern
interactions a small linear feature set has to average away.

This agent uses Matsuzaki's first 4 canonical 6-cell tuple shapes (the
thesis's Figure 5, shapes a-d; the "4x6-tuple network," the smallest
well-documented starting point in that lineage - larger k x 6-tuple
networks add shapes e-h and further improve performance, at the cost of
4x-8x more weights and training time). Each tuple's lookup table is shared
across all 8 dihedral symmetries of the board (4 rotations x 2 reflections)
via symmetric sampling: a single training step updates one table entry per
symmetry the tuple's shape maps to, multiplying the effective training
signal per real game step by 8 for free, with no extra memory (the 8
symmetric shapes of one tuple all read/write the *same* table).

Like LinearQAgent, this learns per-action values Q(s, a) - one full
4-tuple network per action - via 1-step/n-step/Monte Carlo TD, matching the
existing agents' shared select_action/learn/state_dict interface so it
slots into scripts/train.py, scripts/evaluate.py, and src/agents/search.py
unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

TupleShape = Tuple[Tuple[int, int], ...]
LutKey = Tuple[int, ...]

# Matsuzaki's first 4 canonical 6-cell tuple shapes (thesis Figure 5, a-d),
# 0-indexed (row, col), read directly off the figure. The best k x 6-tuple
# network is always "the first k of this list" per the thesis - shapes e-h
# would extend this to an 8x6-tuple network.
TUPLE_SHAPES: Tuple[TupleShape, ...] = (
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),  # (a) 2x3 rectangle
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),  # (b) 2x2 block + tail
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),  # (c) row + 2x2 block
    ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)),  # (d) staircase
)
NUM_TUPLES = len(TUPLE_SHAPES)

# c=16 possible values per cell (empty + log2 values 1..15, i.e. tiles up to
# 2^15) rather than every theoretically possible tile - values this large
# never occur in practice, and capping keeps the LUT key space bounded.
MAX_CELL_VALUE = 15


def _board_symmetries(board: np.ndarray) -> List[np.ndarray]:
    """The 8 dihedral transforms of a square board: 4 rotations x 2 reflections."""
    variants = []
    current = board
    for _ in range(4):
        variants.append(current)
        variants.append(np.fliplr(current))
        current = np.rot90(current)
    return variants


class NTupleQAgent:
    """Q-learning agent using a symmetric-sampled n-tuple network per action."""

    def __init__(
        self,
        action_size: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
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
            learning_rate: TD update step size. The scalar TD error is
                distributed equally across every LUT entry that contributed
                to Q(s, a) - NUM_TUPLES shapes x 8 symmetries = 32 entries -
                so this plays the same role as LinearQAgent's normalized
                step size without needing an explicit norm division (each
                contributing entry's "gradient" is exactly 1, unlike a
                continuous feature's magnitude).
            gamma: Discount factor.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Decay rate (exponential: multiply per step,
                linear: episodes to reach end).
            decay_type: "exponential" or "linear" epsilon decay.
            n_step: How many real rewards to accumulate before bootstrapping
                off Q(s', a') to form the target. None means full Monte
                Carlo. See LinearQAgent's module docstring for the full
                rationale. 1 is plain one-step TD.
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

        # One lookup table per (action, tuple shape), shared across that
        # tuple's 8 symmetric instances. Sparse dict: most of the c^6
        # possible keys are never visited, so there's no benefit to
        # preallocating a dense array.
        self.tables: List[List[Dict[LutKey, float]]] = [
            [dict() for _ in range(NUM_TUPLES)] for _ in range(action_size)
        ]
        self.steps = 0
        self.episodes = 0

        # Rolling trajectory buffer for the in-progress episode: each entry
        # is (state, action, reward). n_step=None (Monte Carlo) buffers the
        # whole episode; a finite n_step only ever needs n_step+1 entries.
        self._trajectory: deque = deque()

    def _tuple_keys(self, state: np.ndarray, shape: TupleShape) -> List[LutKey]:
        """The 8 symmetric LUT keys a tuple shape reads from one board."""
        keys = []
        for variant in _board_symmetries(state):
            key = tuple(min(int(variant[r, c]), MAX_CELL_VALUE) for r, c in shape)
            keys.append(key)
        return keys

    def q_value(self, state: np.ndarray, action: int) -> float:
        """Q(s, a): sum of every tuple's LUT value across all 8 symmetries."""
        total = 0.0
        for tuple_idx, shape in enumerate(TUPLE_SHAPES):
            table = self.tables[action][tuple_idx]
            for key in self._tuple_keys(state, shape):
                total += table.get(key, 0.0)
        return total

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Q(s, a) for every action, e.g. for use as a search leaf evaluator."""
        return np.array(
            [self.q_value(state, action) for action in range(self.action_size)],
            dtype=np.float64,
        )

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
        self._trajectory.append((state, action, reward))

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
        state, action, _ = self._trajectory[0]

        discounted_return = 0.0
        for _, _, r in reversed(self._trajectory):
            discounted_return = r + self.gamma * discounted_return
        self._trajectory.popleft()

        next_q_values = self.q_values(bootstrap_state)
        next_actions = self._normalize_actions(bootstrap_valid_actions)
        bootstrap = float(np.max(next_q_values[next_actions]))

        target = discounted_return + (self.gamma**self.n_step) * bootstrap
        return self._update_tables(state, action, target)

    def _flush_trajectory(self) -> list:
        """Apply a no-bootstrap update to every remaining buffered transition."""
        rewards = [r for _, _, r in self._trajectory]
        results = []
        for i, (state, action, _) in enumerate(self._trajectory):
            discounted_return = 0.0
            for r in reversed(rewards[i:]):
                discounted_return = r + self.gamma * discounted_return
            results.append(self._update_tables(state, action, discounted_return))
        self._trajectory.clear()
        return results

    def _update_tables(self, state: np.ndarray, action: int, target: float) -> Tuple[float, float]:
        """
        Apply one TD update toward target, distributed across every LUT
        entry that contributed to Q(s, a).

        Q(s, a) is a sum of NUM_TUPLES x 8 symmetric LUT lookups, so its
        gradient with respect to any one contributing entry is exactly 1 -
        unlike LinearQAgent's continuous features, there's no need to
        normalize by a feature norm, only to split the scalar TD error
        evenly across however many entries the sum touched.
        """
        current_q = self.q_value(state, action)
        td_error = target - current_q

        all_keys = [
            (tuple_idx, key)
            for tuple_idx, shape in enumerate(TUPLE_SHAPES)
            for key in self._tuple_keys(state, shape)
        ]
        # float() guards against a numpy scalar leaking in via `target`
        # (e.g. if the caller's reward came from a numpy-typed source) -
        # numpy.float64 is a subclass of the builtin float, so an isinstance
        # check wouldn't catch it, but torch.load's weights_only=True
        # unpickler (default since PyTorch 2.6) rejects numpy scalar types
        # outright, so every LUT value must be forced to a plain float
        # before it's ever stored.
        step = float(self.learning_rate * td_error / len(all_keys))
        table_set = self.tables[action]
        for tuple_idx, key in all_keys:
            table_set[tuple_idx][key] = table_set[tuple_idx].get(key, 0.0) + step

        self.steps += 1
        updated_q = self.q_value(state, action)
        return abs(float(td_error)), updated_q

    def update_epsilon(self):
        """Decay epsilon (exponential or linear)."""
        self.episodes += 1

        if self.decay_type == "exponential":
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
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
            "tables": [
                [
                    {",".join(map(str, key)): float(value) for key, value in table.items()}
                    for table in action_tables
                ]
                for action_tables in self.tables
            ],
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
        self.decay_type = state_dict.get("decay_type", "exponential")
        self.n_step = state_dict.get("n_step", None)
        self.steps = state_dict["steps"]
        self.episodes = state_dict["episodes"]
        self.tables = [
            [
                {tuple(int(x) for x in key.split(",")): value for key, value in table.items()}
                for table in action_tables
            ]
            for action_tables in state_dict["tables"]
        ]
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
        """Return the total number of distinct LUT entries visited so far."""
        return sum(len(table) for action_tables in self.tables for table in action_tables)
