"""Gymnasium environment wrapper for 2048 game."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..game import Game2048


class Gym2048Env(gym.Env):
    """
    Gymnasium environment for 2048 game.

    Observation space: 4x4 grid with tile values (log2 representation)
    Action space: Discrete(4) - UP, RIGHT, DOWN, LEFT
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 4,
        render_mode: Optional[str] = None,
        invalid_move_penalty: float = 0.0,
        reward_mode: str = "score",
        max_steps: Optional[int] = None,
    ):
        """
        Initialize environment.

        Args:
            size: Board size (default 4x4)
            render_mode: Rendering mode ("human", "ansi", or None)
            invalid_move_penalty: Penalty for invalid moves (default 0.0)
            reward_mode: How to compute rewards:
                - "score": Reward is score gained from merges
                - "log_score": Reward is log2 of score gained
                - "max_tile": Reward based on max tile reached
                - "shaped": Dense composite reward (new max tile, open-cell
                    count shrinking, score improving, game-over), modeled on
                    a reference agent's hand-tuned reward_calc
                - "shaped_open_reward": Same as "shaped" but the open-cell
                    term is flipped to reward more open cells instead of
                    fewer, matching the "more empty cells is safer" intuition
                    used elsewhere in this project
            max_steps: Maximum steps per episode (None for no limit)
        """
        super().__init__()

        self.size = size
        self.render_mode = render_mode
        self.invalid_move_penalty = invalid_move_penalty
        self.reward_mode = reward_mode
        self.max_steps = max_steps

        # Initialize game
        self.game = Game2048(size=size)

        # Define observation space
        # Using log2 representation: empty=0, 2=1, 4=2, 8=3, ..., 2048=11
        self.observation_space = spaces.Box(
            low=0,
            high=20,  # Allow up to 2^20 = 1M (should be enough)
            shape=(size, size),
            dtype=np.int32,
        )

        # Define action space
        self.action_space = spaces.Discrete(4)

        # Track episode info
        self.steps = 0
        self.episode_reward = 0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed, options=options)

        # Reset game
        self.game.reset(seed=seed)
        self.steps = 0
        self.episode_reward = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.steps += 1

        # Check if action is valid
        is_valid = self.game.is_valid_action(action)

        # Snapshot pre-move state for reward modes that need a before/after
        # comparison (e.g. "shaped")
        old_max = self.game.max_tile
        old_open = int(np.sum(self.game.board == 0))
        old_score = self.game.score

        # Execute action
        board, score_gained, done = self.game.step(action)

        # Compute reward
        if not is_valid:
            reward = -self.invalid_move_penalty
        else:
            reward = self._compute_reward(
                score_gained,
                old_max=old_max,
                old_open=old_open,
                old_score=old_score,
                done=done,
            )

        self.episode_reward += reward

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["valid_action"] = is_valid
        info["score_gained"] = score_gained

        # terminated = game over, truncated = episode limit reached
        terminated = done
        truncated = self.max_steps is not None and self.steps >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            output = str(self.game)
            if self.render_mode == "human":
                print(output)
                return None
            return output
        return None

    def close(self):
        """Clean up resources."""
        pass

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (log2 representation of board).

        Returns:
            Board with log2 values
        """
        obs = np.zeros_like(self.game.board)
        mask = self.game.board > 0
        obs[mask] = np.log2(self.game.board[mask]).astype(np.int32)
        return obs

    def _compute_reward(
        self,
        score_gained: float,
        old_max: int = 0,
        old_open: int = 0,
        old_score: int = 0,
        done: bool = False,
    ) -> float:
        """
        Compute reward based on reward mode.

        Args:
            score_gained: Score gained from last move
            old_max: Max tile before the move (only used by "shaped" modes)
            old_open: Empty cell count before the move (only used by
                "shaped" modes)
            old_score: Score before the move (only used by "shaped" modes)
            done: Whether the move ended the game (only used by "shaped" modes)

        Returns:
            Computed reward
        """
        if self.reward_mode == "score":
            return score_gained
        elif self.reward_mode == "log_score":
            if score_gained > 0:
                return np.log2(score_gained)
            return 0.0
        elif self.reward_mode == "max_tile":
            # Reward when reaching new max tile
            return float(self.game.max_tile)
        elif self.reward_mode in ("shaped", "shaped_open_reward"):
            return self._compute_shaped_reward(old_max, old_open, old_score, done)
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _compute_shaped_reward(
        self, old_max: int, old_open: int, old_score: int, done: bool
    ) -> float:
        """
        Dense composite reward modeled on a reference agent's reward_calc:
        a large bonus for reaching a new max tile, a bonus/penalty for the
        open-cell count improving/worsening, a smaller bonus/penalty for
        score improving/not, and a large penalty for ending the game. Every
        term is evaluated independently and summed (not an if/elif chain),
        so a single move can accumulate multiple bonuses/penalties at once.

        "shaped_open_reward" flips the open-cell term's sign relative to
        "shaped" to reward more empty cells (mobility) instead of fewer,
        for direct comparison against the faithful port of the reference.
        """
        new_max = self.game.max_tile
        new_open = int(np.sum(self.game.board == 0))
        new_score = self.game.score

        reward = 0.0
        if new_max > old_max:
            reward += 100.0

        if self.reward_mode == "shaped_open_reward":
            reward += 25.0 if new_open > old_open else -50.0
        else:
            reward += 25.0 if new_open < old_open else -50.0

        reward += 2.0 if new_score > old_score else -15.0

        if done:
            reward -= 1000.0

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional info.

        Returns:
            Info dictionary
        """
        return {
            "score": self.game.score,
            "max_tile": self.game.max_tile,
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "valid_actions": self.game.get_valid_actions(),
        }

    def get_board(self) -> np.ndarray:
        """Get raw board (not log2 representation)."""
        return self.game.board.copy()
