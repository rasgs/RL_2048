#!/usr/bin/env python
"""Quick demo of the 2048 implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.game import Action, Game2048


def main():
    """Run a quick demo."""
    print("=" * 60)
    print("2048 Game Demo")
    print("=" * 60)

    # Create game
    game = Game2048(seed=42)

    print("\nInitial state:")
    print(game)

    # Make a few random moves
    print("\nMaking random moves...")
    for i in range(10):
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            print("Game over!")
            break

        action = np.random.choice(valid_actions)
        board, reward, done = game.step(action)

        print(f"\nMove {i+1}: {Action(action).name}")
        print(f"Reward: {reward}")
        print(game)

        if done:
            print("\nGame Over!")
            break

    print("\n" + "=" * 60)
    print(f"Final Score: {game.score}")
    print(f"Max Tile: {game.max_tile}")
    print("=" * 60)

    # Test the environment
    print("\n\nTesting Gym Environment...")
    try:
        from src.env import Gym2048Env

        env = Gym2048Env()
        obs, info = env.reset(seed=42)

        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Initial info: {info}")

        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nAfter action {action}:")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Score: {info['score']}")

        print("\nEnvironment test passed! ✓")

    except ImportError as e:
        print("Note: Gymnasium not installed yet. Install with: pip install -e .")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
