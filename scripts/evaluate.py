#!/usr/bin/env python
"""Evaluate a trained RL agent on 2048."""

import argparse
import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game import Game2048
from src.ui import PygameUI


def random_agent(board: np.ndarray) -> int:
    """
    Random agent for testing (placeholder).

    Args:
        board: Current board state

    Returns:
        Random valid action
    """
    # This is a placeholder - replace with actual model inference
    return np.random.randint(0, 4)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained 2048 agent")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to trained model checkpoint"
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--visualize", action="store_true", help="Show visual playback")
    parser.add_argument(
        "--delay", type=int, default=500, help="Delay between moves in ms (for visualization)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Load model if provided
    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        # TODO: Implement model loading and inference
        # For now, use random agent
        print("WARNING: Model loading not yet implemented, using random agent")
        get_action = random_agent
    else:
        print("No model provided, using random agent")
        get_action = random_agent

    # Run evaluation
    scores = []
    max_tiles = []

    for episode in range(args.n_episodes):
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")

        # Create game
        seed = args.seed + episode if args.seed is not None else None
        game = Game2048(seed=seed)

        if args.visualize:
            # Visual playback
            ui = PygameUI(game)
            ui.run_agent_game(get_action, delay_ms=args.delay)
        else:
            # Headless evaluation
            done = False
            steps = 0

            while not done:
                action = get_action(game.board.copy())
                board, reward, done = game.step(action)
                steps += 1

            print(f"Score: {game.score}, Max Tile: {game.max_tile}, Steps: {steps}")

        scores.append(game.score)
        max_tiles.append(game.max_tile)

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Episodes: {args.n_episodes}")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Max Tile: {np.mean(max_tiles):.2f} ± {np.std(max_tiles):.2f}")
    print(f"Best Score: {max(scores)}")
    print(f"Best Max Tile: {max(max_tiles)}")


if __name__ == "__main__":
    main()
