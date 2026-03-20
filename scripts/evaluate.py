#!/usr/bin/env python
"""Evaluate a trained RL agent on 2048."""

import argparse

import numpy as np

from src.agents import MarkovQAgent
from src.game import Game2048
from src.ui import PygameUI


def board_to_observation(board: np.ndarray) -> np.ndarray:
    """Convert a raw board into the environment's log2 observation format."""
    observation = np.zeros_like(board, dtype=np.int32)
    mask = board > 0
    observation[mask] = np.log2(board[mask]).astype(np.int32)
    return observation


def valid_actions_for_board(board: np.ndarray) -> list[int]:
    """Compute valid actions directly from a board snapshot."""
    game = Game2048(size=board.shape[0])
    game.board = board.copy()
    game.score = 0
    game.max_tile = int(np.max(board)) if board.size else 0
    return game.get_valid_actions()


def random_agent(board: np.ndarray) -> int:
    """
    Random valid-action agent for testing.

    Args:
        board: Current board state

    Returns:
        Random valid action
    """
    valid_actions = valid_actions_for_board(board)
    return int(np.random.choice(valid_actions))


def load_markov_agent(model_path: str) -> MarkovQAgent:
    """Load a trained Markov Q agent from disk."""
    agent = MarkovQAgent()
    agent.load(model_path)
    return agent


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained 2048 agent")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to trained model checkpoint"
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument(
        "--agent-type",
        type=str,
        default="markov",
        choices=["markov", "random"],
        help="Agent implementation to evaluate",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visual playback")
    parser.add_argument(
        "--delay", type=int, default=500, help="Delay between moves in ms (for visualization)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Load model if provided
    if args.model_path and args.agent_type == "markov":
        print(f"Loading Markov Q agent from: {args.model_path}")
        agent = load_markov_agent(args.model_path)

        def get_action(board: np.ndarray) -> int:
            observation = board_to_observation(board)
            valid_actions = valid_actions_for_board(board)
            return agent.select_action(
                observation,
                valid_actions=valid_actions,
                use_epsilon=False,
            )

    elif args.model_path and args.agent_type != "random":
        raise ValueError(f"Unsupported agent type: {args.agent_type}")
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
