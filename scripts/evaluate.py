#!/usr/bin/env python
"""Evaluate a trained RL agent on 2048."""

import argparse

import numpy as np

from src.agents import LinearQAgent, MarkovQAgent, NTupleQAgent
from src.agents.search import select_action as expectimax_select_action
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


def load_linear_agent(model_path: str) -> LinearQAgent:
    """Load a trained linear function-approximation Q agent from disk."""
    agent = LinearQAgent()
    agent.load(model_path)
    return agent


def load_ntuple_agent(model_path: str) -> NTupleQAgent:
    """Load a trained n-tuple network Q agent from disk."""
    agent = NTupleQAgent()
    agent.load(model_path)
    return agent


def make_get_action(agent, game: Game2048, use_search: bool, search_depth: int):
    """
    Build a get_action callback bound to one episode's live game.

    `game` is the exact Game2048 instance the caller will step through this
    episode - the callback reads the board straight from it rather than
    trusting whatever board argument it's called with, so it can also reach
    `game` itself (needed to clone() ahead for search). This matches how
    both call sites already use it: PygameUI.run_agent_game and the headless
    loop below always pass a fresh copy of this same game's board.
    """
    if use_search:

        def get_action(_board: np.ndarray) -> int:
            return expectimax_select_action(agent, game, depth=search_depth)

        return get_action

    def get_action(board: np.ndarray) -> int:
        observation = board_to_observation(board)
        valid_actions = valid_actions_for_board(board)
        return agent.select_action(
            observation,
            valid_actions=valid_actions,
            use_epsilon=False,
        )

    return get_action


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
        choices=["markov", "linear", "ntuple", "random"],
        help="Agent implementation to evaluate",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visual playback")
    parser.add_argument(
        "--delay", type=int, default=500, help="Delay between moves in ms (for visualization)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--use-search",
        action="store_true",
        help=(
            "Choose actions via expectimax search over the loaded agent's "
            "value function instead of its own select_action. Requires an "
            "agent that exposes q_values(state) (linear agent type only)."
        ),
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=2,
        help="Number of player-move plies to search when --use-search is set",
    )

    args = parser.parse_args()

    if args.use_search and args.agent_type not in ("linear", "ntuple"):
        raise ValueError(
            "--use-search requires --agent-type linear or ntuple (needs agent.q_values)"
        )

    # Load model if provided
    if args.model_path and args.agent_type == "markov":
        print(f"Loading Markov Q agent from: {args.model_path}")
        agent = load_markov_agent(args.model_path)
    elif args.model_path and args.agent_type == "linear":
        print(f"Loading Linear Q agent from: {args.model_path}")
        agent = load_linear_agent(args.model_path)
    elif args.model_path and args.agent_type == "ntuple":
        print(f"Loading N-Tuple Q agent from: {args.model_path}")
        agent = load_ntuple_agent(args.model_path)
    elif args.model_path and args.agent_type != "random":
        raise ValueError(f"Unsupported agent type: {args.agent_type}")
    else:
        print("No model provided, using random agent")
        agent = None

    # Run evaluation
    scores = []
    max_tiles = []

    for episode in range(args.n_episodes):
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")

        # Create game
        seed = args.seed + episode if args.seed is not None else None
        game = Game2048(seed=seed)

        get_action = (
            random_agent
            if agent is None
            else make_get_action(agent, game, args.use_search, args.search_depth)
        )

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
