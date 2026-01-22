#!/usr/bin/env python
"""CLI interface for playing 2048."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game import Action, Game2048


def get_action_input() -> int:
    """
    Get action from user input.

    Returns:
        Action index, or -1 to quit
    """
    print("\nControls: W=Up, D=Right, S=Down, A=Left, Q=Quit")
    while True:
        key = input("Enter move: ").strip().upper()
        if key == "W":
            return Action.UP
        elif key == "D":
            return Action.RIGHT
        elif key == "S":
            return Action.DOWN
        elif key == "A":
            return Action.LEFT
        elif key == "Q":
            return -1
        else:
            print("Invalid input. Use W/A/S/D or Q.")


def main():
    """Main game loop."""
    print("=" * 40)
    print("Welcome to 2048!")
    print("=" * 40)
    print("\nGoal: Combine tiles to reach 2048!")
    print("Tiles merge when they have the same value.\n")

    game = Game2048()

    while True:
        # Display board
        print("\n" + str(game))

        # Check game state
        if game.is_game_over():
            print("\n" + "=" * 40)
            if game.has_won():
                print("Congratulations! You reached 2048!")
            else:
                print("Game Over!")
            print(f"Final Score: {game.score}")
            print(f"Max Tile: {game.max_tile}")
            print("=" * 40)
            break

        # Get action
        valid_actions = game.get_valid_actions()
        print(f"\nValid moves: {[Action(a).name for a in valid_actions]}")

        action = get_action_input()
        if action == -1:
            print("\nThanks for playing!")
            print(f"Final Score: {game.score}")
            break

        # Execute action
        if action not in valid_actions:
            print("That move is not valid!")
            continue

        board, reward, done = game.step(action)
        if reward > 0:
            print(f"Score gained: +{reward}")


if __name__ == "__main__":
    main()
