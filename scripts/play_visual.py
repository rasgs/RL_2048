#!/usr/bin/env python
"""Visual interface for playing 2048 with pygame."""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game import Game2048
from src.ui import PygameUI


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play 2048 with visual interface")
    parser.add_argument("--size", type=int, default=4, help="Board size (default: 4)")
    parser.add_argument(
        "--cell-size", type=int, default=100, help="Cell size in pixels (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create game
    game = Game2048(size=args.size, seed=args.seed)

    # Create UI
    ui = PygameUI(game, cell_size=args.cell_size)

    print("=" * 50)
    print("2048 - Visual Mode")
    print("=" * 50)
    print("\nControls:")
    print("  Arrow Keys: Move tiles")
    print("  R: Restart game")
    print("  Q/ESC: Quit")
    print("\n" + "=" * 50)

    # Run game
    ui.run_human_game()

    print(f"\nFinal Score: {game.score}")
    print(f"Max Tile: {game.max_tile}")


if __name__ == "__main__":
    main()
