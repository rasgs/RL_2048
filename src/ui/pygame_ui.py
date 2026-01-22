"""Pygame-based visual interface for 2048."""

from typing import Callable, Optional

import numpy as np
import pygame

from ..game import Action, Game2048

# Color scheme
COLORS = {
    "background": (187, 173, 160),
    "empty": (205, 193, 180),
    "text_dark": (119, 110, 101),
    "text_light": (249, 246, 242),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    # For tiles > 2048
    "super": (60, 58, 50),
}


class PygameUI:
    """
    Pygame-based UI for 2048 game.

    Supports both human play and visualizing agent moves.
    """

    def __init__(
        self,
        game: Game2048,
        cell_size: int = 100,
        margin: int = 10,
        fps: int = 10,
    ):
        """
        Initialize UI.

        Args:
            game: Game2048 instance to visualize
            cell_size: Size of each cell in pixels
            margin: Margin between cells in pixels
            fps: Frames per second for animation
        """
        self.game = game
        self.cell_size = cell_size
        self.margin = margin
        self.fps = fps

        # Calculate window size
        board_size = game.size * cell_size + (game.size + 1) * margin
        header_height = 100
        self.width = board_size
        self.height = board_size + header_height
        self.board_offset_y = header_height

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("2048")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

    def run_human_game(self):
        """
        Run game with human player controls.

        Controls:
        - Arrow keys: Move
        - R: Restart
        - Q/ESC: Quit
        """
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    action = None

                    if event.key == pygame.K_UP:
                        action = Action.UP
                    elif event.key == pygame.K_RIGHT:
                        action = Action.RIGHT
                    elif event.key == pygame.K_DOWN:
                        action = Action.DOWN
                    elif event.key == pygame.K_LEFT:
                        action = Action.LEFT
                    elif event.key == pygame.K_r:
                        self.game.reset()
                        print("Game reset!")
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    # Execute action
                    if action is not None:
                        if self.game.is_valid_action(action):
                            board, reward, done = self.game.step(action)
                            if done:
                                print(f"Game Over! Score: {self.game.score}")
                        else:
                            print("Invalid move!")

            # Draw
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

    def run_agent_game(
        self,
        get_action: Callable[[np.ndarray], int],
        delay_ms: int = 500,
        max_steps: Optional[int] = None,
    ):
        """
        Run game with RL agent providing actions.

        Args:
            get_action: Function that takes board state and returns action
            delay_ms: Delay between moves in milliseconds
            max_steps: Maximum steps (None for unlimited)
        """
        running = True
        paused = False
        step = 0
        auto_play = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:
                        self.game.reset()
                        step = 0
                        print("Game reset!")
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # Execute agent action if not paused
            if not paused and auto_play and not self.game.is_game_over():
                if max_steps is None or step < max_steps:
                    action = get_action(self.game.board.copy())
                    if self.game.is_valid_action(action):
                        board, reward, done = self.game.step(action)
                        step += 1

                        if done:
                            print(f"Game Over! Score: {self.game.score}, Steps: {step}")
                            auto_play = False
                    else:
                        print(f"Agent selected invalid action: {action}")

            # Draw
            self.draw()
            pygame.time.wait(delay_ms)
            self.clock.tick(self.fps)

        pygame.quit()

    def draw(self):
        """Draw the current game state."""
        self.screen.fill(COLORS["background"])

        # Draw header
        self._draw_header()

        # Draw board
        self._draw_board()

        pygame.display.flip()

    def _draw_header(self):
        """Draw score and info header."""
        # Score
        score_text = self.font_medium.render(f"Score: {self.game.score}", True, COLORS["text_dark"])
        self.screen.blit(score_text, (20, 20))

        # Max tile
        max_tile_text = self.font_small.render(
            f"Max: {self.game.max_tile}", True, COLORS["text_dark"]
        )
        self.screen.blit(max_tile_text, (20, 60))

        # Game over / won message
        if self.game.is_game_over():
            if self.game.has_won():
                msg = "You Win!"
                color = (0, 200, 0)
            else:
                msg = "Game Over!"
                color = (200, 0, 0)

            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.width - 150, 45))
            self.screen.blit(text, text_rect)

    def _draw_board(self):
        """Draw the game board."""
        for i in range(self.game.size):
            for j in range(self.game.size):
                value = self.game.board[i, j]
                self._draw_cell(i, j, value)

    def _draw_cell(self, row: int, col: int, value: int):
        """
        Draw a single cell.

        Args:
            row: Row index
            col: Column index
            value: Tile value (0 for empty)
        """
        # Calculate position
        x = col * (self.cell_size + self.margin) + self.margin
        y = row * (self.cell_size + self.margin) + self.margin + self.board_offset_y

        # Get color
        if value == 0:
            color = COLORS["empty"]
        elif value in COLORS:
            color = COLORS[value]
        else:
            color = COLORS["super"]

        # Draw cell background
        pygame.draw.rect(
            self.screen, color, (x, y, self.cell_size, self.cell_size), border_radius=5
        )

        # Draw value
        if value > 0:
            # Choose text color
            text_color = COLORS["text_light"] if value > 4 else COLORS["text_dark"]

            # Choose font size based on value length
            if value >= 1000:
                font = self.font_small
            else:
                font = self.font_medium

            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
