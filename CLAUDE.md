# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Reinforcement Learning project for training agents to play the 2048 game. The project provides:
- A complete 2048 game implementation with both CLI and visual interfaces
- A Gymnasium-compatible RL environment for training agents
- MLflow integration for experiment tracking
- Model checkpointing utilities
- Support for both human play and agent visualization

## Language and Environment

- **Language**: Python 3.9+
- **Framework**: PyTorch for RL models
- **Package Manager**: uv (fast Python package installer)
- **Dependencies**: See `pyproject.toml` and `uv.lock` for full list

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package and dependencies
uv sync

# For development (includes pytest, black, ruff)
uv sync --all-extras
```

## Development Commands

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_game.py
```

### Code Quality
```bash
# Format code with black
uv run black src/ tests/ scripts/

# Lint with ruff
uv run ruff check src/ tests/ scripts/

# Fix linting issues
uv run ruff check --fix src/ tests/ scripts/

# Type check with mypy (optional)
uv run mypy src/
```

### Playing the Game
```bash
# CLI version (keyboard input)
uv run python scripts/play_cli.py

# Visual version (pygame, arrow keys)
uv run python scripts/play_visual.py

# With custom board size
uv run python scripts/play_visual.py --size 5
```

### Training Agents
```bash
# Train an agent (to be implemented)
uv run python scripts/train.py

# Evaluate a trained agent
uv run python scripts/evaluate.py --model-path models/best_model.pt --visualize

# Run headless evaluation
uv run python scripts/evaluate.py --model-path models/best_model.pt --n-episodes 100
```

### CI/CD

The project uses GitHub Actions for continuous integration:
- **Automated Testing**: Runs on Python 3.9, 3.10, 3.11, and 3.12
- **Code Quality**: Checks formatting (black) and linting (ruff)
- **Coverage**: Generates coverage reports on Python 3.10
- **Fast Dependencies**: Uses uv for quick dependency installation

See `.github/workflows/ci.yml` for the full CI configuration.

## Architecture

### Core Components

#### 1. Game Logic (`src/game/`)
- **`game_2048.py`**: Core 2048 game implementation
  - `Game2048` class handles all game logic (moves, merges, scoring)
  - Actions: UP (0), RIGHT (1), DOWN (2), LEFT (3)
  - Board is 4x4 grid (configurable size)
  - Supports cloning game state for planning/search algorithms

#### 2. RL Environment (`src/env/`)
- **`gym_2048.py`**: Gymnasium-compatible wrapper
  - `Gym2048Env` provides standard RL interface
  - Observation space: 4x4 grid with log2 tile values
  - Action space: Discrete(4)
  - Configurable reward modes: "score", "log_score", "max_tile"
  - Invalid move penalties optional

#### 3. Visual Interface (`src/ui/`)
- **`pygame_ui.py`**: Pygame-based visualization
  - `PygameUI` class for rendering game state
  - Supports both human play and agent visualization
  - Controls for human play: Arrow keys, R (reset), Q (quit)
  - Smooth animations and color-coded tiles

#### 4. Utilities (`src/utils/`)
- **`checkpoint.py`**: Model checkpointing
  - `ModelCheckpoint` saves/loads PyTorch models
  - Tracks best model based on metrics
  - Saves config files alongside models
- **`logger.py`**: Experiment tracking
  - `MLFlowLogger` wrapper for MLflow
  - Tracks metrics, parameters, artifacts
  - Local storage in `./experiments/mlruns`

#### 5. Agents (`src/agents/`)
- To be implemented: DQN, PPO, A3C, etc.
- Agent implementations will follow a common interface

### Data Flow

1. **Training Loop**:
   - Environment (`Gym2048Env`) wraps game (`Game2048`)
   - Agent selects action based on observation
   - Environment executes action, returns reward and next state
   - Agent updates policy based on experience
   - MLflow logs metrics; checkpoints save models

2. **Evaluation**:
   - Load trained model from checkpoint
   - Run episodes with agent policy
   - Optionally visualize with `PygameUI`
   - Aggregate statistics across episodes

### Key Design Decisions

- **Log2 Representation**: Observations use log2 of tile values (e.g., tile 2→1, 4→2, 8→3) to normalize input range for neural networks
- **Modular Architecture**: Game logic, environment wrapper, and UI are decoupled for flexibility
- **Gym Compatibility**: Following Gymnasium API enables use of standard RL libraries (Stable-Baselines3, RLlib, etc.)
- **MLflow over WandB**: Uses MLflow for experiment tracking (free, local-first, no license required)
