# RL_2048

Reinforcement Learning for the 2048 game.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

For development:
```bash
uv sync --all-extras
```

## Project Structure

- `src/game/` - Core 2048 game logic
- `src/env/` - Gym-compatible RL environment
- `src/agents/` - RL agent implementations
- `src/ui/` - Visual interface (pygame)
- `src/utils/` - Utilities (checkpointing, logging)
- `scripts/` - Training and evaluation scripts
- `tests/` - Unit tests
- `notebooks/` - Analysis notebooks
- `experiments/` - MLflow tracking data
- `models/` - Saved model checkpoints

## Usage

### Play the game yourself (CLI)
```bash
uv run python scripts/play_cli.py
```

### Play with visual interface
```bash
uv run python scripts/play_visual.py
```

### Train an RL agent
```bash
uv run python scripts/train.py
```

### Watch a trained agent play
```bash
uv run python scripts/evaluate.py --model-path models/best_model.pt --visualize
```

## Development

### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
# Format code
uv run black src/ tests/ scripts/

# Lint code
uv run ruff check src/ tests/ scripts/
```

## CI/CD

This project uses GitHub Actions for continuous integration. Tests run automatically on:
- Every push to `main` or `develop` branches
- All pull requests
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
