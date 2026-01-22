# Getting Started with RL_2048

## Quick Start

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and all dependencies
uv sync

# For development (includes test and linting tools)
uv sync --all-extras
```

### 2. Try the Game

```bash
# Run a quick demo to verify installation
uv run python scripts/demo.py

# Play the game yourself (CLI version)
uv run python scripts/play_cli.py

# Play with visual interface (requires pygame)
uv run python scripts/play_visual.py
```

## What's Implemented

### ✓ Core Game Engine
- Complete 2048 game logic in `src/game/game_2048.py`
- Proper tile merging, scoring, and game state management
- Support for cloning game state (useful for MCTS/planning algorithms)

### ✓ Interfaces
- **CLI Interface**: Text-based game playable with WASD keys
- **Visual Interface**: Pygame-based UI with smooth graphics
  - Human play mode (arrow keys)
  - Agent visualization mode (watch RL agent play)

### ✓ RL Environment
- Gymnasium-compatible environment in `src/env/gym_2048.py`
- Standard RL interface (reset, step, render)
- Configurable reward modes
- Log2 observation encoding for neural networks

### ✓ Infrastructure
- MLflow integration for experiment tracking
- Model checkpointing utilities
- Unit tests for game logic
- Code quality tools (black, ruff, pytest)

## Next Steps: Implementing RL Agents

Now that the foundation is ready, you can implement RL agents in `src/agents/`. Here are some suggestions:

### Option 1: Deep Q-Network (DQN)
```python
# Create src/agents/dqn_agent.py
# Implement Q-network, replay buffer, training loop
# Train with: python scripts/train_dqn.py
```

### Option 2: Policy Gradient (PPO)
```python
# Create src/agents/ppo_agent.py
# Implement actor-critic network
# Train with: python scripts/train_ppo.py
```

### Option 3: Use Stable-Baselines3
```bash
# Add to dependencies in pyproject.toml, then:
uv sync

# Create wrapper script
# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", Gym2048Env(), ...)
```

## Project Structure Overview

```
RL_2048/
├── src/
│   ├── game/           # Core 2048 game logic
│   ├── env/            # Gymnasium environment wrapper
│   ├── agents/         # RL agent implementations (TODO)
│   ├── ui/             # Pygame visual interface
│   └── utils/          # Checkpointing, logging
├── scripts/
│   ├── play_cli.py     # Play game in terminal
│   ├── play_visual.py  # Play game with GUI
│   ├── demo.py         # Quick demo
│   └── evaluate.py     # Evaluate trained agents
├── tests/              # Unit tests
├── experiments/        # MLflow tracking data
└── models/             # Saved model checkpoints
```

## Development Workflow

1. **Make changes** to the code
2. **Format**: `uv run black src/ tests/ scripts/`
3. **Lint**: `uv run ruff check --fix src/ tests/ scripts/`
4. **Test**: `uv run pytest`
5. **Commit** your changes

## Continuous Integration

The project uses GitHub Actions for CI:
- **Automated tests** on Python 3.9, 3.10, 3.11, and 3.12
- **Code quality checks** (black formatting, ruff linting)
- **Coverage reporting** on Python 3.10
- **Fast CI runs** using uv for dependency installation

The CI pipeline runs automatically on every push and pull request to `main` or `develop` branches.

## Resources

- [2048 Game](https://play2048.co/) - Original game to understand mechanics
- [Gymnasium Docs](https://gymnasium.farama.org/) - RL environment API
- [PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - DQN tutorial
- [MLflow Docs](https://mlflow.org/docs/latest/index.html) - Experiment tracking

Happy coding! 🎮🤖
