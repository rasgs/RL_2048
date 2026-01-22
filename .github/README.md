# CI/CD Configuration

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### `ci.yml` - Continuous Integration

Runs on every push and pull request to `main` or `develop` branches.

**Test Job:**
- Tests on Python 3.9, 3.10, 3.11, and 3.12
- Uses `uv` for fast dependency installation
- Runs pytest with verbose output
- Generates coverage report on Python 3.10
- Uploads coverage to Codecov (optional)

**Lint Job:**
- Checks code formatting with `black`
- Lints code with `ruff`
- Runs on Python 3.10

## Local Testing

Before pushing, you can run the same checks locally:

```bash
# Run all checks
uv run pytest tests/ -v
uv run black --check src/ tests/ scripts/
uv run ruff check src/ tests/ scripts/

# Auto-fix formatting and linting
uv run black src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/
```

## Adding Codecov (Optional)

To enable coverage reporting:

1. Sign up at https://codecov.io/
2. Add your repository
3. Add `CODECOV_TOKEN` to GitHub repository secrets
4. Coverage reports will be automatically uploaded

## Badge Status

Add these badges to your README.md:

```markdown
![CI](https://github.com/YOUR_USERNAME/RL_2048/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/RL_2048/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/RL_2048)
```
