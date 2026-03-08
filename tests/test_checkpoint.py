"""Tests for ModelCheckpoint."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.checkpoint import ModelCheckpoint


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_initialization(tmp_path):
    """Test checkpoint manager initializes correctly."""
    save_dir = tmp_path / "checkpoints"
    checkpoint = ModelCheckpoint(
        save_dir=str(save_dir),
        filename_prefix="model",
        save_best=True,
        mode="max",
        verbose=False,
    )

    assert checkpoint.save_dir == save_dir
    assert checkpoint.filename_prefix == "model"
    assert checkpoint.save_best
    assert checkpoint.mode == "max"
    assert checkpoint.best_metric == float("-inf")
    assert checkpoint.best_epoch == 0

    # Check directory was created
    assert save_dir.exists()


def test_checkpoint_initialization_min_mode(tmp_path):
    """Test checkpoint with min mode."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        mode="min",
        verbose=False,
    )

    assert checkpoint.mode == "min"
    assert checkpoint.best_metric == float("inf")


def test_save_checkpoint(tmp_path):
    """Test saving a checkpoint."""
    save_dir = tmp_path / "checkpoints"
    checkpoint = ModelCheckpoint(
        save_dir=str(save_dir),
        filename_prefix="test_model",
        save_best=False,
        verbose=False,
    )

    model = DummyModel()
    epoch = 10

    checkpoint_path = checkpoint.save(model, epoch=epoch)

    # Check file was created
    assert Path(checkpoint_path).exists()
    assert "test_model_epoch_10.pt" in checkpoint_path

    # Check latest was created
    latest_path = save_dir / "test_model_latest.pt"
    assert latest_path.exists()


def test_save_with_optimizer(tmp_path):
    """Test saving checkpoint with optimizer state."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint.save(model, epoch=5, optimizer=optimizer)

    # Load and check
    saved_data = torch.load(tmp_path / "checkpoints" / "model_latest.pt")
    assert "optimizer_state_dict" in saved_data


def test_save_best_model_max_mode(tmp_path):
    """Test saving best model in max mode."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        filename_prefix="model",
        save_best=True,
        mode="max",
        verbose=False,
    )

    model = DummyModel()

    # Save with metric=5.0
    checkpoint.save(model, epoch=1, metric=5.0)
    assert checkpoint.best_metric == 5.0
    assert checkpoint.best_epoch == 1

    # Check best model was saved
    best_path = tmp_path / "checkpoints" / "model_best.pt"
    assert best_path.exists()

    # Save with lower metric (shouldn't update best)
    checkpoint.save(model, epoch=2, metric=3.0)
    assert checkpoint.best_metric == 5.0
    assert checkpoint.best_epoch == 1

    # Save with higher metric (should update best)
    checkpoint.save(model, epoch=3, metric=10.0)
    assert checkpoint.best_metric == 10.0
    assert checkpoint.best_epoch == 3


def test_save_best_model_min_mode(tmp_path):
    """Test saving best model in min mode."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        save_best=True,
        mode="min",
        verbose=False,
    )

    model = DummyModel()

    # Save with metric=5.0
    checkpoint.save(model, epoch=1, metric=5.0)
    assert checkpoint.best_metric == 5.0

    # Save with higher metric (shouldn't update best in min mode)
    checkpoint.save(model, epoch=2, metric=10.0)
    assert checkpoint.best_metric == 5.0

    # Save with lower metric (should update best in min mode)
    checkpoint.save(model, epoch=3, metric=2.0)
    assert checkpoint.best_metric == 2.0


def test_save_with_metadata(tmp_path):
    """Test saving checkpoint with metadata."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    model = DummyModel()
    metadata = {"learning_rate": 0.001, "batch_size": 32, "notes": "test run"}

    checkpoint.save(model, epoch=1, metadata=metadata)

    # Load and check
    saved_data = torch.load(tmp_path / "checkpoints" / "model_latest.pt")
    assert "metadata" in saved_data
    assert saved_data["metadata"]["learning_rate"] == 0.001
    assert saved_data["metadata"]["batch_size"] == 32


def test_load_checkpoint(tmp_path):
    """Test loading a checkpoint."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    # Create and save model
    model1 = DummyModel()
    checkpoint.save(model1, epoch=10, metric=5.0)

    # Create new model and load
    model2 = DummyModel()
    loaded_data = checkpoint.load(model2)

    # Check loaded data
    assert loaded_data["epoch"] == 10
    assert loaded_data["metric"] == 5.0

    # Check model weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_load_best_checkpoint(tmp_path):
    """Test loading best checkpoint."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        save_best=True,
        mode="max",
        verbose=False,
    )

    model1 = DummyModel()

    # Save multiple checkpoints
    checkpoint.save(model1, epoch=1, metric=5.0)
    checkpoint.save(model1, epoch=2, metric=3.0)
    checkpoint.save(model1, epoch=3, metric=10.0)  # Best

    # Load best
    model2 = DummyModel()
    loaded_data = checkpoint.load(model2, load_best=True)

    # Should load epoch 3 (best metric)
    assert loaded_data["epoch"] == 3
    assert loaded_data["metric"] == 10.0


def test_load_specific_checkpoint(tmp_path):
    """Test loading a specific checkpoint by path."""
    save_dir = tmp_path / "checkpoints"
    checkpoint = ModelCheckpoint(
        save_dir=str(save_dir),
        filename_prefix="model",
        verbose=False,
    )

    model1 = DummyModel()
    checkpoint.save(model1, epoch=5)

    # Load specific checkpoint
    model2 = DummyModel()
    specific_path = save_dir / "model_epoch_5.pt"
    loaded_data = checkpoint.load(model2, checkpoint_path=str(specific_path))

    assert loaded_data["epoch"] == 5


def test_load_with_optimizer(tmp_path):
    """Test loading checkpoint with optimizer state."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    model = DummyModel()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)

    # Make an optimizer step to change its state
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    optimizer1.step()

    # Save
    checkpoint.save(model, epoch=1, optimizer=optimizer1)

    # Create new optimizer and load
    model2 = DummyModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    checkpoint.load(model2, optimizer=optimizer2)

    # Check optimizer states match
    state1 = optimizer1.state_dict()
    state2 = optimizer2.state_dict()
    assert state1.keys() == state2.keys()


def test_save_config(tmp_path):
    """Test saving configuration."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "model": "DQN",
    }

    checkpoint.save_config(config, filename="train_config.json")

    # Check file exists
    config_path = tmp_path / "checkpoints" / "train_config.json"
    assert config_path.exists()

    # Check contents
    with open(config_path, "r") as f:
        loaded_config = json.load(f)

    assert loaded_config == config


def test_load_config(tmp_path):
    """Test loading configuration."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    config = {"learning_rate": 0.001, "batch_size": 32}

    # Save and load
    checkpoint.save_config(config)
    loaded_config = checkpoint.load_config()

    assert loaded_config == config


def test_checkpoint_not_found(tmp_path):
    """Test error when checkpoint file not found."""
    checkpoint = ModelCheckpoint(
        save_dir=str(tmp_path / "checkpoints"),
        verbose=False,
    )

    model = DummyModel()

    try:
        checkpoint.load(model, checkpoint_path="nonexistent.pt")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as e:
        assert "Checkpoint not found" in str(e)
