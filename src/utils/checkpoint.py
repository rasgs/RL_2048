"""Model checkpointing utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ModelCheckpoint:
    """
    Handles saving and loading of model checkpoints.

    Supports saving best model based on a metric and periodic checkpoints.
    """

    def __init__(
        self,
        save_dir: str,
        filename_prefix: str = "model",
        save_best: bool = True,
        mode: str = "max",
        verbose: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            filename_prefix: Prefix for checkpoint filenames
            save_best: Whether to track and save best model
            mode: "max" or "min" for best model tracking
            verbose: Whether to print save messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.save_best = save_best
        self.mode = mode
        self.verbose = verbose

        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0

    def save(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metric: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            optimizer: Optional optimizer state to save
            metric: Metric value for best model tracking
            metadata: Additional metadata to save
            is_best: Force saving as best model

        Returns:
            Path to saved checkpoint
        """
        # Check if this is the best model
        if metric is not None and self.save_best:
            if self.mode == "max":
                is_best = metric > self.best_metric
            else:
                is_best = metric < self.best_metric

            if is_best:
                self.best_metric = metric
                self.best_epoch = epoch

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metric is not None:
            checkpoint["metric"] = metric

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save checkpoint
        checkpoint_path = self.save_dir / f"{self.filename_prefix}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.save_dir / f"{self.filename_prefix}_best.pt"
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"Saved best model: {best_path} (metric: {metric})")

        # Save latest checkpoint link
        latest_path = self.save_dir / f"{self.filename_prefix}_latest.pt"
        torch.save(checkpoint, latest_path)

        return str(checkpoint_path)

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint (or None to load latest/best)
            load_best: Load best model instead of latest
            optimizer: Optional optimizer to load state into
            device: Device to load model to

        Returns:
            Checkpoint dictionary with metadata
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / f"{self.filename_prefix}_best.pt"
            else:
                checkpoint_path = self.save_dir / f"{self.filename_prefix}_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.verbose:
            epoch = checkpoint.get("epoch", "unknown")
            metric = checkpoint.get("metric", "unknown")
            print(f"Loaded checkpoint from epoch {epoch} (metric: {metric})")

        return checkpoint

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            filename: Filename to save to
        """
        config_path = self.save_dir / filename
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"Saved config: {config_path}")

    def load_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            filename: Filename to load from

        Returns:
            Configuration dictionary
        """
        config_path = self.save_dir / filename
        with open(config_path, "r") as f:
            config = json.load(f)

        return config
