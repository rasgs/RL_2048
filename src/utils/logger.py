"""MLflow logging utilities."""

from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch


class MLFlowLogger:
    """
    Wrapper for MLflow experiment tracking.

    Handles experiment creation, logging metrics, parameters, and artifacts.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (None for local ./mlruns)
            run_name: Optional name for this run
            tags: Optional tags for the run
        """
        # Set tracking URI
        if tracking_uri is None:
            tracking_uri = "./experiments/mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None

    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
        """
        if run_name is None:
            run_name = self.run_name

        self.run = mlflow.start_run(run_name=run_name, nested=nested)

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        return self.run

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any):
        """
        Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to file or directory
            artifact_path: Optional path within artifacts directory
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log PyTorch model to MLflow.

        Args:
            model: PyTorch model
            artifact_path: Path within artifacts
            registered_model_name: Optional name for model registry
        """
        mlflow.pytorch.log_model(model, artifact_path, registered_model_name=registered_model_name)

    def log_figure(self, figure, artifact_file: str):
        """
        Log matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Filename for saved figure
        """
        mlflow.log_figure(figure, artifact_file)

    def end_run(self):
        """End the current MLflow run."""
        if self.run is not None:
            mlflow.end_run()
            self.run = None

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
