"""Model registry and versioning for MLOps.

Tracks model versions, performance metrics, and enables A/B testing.
Supports model rollback and deployment tracking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import shutil
from loguru import logger


@dataclass
class ModelVersion:
    """Represents a versioned model."""

    name: str
    version: str
    model_path: Path
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_status: str = "registered"  # registered, staging, production, archived
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["model_path"] = str(self.model_path)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["model_path"] = Path(data["model_path"])
        return cls(**data)


class ModelRegistry:
    """
    Model registry for tracking and managing model versions.

    Provides:
    - Model versioning with semantic versioning
    - Performance tracking and comparison
    - Deployment status management
    - Model rollback capability
    - A/B testing support
    """

    def __init__(self, registry_dir: Path):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store registry metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.metadata_file = self.registry_dir / "registry.json"

        # Load existing registry
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_registry()

        logger.info(f"Model registry initialized at {self.registry_dir}")

    def register_model(
        self,
        name: str,
        version: str,
        model_path: Path,
        metadata: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name (e.g., 'gfpgan', 'codeformer')
            version: Version string (e.g., '1.4.0', '0.1.0')
            model_path: Path to model file
            metadata: Optional metadata (architecture, training info, etc.)
            performance_metrics: Optional performance metrics
            tags: Optional tags for categorization

        Returns:
            ModelVersion object
        """
        # Copy model to registry
        model_filename = f"{name}_v{version}.pth"
        registry_model_path = self.models_dir / model_filename

        if model_path.exists():
            shutil.copy2(model_path, registry_model_path)
            logger.info(f"Copied model to registry: {registry_model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")

        # Create model version
        model_version = ModelVersion(
            name=name,
            version=version,
            model_path=registry_model_path,
            metadata=metadata or {},
            performance_metrics=performance_metrics or {},
            tags=tags or [],
        )

        # Add to registry
        if name not in self.models:
            self.models[name] = {}

        self.models[name][version] = model_version

        # Save registry
        self._save_registry()

        logger.info(f"Registered model: {name} v{version}")
        return model_version

    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific model version.

        Args:
            name: Model name
            version: Version string (if None, returns latest production version)

        Returns:
            ModelVersion or None
        """
        if name not in self.models:
            return None

        if version is None:
            # Get latest production version
            production_versions = [
                v
                for v in self.models[name].values()
                if v.deployment_status == "production"
            ]

            if production_versions:
                return max(production_versions, key=lambda v: v.created_at)
            else:
                # Fallback to latest version
                return max(self.models[name].values(), key=lambda v: v.created_at)

        return self.models[name].get(version)

    def list_models(
        self,
        name: Optional[str] = None,
        deployment_status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """
        List models with optional filtering.

        Args:
            name: Filter by model name
            deployment_status: Filter by deployment status
            tags: Filter by tags (match any)

        Returns:
            List of ModelVersion objects
        """
        models = []

        for model_name, versions in self.models.items():
            if name is not None and model_name != name:
                continue

            for version in versions.values():
                if deployment_status and version.deployment_status != deployment_status:
                    continue

                if tags and not any(tag in version.tags for tag in tags):
                    continue

                models.append(version)

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def update_deployment_status(
        self, name: str, version: str, status: str
    ) -> bool:
        """
        Update deployment status of a model.

        Args:
            name: Model name
            version: Version string
            status: New status (registered, staging, production, archived)

        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        if model is None:
            logger.error(f"Model not found: {name} v{version}")
            return False

        # If promoting to production, demote current production version
        if status == "production":
            for v in self.models[name].values():
                if v.deployment_status == "production":
                    v.deployment_status = "archived"
                    logger.info(f"Archived previous production: {name} v{v.version}")

        model.deployment_status = status
        self._save_registry()

        logger.info(f"Updated deployment status: {name} v{version} -> {status}")
        return True

    def update_performance_metrics(
        self, name: str, version: str, metrics: Dict[str, float]
    ) -> bool:
        """
        Update performance metrics for a model.

        Args:
            name: Model name
            version: Version string
            metrics: Performance metrics to update

        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        if model is None:
            logger.error(f"Model not found: {name} v{version}")
            return False

        model.performance_metrics.update(metrics)
        self._save_registry()

        logger.info(f"Updated metrics for {name} v{version}")
        return True

    def compare_models(
        self, name: str, versions: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare performance metrics across model versions.

        Args:
            name: Model name
            versions: Version strings to compare (if None, compares all)

        Returns:
            Dictionary mapping version to metrics
        """
        if name not in self.models:
            return {}

        comparison = {}

        for version, model in self.models[name].items():
            if versions is not None and version not in versions:
                continue

            comparison[version] = {
                "metrics": model.performance_metrics,
                "deployment_status": model.deployment_status,
                "created_at": model.created_at.isoformat(),
                "tags": model.tags,
            }

        return comparison

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """
        Get the current production model.

        Args:
            name: Model name

        Returns:
            ModelVersion or None
        """
        if name not in self.models:
            return None

        for model in self.models[name].values():
            if model.deployment_status == "production":
                return model

        return None

    def rollback_production(self, name: str, to_version: str) -> bool:
        """
        Rollback production to a previous version.

        Args:
            name: Model name
            to_version: Version to rollback to

        Returns:
            True if successful
        """
        target_model = self.get_model(name, to_version)
        if target_model is None:
            logger.error(f"Target version not found: {name} v{to_version}")
            return False

        # Archive current production
        current_prod = self.get_production_model(name)
        if current_prod:
            current_prod.deployment_status = "archived"

        # Promote target to production
        target_model.deployment_status = "production"
        self._save_registry()

        logger.info(f"Rolled back {name} to v{to_version}")
        return True

    def delete_version(self, name: str, version: str) -> bool:
        """
        Delete a model version.

        Args:
            name: Model name
            version: Version string

        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        if model is None:
            logger.error(f"Model not found: {name} v{version}")
            return False

        # Don't delete production models
        if model.deployment_status == "production":
            logger.error(f"Cannot delete production model: {name} v{version}")
            return False

        # Delete model file
        if model.model_path.exists():
            model.model_path.unlink()

        # Remove from registry
        del self.models[name][version]
        self._save_registry()

        logger.info(f"Deleted model: {name} v{version}")
        return True

    def export_metadata(self, output_path: Path) -> None:
        """
        Export registry metadata to JSON.

        Args:
            output_path: Output file path
        """
        metadata = {
            model_name: {
                version: model.to_dict()
                for version, model in versions.items()
            }
            for model_name, versions in self.models.items()
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported metadata to {output_path}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        metadata = {
            model_name: {
                version: model.to_dict()
                for version, model in versions.items()
            }
            for model_name, versions in self.models.items()
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

            for model_name, versions in metadata.items():
                self.models[model_name] = {}
                for version, model_data in versions.items():
                    self.models[model_name][version] = ModelVersion.from_dict(
                        model_data
                    )

            logger.info(f"Loaded {len(self.models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")


class ABTestingManager:
    """
    A/B testing manager for model comparison.

    Supports:
    - Traffic splitting between model versions
    - Performance tracking per version
    - Statistical significance testing
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize A/B testing manager.

        Args:
            registry: Model registry
        """
        self.registry = registry
        self.experiments: Dict[str, Dict] = {}
        self.results: Dict[str, List[Dict]] = {}

        logger.info("A/B testing manager initialized")

    def create_experiment(
        self,
        name: str,
        model_a: str,
        version_a: str,
        model_b: str,
        version_b: str,
        traffic_split: float = 0.5,
    ) -> Dict:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            model_a: Name of model A
            version_a: Version of model A
            model_b: Name of model B
            version_b: Version of model B
            traffic_split: Percentage of traffic to model A (0-1)

        Returns:
            Experiment configuration
        """
        experiment = {
            "name": name,
            "model_a": {"name": model_a, "version": version_a},
            "model_b": {"name": model_b, "version": version_b},
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        self.experiments[name] = experiment
        self.results[name] = []

        logger.info(f"Created experiment: {name}")
        return experiment

    def record_result(
        self,
        experiment_name: str,
        variant: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Record a result for an experiment.

        Args:
            experiment_name: Experiment name
            variant: Variant name ('a' or 'b')
            metrics: Performance metrics
        """
        if experiment_name not in self.experiments:
            logger.error(f"Experiment not found: {experiment_name}")
            return

        result = {
            "variant": variant,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.results[experiment_name].append(result)

    def get_experiment_summary(self, experiment_name: str) -> Dict:
        """
        Get summary statistics for an experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            Summary statistics
        """
        if experiment_name not in self.experiments:
            return {}

        results = self.results.get(experiment_name, [])

        # Aggregate by variant
        variant_a_metrics = [r["metrics"] for r in results if r["variant"] == "a"]
        variant_b_metrics = [r["metrics"] for r in results if r["variant"] == "b"]

        def avg_metrics(metrics_list):
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {
                key: sum(m[key] for m in metrics_list) / len(metrics_list)
                for key in keys
            }

        summary = {
            "experiment": self.experiments[experiment_name],
            "total_samples": len(results),
            "variant_a": {
                "samples": len(variant_a_metrics),
                "avg_metrics": avg_metrics(variant_a_metrics),
            },
            "variant_b": {
                "samples": len(variant_b_metrics),
                "avg_metrics": avg_metrics(variant_b_metrics),
            },
        }

        return summary

    def stop_experiment(self, experiment_name: str) -> bool:
        """
        Stop an experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            True if successful
        """
        if experiment_name not in self.experiments:
            logger.error(f"Experiment not found: {experiment_name}")
            return False

        self.experiments[experiment_name]["status"] = "stopped"
        logger.info(f"Stopped experiment: {experiment_name}")
        return True
