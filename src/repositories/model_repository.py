"""
File-system based repository for persisting trained models and associated metadata.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from src.interfaces.base import ModelRepositoryInterface
from src.utils.logging import get_logger
from src.repositories.persistence import ModelArtifactRepository

logger = get_logger(__name__)


@dataclass
class ModelArtifact:
    """Resolvable object returned by the repository when loading a model."""

    model_path: Path
    tokenizer_path: Optional[Path] = None
    extra_files: Optional[Dict[str, Path]] = None


class ModelRepository(ModelRepositoryInterface):
    """Persist models to disk with simple versioning and metadata tracking."""

    def __init__(self, base_dir: str = "./models/registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelRepository initialized at %s", self.base_dir.resolve())
        self._artifact_repository = ModelArtifactRepository()

    # Public API -----------------------------------------------------------------

    def save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Persist a model instance or previously saved directory.

        Supports:
            - HuggingFace models: expects `model` to have `save_pretrained`
            - `IntentClassifier` instances (saves both model + tokenizer)
            - Path / str pointing to an already exported directory
        """
        version_dir = self._prepare_version_dir(model_id)
        logger.info("Saving model '%s' to %s", model_id, version_dir)

        if hasattr(model, "model") and hasattr(model, "tokenizer"):
            # Likely an IntentClassifier instance
            model.model.save_pretrained(version_dir)
            model.tokenizer.save_pretrained(version_dir)
            # Persist label map if available
            label_map_path = getattr(model, "model_path", None)
            if label_map_path:
                label_map_file = Path(label_map_path) / "label_map.json"
                if label_map_file.exists():
                    shutil.copy2(label_map_file, version_dir / "label_map.json")
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(version_dir)
        elif isinstance(model, (str, Path)):
            source_path = Path(model)
            if not source_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {source_path}")
            shutil.copytree(source_path, version_dir, dirs_exist_ok=True)
        else:
            raise TypeError(
                "Unsupported model type. Provide a HuggingFace model/tokenizer, "
                "an IntentClassifier instance, or a path to saved weights."
            )

        # Write metadata for this version
        metadata_enriched = {
            **metadata,
            "model_id": model_id,
            "version": version_dir.name,
            "saved_at": datetime.utcnow().isoformat(),
        }
        self._write_metadata(version_dir, metadata_enriched)

        try:
            self._artifact_repository.save(model_id, version_dir.name, str(version_dir), metadata_enriched)
        except Exception as exc:  # pragma: no cover - persistence best effort
            logger.warning("Failed to register model artifact %s:%s in database: %s", model_id, version_dir.name, exc)

        logger.info("Model '%s' saved successfully (version=%s)", model_id, version_dir.name)
        return str(version_dir)

    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load the most recent version of a model and return its artifact descriptor."""
        latest_version = self._latest_version_dir(model_id)
        if latest_version is None:
            raise FileNotFoundError(f"No saved model found for '{model_id}'")

        metadata = self._read_metadata(latest_version)
        artifact = ModelArtifact(
            model_path=latest_version,
            tokenizer_path=latest_version,
            extra_files={ "metadata": latest_version / "metadata.json" }
        )

        logger.info("Loaded model '%s' (version=%s)", model_id, latest_version.name)
        return artifact, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved model versions with their metadata."""
        entries: List[Dict[str, Any]] = []
        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for version_dir in sorted(model_dir.iterdir()):
                metadata = self._read_metadata(version_dir)
                if metadata:
                    entries.append(metadata)
        return sorted(entries, key=lambda item: item.get("saved_at", ""), reverse=True)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions."""
        model_dir = self.base_dir / model_id
        if not model_dir.exists():
            logger.warning("Model directory not found for '%s'", model_id)
            return False

        shutil.rmtree(model_dir)
        logger.info("Deleted model '%s' from repository", model_id)
        return True

    # Internal helpers -----------------------------------------------------------

    def _prepare_version_dir(self, model_id: str) -> Path:
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        next_version = self._next_version_name(model_dir)
        version_dir = model_dir / next_version
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    @staticmethod
    def _next_version_name(model_dir: Path) -> str:
        existing_versions = [p.name for p in model_dir.iterdir() if p.is_dir()]
        if not existing_versions:
            return "v1"
        indices = []
        for name in existing_versions:
            try:
                if name.startswith("v"):
                    indices.append(int(name[1:]))
            except ValueError:
                continue
        next_index = max(indices, default=0) + 1
        return f"v{next_index}"

    def _latest_version_dir(self, model_id: str) -> Optional[Path]:
        model_dir = self.base_dir / model_id
        if not model_dir.exists():
            return None
        versions = sorted(
            [p for p in model_dir.iterdir() if p.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        return versions[0] if versions else None

    @staticmethod
    def _write_metadata(version_dir: Path, metadata: Dict[str, Any]):
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _read_metadata(version_dir: Path) -> Dict[str, Any]:
        metadata_file = version_dir / "metadata.json"
        if not metadata_file.exists():
            return {}
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
