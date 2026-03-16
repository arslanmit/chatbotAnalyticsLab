"""
Repository layer for data loading and storage.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Banking77Loader": ("src.repositories.dataset_loaders", "Banking77Loader"),
    "BitextLoader": ("src.repositories.dataset_loaders", "BitextLoader"),
    "SchemaGuidedLoader": ("src.repositories.dataset_loaders", "SchemaGuidedLoader"),
    "TwitterSupportLoader": ("src.repositories.dataset_loaders", "TwitterSupportLoader"),
    "SyntheticSupportLoader": ("src.repositories.dataset_loaders", "SyntheticSupportLoader"),
    "DatasetLoaderFactory": ("src.repositories.dataset_loaders", "DatasetLoaderFactory"),
    "ModelRepository": ("src.repositories.model_repository", "ModelRepository"),
    "ExperimentRepository": ("src.repositories.persistence", "ExperimentRepository"),
    "ModelArtifactRepository": ("src.repositories.persistence", "ModelArtifactRepository"),
    "DatasetRepository": ("src.repositories.persistence", "DatasetRepository"),
    "ConversationRepository": ("src.repositories.persistence", "ConversationRepository"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.repositories' has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)


def __dir__() -> list[str]:
    return sorted(__all__)
