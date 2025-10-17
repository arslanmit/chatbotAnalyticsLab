"""
Dataset-related schemas.
"""

from typing import Optional

from pydantic import BaseModel, model_validator

from src.models.core import DatasetType


class DatasetSummary(BaseModel):
    name: str
    dataset_type: DatasetType
    conversations: int
    total_turns: int
    intent_count: int
    metadata: dict


class DatasetUploadResponse(DatasetSummary):
    preprocessed: bool = False
    normalized: bool = False

class DatasetUploadRequest(BaseModel):
    dataset_type: DatasetType
    file_path: Optional[str] = None
    url: Optional[str] = None
    overwrite: bool = False
    preprocess: bool = True
    normalize_text: bool = True

    @model_validator(mode="after")
    def validate_source(cls, values: "DatasetUploadRequest") -> "DatasetUploadRequest":
        if not values.file_path and not values.url:
            raise ValueError("Either file_path or url must be provided.")
        return values
