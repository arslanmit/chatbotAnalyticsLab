"""
Dataset management endpoints.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas.datasets import (
    DatasetUploadRequest,
    DatasetUploadResponse,
)
from src.api.dependencies import get_data_preprocessor
from src.repositories.dataset_loaders import DatasetLoaderFactory

router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    request: DatasetUploadRequest,
    preprocessor=Depends(get_data_preprocessor),
) -> DatasetUploadResponse:
    """
    Load (and optionally preprocess) a dataset, returning summary metadata.
    """
    if not request.file_path and not request.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either file_path or url must be provided.",
        )
    if request.url:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="URL-based dataset ingestion is not yet supported.",
        )

    dataset_path = Path(request.file_path).expanduser().resolve()
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset path not found: {dataset_path}",
        )

    loader = DatasetLoaderFactory.get_loader(request.dataset_type)
    dataset = loader.load(dataset_path)

    preprocessed_flag = False
    normalized_flag = False

    if request.preprocess:
        dataset = preprocessor.preprocess_dataset(dataset, normalize=request.normalize_text)
        preprocessed_flag = True
        normalized_flag = request.normalize_text

    intents = dataset.get_intents()

    return DatasetUploadResponse(
        name=dataset.name,
        dataset_type=dataset.dataset_type,
        conversations=dataset.size,
        total_turns=dataset.total_turns,
        intent_count=len(intents),
        metadata=dataset.metadata,
        preprocessed=preprocessed_flag,
        normalized=normalized_flag,
    )
