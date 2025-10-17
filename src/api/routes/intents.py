"""
Intent classification endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from src.api.dependencies import get_intent_classifier
from src.api.schemas.intents import (
    IntentQuery,
    BatchIntentQuery,
    IntentPredictionResponse,
    BatchIntentPredictionResponse,
)

router = APIRouter()


@router.post("/predict", response_model=IntentPredictionResponse)
async def predict_intent(query: IntentQuery) -> IntentPredictionResponse:
    """
    Classify a single utterance using the specified intent classifier.
    """
    try:
        classifier = get_intent_classifier(query.model_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{query.model_id}' not found. Train and register the model before prediction.",
        )

    prediction = classifier.predict(query.text)
    return IntentPredictionResponse(
        intent=prediction.intent,
        confidence=prediction.confidence,
        alternatives=prediction.alternatives,
    )


@router.post("/predict/batch", response_model=BatchIntentPredictionResponse)
async def predict_intent_batch(request: BatchIntentQuery) -> BatchIntentPredictionResponse:
    """
    Perform batch classification on multiple utterances.
    """
    try:
        classifier = get_intent_classifier(request.model_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_id}' not found. Train and register the model before prediction.",
        )

    predictions = classifier.predict_batch(request.texts)
    return BatchIntentPredictionResponse(
        predictions=[
            IntentPredictionResponse(
                intent=prediction.intent,
                confidence=prediction.confidence,
                alternatives=prediction.alternatives,
            )
            for prediction in predictions
        ]
    )
