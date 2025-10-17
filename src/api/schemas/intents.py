"""
Intent classification schemas.
"""

from typing import List, Tuple

from pydantic import BaseModel, Field


class IntentQuery(BaseModel):
    text: str = Field(..., description="User utterance to classify.")
    model_id: str = Field("intent_classifier", description="Model identifier to use for prediction.")


class BatchIntentQuery(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Utterances to classify in batch.")
    model_id: str = Field("intent_classifier", description="Model identifier to use for prediction.")


class IntentPredictionResponse(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: List[Tuple[str, float]] = Field(default_factory=list)


class BatchIntentPredictionResponse(BaseModel):
    predictions: List[IntentPredictionResponse]
