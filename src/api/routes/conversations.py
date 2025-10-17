"""
Conversation analysis endpoints.
"""

from pathlib import Path
from typing import Iterable, List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_data_preprocessor,
    get_conversation_flow_analyzer,
    get_sentiment_analyzer,
    get_performance_analyzer,
    get_response_cache,
)
from src.api.schemas.conversations import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    ConversationTrendRequest,
    ConversationTrendResponse,
)
from src.config.settings import settings
from src.models.core import Conversation, IntentPrediction
from src.repositories.dataset_loaders import DatasetLoaderFactory

router = APIRouter()


@router.post("/analyze", response_model=ConversationAnalysisResponse)
async def analyze_conversations(
    request: ConversationAnalysisRequest,
    preprocessor=Depends(get_data_preprocessor),
    flow_analyzer=Depends(get_conversation_flow_analyzer),
    sentiment_analyzer=Depends(get_sentiment_analyzer),
    performance_analyzer=Depends(get_performance_analyzer),
) -> ConversationAnalysisResponse:
    """
    Perform conversation analytics for the requested dataset subset.
    """
    dataset = _load_dataset(request.dataset_type, request.dataset_path)

    if request.preprocess:
        dataset = preprocessor.preprocess_dataset(dataset, normalize=request.normalize_text)

    conversations = _filter_conversations(dataset.conversations, request.conversation_ids)

    if not conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No conversations matched the provided criteria.",
        )

    flow_result = flow_analyzer.analyze_dialogue_flow(conversations) if request.include_flow else None
    sentiment_result = sentiment_analyzer.analyze_conversations(conversations) if request.include_sentiment else None

    performance_result = None
    if request.include_performance:
        success_metrics = flow_analyzer.calculate_success_metrics(conversations)
        response_times = performance_analyzer.compute_response_times(conversations)
        predictions = _collect_intent_predictions(conversations)
        intent_distribution = performance_analyzer.calculate_intent_distribution(predictions)
        performance_payload = {
            "success_metrics": success_metrics,
            "response_times": response_times,
            "intent_distribution": intent_distribution,
            "sentiment": sentiment_result,
        }
        performance_result = performance_analyzer.generate_performance_report(performance_payload)

    return ConversationAnalysisResponse(
        conversation_count=len(conversations),
        flow=flow_result,
        sentiment=sentiment_result,
        performance=performance_result,
    )


@router.post("/trends", response_model=ConversationTrendResponse)
async def conversation_trends(
    request: ConversationTrendRequest,
    preprocessor=Depends(get_data_preprocessor),
    sentiment_analyzer=Depends(get_sentiment_analyzer),
    cache=Depends(get_response_cache),
) -> ConversationTrendResponse:
    """
    Generate sentiment trends over time for the requested dataset subset.
    """
    dataset = _load_dataset(request.dataset_type, request.dataset_path)

    if request.preprocess:
        dataset = preprocessor.preprocess_dataset(dataset, normalize=request.normalize_text)

    cache_key = ";".join(
        [
            "trend",
            request.dataset_type.value,
            request.dataset_path or "default",
            request.granularity,
            str(request.preprocess),
            str(request.normalize_text),
        ]
    )

    cached = cache.get(cache_key)
    if cached:
        return ConversationTrendResponse(**cached)

    trend = sentiment_analyzer.calculate_sentiment_trend(
        dataset.conversations,
        granularity=request.granularity,
    )

    response = ConversationTrendResponse(
        granularity=trend["granularity"],
        trend=trend["trend"],
    )
    cache.set(cache_key, response.dict())
    return response


def _load_dataset(dataset_type, dataset_path: str | None):
    path = _resolve_dataset_path(dataset_type.value, dataset_path)
    loader = DatasetLoaderFactory.get_loader(dataset_type)
    return loader.load(path)


def _resolve_dataset_path(dataset_name: str, explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset path not found: {candidate}",
            )
        return candidate

    base_dir = Path(settings.data.dataset_dir)
    candidates = [
        base_dir / dataset_name,
        base_dir / dataset_name.lower(),
        base_dir / dataset_name.upper(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Dataset for '{dataset_name}' not found under {base_dir}",
    )


def _filter_conversations(
    conversations: Iterable[Conversation],
    conversation_ids: List[str] | None,
) -> List[Conversation]:
    if not conversation_ids:
        return list(conversations)

    indexed = {conv.id: conv for conv in conversations}
    missing = [cid for cid in conversation_ids if cid not in indexed]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversations not found: {', '.join(missing)}",
        )
    return [indexed[cid] for cid in conversation_ids]


def _collect_intent_predictions(conversations: Iterable[Conversation]) -> List[IntentPrediction]:
    predictions: List[IntentPrediction] = []
    for conversation in conversations:
        for turn in conversation.turns:
            if turn.intent:
                predictions.append(
                    IntentPrediction(
                        intent=turn.intent,
                        confidence=turn.confidence or 0.0,
                        alternatives=[],
                    )
                )
    return predictions
