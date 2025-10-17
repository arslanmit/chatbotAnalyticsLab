"""
Conversation analytics schemas.
"""

from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from src.models.core import DatasetType


class ConversationAnalysisRequest(BaseModel):
    dataset_type: DatasetType
    dataset_path: Optional[str] = None
    conversation_ids: Optional[List[str]] = Field(None, min_length=1)
    include_sentiment: bool = False
    include_flow: bool = True
    include_performance: bool = False
    preprocess: bool = False
    normalize_text: bool = True


class ConversationAnalysisResponse(BaseModel):
    conversation_count: int
    flow: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


class ConversationTrendRequest(BaseModel):
    dataset_type: DatasetType
    dataset_path: Optional[str] = None
    granularity: str = Field("daily", pattern="^(hourly|daily|conversation)$")
    preprocess: bool = False
    normalize_text: bool = True


class ConversationTrendResponse(BaseModel):
    granularity: str
    trend: List[Dict[str, Any]]
