"""
Core data models and interfaces for the Chatbot Analytics System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


class DatasetType(Enum):
    """Supported dataset types."""
    BANKING77 = "banking77"
    BITEXT = "bitext"
    SCHEMA_GUIDED = "schema_guided"
    TWITTER_SUPPORT = "twitter_support"
    SYNTHETIC_SUPPORT = "synthetic_support"


class Speaker(Enum):
    """Conversation participants."""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    speaker: Speaker
    text: str
    timestamp: Optional[datetime] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.speaker, str):
            self.speaker = Speaker(self.speaker)


@dataclass
class Conversation:
    """Represents a complete conversation with multiple turns."""
    id: str
    turns: List[ConversationTurn]
    source_dataset: DatasetType
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    
    def __post_init__(self):
        if isinstance(self.source_dataset, str):
            self.source_dataset = DatasetType(self.source_dataset)
    
    @property
    def turn_count(self) -> int:
        """Get the number of turns in the conversation."""
        return len(self.turns)
    
    @property
    def user_turns(self) -> List[ConversationTurn]:
        """Get only user turns from the conversation."""
        return [turn for turn in self.turns if turn.speaker == Speaker.USER]
    
    @property
    def assistant_turns(self) -> List[ConversationTurn]:
        """Get only assistant turns from the conversation."""
        return [turn for turn in self.turns if turn.speaker == Speaker.ASSISTANT]


@dataclass
class IntentPrediction:
    """Represents an intent classification prediction."""
    intent: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure confidence is between 0 and 1
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class Dataset:
    """Represents a loaded and processed dataset."""
    name: str
    dataset_type: DatasetType
    conversations: List[Conversation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.dataset_type, str):
            self.dataset_type = DatasetType(self.dataset_type)
    
    @property
    def size(self) -> int:
        """Get the number of conversations in the dataset."""
        return len(self.conversations)
    
    @property
    def total_turns(self) -> int:
        """Get the total number of turns across all conversations."""
        return sum(conv.turn_count for conv in self.conversations)
    
    def get_intents(self) -> List[str]:
        """Get all unique intents from the dataset."""
        intents = set()
        for conv in self.conversations:
            for turn in conv.turns:
                if turn.intent:
                    intents.add(turn.intent)
        return sorted(list(intents))


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)


@dataclass
class QualityReport:
    """Data quality assessment report."""
    completeness_score: float  # 0-1 score for data completeness
    consistency_score: float   # 0-1 score for data consistency
    total_records: int
    valid_records: int
    missing_fields: Dict[str, int] = field(default_factory=dict)
    duplicate_records: int = 0
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness_score + self.consistency_score) / 2


@dataclass
class PerformanceMetrics:
    """Model performance evaluation metrics."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    
    @property
    def macro_precision(self) -> float:
        """Calculate macro-averaged precision."""
        return sum(self.precision.values()) / len(self.precision) if self.precision else 0.0
    
    @property
    def macro_recall(self) -> float:
        """Calculate macro-averaged recall."""
        return sum(self.recall.values()) / len(self.recall) if self.recall else 0.0
    
    @property
    def macro_f1(self) -> float:
        """Calculate macro-averaged F1 score."""
        return sum(self.f1_score.values()) / len(self.f1_score) if self.f1_score else 0.0


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 512
    train_test_split: float = 0.8
    validation_split: float = 0.1
    random_seed: int = 42
    evaluation_strategy: str = "epoch"
    save_total_limit: Optional[int] = 2
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'max_length': self.max_length,
            'train_test_split': self.train_test_split,
            'validation_split': self.validation_split,
            'random_seed': self.random_seed,
            'evaluation_strategy': self.evaluation_strategy,
            'save_total_limit': self.save_total_limit,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold
        }


@dataclass
class TrainingResult:
    """Result of model training."""
    model_path: str
    training_metrics: PerformanceMetrics
    validation_metrics: PerformanceMetrics
    training_time: float  # in seconds
    config: TrainingConfig
    
    @property
    def is_successful(self) -> bool:
        """Check if training was successful based on validation accuracy."""
        return self.validation_metrics.accuracy > 0.5  # Minimum threshold
