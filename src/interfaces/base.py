"""
Base interfaces and abstract classes for the Chatbot Analytics System.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.models.core import (
    Dataset, Conversation, IntentPrediction, ValidationResult, 
    QualityReport, PerformanceMetrics, TrainingConfig, TrainingResult
)


class DatasetLoaderInterface(ABC):
    """Abstract interface for dataset loaders."""
    
    @abstractmethod
    def load(self, path: Path) -> Dataset:
        """
        Load a dataset from the given path.
        
        Args:
            path: Path to the dataset
            
        Returns:
            Loaded dataset
            
        Raises:
            FileNotFoundError: If the dataset path doesn't exist
            ValueError: If the dataset format is invalid
        """
        pass
    
    @abstractmethod
    def validate_format(self, path: Path) -> bool:
        """
        Validate if the dataset format is supported.
        
        Args:
            path: Path to the dataset
            
        Returns:
            True if format is valid, False otherwise
        """
        pass


class DataValidatorInterface(ABC):
    """Abstract interface for data validation."""
    
    @abstractmethod
    def validate_schema(self, dataset: Dataset) -> ValidationResult:
        """
        Validate dataset schema and structure.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation result with errors and warnings
        """
        pass
    
    @abstractmethod
    def check_data_quality(self, dataset: Dataset) -> QualityReport:
        """
        Assess data quality metrics.
        
        Args:
            dataset: Dataset to assess
            
        Returns:
            Quality report with metrics and scores
        """
        pass


class DataPreprocessorInterface(ABC):
    """Abstract interface for data preprocessing."""
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """
        Normalize and clean text data.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        pass
    
    @abstractmethod
    def create_train_test_split(
        self, 
        dataset: Dataset, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass


class IntentClassifierInterface(ABC):
    """Abstract interface for intent classification."""
    
    @abstractmethod
    def train(self, train_data: Dataset, val_data: Dataset, config: TrainingConfig) -> TrainingResult:
        """
        Train the intent classifier.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            config: Training configuration
            
        Returns:
            Training result with metrics and model path
        """
        pass
    
    @abstractmethod
    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text to classify
            
        Returns:
            Intent prediction with confidence score
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of intent predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Dataset) -> PerformanceMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Performance metrics
        """
        pass


class ConversationAnalyzerInterface(ABC):
    """Abstract interface for conversation analysis."""
    
    @abstractmethod
    def analyze_dialogue_flow(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """
        Analyze dialogue flow patterns.
        
        Args:
            conversations: List of conversations to analyze
            
        Returns:
            Flow analysis results
        """
        pass
    
    @abstractmethod
    def detect_failure_points(self, conversations: List[Conversation]) -> List[Dict[str, Any]]:
        """
        Detect conversation failure points.
        
        Args:
            conversations: List of conversations to analyze
            
        Returns:
            List of detected failure points
        """
        pass
    
    @abstractmethod
    def calculate_success_metrics(self, conversations: List[Conversation]) -> Dict[str, float]:
        """
        Calculate conversation success metrics.
        
        Args:
            conversations: List of conversations to analyze
            
        Returns:
            Dictionary of success metrics
        """
        pass


class PerformanceAnalyzerInterface(ABC):
    """Abstract interface for performance analysis."""
    
    @abstractmethod
    def calculate_intent_distribution(self, predictions: List[IntentPrediction]) -> Dict[str, int]:
        """
        Calculate intent distribution from predictions.
        
        Args:
            predictions: List of intent predictions
            
        Returns:
            Dictionary mapping intents to counts
        """
        pass
    
    @abstractmethod
    def compute_response_times(self, conversations: List[Conversation]) -> Dict[str, float]:
        """
        Compute response time statistics.
        
        Args:
            conversations: List of conversations with timestamps
            
        Returns:
            Dictionary of response time statistics
        """
        pass
    
    @abstractmethod
    def generate_performance_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Dictionary of various metrics
            
        Returns:
            Formatted performance report
        """
        pass


class ModelRepositoryInterface(ABC):
    """Abstract interface for model storage and retrieval."""
    
    @abstractmethod
    def save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model object
            model_id: Unique identifier for the model
            metadata: Model metadata and configuration
            
        Returns:
            Path where model was saved
        """
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a saved model with its metadata.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Tuple of (model, metadata)
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model metadata dictionaries
        """
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a saved model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass


class DataRepositoryInterface(ABC):
    """Abstract interface for data storage and retrieval."""
    
    @abstractmethod
    def save_dataset(self, dataset: Dataset) -> str:
        """
        Save a processed dataset.
        
        Args:
            dataset: Dataset to save
            
        Returns:
            Dataset identifier
        """
        pass
    
    @abstractmethod
    def load_dataset(self, dataset_id: str) -> Dataset:
        """
        Load a saved dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Loaded dataset
        """
        pass
    
    @abstractmethod
    def save_conversations(self, conversations: List[Conversation]) -> List[str]:
        """
        Save conversations to storage.
        
        Args:
            conversations: List of conversations to save
            
        Returns:
            List of conversation IDs
        """
        pass
    
    @abstractmethod
    def load_conversations(self, conversation_ids: List[str]) -> List[Conversation]:
        """
        Load conversations from storage.
        
        Args:
            conversation_ids: List of conversation IDs to load
            
        Returns:
            List of loaded conversations
        """
        pass