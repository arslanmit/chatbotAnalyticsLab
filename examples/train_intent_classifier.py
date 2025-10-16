"""
Example script demonstrating intent classifier training and prediction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.intent_classifier import IntentClassifier
from src.models.core import TrainingConfig
from src.repositories.dataset_loaders import Banking77Loader
from src.services.data_preprocessor import DataPreprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main training example."""
    
    # Load BANKING77 dataset
    logger.info("Loading BANKING77 dataset...")
    loader = Banking77Loader()
    dataset_path = Path("Dataset/BANKING77")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    dataset = loader.load(dataset_path)
    logger.info(f"Loaded dataset with {dataset.size} conversations")
    logger.info(f"Found {len(dataset.get_intents())} unique intents")
    
    # Split dataset
    logger.info("Splitting dataset into train/val/test sets...")
    preprocessor = DataPreprocessor()
    train_data, val_data, test_data = preprocessor.create_train_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    logger.info(f"Train size: {train_data.size}")
    logger.info(f"Validation size: {val_data.size}")
    logger.info(f"Test size: {test_data.size}")
    
    # Configure training
    config = TrainingConfig(
        model_name="bert-base-uncased",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=3,
        max_length=128,
        random_seed=42
    )
    
    # Initialize and train classifier
    logger.info("Initializing intent classifier...")
    classifier = IntentClassifier(model_name=config.model_name)
    
    logger.info("Starting training...")
    result = classifier.train(train_data, val_data, config)
    
    logger.info("\n" + "="*50)
    logger.info("Training Results:")
    logger.info("="*50)
    logger.info(f"Model saved to: {result.model_path}")
    logger.info(f"Training time: {result.training_time:.2f} seconds")
    logger.info(f"Training accuracy: {result.training_metrics.accuracy:.4f}")
    logger.info(f"Validation accuracy: {result.validation_metrics.accuracy:.4f}")
    logger.info(f"Validation macro F1: {result.validation_metrics.macro_f1:.4f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = classifier.evaluate(test_data)
    
    logger.info("\n" + "="*50)
    logger.info("Test Results:")
    logger.info("="*50)
    logger.info(f"Test accuracy: {test_metrics.accuracy:.4f}")
    logger.info(f"Test macro precision: {test_metrics.macro_precision:.4f}")
    logger.info(f"Test macro recall: {test_metrics.macro_recall:.4f}")
    logger.info(f"Test macro F1: {test_metrics.macro_f1:.4f}")
    
    # Test predictions
    logger.info("\n" + "="*50)
    logger.info("Sample Predictions:")
    logger.info("="*50)
    
    test_queries = [
        "I want to transfer money to another account",
        "How do I check my account balance?",
        "My card is not working",
        "I need to change my PIN",
        "What are the interest rates for savings accounts?"
    ]
    
    for query in test_queries:
        prediction = classifier.predict(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Predicted Intent: {prediction.intent}")
        logger.info(f"Confidence: {prediction.confidence:.4f}")
        logger.info(f"Top alternatives: {prediction.alternatives[:3]}")
    
    # Test batch prediction
    logger.info("\n" + "="*50)
    logger.info("Batch Prediction Test:")
    logger.info("="*50)
    
    batch_predictions = classifier.predict_batch(test_queries)
    for query, pred in zip(test_queries, batch_predictions):
        logger.info(f"{query[:50]:50s} -> {pred.intent:30s} ({pred.confidence:.3f})")
    
    logger.info("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
