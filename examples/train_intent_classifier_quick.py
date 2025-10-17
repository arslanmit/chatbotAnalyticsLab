#!/usr/bin/env python3
"""
Quick Intent Classifier Training Script (CPU-only, 1 epoch for demo)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Disable MPS

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.repositories.dataset_loaders import DatasetLoaderFactory
from src.models.core import DatasetType, TrainingConfig
from src.models.intent_classifier import IntentClassifier
from src.services.data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load BANKING77 dataset
    logger.info("Loading BANKING77 dataset...")
    dataset_path = project_root / "Dataset" / "BANKING77"
    loader = DatasetLoaderFactory.get_loader(DatasetType.BANKING77)
    dataset = loader.load(dataset_path)
    
    logger.info(f"Loaded dataset with {len(dataset.conversations)} conversations")
    
    # Get unique intents
    intents = set()
    for conv in dataset.conversations:
        for turn in conv.turns:
            if turn.intent:
                intents.add(turn.intent)
    
    logger.info(f"Found {len(intents)} unique intents")
    
    # Split dataset
    logger.info("Splitting dataset into train/val/test sets...")
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.create_train_test_split(
        dataset, train_ratio=0.7, val_ratio=0.15
    )
    
    logger.info(f"Train size: {len(train_dataset.conversations)}")
    logger.info(f"Validation size: {len(val_dataset.conversations)}")
    logger.info(f"Test size: {len(test_dataset.conversations)}")
    
    # Prepare training data
    train_data = [(turn.text, turn.intent) 
                  for conv in train_dataset.conversations 
                  for turn in conv.turns if turn.intent]
    val_data = [(turn.text, turn.intent) 
                for conv in val_dataset.conversations 
                for turn in conv.turns if turn.intent]
    test_data = [(turn.text, turn.intent) 
                 for conv in test_dataset.conversations 
                 for turn in conv.turns if turn.intent]
    
    # Training configuration - QUICK VERSION
    config = TrainingConfig(
        model_name="bert-base-uncased",
        learning_rate=2e-5,
        batch_size=8,  # Smaller batch size
        num_epochs=1,  # Just 1 epoch for quick demo
        max_length=128,
        random_seed=42
    )
    
    # Initialize and train classifier
    logger.info("Initializing intent classifier (CPU-only)...")
    classifier = IntentClassifier(model_name=config.model_name)
    
    logger.info("Starting quick training (1 epoch)...")
    result = classifier.train(train_data, val_data, config)
    
    logger.info("\n" + "="*50)
    logger.info("Training Results:")
    logger.info("="*50)
    logger.info(f"Model saved to: {result.model_path}")
    logger.info(f"Training time: {result.training_time:.2f} seconds")
    logger.info(f"Training accuracy: {result.training_metrics.accuracy:.4f}")
    logger.info(f"Validation accuracy: {result.validation_metrics.accuracy:.4f}")
    logger.info(f"Validation macro F1: {result.validation_metrics.macro_f1:.4f}")
    
    # Quick test on a few examples
    logger.info("\n" + "="*50)
    logger.info("Testing on sample queries:")
    logger.info("="*50)
    
    test_queries = [
        "I want to check my account balance",
        "How do I transfer money to another account?",
        "My card is not working",
    ]
    
    for query in test_queries:
        prediction = classifier.predict(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Predicted intent: {prediction.intent}")
        logger.info(f"Confidence: {prediction.confidence:.4f}")
    
    logger.info("\n" + "="*50)
    logger.info("✓ Quick training complete!")
    logger.info("="*50)

if __name__ == "__main__":
    main()
