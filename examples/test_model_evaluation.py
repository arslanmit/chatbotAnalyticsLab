"""
Test script for model evaluation and metrics functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.intent_classifier import IntentClassifier
from src.services.model_evaluator import ModelEvaluator
from src.models.core import (
    Dataset, Conversation, ConversationTurn, DatasetType, Speaker, TrainingConfig
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_sample_dataset(num_samples: int = 200) -> Dataset:
    """Create a sample dataset for testing."""
    intents = [
        "transfer_money",
        "check_balance",
        "card_issue",
        "change_pin",
        "interest_rates",
        "open_account",
        "close_account",
        "loan_inquiry"
    ]
    
    sample_texts = {
        "transfer_money": [
            "I want to transfer money",
            "How do I send money to another account",
            "Transfer funds please",
            "Send money to my friend"
        ],
        "check_balance": [
            "What is my balance",
            "Check my account balance",
            "How much money do I have",
            "Show me my balance"
        ],
        "card_issue": [
            "My card is not working",
            "Card declined",
            "Problem with my debit card",
            "Card is blocked"
        ],
        "change_pin": [
            "I need to change my PIN",
            "Update my PIN number",
            "Reset my card PIN",
            "Modify PIN"
        ],
        "interest_rates": [
            "What are the interest rates",
            "Tell me about savings account rates",
            "Current interest rates",
            "Interest rate information"
        ],
        "open_account": [
            "I want to open a new account",
            "How do I create an account",
            "Open savings account",
            "Start new account"
        ],
        "close_account": [
            "Close my account",
            "I want to terminate my account",
            "Cancel my account",
            "Deactivate account"
        ],
        "loan_inquiry": [
            "Tell me about loans",
            "How do I apply for a loan",
            "Loan information",
            "Personal loan rates"
        ]
    }
    
    conversations = []
    for i in range(num_samples):
        intent = intents[i % len(intents)]
        text = sample_texts[intent][i % len(sample_texts[intent])]
        
        turn = ConversationTurn(
            speaker=Speaker.USER,
            text=text,
            intent=intent
        )
        
        conv = Conversation(
            id=f"conv_{i}",
            turns=[turn],
            source_dataset=DatasetType.BANKING77,
            metadata={"sample": True}
        )
        conversations.append(conv)
    
    return Dataset(
        name="sample_dataset",
        dataset_type=DatasetType.BANKING77,
        conversations=conversations
    )


def main():
    """Test model evaluation functionality."""
    
    logger.info("="*80)
    logger.info("Testing Model Evaluation and Metrics")
    logger.info("="*80)
    
    # Create sample dataset
    logger.info("\n1. Creating sample dataset...")
    dataset = create_sample_dataset(num_samples=200)
    logger.info(f"✓ Created dataset with {dataset.size} conversations")
    logger.info(f"  Unique intents: {len(dataset.get_intents())}")
    
    # Split dataset
    train_size = int(0.7 * dataset.size)
    val_size = int(0.15 * dataset.size)
    
    train_data = Dataset(
        name="train",
        dataset_type=DatasetType.BANKING77,
        conversations=dataset.conversations[:train_size]
    )
    
    val_data = Dataset(
        name="val",
        dataset_type=DatasetType.BANKING77,
        conversations=dataset.conversations[train_size:train_size + val_size]
    )
    
    test_data = Dataset(
        name="test",
        dataset_type=DatasetType.BANKING77,
        conversations=dataset.conversations[train_size + val_size:]
    )
    
    logger.info(f"  Train: {train_data.size}, Val: {val_data.size}, Test: {test_data.size}")
    
    # Initialize classifier
    logger.info("\n2. Initializing intent classifier...")
    classifier = IntentClassifier(
        model_name="bert-base-uncased",
        batch_size=16
    )
    logger.info("✓ Classifier initialized")
    
    # Train model (quick training for testing)
    logger.info("\n3. Training model (quick training for testing)...")
    config = TrainingConfig(
        model_name="bert-base-uncased",
        num_epochs=1,  # Quick training
        batch_size=16,
        learning_rate=2e-5
    )
    
    training_result = classifier.train(train_data, val_data, config)
    logger.info(f"✓ Training completed in {training_result.training_time:.2f} seconds")
    logger.info(f"  Training accuracy: {training_result.training_metrics.accuracy:.4f}")
    logger.info(f"  Validation accuracy: {training_result.validation_metrics.accuracy:.4f}")
    
    # Initialize evaluator
    logger.info("\n4. Initializing model evaluator...")
    evaluator = ModelEvaluator(output_dir="./evaluation_results")
    logger.info("✓ Evaluator initialized")
    
    # Evaluate model
    logger.info("\n5. Evaluating model on test set...")
    evaluation_results = evaluator.evaluate_model(
        classifier=classifier,
        test_data=test_data,
        save_results=True,
        generate_visualizations=True
    )
    
    logger.info("✓ Evaluation completed")
    logger.info(f"\n  Overall Metrics:")
    logger.info(f"  - Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"  - Macro F1: {evaluation_results['macro_f1']:.4f}")
    logger.info(f"  - Weighted F1: {evaluation_results['weighted_f1']:.4f}")
    logger.info(f"  - Cohen's Kappa: {evaluation_results['cohen_kappa']:.4f}")
    logger.info(f"  - Matthews Correlation: {evaluation_results['matthews_corrcoef']:.4f}")
    
    logger.info(f"\n  Confidence Statistics:")
    logger.info(f"  - Mean confidence: {evaluation_results['confidence_statistics']['mean_confidence']:.4f}")
    logger.info(f"  - Confidence (correct): {evaluation_results['average_confidence_correct']:.4f}")
    logger.info(f"  - Confidence (incorrect): {evaluation_results['average_confidence_incorrect']:.4f}")
    
    # Test per-class metrics
    logger.info("\n6. Testing per-class metrics...")
    per_class = evaluation_results['per_class']
    logger.info(f"✓ Per-class metrics calculated for {len(per_class)} classes")
    
    # Show top 3 and bottom 3 classes by F1
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    logger.info("\n  Top 3 classes by F1 score:")
    for intent, metrics in sorted_classes[:3]:
        logger.info(f"  - {intent}: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    if len(sorted_classes) > 3:
        logger.info("\n  Bottom 3 classes by F1 score:")
        for intent, metrics in sorted_classes[-3:]:
            logger.info(f"  - {intent}: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Test classification report
    logger.info("\n7. Generating classification report...")
    texts = []
    true_labels = []
    for conv in test_data.conversations:
        for turn in conv.turns:
            if turn.intent:
                texts.append(turn.text)
                true_labels.append(turn.intent)
    
    predictions = classifier.predict_batch(texts)
    predicted_labels = [pred.intent for pred in predictions]
    
    report = evaluator.generate_classification_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        save_report=True,
        dataset_name="test"
    )
    logger.info("✓ Classification report generated")
    
    # Test misclassification analysis
    logger.info("\n8. Analyzing misclassifications...")
    misclass_analysis = evaluator.analyze_misclassifications(
        classifier=classifier,
        test_data=test_data,
        top_n=5
    )
    
    logger.info(f"✓ Misclassification analysis completed")
    logger.info(f"  Total misclassifications: {misclass_analysis['total_misclassifications']}")
    logger.info(f"  Misclassification rate: {misclass_analysis['misclassification_rate']:.2%}")
    
    if misclass_analysis['top_patterns']:
        logger.info(f"\n  Top misclassification patterns:")
        for pattern_info in misclass_analysis['top_patterns'][:3]:
            logger.info(f"  - {pattern_info['pattern']}: {pattern_info['count']} occurrences ({pattern_info['percentage']:.1f}%)")
    
    # Test model comparison (compare with itself for demonstration)
    logger.info("\n9. Testing model comparison...")
    comparison = evaluator.compare_models(
        evaluation_results=[evaluation_results, evaluation_results],
        model_names=["Model_A", "Model_B"],
        save_comparison=True
    )
    logger.info("✓ Model comparison completed")
    logger.info(f"  Best model: {comparison['best_model']['name']}")
    
    logger.info("\n" + "="*80)
    logger.info("All Model Evaluation Tests Passed!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: ./evaluation_results/")
    logger.info("Check the directory for:")
    logger.info("  - Evaluation metrics (JSON)")
    logger.info("  - Confusion matrix visualization")
    logger.info("  - Per-class F1 scores plot")
    logger.info("  - Confidence distribution plot")
    logger.info("  - Classification report (TXT)")
    logger.info("  - Model comparison results")


if __name__ == "__main__":
    main()
