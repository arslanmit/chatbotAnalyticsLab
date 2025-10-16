"""
Basic test script to verify intent classifier implementation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.intent_classifier import IntentClassifier
from src.models.core import (
    Dataset, Conversation, ConversationTurn, DatasetType, Speaker, TrainingConfig
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_sample_dataset(num_samples: int = 100) -> Dataset:
    """Create a small sample dataset for testing."""
    intents = [
        "transfer_money",
        "check_balance",
        "card_issue",
        "change_pin",
        "interest_rates"
    ]
    
    sample_texts = {
        "transfer_money": [
            "I want to transfer money",
            "How do I send money to another account",
            "Transfer funds please"
        ],
        "check_balance": [
            "What is my balance",
            "Check my account balance",
            "How much money do I have"
        ],
        "card_issue": [
            "My card is not working",
            "Card declined",
            "Problem with my debit card"
        ],
        "change_pin": [
            "I need to change my PIN",
            "Update my PIN number",
            "Reset my card PIN"
        ],
        "interest_rates": [
            "What are the interest rates",
            "Tell me about savings account rates",
            "Current interest rates"
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
    """Test basic intent classifier functionality."""
    
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset(num_samples=150)
    logger.info(f"Created dataset with {dataset.size} conversations")
    logger.info(f"Unique intents: {dataset.get_intents()}")
    
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
    
    logger.info(f"Train: {train_data.size}, Val: {val_data.size}, Test: {test_data.size}")
    
    # Test classifier initialization
    logger.info("\nTesting IntentClassifier initialization...")
    classifier = IntentClassifier(model_name="bert-base-uncased")
    logger.info("✓ Classifier initialized successfully")
    
    # Test label preparation
    logger.info("\nTesting label preparation...")
    label2id, id2label = classifier._prepare_labels(train_data)
    logger.info(f"✓ Created mappings for {len(label2id)} labels")
    logger.info(f"  Labels: {list(label2id.keys())}")
    
    # Test dataset conversion
    logger.info("\nTesting dataset conversion...")
    hf_dataset = classifier._dataset_to_hf_format(train_data)
    logger.info(f"✓ Converted to HuggingFace format: {len(hf_dataset)} examples")
    
    logger.info("\n" + "="*50)
    logger.info("Basic functionality tests passed!")
    logger.info("="*50)
    logger.info("\nNote: Full training test requires GPU/CPU resources and time.")
    logger.info("To run full training, use: python examples/train_intent_classifier.py")


if __name__ == "__main__":
    main()
