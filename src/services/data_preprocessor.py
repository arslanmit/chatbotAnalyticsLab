"""
Data preprocessing and normalization for datasets.
"""

import re
import random
from typing import Tuple, List

from src.models.core import Dataset, Conversation, ConversationTurn
from src.interfaces.base import DataPreprocessorInterface
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor(DataPreprocessorInterface):
    """Preprocesses and normalizes text data."""
    
    def __init__(self):
        self.lowercase = True
        self.remove_urls = True
        self.remove_emails = True
        self.remove_extra_whitespace = True
        self.remove_special_chars = False
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize and clean text data.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        normalized = text
        
        # Remove URLs
        if self.remove_urls:
            normalized = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '',
                normalized
            )
        
        # Remove email addresses
        if self.remove_emails:
            normalized = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '',
                normalized
            )
        
        # Remove special characters (optional)
        if self.remove_special_chars:
            normalized = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', normalized)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.strip()
        
        # Convert to lowercase (optional)
        if self.lowercase:
            normalized = normalized.lower()
        
        return normalized
    
    def preprocess_dataset(self, dataset: Dataset, normalize: bool = True) -> Dataset:
        """
        Preprocess entire dataset.
        
        Args:
            dataset: Dataset to preprocess
            normalize: Whether to normalize text
            
        Returns:
            Preprocessed dataset
        """
        preprocessed_conversations = []
        
        for conversation in dataset.conversations:
            preprocessed_turns = []
            
            for turn in conversation.turns:
                text = turn.text
                if normalize:
                    text = self.normalize_text(text)
                
                preprocessed_turn = ConversationTurn(
                    speaker=turn.speaker,
                    text=text,
                    timestamp=turn.timestamp,
                    intent=turn.intent,
                    confidence=turn.confidence
                )
                
                preprocessed_turns.append(preprocessed_turn)
            
            preprocessed_conversation = Conversation(
                id=conversation.id,
                turns=preprocessed_turns,
                source_dataset=conversation.source_dataset,
                metadata=conversation.metadata,
                success=conversation.success
            )
            
            preprocessed_conversations.append(preprocessed_conversation)
        
        preprocessed_dataset = Dataset(
            name=dataset.name,
            dataset_type=dataset.dataset_type,
            conversations=preprocessed_conversations,
            metadata={
                **dataset.metadata,
                'preprocessed': True,
                'normalized': normalize
            }
        )
        
        logger.info(f"Preprocessed dataset {dataset.name} with {len(preprocessed_conversations)} conversations")
        
        return preprocessed_dataset
    
    def create_train_test_split(
        self,
        dataset: Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set (default 0.7)
            val_ratio: Ratio for validation set (default 0.15)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0 or test_ratio > 1:
            raise ValueError(
                f"Invalid split ratios: train={train_ratio}, val={val_ratio}, "
                f"test={test_ratio}. Must sum to 1.0"
            )
        
        # Shuffle conversations
        conversations = dataset.conversations.copy()
        random.seed(random_seed)
        random.shuffle(conversations)
        
        # Calculate split indices
        total = len(conversations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split conversations
        train_conversations = conversations[:train_end]
        val_conversations = conversations[train_end:val_end]
        test_conversations = conversations[val_end:]
        
        # Create datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train",
            dataset_type=dataset.dataset_type,
            conversations=train_conversations,
            metadata={
                **dataset.metadata,
                'split': 'train',
                'split_ratio': train_ratio
            }
        )
        
        val_dataset = Dataset(
            name=f"{dataset.name}_val",
            dataset_type=dataset.dataset_type,
            conversations=val_conversations,
            metadata={
                **dataset.metadata,
                'split': 'validation',
                'split_ratio': val_ratio
            }
        )
        
        test_dataset = Dataset(
            name=f"{dataset.name}_test",
            dataset_type=dataset.dataset_type,
            conversations=test_conversations,
            metadata={
                **dataset.metadata,
                'split': 'test',
                'split_ratio': test_ratio
            }
        )
        
        logger.info(
            f"Split dataset {dataset.name}: "
            f"train={len(train_conversations)}, "
            f"val={len(val_conversations)}, "
            f"test={len(test_conversations)}"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def extract_conversation_turns(
        self,
        conversations: List[Conversation]
    ) -> List[Tuple[str, str]]:
        """
        Extract conversation turns as (user_text, assistant_text) pairs.
        
        Args:
            conversations: List of conversations
            
        Returns:
            List of (user_text, assistant_text) tuples
        """
        pairs = []
        
        for conversation in conversations:
            user_texts = []
            assistant_texts = []
            
            for turn in conversation.turns:
                if str(turn.speaker) == 'Speaker.USER':
                    user_texts.append(turn.text)
                elif str(turn.speaker) == 'Speaker.ASSISTANT':
                    assistant_texts.append(turn.text)
            
            # Create pairs (simple approach: match by index)
            for i in range(min(len(user_texts), len(assistant_texts))):
                pairs.append((user_texts[i], assistant_texts[i]))
        
        return pairs
    
    def balance_dataset_by_intent(
        self,
        dataset: Dataset,
        max_samples_per_intent: int = 1000
    ) -> Dataset:
        """
        Balance dataset by limiting samples per intent.
        
        Args:
            dataset: Dataset to balance
            max_samples_per_intent: Maximum samples per intent
            
        Returns:
            Balanced dataset
        """
        from collections import defaultdict
        
        # Group conversations by intent
        intent_groups = defaultdict(list)
        no_intent_conversations = []
        
        for conversation in dataset.conversations:
            # Get intent from first user turn
            intent = None
            for turn in conversation.turns:
                if str(turn.speaker) == 'Speaker.USER' and turn.intent:
                    intent = turn.intent
                    break
            
            if intent:
                intent_groups[intent].append(conversation)
            else:
                no_intent_conversations.append(conversation)
        
        # Balance by sampling
        balanced_conversations = []
        
        for intent, convs in intent_groups.items():
            if len(convs) > max_samples_per_intent:
                sampled = random.sample(convs, max_samples_per_intent)
                balanced_conversations.extend(sampled)
            else:
                balanced_conversations.extend(convs)
        
        # Add conversations without intents
        balanced_conversations.extend(no_intent_conversations)
        
        balanced_dataset = Dataset(
            name=f"{dataset.name}_balanced",
            dataset_type=dataset.dataset_type,
            conversations=balanced_conversations,
            metadata={
                **dataset.metadata,
                'balanced': True,
                'max_samples_per_intent': max_samples_per_intent,
                'original_size': dataset.size
            }
        )
        
        logger.info(
            f"Balanced dataset {dataset.name}: "
            f"{dataset.size} -> {balanced_dataset.size} conversations"
        )
        
        return balanced_dataset


class ConversationExtractor:
    """Utilities for extracting structured data from conversations."""
    
    @staticmethod
    def extract_intent_examples(
        dataset: Dataset,
        intent: str
    ) -> List[str]:
        """Extract all text examples for a specific intent."""
        examples = []
        
        for conversation in dataset.conversations:
            for turn in conversation.turns:
                if turn.intent == intent:
                    examples.append(turn.text)
        
        return examples
    
    @staticmethod
    def extract_user_queries(dataset: Dataset) -> List[Tuple[str, str]]:
        """Extract user queries with their intents."""
        queries = []
        
        for conversation in dataset.conversations:
            for turn in conversation.turns:
                if str(turn.speaker) == 'Speaker.USER':
                    queries.append((turn.text, turn.intent or 'unknown'))
        
        return queries
    
    @staticmethod
    def extract_multi_turn_dialogues(
        dataset: Dataset,
        min_turns: int = 3
    ) -> List[Conversation]:
        """Extract conversations with minimum number of turns."""
        return [
            conv for conv in dataset.conversations
            if conv.turn_count >= min_turns
        ]
    
    @staticmethod
    def create_intent_classification_dataset(
        dataset: Dataset
    ) -> List[Tuple[str, str]]:
        """
        Create intent classification dataset as (text, intent) pairs.
        
        Args:
            dataset: Source dataset
            
        Returns:
            List of (text, intent) tuples
        """
        classification_data = []
        
        for conversation in dataset.conversations:
            for turn in conversation.turns:
                if str(turn.speaker) == 'Speaker.USER' and turn.intent:
                    classification_data.append((turn.text, turn.intent))
        
        return classification_data


class DataAugmentor:
    """Data augmentation utilities for increasing dataset size."""
    
    @staticmethod
    def augment_by_paraphrasing(text: str) -> List[str]:
        """
        Simple paraphrasing augmentation (placeholder for more advanced methods).
        
        Args:
            text: Original text
            
        Returns:
            List of paraphrased versions
        """
        # This is a simple placeholder
        # In production, use models like T5, BART, or GPT for paraphrasing
        augmented = [text]
        
        # Simple word substitutions (very basic)
        substitutions = {
            'can you': ['could you', 'would you'],
            'help me': ['assist me', 'support me'],
            'i want to': ['i would like to', "i'd like to"],
            'how do i': ['how can i', 'what is the way to']
        }
        
        for original, replacements in substitutions.items():
            if original in text.lower():
                for replacement in replacements:
                    augmented_text = text.lower().replace(original, replacement)
                    augmented.append(augmented_text)
        
        return augmented
    
    @staticmethod
    def augment_dataset(
        dataset: Dataset,
        augmentation_factor: int = 2
    ) -> Dataset:
        """
        Augment dataset by creating variations.
        
        Args:
            dataset: Original dataset
            augmentation_factor: How many variations to create per sample
            
        Returns:
            Augmented dataset
        """
        augmented_conversations = list(dataset.conversations)
        
        for conversation in dataset.conversations[:]:
            for _ in range(augmentation_factor - 1):
                # Create augmented turns
                augmented_turns = []
                for turn in conversation.turns:
                    if str(turn.speaker) == 'Speaker.USER':
                        # Augment user turns
                        variations = DataAugmentor.augment_by_paraphrasing(turn.text)
                        if len(variations) > 1:
                            augmented_text = variations[1]
                        else:
                            augmented_text = turn.text
                    else:
                        augmented_text = turn.text
                    
                    augmented_turn = ConversationTurn(
                        speaker=turn.speaker,
                        text=augmented_text,
                        timestamp=turn.timestamp,
                        intent=turn.intent,
                        confidence=turn.confidence
                    )
                    augmented_turns.append(augmented_turn)
                
                # Create augmented conversation
                augmented_conv = Conversation(
                    id=f"{conversation.id}_aug",
                    turns=augmented_turns,
                    source_dataset=conversation.source_dataset,
                    metadata={
                        **conversation.metadata,
                        'augmented': True
                    },
                    success=conversation.success
                )
                
                augmented_conversations.append(augmented_conv)
        
        augmented_dataset = Dataset(
            name=f"{dataset.name}_augmented",
            dataset_type=dataset.dataset_type,
            conversations=augmented_conversations,
            metadata={
                **dataset.metadata,
                'augmented': True,
                'augmentation_factor': augmentation_factor,
                'original_size': dataset.size
            }
        )
        
        logger.info(
            f"Augmented dataset {dataset.name}: "
            f"{dataset.size} -> {augmented_dataset.size} conversations"
        )
        
        return augmented_dataset
