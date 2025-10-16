"""
Data validation and quality assessment for datasets.
"""

from typing import Dict, Set
from collections import Counter

from src.models.core import (
    Dataset, Conversation, ValidationResult, QualityReport, DatasetType
)
from src.interfaces.base import DataValidatorInterface
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataValidator(DataValidatorInterface):
    """Validates dataset schema and assesses data quality."""
    
    def validate_schema(self, dataset: Dataset) -> ValidationResult:
        """
        Validate dataset schema and structure.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation result with errors and warnings
        """
        result = ValidationResult(is_valid=True)
        
        # Check if dataset has conversations
        if not dataset.conversations:
            result.add_error("Dataset contains no conversations")
            return result
        
        # Validate each conversation
        for idx, conversation in enumerate(dataset.conversations):
            self._validate_conversation(conversation, idx, result)
        
        # Check for dataset-specific requirements
        self._validate_dataset_type_requirements(dataset, result)
        
        if result.is_valid:
            logger.info(f"Dataset validation passed for {dataset.name}")
        else:
            logger.warning(f"Dataset validation failed for {dataset.name} with {len(result.errors)} errors")
        
        return result
    
    def _validate_conversation(
        self, 
        conversation: Conversation, 
        idx: int, 
        result: ValidationResult
    ) -> None:
        """Validate a single conversation."""
        # Check conversation ID
        if not conversation.id:
            result.add_error(f"Conversation at index {idx} has no ID")
        
        # Check turns
        if not conversation.turns:
            result.add_warning(f"Conversation {conversation.id} has no turns")
            return
        
        # Validate each turn
        for turn_idx, turn in enumerate(conversation.turns):
            if not turn.text or not turn.text.strip():
                result.add_warning(
                    f"Conversation {conversation.id}, turn {turn_idx} has empty text"
                )
            
            # Check text length
            if len(turn.text) > 10000:
                result.add_warning(
                    f"Conversation {conversation.id}, turn {turn_idx} has very long text ({len(turn.text)} chars)"
                )
            
            # Check confidence score if present
            if turn.confidence is not None:
                if not 0 <= turn.confidence <= 1:
                    result.add_error(
                        f"Conversation {conversation.id}, turn {turn_idx} has invalid confidence: {turn.confidence}"
                    )
    
    def _validate_dataset_type_requirements(
        self, 
        dataset: Dataset, 
        result: ValidationResult
    ) -> None:
        """Validate dataset-specific requirements."""
        if dataset.dataset_type == DatasetType.BANKING77:
            # BANKING77 should have intent labels
            intents = dataset.get_intents()
            if not intents:
                result.add_warning("BANKING77 dataset has no intent labels")
            elif len(intents) < 70:
                result.add_warning(
                    f"BANKING77 dataset has only {len(intents)} intents (expected ~77)"
                )
        
        elif dataset.dataset_type == DatasetType.BITEXT:
            # Bitext should have multi-turn conversations
            single_turn_count = sum(1 for conv in dataset.conversations if conv.turn_count == 2)
            if single_turn_count == len(dataset.conversations):
                logger.info("Bitext dataset contains Q&A pairs (2 turns per conversation)")
        
        elif dataset.dataset_type == DatasetType.SCHEMA_GUIDED:
            # Schema-Guided should have multi-turn dialogues
            avg_turns = dataset.total_turns / dataset.size if dataset.size > 0 else 0
            if avg_turns < 3:
                result.add_warning(
                    f"Schema-Guided dataset has low average turns per conversation: {avg_turns:.1f}"
                )
    
    def check_data_quality(self, dataset: Dataset) -> QualityReport:
        """
        Assess data quality metrics.
        
        Args:
            dataset: Dataset to assess
            
        Returns:
            Quality report with metrics and scores
        """
        total_records = dataset.size
        valid_records = 0
        missing_fields: Dict[str, int] = {
            'empty_text': 0,
            'missing_intent': 0,
            'missing_speaker': 0,
            'missing_metadata': 0
        }
        
        # Track unique conversation IDs for duplicate detection
        conversation_ids: Set[str] = set()
        duplicate_records = 0
        
        # Analyze each conversation
        for conversation in dataset.conversations:
            is_valid = True
            
            # Check for duplicate IDs
            if conversation.id in conversation_ids:
                duplicate_records += 1
            else:
                conversation_ids.add(conversation.id)
            
            # Check metadata
            if not conversation.metadata:
                missing_fields['missing_metadata'] += 1
            
            # Check turns
            for turn in conversation.turns:
                if not turn.text or not turn.text.strip():
                    missing_fields['empty_text'] += 1
                    is_valid = False
                
                if turn.speaker is None:
                    missing_fields['missing_speaker'] += 1
                    is_valid = False
                
                # Check intent for user turns (dataset-specific)
                if dataset.dataset_type in [DatasetType.BANKING77, DatasetType.BITEXT]:
                    if turn.intent is None and str(turn.speaker) == 'Speaker.USER':
                        missing_fields['missing_intent'] += 1
            
            if is_valid:
                valid_records += 1
        
        # Calculate completeness score
        total_fields = total_records * 3  # text, speaker, metadata per conversation
        missing_count = sum(missing_fields.values())
        completeness_score = max(0.0, 1.0 - (missing_count / total_fields)) if total_fields > 0 else 0.0
        
        # Calculate consistency score
        consistency_score = (valid_records / total_records) if total_records > 0 else 0.0
        
        report = QualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            total_records=total_records,
            valid_records=valid_records,
            missing_fields=missing_fields,
            duplicate_records=duplicate_records
        )
        
        logger.info(
            f"Quality assessment for {dataset.name}: "
            f"Overall={report.overall_quality:.2f}, "
            f"Completeness={completeness_score:.2f}, "
            f"Consistency={consistency_score:.2f}"
        )
        
        return report


class DataQualityAnalyzer:
    """Advanced data quality analysis utilities."""
    
    @staticmethod
    def analyze_text_statistics(dataset: Dataset) -> Dict[str, float]:
        """Analyze text statistics across the dataset."""
        text_lengths = []
        word_counts = []
        
        for conversation in dataset.conversations:
            for turn in conversation.turns:
                text = turn.text.strip()
                text_lengths.append(len(text))
                word_counts.append(len(text.split()))
        
        if not text_lengths:
            return {
                'avg_text_length': 0.0,
                'max_text_length': 0.0,
                'min_text_length': 0.0,
                'avg_word_count': 0.0,
                'max_word_count': 0.0,
                'min_word_count': 0.0
            }
        
        return {
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'max_text_length': float(max(text_lengths)),
            'min_text_length': float(min(text_lengths)),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'max_word_count': float(max(word_counts)),
            'min_word_count': float(min(word_counts))
        }
    
    @staticmethod
    def analyze_intent_distribution(dataset: Dataset) -> Dict[str, int]:
        """Analyze intent distribution across the dataset."""
        intent_counts: Counter = Counter()
        
        for conversation in dataset.conversations:
            for turn in conversation.turns:
                if turn.intent:
                    intent_counts[turn.intent] += 1
        
        return dict(intent_counts)
    
    @staticmethod
    def analyze_conversation_patterns(dataset: Dataset) -> Dict[str, float]:
        """Analyze conversation patterns."""
        turn_counts = [conv.turn_count for conv in dataset.conversations]
        user_turn_counts = [len(conv.user_turns) for conv in dataset.conversations]
        assistant_turn_counts = [len(conv.assistant_turns) for conv in dataset.conversations]
        
        if not turn_counts:
            return {
                'avg_turns_per_conversation': 0.0,
                'max_turns': 0.0,
                'min_turns': 0.0,
                'avg_user_turns': 0.0,
                'avg_assistant_turns': 0.0
            }
        
        return {
            'avg_turns_per_conversation': sum(turn_counts) / len(turn_counts),
            'max_turns': float(max(turn_counts)),
            'min_turns': float(min(turn_counts)),
            'avg_user_turns': sum(user_turn_counts) / len(user_turn_counts),
            'avg_assistant_turns': sum(assistant_turn_counts) / len(assistant_turn_counts)
        }
    
    @staticmethod
    def generate_quality_summary(dataset: Dataset, validator: DataValidator) -> Dict:
        """Generate comprehensive quality summary."""
        validation_result = validator.validate_schema(dataset)
        quality_report = validator.check_data_quality(dataset)
        text_stats = DataQualityAnalyzer.analyze_text_statistics(dataset)
        intent_dist = DataQualityAnalyzer.analyze_intent_distribution(dataset)
        conversation_patterns = DataQualityAnalyzer.analyze_conversation_patterns(dataset)
        
        return {
            'dataset_name': dataset.name,
            'dataset_type': dataset.dataset_type.value,
            'validation': {
                'is_valid': validation_result.is_valid,
                'num_errors': len(validation_result.errors),
                'num_warnings': len(validation_result.warnings),
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            },
            'quality': {
                'overall_quality': quality_report.overall_quality,
                'completeness_score': quality_report.completeness_score,
                'consistency_score': quality_report.consistency_score,
                'total_records': quality_report.total_records,
                'valid_records': quality_report.valid_records,
                'duplicate_records': quality_report.duplicate_records,
                'missing_fields': quality_report.missing_fields
            },
            'text_statistics': text_stats,
            'intent_distribution': {
                'num_unique_intents': len(intent_dist),
                'top_10_intents': dict(sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            'conversation_patterns': conversation_patterns
        }
