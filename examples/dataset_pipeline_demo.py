"""
Demonstration of the complete dataset loading and processing pipeline.

This script shows how to:
1. Load datasets from various formats
2. Validate data quality
3. Preprocess and normalize text
4. Split data for training
5. Extract structured data for ML tasks
"""

from pathlib import Path
from src.repositories.dataset_loaders import DatasetLoaderFactory
from src.services.data_validator import DataValidator, DataQualityAnalyzer
from src.services.data_preprocessor import DataPreprocessor, ConversationExtractor
from src.models.core import DatasetType


def main():
    print("=" * 70)
    print("Dataset Loading and Processing Pipeline Demo")
    print("=" * 70)
    
    # 1. Load Bitext dataset
    print("\n1. Loading Bitext Retail Banking Dataset...")
    print("-" * 70)
    loader = DatasetLoaderFactory.get_loader(DatasetType.BITEXT)
    bitext_dataset = loader.load(Path("Dataset/BitextRetailBanking"))
    
    print(f"✓ Loaded {bitext_dataset.size} conversations")
    print(f"  Total turns: {bitext_dataset.total_turns}")
    print(f"  Unique intents: {len(bitext_dataset.get_intents())}")
    
    # 2. Validate data quality
    print("\n2. Validating Data Quality...")
    print("-" * 70)
    validator = DataValidator()
    validation_result = validator.validate_schema(bitext_dataset)
    quality_report = validator.check_data_quality(bitext_dataset)
    
    print(f"✓ Validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    print(f"  Errors: {len(validation_result.errors)}")
    print(f"  Warnings: {len(validation_result.warnings)}")
    print(f"  Overall Quality: {quality_report.overall_quality:.2%}")
    print(f"  Completeness: {quality_report.completeness_score:.2%}")
    print(f"  Consistency: {quality_report.consistency_score:.2%}")
    
    # 3. Analyze dataset statistics
    print("\n3. Analyzing Dataset Statistics...")
    print("-" * 70)
    text_stats = DataQualityAnalyzer.analyze_text_statistics(bitext_dataset)
    intent_dist = DataQualityAnalyzer.analyze_intent_distribution(bitext_dataset)
    
    print(f"✓ Text Statistics:")
    print(f"  Avg text length: {text_stats['avg_text_length']:.1f} chars")
    print(f"  Avg word count: {text_stats['avg_word_count']:.1f} words")
    print(f"✓ Intent Distribution:")
    print(f"  Unique intents: {len(intent_dist)}")
    top_intents = sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    for intent, count in top_intents:
        print(f"  - {intent}: {count} examples")
    
    # 4. Preprocess dataset
    print("\n4. Preprocessing Dataset...")
    print("-" * 70)
    preprocessor = DataPreprocessor()
    
    # Sample original text
    sample_conv = bitext_dataset.conversations[0]
    original_text = sample_conv.turns[0].text
    print(f"Original text: {original_text[:80]}...")
    
    # Preprocess
    preprocessed_dataset = preprocessor.preprocess_dataset(bitext_dataset, normalize=True)
    preprocessed_text = preprocessed_dataset.conversations[0].turns[0].text
    print(f"Preprocessed: {preprocessed_text[:80]}...")
    
    # 5. Split dataset
    print("\n5. Splitting Dataset for Training...")
    print("-" * 70)
    train, val, test = preprocessor.create_train_test_split(
        preprocessed_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42
    )
    
    print(f"✓ Dataset split:")
    print(f"  Training: {train.size} conversations ({train.size/preprocessed_dataset.size:.1%})")
    print(f"  Validation: {val.size} conversations ({val.size/preprocessed_dataset.size:.1%})")
    print(f"  Test: {test.size} conversations ({test.size/preprocessed_dataset.size:.1%})")
    
    # 6. Extract data for ML tasks
    print("\n6. Extracting Data for ML Tasks...")
    print("-" * 70)
    
    # Extract intent classification data
    classification_data = ConversationExtractor.create_intent_classification_dataset(train)
    print(f"✓ Intent classification dataset: {len(classification_data)} examples")
    
    # Extract user queries
    user_queries = ConversationExtractor.extract_user_queries(train)
    print(f"✓ User queries extracted: {len(user_queries)}")
    
    # Sample classification example
    if classification_data:
        text, intent = classification_data[0]
        print(f"\nSample classification example:")
        print(f"  Text: {text[:60]}...")
        print(f"  Intent: {intent}")
    
    # 7. Load Schema-Guided dataset
    print("\n7. Loading Schema-Guided Dialogue Dataset...")
    print("-" * 70)
    schema_loader = DatasetLoaderFactory.get_loader(DatasetType.SCHEMA_GUIDED)
    schema_dataset = schema_loader.load(Path("Dataset/SchemaGuidedDialogue/banking_only/train"))
    
    print(f"✓ Loaded {schema_dataset.size} multi-turn conversations")
    print(f"  Total turns: {schema_dataset.total_turns}")
    print(f"  Avg turns per conversation: {schema_dataset.total_turns/schema_dataset.size:.1f}")
    
    # Extract multi-turn dialogues
    multi_turn = ConversationExtractor.extract_multi_turn_dialogues(schema_dataset, min_turns=5)
    print(f"✓ Conversations with 5+ turns: {len(multi_turn)}")
    
    # Sample multi-turn conversation
    if multi_turn:
        sample = multi_turn[0]
        print(f"\nSample multi-turn conversation ({sample.turn_count} turns):")
        for i, turn in enumerate(sample.turns[:4]):
            speaker = "User" if str(turn.speaker) == "Speaker.USER" else "Assistant"
            print(f"  {speaker}: {turn.text[:60]}...")
    
    # 8. Generate comprehensive quality summary
    print("\n8. Generating Comprehensive Quality Summary...")
    print("-" * 70)
    summary = DataQualityAnalyzer.generate_quality_summary(bitext_dataset, validator)
    
    print(f"✓ Quality Summary Generated:")
    print(f"  Dataset: {summary['dataset_name']}")
    print(f"  Type: {summary['dataset_type']}")
    print(f"  Records: {summary['quality']['total_records']}")
    print(f"  Valid: {summary['quality']['valid_records']}")
    print(f"  Quality Score: {summary['quality']['overall_quality']:.2%}")
    
    print("\n" + "=" * 70)
    print("✓ Pipeline Demo Complete!")
    print("=" * 70)
    print("\nAll dataset loading and processing components are working correctly.")
    print("The pipeline is ready for:")
    print("  - Intent classification model training")
    print("  - Conversation analysis")
    print("  - Multi-turn dialogue modeling")
    print("  - Quality assessment and monitoring")


if __name__ == "__main__":
    main()
