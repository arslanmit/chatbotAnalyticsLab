# Intent Classification Module

This module provides BERT-based intent classification for banking domain queries.

## Overview

The `IntentClassifier` class implements a complete pipeline for training and deploying intent classification models using HuggingFace transformers. It supports:

- **BERT-based architecture**: Uses pretrained BERT models for transfer learning
- **Training pipeline**: Complete training workflow with validation and metrics
- **Prediction interface**: Single and batch prediction with confidence scores
- **Model persistence**: Save and load trained models
- **Comprehensive evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix

## Requirements

The intent classifier satisfies the following system requirements:

- **Requirement 2.1**: Classify customer text into 77 banking intent categories
- **Requirement 2.2**: Achieve minimum 85% accuracy on test dataset
- **Requirement 2.4**: Provide confidence scores for predictions

## Usage

### Basic Training

```python
from src.models.intent_classifier import IntentClassifier
from src.models.core import TrainingConfig
from src.repositories.dataset_loaders import Banking77Loader

# Load dataset
loader = Banking77Loader()
dataset = loader.load("Dataset/BANKING77")

# Split dataset (using preprocessor)
from src.services.data_preprocessor import DataPreprocessor
preprocessor = DataPreprocessor()
train_data, val_data, test_data = preprocessor.create_train_test_split(dataset)

# Configure training
config = TrainingConfig(
    model_name="bert-base-uncased",
    learning_rate=2e-5,
    batch_size=16,
    num_epochs=3,
    max_length=128
)

# Train model
classifier = IntentClassifier(model_name=config.model_name)
result = classifier.train(train_data, val_data, config)

print(f"Training accuracy: {result.training_metrics.accuracy:.4f}")
print(f"Validation accuracy: {result.validation_metrics.accuracy:.4f}")
```

### Making Predictions

```python
# Single prediction
prediction = classifier.predict("I want to transfer money to another account")
print(f"Intent: {prediction.intent}")
print(f"Confidence: {prediction.confidence:.4f}")
print(f"Alternatives: {prediction.alternatives[:3]}")

# Batch prediction
queries = [
    "How do I check my balance?",
    "My card is not working",
    "What are the interest rates?"
]
predictions = classifier.predict_batch(queries)
for query, pred in zip(queries, predictions):
    print(f"{query} -> {pred.intent} ({pred.confidence:.3f})")
```

### Loading a Saved Model

```python
# Load previously trained model
classifier = IntentClassifier(model_path="./models/intent_classifier_20241017_120000")

# Use for predictions
prediction = classifier.predict("I need to change my PIN")
```

### Model Evaluation

```python
# Evaluate on test set
metrics = classifier.evaluate(test_data)

print(f"Test accuracy: {metrics.accuracy:.4f}")
print(f"Macro precision: {metrics.macro_precision:.4f}")
print(f"Macro recall: {metrics.macro_recall:.4f}")
print(f"Macro F1: {metrics.macro_f1:.4f}")

# Per-class metrics
for intent, f1 in metrics.f1_score.items():
    print(f"{intent}: F1={f1:.4f}")
```

## Architecture

### Model Components

1. **Tokenizer**: Converts text to token IDs using BERT tokenizer
2. **BERT Encoder**: Pretrained BERT model for contextual embeddings
3. **Classification Head**: Linear layer for intent classification
4. **Softmax Layer**: Converts logits to probability distribution

### Training Pipeline

1. **Data Preparation**:
   - Load and validate dataset
   - Create label mappings (intent -> ID)
   - Convert to HuggingFace Dataset format
   - Tokenize text inputs

2. **Model Training**:
   - Initialize BERT model with classification head
   - Train using AdamW optimizer
   - Validate after each epoch
   - Save best model based on validation accuracy

3. **Evaluation**:
   - Calculate accuracy, precision, recall, F1-score
   - Generate confusion matrix
   - Compute per-class and macro-averaged metrics

### Prediction Pipeline

1. **Input Processing**:
   - Tokenize input text
   - Convert to tensor format
   - Move to appropriate device (CPU/GPU)

2. **Inference**:
   - Forward pass through model
   - Apply softmax to get probabilities
   - Extract top prediction and alternatives

3. **Output Formatting**:
   - Return IntentPrediction with confidence scores
   - Include top 5 alternative intents

## Configuration

### TrainingConfig Parameters

- `model_name`: Pretrained model to use (default: "bert-base-uncased")
- `learning_rate`: Learning rate for optimizer (default: 2e-5)
- `batch_size`: Batch size for training (default: 16)
- `num_epochs`: Number of training epochs (default: 3)
- `max_length`: Maximum sequence length (default: 512)
- `random_seed`: Random seed for reproducibility (default: 42)

### Device Selection

The classifier automatically detects and uses GPU if available:

- CUDA GPU: Used if available for faster training
- CPU: Fallback for systems without GPU

## Performance Optimization

### Training Optimization

- **Batch Processing**: Process multiple examples simultaneously
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Mixed Precision**: Use FP16 for faster training (if supported)
- **Early Stopping**: Stop training when validation performance plateaus

### Inference Optimization

- **Batch Prediction**: Process multiple queries in parallel
- **Model Caching**: Keep model in memory for repeated predictions
- **GPU Acceleration**: Use GPU for faster inference when available

## Model Persistence

### Saved Model Structure

```
models/intent_classifier_YYYYMMDD_HHMMSS/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.txt                # Vocabulary
└── label_map.json          # Intent label mappings
```

### Loading Models

Models can be loaded by providing the path to the saved model directory:

```python
classifier = IntentClassifier(model_path="./models/intent_classifier_20241017_120000")
```

## Error Handling

The classifier includes comprehensive error handling:

- **Model Not Loaded**: Raises RuntimeError if predict/evaluate called before training/loading
- **Invalid Confidence**: Validates confidence scores are between 0 and 1
- **Missing Files**: Checks for required model files when loading
- **Device Errors**: Handles GPU/CPU device mismatches

## Logging

The classifier uses structured logging for:

- Training progress and metrics
- Model loading/saving operations
- Prediction requests
- Error conditions

## Examples

See the `examples/` directory for complete examples:

- `train_intent_classifier.py`: Full training pipeline example
- `test_intent_classifier_basic.py`: Basic functionality test
- `verify_intent_classifier.py`: Code structure verification

## Testing

Run the verification script to check implementation:

```bash
python examples/verify_intent_classifier.py
```

For full training (requires dependencies):

```bash
pip install -r requirements.txt
python examples/train_intent_classifier.py
```
