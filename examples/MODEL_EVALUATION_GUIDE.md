# Model Evaluation and Metrics Guide

## Overview

The `ModelEvaluator` service provides comprehensive evaluation capabilities for intent classification models, including detailed metrics calculation, confusion matrix analysis, and model comparison tools.

## Features

### 1. Comprehensive Metrics Calculation

The evaluator calculates a wide range of metrics:

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Aggregate Metrics**: Macro and Weighted averages
- **Statistical Metrics**: Cohen's Kappa, Matthews Correlation Coefficient
- **Confidence Metrics**: Average confidence for correct/incorrect predictions

### 2. Per-Class Analysis

Detailed metrics for each intent class:

- Precision, Recall, F1-Score per class
- True Positives, False Positives, False Negatives, True Negatives
- Support (number of samples) per class

### 3. Confusion Matrix

- Generation of confusion matrix
- Visualization as heatmap
- Automatic handling of large number of classes (shows top 20)

### 4. Confidence Analysis

- Confidence distribution across prediction bins
- Accuracy by confidence range
- Statistics: min, max, mean, median, std deviation

### 5. Model Comparison

- Compare multiple models side-by-side
- Identify best performing model
- Statistical comparison across metrics
- Visualization of model performance

### 6. Misclassification Analysis

- Identify most common misclassification patterns
- Provide examples of misclassified samples
- Calculate misclassification rate
- Pattern frequency analysis

### 7. Visualizations

Automatically generated plots:

- Confusion matrix heatmap
- Per-class F1 scores bar chart
- Confidence distribution histogram
- Model comparison bar chart

## Usage

### Basic Evaluation

```python
from src.services.model_evaluator import ModelEvaluator
from src.models.intent_classifier import IntentClassifier

# Initialize evaluator
evaluator = ModelEvaluator(output_dir="./evaluation_results")

# Evaluate a trained model
results = evaluator.evaluate_model(
    classifier=trained_classifier,
    test_data=test_dataset,
    save_results=True,
    generate_visualizations=True
)

# Access metrics
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"Cohen's Kappa: {results['cohen_kappa']:.4f}")
```

### Per-Class Metrics

```python
# Get per-class metrics
per_class = results['per_class']

for intent, metrics in per_class.items():
    print(f"{intent}:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Support: {metrics['support']}")
```

### Model Comparison

```python
# Evaluate multiple models
results_model_a = evaluator.evaluate_model(classifier_a, test_data)
results_model_b = evaluator.evaluate_model(classifier_b, test_data)

# Compare models
comparison = evaluator.compare_models(
    evaluation_results=[results_model_a, results_model_b],
    model_names=["BERT-base", "DistilBERT"],
    save_comparison=True
)

print(f"Best model: {comparison['best_model']['name']}")
print(f"Weighted F1: {comparison['best_model']['weighted_f1']:.4f}")
```

### Misclassification Analysis

```python
# Analyze misclassifications
misclass_analysis = evaluator.analyze_misclassifications(
    classifier=trained_classifier,
    test_data=test_dataset,
    top_n=10
)

print(f"Total misclassifications: {misclass_analysis['total_misclassifications']}")
print(f"Misclassification rate: {misclass_analysis['misclassification_rate']:.2%}")

# Show top patterns
for pattern in misclass_analysis['top_patterns'][:5]:
    print(f"{pattern['pattern']}: {pattern['count']} occurrences")
    print(f"  Examples: {pattern['examples'][:2]}")
```

### Classification Report

```python
# Generate detailed classification report
report = evaluator.generate_classification_report(
    true_labels=true_labels,
    predicted_labels=predicted_labels,
    save_report=True,
    dataset_name="banking77_test"
)

print(report)
```

## Output Files

The evaluator saves results to the specified output directory:

### JSON Files

- `evaluation_{dataset}_{timestamp}.json` - Complete evaluation metrics
- `model_comparison_{timestamp}.json` - Model comparison results

### Visualizations

- `{dataset}_{timestamp}_confusion_matrix.png` - Confusion matrix heatmap
- `{dataset}_{timestamp}_f1_scores.png` - Per-class F1 scores
- `{dataset}_{timestamp}_confidence_dist.png` - Confidence distribution
- `model_comparison_{timestamp}.png` - Model comparison chart

### Text Reports

- `classification_report_{dataset}_{timestamp}.txt` - Detailed classification report

## Metrics Explained

### Accuracy

Overall percentage of correct predictions.

### Precision

Of all predictions for a class, how many were correct.

- High precision = Few false positives

### Recall

Of all actual instances of a class, how many were correctly identified.

- High recall = Few false negatives

### F1-Score

Harmonic mean of precision and recall.

- Balances precision and recall
- Good for imbalanced datasets

### Cohen's Kappa

Measures agreement between predictions and true labels, accounting for chance.

- Range: -1 to 1
- > 0.8 = Strong agreement
- 0.6-0.8 = Moderate agreement

### Matthews Correlation Coefficient (MCC)

Correlation between predicted and true labels.

- Range: -1 to 1
- 1 = Perfect prediction
- 0 = Random prediction
- -1 = Total disagreement

## Requirements Satisfied

This implementation satisfies the following requirements:

### Requirement 2.2

- ✓ Accuracy calculation
- ✓ Precision, recall, F1-score for each class
- ✓ Aggregate metrics (macro, weighted)
- ✓ Confidence scoring analysis

### Requirement 4.4

- ✓ Model comparison functionality
- ✓ Benchmarking tools
- ✓ Performance tracking
- ✓ Statistical comparison

## Example Output

```
Overall Metrics:
  - Accuracy: 0.8750
  - Macro F1: 0.8542
  - Weighted F1: 0.8698
  - Cohen's Kappa: 0.8456
  - Matthews Correlation: 0.8512

Confidence Statistics:
  - Mean confidence: 0.9234
  - Confidence (correct): 0.9456
  - Confidence (incorrect): 0.7823

Top 3 Classes by F1:
  - transfer_money: F1=0.95, Precision=0.94, Recall=0.96
  - check_balance: F1=0.93, Precision=0.95, Recall=0.91
  - card_issue: F1=0.89, Precision=0.87, Recall=0.91
```

## Integration with Training Pipeline

The evaluator integrates seamlessly with the training pipeline:

```python
# Train model
training_result = classifier.train(train_data, val_data, config)

# Evaluate on test set
evaluator = ModelEvaluator()
test_results = evaluator.evaluate_model(classifier, test_data)

# Compare with validation results
comparison = evaluator.compare_models(
    evaluation_results=[
        training_result.validation_metrics.__dict__,
        test_results
    ],
    model_names=["Validation", "Test"]
)
```

## Best Practices

1. **Always evaluate on held-out test data** - Don't use training or validation data
2. **Check per-class metrics** - Overall accuracy can be misleading with imbalanced data
3. **Analyze confidence** - Low confidence on correct predictions may indicate model uncertainty
4. **Review misclassifications** - Understand common error patterns
5. **Compare multiple models** - Use comparison tools to select best model
6. **Save results** - Keep evaluation results for reproducibility and tracking

## Testing

Run the verification script to test the implementation:

```bash
python examples/verify_model_evaluator.py
```

Run the full integration test (requires dependencies):

```bash
python examples/test_model_evaluation.py
```
