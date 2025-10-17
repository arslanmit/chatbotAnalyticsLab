"""
Model evaluation and metrics service for intent classification.
Provides comprehensive evaluation metrics, confusion matrix analysis, and model comparison.
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.core import Dataset, PerformanceMetrics, IntentPrediction
from src.interfaces.base import IntentClassifierInterface
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation service.
    
    Features:
    - Detailed accuracy, precision, recall, F1-score calculations
    - Confusion matrix generation and visualization
    - Per-class and aggregate metrics
    - Model comparison and benchmarking
    - Statistical significance testing
    """
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelEvaluator initialized with output directory: {output_dir}")
    
    def evaluate_model(
        self,
        classifier: IntentClassifierInterface,
        test_data: Dataset,
        save_results: bool = True,
        generate_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a model.
        
        Args:
            classifier: Trained intent classifier
            test_data: Test dataset
            save_results: Whether to save results to disk
            generate_visualizations: Whether to generate visualization plots
            
        Returns:
            Dictionary containing all evaluation metrics and analysis
        """
        logger.info(f"Evaluating model on {test_data.size} conversations")
        
        # Get predictions for all test data
        texts = []
        true_labels = []
        
        for conv in test_data.conversations:
            for turn in conv.turns:
                if turn.intent:
                    texts.append(turn.text)
                    true_labels.append(turn.intent)
        
        logger.info(f"Predicting intents for {len(texts)} samples...")
        predictions = classifier.predict_batch(texts)
        predicted_labels = [pred.intent for pred in predictions]
        confidences = [pred.confidence for pred in predictions]
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            true_labels, 
            predicted_labels, 
            confidences
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Get unique labels
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        metrics['labels'] = unique_labels
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(
            true_labels,
            predicted_labels,
            unique_labels
        )
        metrics['per_class'] = per_class_metrics
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_statistics(
            true_labels,
            predicted_labels,
            confidences
        )
        metrics['confidence_statistics'] = confidence_stats
        
        # Add metadata
        metrics['metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'test_size': len(texts),
            'num_classes': len(unique_labels),
            'dataset_name': test_data.name
        }
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(metrics, test_data.name)
        
        # Generate visualizations if requested
        if generate_visualizations:
            self._generate_visualizations(
                cm,
                unique_labels,
                per_class_metrics,
                confidence_stats,
                test_data.name
            )
        
        logger.info(f"Evaluation complete. Overall accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _calculate_comprehensive_metrics(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Precision, recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average=None,
            zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average='macro',
            zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average='weighted',
            zero_division=0
        )
        
        # Additional metrics
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        mcc = matthews_corrcoef(true_labels, predicted_labels)
        
        # Confidence metrics
        avg_confidence = np.mean(confidences)
        correct_mask = np.array(true_labels) == np.array(predicted_labels)
        avg_confidence_correct = np.mean(np.array(confidences)[correct_mask])
        avg_confidence_incorrect = np.mean(np.array(confidences)[~correct_mask]) if not all(correct_mask) else 0.0
        
        return {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'cohen_kappa': float(kappa),
            'matthews_corrcoef': float(mcc),
            'average_confidence': float(avg_confidence),
            'average_confidence_correct': float(avg_confidence_correct),
            'average_confidence_incorrect': float(avg_confidence_incorrect),
            'total_samples': len(true_labels),
            'correct_predictions': int(np.sum(correct_mask)),
            'incorrect_predictions': int(np.sum(~correct_mask))
        }
    
    def _calculate_per_class_metrics(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        unique_labels: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each class."""
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=unique_labels,
            average=None,
            zero_division=0
        )
        
        per_class = {}
        for i, label in enumerate(unique_labels):
            per_class[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i]),
                'true_positives': int(np.sum((np.array(true_labels) == label) & (np.array(predicted_labels) == label))),
                'false_positives': int(np.sum((np.array(true_labels) != label) & (np.array(predicted_labels) == label))),
                'false_negatives': int(np.sum((np.array(true_labels) == label) & (np.array(predicted_labels) != label))),
                'true_negatives': int(np.sum((np.array(true_labels) != label) & (np.array(predicted_labels) != label)))
            }
        
        return per_class
    
    def _calculate_confidence_statistics(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        confidences: List[float]
    ) -> Dict[str, Any]:
        """Calculate statistics about prediction confidence."""
        
        confidences_array = np.array(confidences)
        correct_mask = np.array(true_labels) == np.array(predicted_labels)
        
        # Confidence distribution
        confidence_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        confidence_distribution = {}
        
        for i in range(len(confidence_bins) - 1):
            bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
            mask = (confidences_array >= confidence_bins[i]) & (confidences_array < confidence_bins[i+1])
            if i == len(confidence_bins) - 2:  # Last bin includes upper bound
                mask = (confidences_array >= confidence_bins[i]) & (confidences_array <= confidence_bins[i+1])
            
            count = int(np.sum(mask))
            correct_count = int(np.sum(mask & correct_mask))
            
            confidence_distribution[bin_name] = {
                'count': count,
                'correct': correct_count,
                'accuracy': float(correct_count / count) if count > 0 else 0.0
            }
        
        return {
            'min_confidence': float(np.min(confidences_array)),
            'max_confidence': float(np.max(confidences_array)),
            'mean_confidence': float(np.mean(confidences_array)),
            'median_confidence': float(np.median(confidences_array)),
            'std_confidence': float(np.std(confidences_array)),
            'distribution': confidence_distribution
        }
    
    def _save_evaluation_results(self, metrics: Dict[str, Any], dataset_name: str):
        """Save evaluation results to JSON file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluation_{dataset_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_copy = metrics.copy()
        if 'confusion_matrix' in metrics_copy and isinstance(metrics_copy['confusion_matrix'], np.ndarray):
            metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_copy, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def _generate_visualizations(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        per_class_metrics: Dict[str, Dict[str, float]],
        confidence_stats: Dict[str, Any],
        dataset_name: str
    ):
        """Generate and save visualization plots."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Confusion Matrix Heatmap
        self._plot_confusion_matrix(
            confusion_matrix,
            labels,
            f"{dataset_name}_{timestamp}_confusion_matrix.png"
        )
        
        # 2. Per-class F1 scores
        self._plot_per_class_f1(
            per_class_metrics,
            f"{dataset_name}_{timestamp}_f1_scores.png"
        )
        
        # 3. Confidence distribution
        self._plot_confidence_distribution(
            confidence_stats,
            f"{dataset_name}_{timestamp}_confidence_dist.png"
        )
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        filename: str
    ):
        """Plot and save confusion matrix heatmap."""
        
        # Limit display to top classes if too many
        max_display_classes = 20
        if len(labels) > max_display_classes:
            # Show top classes by support
            class_totals = cm.sum(axis=1)
            top_indices = np.argsort(class_totals)[-max_display_classes:]
            cm = cm[top_indices][:, top_indices]
            labels = [labels[i] for i in top_indices]
            logger.info(f"Displaying top {max_display_classes} classes in confusion matrix")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {filepath}")
    
    def _plot_per_class_f1(
        self,
        per_class_metrics: Dict[str, Dict[str, float]],
        filename: str
    ):
        """Plot per-class F1 scores."""
        
        # Sort by F1 score
        sorted_classes = sorted(
            per_class_metrics.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        # Limit to top/bottom classes if too many
        max_display = 30
        if len(sorted_classes) > max_display:
            # Show top 15 and bottom 15
            sorted_classes = sorted_classes[:15] + sorted_classes[-15:]
        
        classes = [item[0] for item in sorted_classes]
        f1_scores = [item[1]['f1_score'] for item in sorted_classes]
        
        plt.figure(figsize=(12, max(8, len(classes) * 0.3)))
        bars = plt.barh(range(len(classes)), f1_scores)
        
        # Color bars based on F1 score
        for i, bar in enumerate(bars):
            if f1_scores[i] >= 0.8:
                bar.set_color('green')
            elif f1_scores[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"F1 scores plot saved to {filepath}")
    
    def _plot_confidence_distribution(
        self,
        confidence_stats: Dict[str, Any],
        filename: str
    ):
        """Plot confidence distribution and accuracy by confidence bin."""
        
        distribution = confidence_stats['distribution']
        bins = list(distribution.keys())
        counts = [distribution[bin]['count'] for bin in bins]
        accuracies = [distribution[bin]['accuracy'] for bin in bins]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Count distribution
        ax1.bar(bins, counts, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Range')
        ax1.set_ylabel('Number of Predictions')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Accuracy by confidence
        ax2.bar(bins, accuracies, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Confidence Range')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy by Confidence Range')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=confidence_stats['mean_confidence'], color='red', linestyle='--', label='Mean Confidence')
        ax2.legend()
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence distribution plot saved to {filepath}")
    
    def compare_models(
        self,
        evaluation_results: List[Dict[str, Any]],
        model_names: List[str],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            model_names: Names of the models being compared
            save_comparison: Whether to save comparison results
            
        Returns:
            Dictionary containing comparison analysis
        """
        logger.info(f"Comparing {len(model_names)} models")
        
        if len(evaluation_results) != len(model_names):
            raise ValueError("Number of evaluation results must match number of model names")
        
        comparison = {
            'models': model_names,
            'metrics_comparison': {},
            'best_model': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Compare key metrics
        metrics_to_compare = [
            'accuracy',
            'macro_f1',
            'weighted_f1',
            'macro_precision',
            'macro_recall',
            'cohen_kappa',
            'matthews_corrcoef'
        ]
        
        for metric in metrics_to_compare:
            values = [result.get(metric, 0.0) for result in evaluation_results]
            best_idx = np.argmax(values)
            
            comparison['metrics_comparison'][metric] = {
                'values': {model_names[i]: float(values[i]) for i in range(len(model_names))},
                'best_model': model_names[best_idx],
                'best_value': float(values[best_idx]),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Determine overall best model (by weighted F1)
        weighted_f1_values = [result.get('weighted_f1', 0.0) for result in evaluation_results]
        best_model_idx = np.argmax(weighted_f1_values)
        comparison['best_model'] = {
            'name': model_names[best_model_idx],
            'weighted_f1': float(weighted_f1_values[best_model_idx]),
            'accuracy': float(evaluation_results[best_model_idx].get('accuracy', 0.0))
        }
        
        # Save comparison if requested
        if save_comparison:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_comparison_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"Model comparison saved to {filepath}")
            
            # Generate comparison visualization
            self._plot_model_comparison(comparison, f"model_comparison_{timestamp}.png")
        
        logger.info(f"Best model: {comparison['best_model']['name']} (Weighted F1: {comparison['best_model']['weighted_f1']:.4f})")
        
        return comparison
    
    def _plot_model_comparison(self, comparison: Dict[str, Any], filename: str):
        """Plot model comparison visualization."""
        
        models = comparison['models']
        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        # Prepare data
        data = []
        for metric in metrics:
            if metric in comparison['metrics_comparison']:
                values = [comparison['metrics_comparison'][metric]['values'][model] for model in models]
                data.append(values)
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics) / 2)
            ax.bar(x + offset, data[i], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {filepath}")
    
    def generate_classification_report(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        save_report: bool = True,
        dataset_name: str = "test"
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            true_labels: True labels
            predicted_labels: Predicted labels
            save_report: Whether to save report to file
            dataset_name: Name of the dataset
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            true_labels,
            predicted_labels,
            zero_division=0
        )
        
        if save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"classification_report_{dataset_name}_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(f"Classification Report - {dataset_name}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)
            
            logger.info(f"Classification report saved to {filepath}")
        
        return report
    
    def analyze_misclassifications(
        self,
        classifier: IntentClassifierInterface,
        test_data: Dataset,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze the most common misclassifications.
        
        Args:
            classifier: Trained intent classifier
            test_data: Test dataset
            top_n: Number of top misclassifications to return
            
        Returns:
            Dictionary containing misclassification analysis
        """
        logger.info("Analyzing misclassifications...")
        
        # Get predictions
        texts = []
        true_labels = []
        
        for conv in test_data.conversations:
            for turn in conv.turns:
                if turn.intent:
                    texts.append(turn.text)
                    true_labels.append(turn.intent)
        
        predictions = classifier.predict_batch(texts)
        predicted_labels = [pred.intent for pred in predictions]
        
        # Find misclassifications
        misclassifications = []
        for i, (true, pred, text, prediction) in enumerate(zip(true_labels, predicted_labels, texts, predictions)):
            if true != pred:
                misclassifications.append({
                    'index': i,
                    'text': text,
                    'true_label': true,
                    'predicted_label': pred,
                    'confidence': prediction.confidence,
                    'alternatives': prediction.alternatives[:3]
                })
        
        # Count misclassification patterns
        pattern_counts = {}
        for misc in misclassifications:
            pattern = f"{misc['true_label']} -> {misc['predicted_label']}"
            if pattern not in pattern_counts:
                pattern_counts[pattern] = {
                    'count': 0,
                    'examples': []
                }
            pattern_counts[pattern]['count'] += 1
            if len(pattern_counts[pattern]['examples']) < 3:
                pattern_counts[pattern]['examples'].append(misc['text'])
        
        # Sort by frequency
        top_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:top_n]
        
        analysis = {
            'total_misclassifications': len(misclassifications),
            'misclassification_rate': len(misclassifications) / len(texts),
            'top_patterns': [
                {
                    'pattern': pattern,
                    'count': data['count'],
                    'percentage': data['count'] / len(misclassifications) * 100,
                    'examples': data['examples']
                }
                for pattern, data in top_patterns
            ],
            'sample_misclassifications': misclassifications[:10]
        }
        
        logger.info(f"Found {len(misclassifications)} misclassifications ({analysis['misclassification_rate']:.2%})")
        
        return analysis
