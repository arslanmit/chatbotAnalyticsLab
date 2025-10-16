"""
BERT-based intent classification model with training and prediction capabilities.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models.core import (
    Dataset, IntentPrediction, TrainingConfig, TrainingResult, PerformanceMetrics
)
from src.interfaces.base import IntentClassifierInterface
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IntentClassifier(IntentClassifierInterface):
    """BERT-based intent classifier for banking domain queries."""
    
    def __init__(self, model_name: str = "bert-base-uncased", model_path: Optional[str] = None):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: Name of the pretrained model to use
            model_path: Path to a saved model (if loading existing model)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing IntentClassifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load a saved model and tokenizer."""
        logger.info(f"Loading model from {model_path}")
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load label mappings
        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                label_data = json.load(f)
                self.label2id = label_data['label2id']
                self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully with {len(self.label2id)} labels")
    
    def _prepare_labels(self, dataset: Dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create label mappings from dataset.
        
        Args:
            dataset: Dataset containing conversations with intents
            
        Returns:
            Tuple of (label2id, id2label) mappings
        """
        intents = dataset.get_intents()
        label2id = {intent: idx for idx, intent in enumerate(sorted(intents))}
        id2label = {idx: intent for intent, idx in label2id.items()}
        
        logger.info(f"Created label mappings for {len(label2id)} intents")
        return label2id, id2label
    
    def _dataset_to_hf_format(self, dataset: Dataset) -> HFDataset:
        """
        Convert Dataset to HuggingFace Dataset format.
        
        Args:
            dataset: Dataset to convert
            
        Returns:
            HuggingFace Dataset
        """
        texts = []
        labels = []
        
        for conv in dataset.conversations:
            for turn in conv.turns:
                if turn.intent:
                    texts.append(turn.text)
                    labels.append(self.label2id[turn.intent])
        
        return HFDataset.from_dict({
            'text': texts,
            'label': labels
        })
    
    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize text examples."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def _compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            pred: Predictions from the model
            
        Returns:
            Dictionary of metrics
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self, 
        train_data: Dataset, 
        val_data: Dataset, 
        config: TrainingConfig
    ) -> TrainingResult:
        """
        Train the intent classifier.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            config: Training configuration
            
        Returns:
            Training result with metrics and model path
        """
        logger.info("Starting model training")
        start_time = time.time()
        
        # Prepare labels
        self.label2id, self.id2label = self._prepare_labels(train_data)
        num_labels = len(self.label2id)
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        
        # Convert datasets to HuggingFace format
        logger.info("Converting datasets to HuggingFace format")
        train_hf = self._dataset_to_hf_format(train_data)
        val_hf = self._dataset_to_hf_format(val_data)
        
        # Tokenize datasets
        logger.info("Tokenizing datasets")
        train_tokenized = train_hf.map(self._tokenize_function, batched=True)
        val_tokenized = val_hf.map(self._tokenize_function, batched=True)
        
        # Set up training arguments
        output_dir = f"./models/intent_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            seed=config.random_seed,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            compute_metrics=self._compute_metrics
        )
        
        # Train the model
        logger.info("Training model...")
        train_result = trainer.train()
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set")
        val_metrics = trainer.evaluate()
        
        # Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        label_map_path = Path(output_dir) / "label_map.json"
        with open(label_map_path, 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Get detailed metrics
        train_predictions = trainer.predict(train_tokenized)
        train_preds = train_predictions.predictions.argmax(-1)
        train_labels = train_predictions.label_ids
        
        val_predictions = trainer.predict(val_tokenized)
        val_preds = val_predictions.predictions.argmax(-1)
        val_labels = val_predictions.label_ids
        
        # Calculate per-class metrics
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average=None, zero_division=0
        )
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0
        )
        
        # Create performance metrics
        training_metrics = PerformanceMetrics(
            accuracy=accuracy_score(train_labels, train_preds),
            precision={self.id2label[i]: float(p) for i, p in enumerate(train_precision)},
            recall={self.id2label[i]: float(r) for i, r in enumerate(train_recall)},
            f1_score={self.id2label[i]: float(f) for i, f in enumerate(train_f1)},
            confusion_matrix=confusion_matrix(train_labels, train_preds)
        )
        
        validation_metrics = PerformanceMetrics(
            accuracy=accuracy_score(val_labels, val_preds),
            precision={self.id2label[i]: float(p) for i, p in enumerate(val_precision)},
            recall={self.id2label[i]: float(r) for i, r in enumerate(val_recall)},
            f1_score={self.id2label[i]: float(f) for i, f in enumerate(val_f1)},
            confusion_matrix=confusion_matrix(val_labels, val_preds)
        )
        
        self.model_path = output_dir
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training accuracy: {training_metrics.accuracy:.4f}")
        logger.info(f"Validation accuracy: {validation_metrics.accuracy:.4f}")
        
        return TrainingResult(
            model_path=output_dir,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            training_time=training_time,
            config=config
        )
    
    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text to classify
            
        Returns:
            Intent prediction with confidence score
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get top prediction and alternatives
        probs_np = probs.cpu().numpy()[0]
        top_indices = np.argsort(probs_np)[::-1]
        
        top_intent_id = top_indices[0]
        top_confidence = float(probs_np[top_intent_id])
        top_intent = self.id2label[top_intent_id]
        
        # Get top 5 alternatives
        alternatives = [
            (self.id2label[idx], float(probs_np[idx]))
            for idx in top_indices[1:6]
        ]
        
        return IntentPrediction(
            intent=top_intent,
            confidence=top_confidence,
            alternatives=alternatives
        )
    
    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of intent predictions
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        if not texts:
            return []
        
        # Tokenize all inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Process each prediction
        probs_np = probs.cpu().numpy()
        predictions = []
        
        for i in range(len(texts)):
            prob_dist = probs_np[i]
            top_indices = np.argsort(prob_dist)[::-1]
            
            top_intent_id = top_indices[0]
            top_confidence = float(prob_dist[top_intent_id])
            top_intent = self.id2label[top_intent_id]
            
            alternatives = [
                (self.id2label[idx], float(prob_dist[idx]))
                for idx in top_indices[1:6]
            ]
            
            predictions.append(IntentPrediction(
                intent=top_intent,
                confidence=top_confidence,
                alternatives=alternatives
            ))
        
        return predictions
    
    def evaluate(self, test_data: Dataset) -> PerformanceMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Performance metrics
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        logger.info("Evaluating model on test data")
        
        # Convert to HuggingFace format
        test_hf = self._dataset_to_hf_format(test_data)
        test_tokenized = test_hf.map(self._tokenize_function, batched=True)
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            compute_metrics=self._compute_metrics
        )
        
        # Get predictions
        predictions = trainer.predict(test_tokenized)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            accuracy=accuracy_score(labels, preds),
            precision={self.id2label[i]: float(p) for i, p in enumerate(precision)},
            recall={self.id2label[i]: float(r) for i, r in enumerate(recall)},
            f1_score={self.id2label[i]: float(f) for i, f in enumerate(f1)},
            confusion_matrix=confusion_matrix(labels, preds)
        )
        
        logger.info(f"Test accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Test macro F1: {metrics.macro_f1:.4f}")
        
        return metrics
