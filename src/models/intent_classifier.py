"""
BERT-based intent classification model with training and prediction capabilities.
Includes batch processing, GPU acceleration, and model caching optimizations.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import hashlib
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
    """
    BERT-based intent classifier for banking domain queries.
    
    Features:
    - Batch processing for efficient inference
    - GPU acceleration support
    - Model caching and warm-up mechanisms
    - Optimized memory usage
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        model_path: Optional[str] = None,
        batch_size: int = 32,
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: Name of the pretrained model to use
            model_path: Path to a saved model (if loading existing model)
            batch_size: Default batch size for batch predictions
            enable_cache: Whether to enable prediction caching
            cache_size: Maximum number of cached predictions
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._prediction_cache: Dict[str, IntentPrediction] = {}
        self._is_warmed_up = False
        
        logger.info(f"Initializing IntentClassifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Cache enabled: {enable_cache} (size: {cache_size})")
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load a saved model and tokenizer with GPU optimization."""
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
        
        # Move model to device and optimize
        self.model.to(self.device)
        self.model.eval()
        
        # Enable GPU optimizations if available
        if torch.cuda.is_available():
            # Enable cuDNN autotuner for better performance
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN autotuner for GPU optimization")
        
        logger.info(f"Model loaded successfully with {len(self.label2id)} labels")
        
        # Clear cache
        self._prediction_cache.clear()
        self._is_warmed_up = False
    
    def warm_up(self, sample_texts: Optional[List[str]] = None):
        """
        Warm up the model by running sample predictions.
        This helps optimize GPU memory allocation and compilation.
        
        Args:
            sample_texts: Optional list of sample texts. If None, uses default samples.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        if self._is_warmed_up:
            logger.info("Model already warmed up")
            return
        
        logger.info("Warming up model...")
        start_time = time.time()
        
        # Use provided samples or create default ones
        if sample_texts is None:
            sample_texts = [
                "I want to check my account balance",
                "How do I transfer money?",
                "What are the interest rates?",
                "I need to report a lost card"
            ]
        
        # Run warm-up predictions
        with torch.no_grad():
            for text in sample_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_warmed_up = True
        warm_up_time = time.time() - start_time
        logger.info(f"Model warm-up completed in {warm_up_time:.2f} seconds")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text input."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[IntentPrediction]:
        """Retrieve prediction from cache if available."""
        if not self.enable_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        return self._prediction_cache.get(cache_key)
    
    def _add_to_cache(self, text: str, prediction: IntentPrediction):
        """Add prediction to cache with size limit."""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(text)
        
        # Implement simple LRU by removing oldest entries
        if len(self._prediction_cache) >= self.cache_size:
            # Remove first (oldest) entry
            first_key = next(iter(self._prediction_cache))
            del self._prediction_cache[first_key]
        
        self._prediction_cache[cache_key] = prediction
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._prediction_cache),
            'cache_limit': self.cache_size,
            'cache_usage_percent': (len(self._prediction_cache) / self.cache_size * 100) if self.cache_size > 0 else 0
        }
    
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
        
        # Set up training arguments with GPU optimization
        output_dir = f"./models/intent_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine if we can use GPU acceleration
        use_gpu = torch.cuda.is_available()
        fp16 = use_gpu  # Enable mixed precision training on GPU
        
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
            report_to="none",
            fp16=fp16,  # Enable mixed precision training for faster training on GPU
            dataloader_num_workers=4 if use_gpu else 0,  # Parallel data loading
            gradient_accumulation_steps=1,
            warmup_steps=100,  # Learning rate warmup
        )
        
        if use_gpu:
            logger.info("GPU acceleration enabled for training")
            logger.info(f"Mixed precision (FP16) training: {fp16}")
        
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
    
    def predict(self, text: str, use_cache: bool = True) -> IntentPrediction:
        """
        Predict intent for a single text input with caching support.
        
        Args:
            text: Input text to classify
            use_cache: Whether to use cached predictions
            
        Returns:
            Intent prediction with confidence score
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        # Check cache first
        if use_cache:
            cached_prediction = self._get_from_cache(text)
            if cached_prediction is not None:
                return cached_prediction
        
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
        
        prediction = IntentPrediction(
            intent=top_intent,
            confidence=top_confidence,
            alternatives=alternatives
        )
        
        # Add to cache
        if use_cache:
            self._add_to_cache(text, prediction)
        
        return prediction
    
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = False
    ) -> List[IntentPrediction]:
        """
        Predict intents for multiple text inputs with optimized batch processing.
        
        Args:
            texts: List of input texts to classify
            batch_size: Batch size for processing (uses default if None)
            use_cache: Whether to use cached predictions
            show_progress: Whether to log progress information
            
        Returns:
            List of intent predictions
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        predictions = []
        
        # Check cache for all texts first
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        cached_predictions: List[Optional[IntentPrediction]] = [None] * len(texts)
        
        if use_cache:
            for i, text in enumerate(texts):
                cached_pred = self._get_from_cache(text)
                if cached_pred is not None:
                    cached_predictions[i] = cached_pred
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        if show_progress:
            logger.info(f"Processing {len(texts)} texts: {len(cached_predictions) - len(uncached_texts)} from cache, {len(uncached_texts)} to predict")
        
        # Process uncached texts in batches
        if uncached_texts:
            start_time = time.time()
            
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
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
                
                # Process each prediction in batch
                probs_np = probs.cpu().numpy()
                
                for i in range(len(batch_texts)):
                    prob_dist = probs_np[i]
                    top_indices = np.argsort(prob_dist)[::-1]
                    
                    top_intent_id = top_indices[0]
                    top_confidence = float(prob_dist[top_intent_id])
                    top_intent = self.id2label[top_intent_id]
                    
                    alternatives = [
                        (self.id2label[idx], float(prob_dist[idx]))
                        for idx in top_indices[1:6]
                    ]
                    
                    prediction = IntentPrediction(
                        intent=top_intent,
                        confidence=top_confidence,
                        alternatives=alternatives
                    )
                    
                    # Store prediction
                    original_idx = uncached_indices[batch_start + i]
                    cached_predictions[original_idx] = prediction
                    
                    # Add to cache
                    if use_cache:
                        self._add_to_cache(batch_texts[i], prediction)
                
                if show_progress and len(uncached_texts) > batch_size:
                    logger.info(f"Processed batch {batch_start // batch_size + 1}/{(len(uncached_texts) + batch_size - 1) // batch_size}")
            
            # Clear GPU cache after batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            if show_progress:
                throughput = len(uncached_texts) / processing_time if processing_time > 0 else 0
                logger.info(f"Batch processing completed: {len(uncached_texts)} predictions in {processing_time:.2f}s ({throughput:.1f} predictions/sec)")
        
        # Return predictions, ensuring all are non-None
        result: List[IntentPrediction] = []
        for pred in cached_predictions:
            if pred is not None:
                result.append(pred)
        return result
    
    def predict_batch_streaming(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ):
        """
        Generator that yields predictions in batches for memory-efficient processing.
        
        Args:
            texts: List of input texts to classify
            batch_size: Batch size for processing (uses default if None)
            
        Yields:
            Tuples of (batch_start_index, batch_predictions)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        if not texts:
            return
        
        batch_size = batch_size or self.batch_size
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
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
            
            # Process predictions
            probs_np = probs.cpu().numpy()
            batch_predictions = []
            
            for i in range(len(batch_texts)):
                prob_dist = probs_np[i]
                top_indices = np.argsort(prob_dist)[::-1]
                
                top_intent_id = top_indices[0]
                top_confidence = float(prob_dist[top_intent_id])
                top_intent = self.id2label[top_intent_id]
                
                alternatives = [
                    (self.id2label[idx], float(prob_dist[idx]))
                    for idx in top_indices[1:6]
                ]
                
                batch_predictions.append(IntentPrediction(
                    intent=top_intent,
                    confidence=top_confidence,
                    alternatives=alternatives
                ))
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield batch_start, batch_predictions
    
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
    
    def get_gpu_memory_stats(self) -> Dict[str, Any]:
        """
        Get GPU memory usage statistics.
        
        Returns:
            Dictionary with GPU memory information
        """
        if not torch.cuda.is_available():
            return {
                'gpu_available': False,
                'message': 'CUDA not available'
            }
        
        return {
            'gpu_available': True,
            'device_name': torch.cuda.get_device_name(0),
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'allocated_memory_gb': torch.cuda.memory_allocated(0) / 1e9,
            'cached_memory_gb': torch.cuda.memory_reserved(0) / 1e9,
            'free_memory_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        }
    
    def optimize_for_inference(self):
        """
        Optimize model for inference by applying various optimizations.
        This includes setting eval mode, disabling gradients, and other optimizations.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        logger.info("Optimizing model for inference...")
        
        # Set to eval mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable inference mode optimizations
        if torch.cuda.is_available():
            # Use TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for faster inference on compatible GPUs")
        
        logger.info("Model optimized for inference")
    
    def benchmark_performance(
        self, 
        sample_texts: Optional[List[str]] = None,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark model performance with various batch sizes.
        
        Args:
            sample_texts: Sample texts to use for benchmarking
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        if sample_texts is None:
            sample_texts = [
                "I want to check my account balance",
                "How do I transfer money to another account?",
                "What are the current interest rates?",
                "I need to report a lost credit card",
                "Can you help me set up online banking?"
            ]
        
        logger.info(f"Running performance benchmark with {num_iterations} iterations...")
        
        results = {
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'num_iterations': num_iterations,
            'sample_size': len(sample_texts),
            'benchmarks': {}
        }
        
        # Benchmark single prediction
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.predict(sample_texts[0], use_cache=False)
        single_time = time.time() - start_time
        results['benchmarks']['single_prediction'] = {
            'total_time': single_time,
            'avg_time': single_time / num_iterations,
            'throughput': num_iterations / single_time
        }
        
        # Benchmark batch prediction with different sizes
        for batch_size in [1, 4, 8, 16, 32]:
            if batch_size > len(sample_texts):
                # Replicate samples to reach batch size
                batch_texts = sample_texts * (batch_size // len(sample_texts) + 1)
                batch_texts = batch_texts[:batch_size]
            else:
                batch_texts = sample_texts[:batch_size]
            
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.predict_batch(batch_texts, batch_size=batch_size, use_cache=False, show_progress=False)
            batch_time = time.time() - start_time
            
            results['benchmarks'][f'batch_size_{batch_size}'] = {
                'total_time': batch_time,
                'avg_time': batch_time / num_iterations,
                'throughput': (num_iterations * batch_size) / batch_time,
                'time_per_sample': batch_time / (num_iterations * batch_size)
            }
        
        # Add GPU memory stats if available
        if torch.cuda.is_available():
            results['gpu_memory'] = self.get_gpu_memory_stats()
        
        logger.info("Benchmark completed")
        return results
