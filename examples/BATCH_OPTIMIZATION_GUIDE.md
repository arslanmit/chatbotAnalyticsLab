# Batch Processing and Performance Optimization Guide

This guide explains the batch processing and performance optimization features added to the Intent Classifier.

## Features Overview

### 1. Batch Processing

The classifier now supports efficient batch prediction for processing multiple queries simultaneously:

```python
from src.models.intent_classifier import IntentClassifier

# Initialize with batch size
classifier = IntentClassifier(
    model_path="./models/intent_classifier_latest",
    batch_size=32  # Default batch size for predictions
)

# Batch prediction
queries = [
    "Check my balance",
    "Transfer money",
    "Apply for loan"
]

predictions = classifier.predict_batch(
    queries,
    batch_size=32,
    use_cache=True,
    show_progress=True
)
```

**Benefits:**

- Process multiple queries in parallel
- Reduced overhead compared to individual predictions
- Automatic batching and memory management

### 2. GPU Acceleration

Automatic GPU detection and optimization:

```python
# GPU is automatically detected and used if available
classifier = IntentClassifier(model_path="./models/model")

# Check GPU status
gpu_stats = classifier.get_gpu_memory_stats()
print(f"GPU Available: {gpu_stats['gpu_available']}")
print(f"Device: {gpu_stats.get('device_name', 'CPU')}")
```

**GPU Optimizations:**

- Automatic CUDA device selection
- Mixed precision (FP16) training for faster training
- cuDNN autotuner for optimal performance
- TF32 support for Ampere GPUs
- Efficient memory management

### 3. Model Caching

Prediction caching for frequently queried texts:

```python
# Enable caching (enabled by default)
classifier = IntentClassifier(
    model_path="./models/model",
    enable_cache=True,
    cache_size=1000  # Maximum cached predictions
)

# First prediction (not cached)
pred1 = classifier.predict("Check balance", use_cache=True)

# Second prediction (cached - much faster!)
pred2 = classifier.predict("Check balance", use_cache=True)

# Check cache statistics
stats = classifier.get_cache_stats()
print(f"Cache usage: {stats['cache_size']}/{stats['cache_limit']}")
```

**Benefits:**

- Instant responses for repeated queries
- Configurable cache size
- Automatic LRU eviction
- Cache statistics monitoring

### 4. Model Warm-up

Pre-load and optimize model before production use:

```python
# Warm up with sample queries
classifier.warm_up(sample_texts=[
    "Sample query 1",
    "Sample query 2"
])

# Or use default warm-up samples
classifier.warm_up()
```

**Benefits:**

- Optimizes GPU memory allocation
- Pre-compiles CUDA kernels
- Reduces first-prediction latency

### 5. Streaming Batch Processing

Memory-efficient processing for large datasets:

```python
# Process large dataset in streaming mode
large_query_list = [...]  # Thousands of queries

for batch_idx, batch_predictions in classifier.predict_batch_streaming(
    large_query_list,
    batch_size=32
):
    # Process each batch as it completes
    print(f"Batch {batch_idx}: {len(batch_predictions)} predictions")
    # Save or process predictions immediately
```

**Benefits:**

- Constant memory usage
- Process unlimited queries
- Real-time processing feedback

### 6. Inference Optimization

Optimize model specifically for inference:

```python
# Apply inference optimizations
classifier.optimize_for_inference()
```

**Optimizations Applied:**

- Disable gradient computation
- Enable eval mode
- TF32 acceleration (on compatible GPUs)
- Memory optimization

### 7. Performance Benchmarking

Built-in benchmarking tools:

```python
# Run comprehensive benchmark
results = classifier.benchmark_performance(
    sample_texts=["Query 1", "Query 2"],
    num_iterations=100
)

print(f"Single prediction: {results['benchmarks']['single_prediction']['throughput']:.1f} pred/sec")
print(f"Batch size 32: {results['benchmarks']['batch_size_32']['throughput']:.1f} pred/sec")
```

## Performance Comparison

### Single vs Batch Prediction

| Method | Throughput | Use Case |
|--------|-----------|----------|
| Single prediction | ~10-50 pred/sec | Real-time user queries |
| Batch (size 8) | ~100-200 pred/sec | Small batches |
| Batch (size 32) | ~300-500 pred/sec | Large batch processing |
| Streaming | ~300-500 pred/sec | Unlimited dataset size |

*Note: Actual performance depends on hardware and model size*

### GPU vs CPU

| Device | Training Speed | Inference Speed |
|--------|---------------|-----------------|
| CPU | 1x (baseline) | 1x (baseline) |
| GPU (CUDA) | 5-10x faster | 3-5x faster |
| GPU (FP16) | 10-20x faster | 5-8x faster |

### Caching Impact

| Scenario | First Call | Cached Call | Speedup |
|----------|-----------|-------------|---------|
| Single prediction | 10-50ms | <1ms | 10-50x |
| Repeated queries | Normal | Instant | 50-100x |

## Best Practices

### 1. Batch Size Selection

```python
# For real-time applications
classifier = IntentClassifier(batch_size=8)

# For batch processing
classifier = IntentClassifier(batch_size=32)

# For large-scale processing
classifier = IntentClassifier(batch_size=64)
```

**Guidelines:**

- Smaller batches (4-8): Lower latency, real-time apps
- Medium batches (16-32): Balanced throughput/latency
- Large batches (32-64): Maximum throughput, batch jobs

### 2. Memory Management

```python
# Monitor GPU memory
stats = classifier.get_gpu_memory_stats()
if stats['free_memory_gb'] < 1.0:
    # Reduce batch size or clear cache
    classifier.clear_cache()
```

### 3. Production Deployment

```python
# Initialize for production
classifier = IntentClassifier(
    model_path="./models/production_model",
    batch_size=32,
    enable_cache=True,
    cache_size=5000
)

# Warm up before serving
classifier.warm_up()

# Optimize for inference
classifier.optimize_for_inference()

# Ready to serve predictions!
```

### 4. Large Dataset Processing

```python
# Use streaming for large datasets
def process_large_dataset(queries):
    results = []
    for batch_idx, batch_preds in classifier.predict_batch_streaming(
        queries,
        batch_size=32
    ):
        # Process and save each batch
        results.extend(batch_preds)
        
        # Optional: Save intermediate results
        if batch_idx % 100 == 0:
            save_checkpoint(results)
    
    return results
```

## Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size
classifier.batch_size = 16

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Disable caching if needed
classifier.enable_cache = False
```

### Slow Performance

```python
# Check if GPU is being used
stats = classifier.get_gpu_memory_stats()
if not stats['gpu_available']:
    print("Warning: Running on CPU, consider using GPU")

# Warm up the model
classifier.warm_up()

# Optimize for inference
classifier.optimize_for_inference()

# Use larger batch sizes
predictions = classifier.predict_batch(queries, batch_size=32)
```

### Cache Not Working

```python
# Verify cache is enabled
stats = classifier.get_cache_stats()
print(f"Cache enabled: {stats['cache_enabled']}")

# Clear and rebuild cache
classifier.clear_cache()

# Ensure use_cache=True in predictions
pred = classifier.predict(text, use_cache=True)
```

## Example Scripts

See the following example scripts for complete demonstrations:

- `examples/test_batch_optimization.py` - Comprehensive demo of all features
- `examples/train_intent_classifier.py` - Training with GPU acceleration
- `examples/verify_intent_classifier.py` - Model verification and testing

## API Reference

### IntentClassifier Constructor

```python
IntentClassifier(
    model_name: str = "bert-base-uncased",
    model_path: Optional[str] = None,
    batch_size: int = 32,
    enable_cache: bool = True,
    cache_size: int = 1000
)
```

### Key Methods

- `predict(text, use_cache=True)` - Single prediction with caching
- `predict_batch(texts, batch_size, use_cache, show_progress)` - Batch prediction
- `predict_batch_streaming(texts, batch_size)` - Streaming batch prediction
- `warm_up(sample_texts)` - Model warm-up
- `optimize_for_inference()` - Apply inference optimizations
- `benchmark_performance(sample_texts, num_iterations)` - Performance benchmarking
- `get_gpu_memory_stats()` - GPU memory information
- `get_cache_stats()` - Cache statistics
- `clear_cache()` - Clear prediction cache

## Performance Tips

1. **Always warm up** the model before production use
2. **Use batch prediction** for multiple queries
3. **Enable caching** for repeated queries
4. **Monitor GPU memory** to avoid OOM errors
5. **Use streaming** for very large datasets
6. **Optimize batch size** based on your hardware
7. **Apply inference optimizations** for production deployment

## Requirements

- PyTorch with CUDA support (for GPU acceleration)
- transformers library
- CUDA-compatible GPU (optional but recommended)
- Sufficient GPU memory (4GB+ recommended)
