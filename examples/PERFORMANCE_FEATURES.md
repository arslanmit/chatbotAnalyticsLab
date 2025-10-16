# Performance Features Overview

## Quick Reference Guide

### 🚀 Batch Processing

Process multiple queries efficiently:

```python
# Initialize with batch settings
classifier = IntentClassifier(
    model_path="./models/model",
    batch_size=32
)

# Batch prediction
queries = ["Query 1", "Query 2", ...]
predictions = classifier.predict_batch(
    queries,
    batch_size=32,
    show_progress=True
)
```

**Performance:** 300-500+ predictions/second on GPU

---

### 💾 Prediction Caching

Cache repeated queries for instant responses:

```python
# Enable caching (default)
classifier = IntentClassifier(
    enable_cache=True,
    cache_size=1000
)

# First call: ~10-50ms
pred1 = classifier.predict("Check balance", use_cache=True)

# Second call: <1ms (cached!)
pred2 = classifier.predict("Check balance", use_cache=True)

# Check cache stats
stats = classifier.get_cache_stats()
```

**Performance:** 50-100x speedup for repeated queries

---

### 🎮 GPU Acceleration

Automatic GPU detection and optimization:

```python
# GPU automatically detected
classifier = IntentClassifier(model_path="./models/model")

# Check GPU status
gpu_stats = classifier.get_gpu_memory_stats()
print(f"GPU: {gpu_stats['gpu_available']}")
print(f"Device: {gpu_stats.get('device_name', 'CPU')}")
print(f"Free Memory: {gpu_stats.get('free_memory_gb', 0):.2f} GB")
```

**Performance:** 5-10x faster training, 3-5x faster inference

---

### 🔥 Model Warm-up

Pre-optimize model before production:

```python
# Warm up with default samples
classifier.warm_up()

# Or with custom samples
classifier.warm_up(sample_texts=[
    "Sample query 1",
    "Sample query 2"
])
```

**Benefit:** Reduces first-prediction latency

---

### 🌊 Streaming Processing

Memory-efficient processing for large datasets:

```python
# Process unlimited queries
large_dataset = [...]  # Millions of queries

for batch_idx, batch_preds in classifier.predict_batch_streaming(
    large_dataset,
    batch_size=32
):
    # Process each batch
    save_results(batch_preds)
```

**Benefit:** Constant memory usage, unlimited dataset size

---

### ⚡ Inference Optimization

Optimize model for production:

```python
# Apply all inference optimizations
classifier.optimize_for_inference()
```

**Optimizations:**

- Disable gradient computation
- Enable eval mode
- TF32 acceleration (on compatible GPUs)
- Memory optimization

---

### 📊 Performance Benchmarking

Built-in benchmarking tools:

```python
# Run comprehensive benchmark
results = classifier.benchmark_performance(
    sample_texts=["Query 1", "Query 2"],
    num_iterations=100
)

# View results
for name, data in results['benchmarks'].items():
    print(f"{name}: {data['throughput']:.1f} pred/sec")
```

**Metrics:**

- Throughput (predictions/second)
- Latency (time per prediction)
- GPU memory usage

---

## Performance Comparison Table

| Feature | Without Optimization | With Optimization | Improvement |
|---------|---------------------|-------------------|-------------|
| Single Prediction | 10-50 pred/sec | 10-50 pred/sec | Baseline |
| Batch Processing | N/A | 300-500 pred/sec | 6-10x |
| Cached Queries | 10-50ms | <1ms | 50-100x |
| GPU Training | 1x (CPU) | 5-10x (GPU) | 5-10x |
| GPU Inference | 1x (CPU) | 3-5x (GPU) | 3-5x |

---

## Usage Patterns

### Real-time Applications

```python
# Small batches, caching enabled
classifier = IntentClassifier(
    batch_size=8,
    enable_cache=True,
    cache_size=5000
)
classifier.warm_up()
classifier.optimize_for_inference()
```

### Batch Processing Jobs

```python
# Large batches, no caching needed
classifier = IntentClassifier(
    batch_size=64,
    enable_cache=False
)
classifier.optimize_for_inference()

# Process in batches
predictions = classifier.predict_batch(
    queries,
    batch_size=64,
    show_progress=True
)
```

### Large Dataset Processing

```python
# Streaming mode
classifier = IntentClassifier(batch_size=32)
classifier.optimize_for_inference()

# Stream through dataset
for batch_idx, preds in classifier.predict_batch_streaming(
    large_dataset,
    batch_size=32
):
    process_and_save(preds)
```

---

## Monitoring

### Cache Statistics

```python
stats = classifier.get_cache_stats()
print(f"Cache: {stats['cache_size']}/{stats['cache_limit']}")
print(f"Usage: {stats['cache_usage_percent']:.1f}%")
```

### GPU Memory

```python
gpu_stats = classifier.get_gpu_memory_stats()
if gpu_stats['gpu_available']:
    print(f"Allocated: {gpu_stats['allocated_memory_gb']:.2f} GB")
    print(f"Free: {gpu_stats['free_memory_gb']:.2f} GB")
```

---

## Best Practices

1. ✅ **Always warm up** before production use
2. ✅ **Use batch processing** for multiple queries
3. ✅ **Enable caching** for repeated queries
4. ✅ **Monitor GPU memory** to avoid OOM
5. ✅ **Use streaming** for very large datasets
6. ✅ **Optimize batch size** based on hardware
7. ✅ **Apply inference optimizations** for production

---

## Complete Example

```python
from src.models.intent_classifier import IntentClassifier

# 1. Initialize with optimizations
classifier = IntentClassifier(
    model_path="./models/production_model",
    batch_size=32,
    enable_cache=True,
    cache_size=5000
)

# 2. Warm up
classifier.warm_up()

# 3. Optimize for inference
classifier.optimize_for_inference()

# 4. Check GPU
gpu_stats = classifier.get_gpu_memory_stats()
print(f"GPU: {gpu_stats['gpu_available']}")

# 5. Process queries
queries = ["Query 1", "Query 2", ...]
predictions = classifier.predict_batch(
    queries,
    batch_size=32,
    use_cache=True,
    show_progress=True
)

# 6. Monitor performance
cache_stats = classifier.get_cache_stats()
print(f"Cache hits: {cache_stats['cache_size']}")

# 7. Benchmark
results = classifier.benchmark_performance()
print(f"Throughput: {results['benchmarks']['batch_size_32']['throughput']:.1f} pred/sec")
```

---

## See Also

- **[Batch Optimization Guide](BATCH_OPTIMIZATION_GUIDE.md)** - Detailed documentation
- **[test_batch_optimization.py](test_batch_optimization.py)** - Complete demo script
- **[README.md](README.md)** - Examples overview
