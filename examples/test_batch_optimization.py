"""
Example script demonstrating batch processing and performance optimization features.

This script shows:
1. Model warm-up for optimal performance
2. Batch prediction with different batch sizes
3. Caching mechanisms
4. GPU acceleration (if available)
5. Performance benchmarking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.intent_classifier import IntentClassifier
from src.repositories.dataset_loaders import BANKING77Loader
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Demonstrate batch processing and optimization features."""
    
    print("=" * 80)
    print("Intent Classifier - Batch Processing & Optimization Demo")
    print("=" * 80)
    
    # Load a trained model (you need to have trained one first)
    model_path = "./models/intent_classifier_latest"
    
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first using train_intent_classifier.py")
        return
    
    # Initialize classifier with optimization settings
    print("\n1. Initializing classifier with optimization settings...")
    classifier = IntentClassifier(
        model_path=model_path,
        batch_size=32,
        enable_cache=True,
        cache_size=1000
    )
    
    # Display GPU information
    print("\n2. GPU Information:")
    gpu_stats = classifier.get_gpu_memory_stats()
    if gpu_stats['gpu_available']:
        print(f"   Device: {gpu_stats['device_name']}")
        print(f"   Total Memory: {gpu_stats['total_memory_gb']:.2f} GB")
        print(f"   Free Memory: {gpu_stats['free_memory_gb']:.2f} GB")
    else:
        print("   GPU not available, using CPU")
    
    # Warm up the model
    print("\n3. Warming up model...")
    classifier.warm_up()
    
    # Optimize for inference
    print("\n4. Optimizing model for inference...")
    classifier.optimize_for_inference()
    
    # Test sample queries
    sample_queries = [
        "I want to check my account balance",
        "How do I transfer money to another account?",
        "What are the current interest rates for savings accounts?",
        "I need to report a lost credit card immediately",
        "Can you help me set up online banking?",
        "What is the minimum balance requirement?",
        "How do I apply for a personal loan?",
        "I want to dispute a transaction on my statement",
        "What are your business hours?",
        "How can I update my contact information?"
    ]
    
    # Single prediction with caching
    print("\n5. Testing single prediction with caching...")
    print(f"   Query: '{sample_queries[0]}'")
    
    # First prediction (not cached)
    import time
    start = time.time()
    pred1 = classifier.predict(sample_queries[0], use_cache=True)
    time1 = time.time() - start
    print(f"   First call (uncached): {time1*1000:.2f}ms")
    print(f"   Intent: {pred1.intent} (confidence: {pred1.confidence:.4f})")
    
    # Second prediction (cached)
    start = time.time()
    pred2 = classifier.predict(sample_queries[0], use_cache=True)
    time2 = time.time() - start
    print(f"   Second call (cached): {time2*1000:.2f}ms")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    
    # Cache statistics
    cache_stats = classifier.get_cache_stats()
    print(f"\n   Cache stats: {cache_stats['cache_size']}/{cache_stats['cache_limit']} entries")
    
    # Batch prediction
    print("\n6. Testing batch prediction...")
    print(f"   Processing {len(sample_queries)} queries in batch...")
    
    start = time.time()
    batch_predictions = classifier.predict_batch(
        sample_queries,
        batch_size=32,
        use_cache=False,
        show_progress=True
    )
    batch_time = time.time() - start
    
    print(f"\n   Batch processing completed in {batch_time:.2f}s")
    print(f"   Throughput: {len(sample_queries)/batch_time:.1f} predictions/sec")
    
    print("\n   Sample predictions:")
    for i, (query, pred) in enumerate(zip(sample_queries[:3], batch_predictions[:3])):
        print(f"   {i+1}. '{query[:50]}...'")
        print(f"      → {pred.intent} ({pred.confidence:.4f})")
    
    # Streaming batch prediction
    print("\n7. Testing streaming batch prediction...")
    print(f"   Processing {len(sample_queries) * 5} queries in streaming mode...")
    
    large_query_list = sample_queries * 5
    total_processed = 0
    
    start = time.time()
    for batch_idx, batch_preds in classifier.predict_batch_streaming(
        large_query_list,
        batch_size=16
    ):
        total_processed += len(batch_preds)
        print(f"   Processed batch starting at index {batch_idx}: {len(batch_preds)} predictions")
    
    stream_time = time.time() - start
    print(f"\n   Streaming completed: {total_processed} predictions in {stream_time:.2f}s")
    print(f"   Throughput: {total_processed/stream_time:.1f} predictions/sec")
    
    # Performance benchmark
    print("\n8. Running performance benchmark...")
    print("   This may take a minute...")
    
    benchmark_results = classifier.benchmark_performance(
        sample_texts=sample_queries[:5],
        num_iterations=50
    )
    
    print(f"\n   Device: {benchmark_results['device']}")
    print(f"   GPU Available: {benchmark_results['gpu_available']}")
    print("\n   Benchmark Results:")
    
    for bench_name, bench_data in benchmark_results['benchmarks'].items():
        if 'throughput' in bench_data:
            print(f"   {bench_name}:")
            print(f"      Throughput: {bench_data['throughput']:.1f} predictions/sec")
            print(f"      Avg time: {bench_data['avg_time']*1000:.2f}ms")
    
    # GPU memory after operations
    if gpu_stats['gpu_available']:
        print("\n9. Final GPU memory usage:")
        final_gpu_stats = classifier.get_gpu_memory_stats()
        print(f"   Allocated: {final_gpu_stats['allocated_memory_gb']:.2f} GB")
        print(f"   Cached: {final_gpu_stats['cached_memory_gb']:.2f} GB")
        print(f"   Free: {final_gpu_stats['free_memory_gb']:.2f} GB")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
