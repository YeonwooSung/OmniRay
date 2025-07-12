"""
OmniRay Framework - High-Performance Ray-based ML/DL Data Processing Framework

OmniRay is an optimized framework for efficient data loading, processing, and management
in machine learning and deep learning pipelines using Ray Data. It provides:

- Automatic loader selection based on data format
- Pinterest-inspired performance optimizations
- Intelligent caching and memory management
- Real-time performance monitoring
- GPU optimization for ML workloads
- Distributed processing capabilities

Key Components:
- BaseDataLoader: Foundation class with core optimizations
- Specialized loaders: HuggingFace, NumPy, Pandas, PyTorch
- OmniRayManager: Centralized framework management
- PerformanceMonitor: Real-time monitoring and benchmarking

Usage:
    from omniray.dataloader.manager import OmniRayManager, PerformanceMonitor

    # Quick start
    with OmniRayManager() as manager:
        loader = manager.auto_detect_loader("data.csv")
        loader.load("data.csv")

    # With monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    # ... your data processing code ...
    recommendations = monitor.generate_optimization_recommendations()
"""

# Base components
from .base import BaseDataLoader

# Specialized loaders
from .huggingface_loader import HuggingfaceDataLoader
from .numpy_loader import NumpyDataLoader
from .pandas_loader import PandasDataLoader
from .torch_loader import TorchDataLoader
from .kafka_streaming_loader import KafkaStreamingDataLoader

# Framework management
from .manager import OmniRayManager


__all__ = [
    "BaseDataLoader",
    "HuggingfaceDataLoader",
    "NumpyDataLoader",
    "PandasDataLoader",
    "TorchDataLoader",
    "KafkaStreamingDataLoader",
    "OmniRayManager"
]