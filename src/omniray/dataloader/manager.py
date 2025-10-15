import ray
from ray.data.context import DataContext
from typing import Dict, Any, Optional, List
import logging
import time
from pathlib import Path
import psutil
import json

from .base import BaseDataLoader
from .huggingface_loader import HuggingfaceDataLoader
from .numpy_loader import NumpyDataLoader
from .pandas_loader import PandasDataLoader
from .torch_loader import TorchDataLoader


class OmniRayManager:
    """
    Centralized manager for OmniRay framework with automatic loader selection,
    performance monitoring, and resource optimization.
    """
    
    def __init__(
        self,
        ray_config: Optional[Dict] = None,
        enable_monitoring: bool = True,
        auto_optimize: bool = True,
        cache_dir: str = "/tmp/omniray_cache"
    ):
        """
        Initialize OmniRay framework manager.
        
        Args:
            ray_config: Ray initialization configuration
            enable_monitoring: Enable performance monitoring
            auto_optimize: Enable automatic optimizations
            cache_dir: Global cache directory
        """
        self.enable_monitoring = enable_monitoring
        self.auto_optimize = auto_optimize
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('OmniRayManager')
        
        # Initialize Ray
        self._initialize_ray(ray_config)
        
        # Register loaders
        self.loaders = {
            'huggingface': HuggingfaceDataLoader,
            'numpy': NumpyDataLoader,
            'pandas': PandasDataLoader,
            'torch': TorchDataLoader,
            'base': BaseDataLoader
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.active_loaders = {}
        
        # System information
        self.system_info = self._collect_system_info()
        
        if self.auto_optimize:
            self._apply_global_optimizations()
            
        self.logger.info("OmniRay Manager initialized successfully")

    def _initialize_ray(self, ray_config: Optional[Dict]):
        """Initialize Ray with optimal configuration."""
        try:
            if ray.is_initialized():
                self.logger.info("Ray already initialized")
                return
            
            # Default configuration optimized for data processing
            default_config = {
                'ignore_reinit_error': True,
                'log_to_driver': False,
                'include_dashboard': True,
                'dashboard_host': '0.0.0.0',
                'object_store_memory': self._calculate_object_store_memory(),
            }
            
            # Merge with user config
            final_config = {**default_config, **(ray_config or {})}
            
            self.logger.info(f"Initializing Ray with config: {final_config}")
            ray.init(**final_config)
            
            # Log cluster information
            self._log_cluster_info()
            
        except Exception as e:
            self.logger.error(f"Ray initialization failed: {e}")
            raise

    def _calculate_object_store_memory(self) -> int:
        """Calculate optimal object store memory size."""
        try:
            total_memory = psutil.virtual_memory().total
            # Use 30% of total memory for object store
            object_store_memory = int(total_memory * 0.3)
            self.logger.info(f"Setting object store memory to {object_store_memory/(1024**3):.1f}GB")
            return object_store_memory
        except Exception as e:
            self.logger.warning(f"Failed to calculate object store memory: {e}")
            # Fallback to Ray's default
            return None

    def _log_cluster_info(self):
        """Log Ray cluster information."""
        try:
            cluster_resources = ray.cluster_resources()
            self.logger.info(f"Ray cluster resources: {cluster_resources}")
            
            nodes = ray.nodes()
            self.logger.info(f"Ray cluster nodes: {len(nodes)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get cluster info: {e}")

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        try:
            import platform
            import torch
            
            info = {
                'platform': {
                    'system': platform.system(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'python_version': platform.python_version()
                },
                'memory': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3),
                    'percent_used': psutil.virtual_memory().percent
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'count_logical': psutil.cpu_count(logical=True),
                    'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3)
                }
            }
            
            # GPU information
            if torch.cuda.is_available():
                info['gpu'] = {
                    'count': torch.cuda.device_count(),
                    'devices': [
                        {
                            'name': torch.cuda.get_device_name(i),
                            'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        }
                        for i in range(torch.cuda.device_count())
                    ]
                }
            else:
                info['gpu'] = {'count': 0, 'devices': []}
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system info: {e}")
            return {}

    def _apply_global_optimizations(self):
        """Apply global Ray Data optimizations."""
        try:
            ctx = DataContext.get_current()
            
            # Pinterest-inspired optimizations
            # Increase block size for better performance on large datasets
            ctx.target_max_block_size = 256 * 1024 * 1024  # 256MB
            ctx.target_min_block_size = 16 * 1024 * 1024   # 16MB
            
            # Optimize scheduling for distributed workloads
            ctx.scheduling_strategy = "SPREAD"
            
            # Enable memory optimizations
            ctx.enable_pandas_block = True
            ctx.actor_prefetcher_enabled = True
            
            # Increase batch size for shuffle operations
            ctx.target_shuffle_max_block_size = 512 * 1024 * 1024  # 512MB
            
            # Enable progress tracking
            ctx.enable_progress_bars = True
            ctx.enable_auto_log_stats = True
            
            self.logger.info("Applied global Ray Data optimizations")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply global optimizations: {e}")

    def create_loader(
        self, 
        loader_type: str,
        loader_id: Optional[str] = None,
        **kwargs
    ) -> BaseDataLoader:
        """
        Create a data loader with automatic optimization.
        
        Args:
            loader_type: Type of loader ('huggingface', 'numpy', 'pandas', 'torch')
            loader_id: Unique identifier for the loader
            **kwargs: Loader-specific arguments
        
        Returns:
            Configured data loader instance
        """
        if loader_type not in self.loaders:
            raise ValueError(f"Unknown loader type: {loader_type}. "
                           f"Available: {list(self.loaders.keys())}")
        
        try:
            # Generate unique ID if not provided
            if loader_id is None:
                loader_id = f"{loader_type}_{int(time.time())}"
            
            # Add global cache directory to kwargs
            if 'cache_dir' not in kwargs:
                kwargs['cache_dir'] = str(self.cache_dir / loader_id)
            
            # Create loader instance
            loader_class = self.loaders[loader_type]
            loader = loader_class(**kwargs)
            
            # Register loader for monitoring
            self.active_loaders[loader_id] = {
                'loader': loader,
                'type': loader_type,
                'created_at': time.time(),
                'metrics': {}
            }
            
            self.logger.info(f"Created {loader_type} loader with ID: {loader_id}")
            return loader
            
        except Exception as e:
            self.logger.error(f"Failed to create {loader_type} loader: {e}")
            raise

    def auto_detect_loader(self, path: str, **kwargs) -> BaseDataLoader:
        """
        Automatically detect and create the appropriate loader for a file.
        
        Args:
            path: File path
            **kwargs: Additional arguments for the loader
        
        Returns:
            Appropriate data loader
        """
        try:
            file_path = Path(path)
            extension = file_path.suffix.lower()
            
            # File extension to loader mapping
            extension_map = {
                '.csv': 'pandas',
                '.parquet': 'pandas', 
                '.xlsx': 'pandas',
                '.xls': 'pandas',
                '.json': 'pandas',
                '.npy': 'numpy',
                '.npz': 'numpy',
                '.pt': 'torch',
                '.pth': 'torch',
                '.pkl': 'torch',
                '.pickle': 'torch'
            }
            
            if extension in extension_map:
                loader_type = extension_map[extension]
                self.logger.info(f"Auto-detected loader type: {loader_type} for {path}")
                return self.create_loader(loader_type, **kwargs)
            
            # Check if it's a HuggingFace dataset path (no extension, likely a dataset name)
            elif '/' in path and '.' not in file_path.name:
                self.logger.info(f"Auto-detected HuggingFace dataset: {path}")
                return self.create_loader('huggingface', **kwargs)
            
            else:
                self.logger.warning(f"Could not auto-detect loader for: {path}, using base loader")
                return self.create_loader('base', **kwargs)
                
        except Exception as e:
            self.logger.error(f"Auto-detection failed for {path}: {e}")
            return self.create_loader('base', **kwargs)

    def benchmark_loader(
        self, 
        loader: BaseDataLoader,
        operations: List[str] = None,
        dataset_size_limit: int = 10000
    ) -> Dict[str, Any]:
        """
        Benchmark loader performance.
        
        Args:
            loader: Loader to benchmark
            operations: List of operations to benchmark
            dataset_size_limit: Limit dataset size for benchmarking
        
        Returns:
            Benchmark results
        """
        if operations is None:
            operations = ['load', 'map', 'map_batch']
        
        results = {
            'loader_type': type(loader).__name__,
            'timestamp': time.time(),
            'system_info': self.system_info,
            'operations': {}
        }
        
        try:
            if 'load' in operations and hasattr(loader, 'dataset'):
                # Benchmark basic operations
                if loader.ray_dataset is not None:
                    dataset = loader.ray_dataset
                    
                    # Limit dataset size for benchmarking
                    if dataset.count() > dataset_size_limit:
                        dataset = dataset.limit(dataset_size_limit)
                    
                    # Benchmark take operation
                    start_time = time.time()
                    sample = dataset.take(100)
                    results['operations']['take_100'] = {
                        'time_seconds': time.time() - start_time,
                        'sample_size': len(sample)
                    }
                    
                    # Benchmark count operation
                    start_time = time.time()
                    count = dataset.count()
                    results['operations']['count'] = {
                        'time_seconds': time.time() - start_time,
                        'count': count
                    }

                    # Benchmark map operation
                    if 'map' in operations:
                        start_time = time.time()
                        _ = dataset.map(lambda x: x).materialize()
                        results['operations']['map_identity'] = {
                            'time_seconds': time.time() - start_time,
                            'throughput_rows_per_sec': count / (time.time() - start_time)
                        }

                    # Benchmark batch operation
                    if 'map_batch' in operations:
                        start_time = time.time()
                        _ = dataset.map_batches(
                            lambda batch: batch, 
                            batch_size=1000
                        ).materialize()
                        results['operations']['map_batch_identity'] = {
                            'time_seconds': time.time() - start_time,
                            'throughput_rows_per_sec': count / (time.time() - start_time)
                        }

            self.logger.info(f"Benchmark completed for {type(loader).__name__}")
            return results

        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            results['error'] = str(e)
            return results


    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        try:
            status = {
                'timestamp': time.time(),
                'ray_initialized': ray.is_initialized(),
                'system_info': self.system_info
            }
            
            if ray.is_initialized():
                status.update({
                    'cluster_resources': ray.cluster_resources(),
                    'available_resources': ray.available_resources(),
                    'nodes': len(ray.nodes()),
                    'node_details': ray.nodes()
                })
                
                # Get Ray Data context info
                ctx = DataContext.get_current()
                status['ray_data_context'] = {
                    'target_max_block_size_mb': ctx.target_max_block_size / (1024 * 1024),
                    'target_min_block_size_mb': ctx.target_min_block_size / (1024 * 1024),
                    'scheduling_strategy': ctx.scheduling_strategy,
                    'enable_pandas_block': ctx.enable_pandas_block,
                    'actor_prefetcher_enabled': ctx.actor_prefetcher_enabled
                }
            
            # Active loaders info
            status['active_loaders'] = {
                loader_id: {
                    'type': info['type'],
                    'created_at': info['created_at'],
                    'age_seconds': time.time() - info['created_at']
                }
                for loader_id, info in self.active_loaders.items()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster status: {e}")
            return {'error': str(e)}

    def optimize_for_workload(
        self, 
        workload_type: str,
        dataset_size_gb: Optional[float] = None,
        num_workers: Optional[int] = None
    ):
        """
        Optimize Ray configuration for specific workload types.
        
        Args:
            workload_type: Type of workload ('training', 'inference', 'preprocessing', 'etl')
            dataset_size_gb: Expected dataset size in GB
            num_workers: Number of workers to optimize for
        """
        try:
            ctx = DataContext.get_current()
            
            if workload_type == 'training':
                # Optimize for ML training workloads
                ctx.target_max_block_size = 128 * 1024 * 1024  # 128MB for stable memory
                ctx.scheduling_strategy = "SPREAD"
                ctx.actor_prefetcher_enabled = True
                ctx.enable_progress_bars = True
                
            elif workload_type == 'inference':
                # Optimize for batch inference
                ctx.target_max_block_size = 64 * 1024 * 1024   # 64MB for low latency
                ctx.scheduling_strategy = "DEFAULT"
                ctx.actor_prefetcher_enabled = True
                
            elif workload_type == 'preprocessing':
                # Optimize for data preprocessing
                ctx.target_max_block_size = 256 * 1024 * 1024  # 256MB for throughput
                ctx.scheduling_strategy = "SPREAD"
                ctx.enable_pandas_block = True
                
            elif workload_type == 'etl':
                # Optimize for ETL workloads
                ctx.target_max_block_size = 512 * 1024 * 1024  # 512MB for large data
                ctx.target_shuffle_max_block_size = 1024 * 1024 * 1024  # 1GB
                ctx.scheduling_strategy = "SPREAD"
                
            # Adjust based on dataset size
            if dataset_size_gb:
                if dataset_size_gb > 100:  # Very large datasets
                    ctx.target_max_block_size = min(ctx.target_max_block_size * 2, 1024 * 1024 * 1024)
                elif dataset_size_gb < 1:  # Small datasets
                    ctx.target_max_block_size = max(ctx.target_max_block_size // 2, 16 * 1024 * 1024)
            
            self.logger.info(f"Optimized configuration for {workload_type} workload")
            
        except Exception as e:
            self.logger.warning(f"Workload optimization failed: {e}")

    def save_performance_report(self, output_path: str):
        """Save comprehensive performance report."""
        try:
            report = {
                'timestamp': time.time(),
                'omniray_version': '1.0.0',
                'cluster_status': self.get_cluster_status(),
                'performance_metrics': self.performance_metrics,
                'active_loaders': {
                    loader_id: {
                        'type': info['type'],
                        'created_at': info['created_at'],
                        'metrics': info['metrics']
                    }
                    for loader_id, info in self.active_loaders.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.logger.info(f"Performance report saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")


    def cleanup(self):
        """Clean up resources and shutdown Ray."""
        try:
            # Clear loader references
            self.active_loaders.clear()
            
            # Clear performance metrics
            self.performance_metrics.clear()
            
            # Shutdown Ray if initialized
            if ray.is_initialized():
                ray.shutdown()
                
            self.logger.info("OmniRay Manager cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
