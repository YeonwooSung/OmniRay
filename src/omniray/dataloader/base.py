import ray
from ray.data.dataset import MaterializedDataset
from ray.data.context import DataContext
from typing import Callable, Union, Optional, Dict, Any
import torch
import time
import logging
import hashlib
import pickle
from pathlib import Path


class BaseDataLoader:
    """
    Optimized base data loader for OmniRay framework with performance enhancements
    inspired by Pinterest's Ray Data optimizations.
    """
    
    def __init__(
        self, 
        init_ray: bool = False,
        enable_caching: bool = True,
        cache_dir: str = "/tmp/omniray_cache",
        optimize_for_large_workloads: bool = True,
        target_max_block_size: Optional[int] = None,
        scheduling_strategy: str = "SPREAD"
    ):
        """
        Initialize BaseDataLoader with optimization settings.
        
        Args:
            init_ray: Whether to initialize Ray
            enable_caching: Enable intelligent caching
            cache_dir: Directory for caching datasets
            optimize_for_large_workloads: Apply Pinterest-style optimizations
            target_max_block_size: Custom block size (None uses optimized default)
            scheduling_strategy: Ray scheduling strategy
        """
        if init_ray:
            ray.init(ignore_reinit_error=True)

        self.ray_dataset: Union[MaterializedDataset, None] = None
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Apply performance optimizations
        if optimize_for_large_workloads:
            self._apply_performance_optimizations(target_max_block_size, scheduling_strategy)
        
        # Cache for loaded datasets
        self._dataset_cache: Dict[str, MaterializedDataset] = {}
        self._cache_metadata: Dict[str, Dict] = {}

    def _apply_performance_optimizations(
        self, 
        target_max_block_size: Optional[int], 
        scheduling_strategy: str
    ):
        """Apply Pinterest-inspired performance optimizations."""
        ctx = DataContext.get_current()
        
        # Optimize block size for large workloads
        if target_max_block_size is None:
            # Use larger blocks for better performance on large datasets
            # Pinterest found 2-3x speedup with optimized block sizes
            target_max_block_size = 256 * 1024 * 1024  # 256MB instead of default 128MB
        
        ctx.target_max_block_size = target_max_block_size
        
        # Optimize for memory efficiency
        ctx.target_min_block_size = 16 * 1024 * 1024  # 16MB minimum
        
        # Set scheduling strategy for better distribution
        ctx.scheduling_strategy = scheduling_strategy
        
        # Enable optimizations that reduce overhead
        ctx.enable_pandas_block = True
        ctx.actor_prefetcher_enabled = True
        
        # Optimize for large args threshold
        ctx.large_args_threshold = 100 * 1024 * 1024  # 100MB
        
        self.logger.info(f"Applied performance optimizations: "
                        f"max_block_size={target_max_block_size}, "
                        f"scheduling={scheduling_strategy}")

    def _generate_cache_key(self, path: str, **kwargs) -> str:
        """Generate unique cache key for dataset."""
        key_data = f"{path}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.ray_dataset"

    def _cache_dataset(self, cache_key: str, dataset: MaterializedDataset, metadata: Dict):
        """Cache dataset to disk."""
        if not self.enable_caching:
            return
            
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Save dataset using Ray's native serialization
            dataset.write_parquet(str(cache_path))
            
            # Save metadata
            metadata_path = cache_path.with_suffix('.metadata')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            self.logger.info(f"Cached dataset to {cache_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache dataset: {e}")

    def _load_cached_dataset(self, cache_key: str) -> Optional[MaterializedDataset]:
        """Load dataset from cache."""
        if not self.enable_caching:
            return None
            
        try:
            cache_path = self._get_cache_path(cache_key)
            metadata_path = cache_path.with_suffix('.metadata')
            
            if cache_path.exists() and metadata_path.exists():
                # Load metadata to check validity
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Check if cache is still valid (e.g., within 24 hours)
                cache_age = time.time() - metadata.get('timestamp', 0)
                if cache_age < 24 * 3600:  # 24 hours
                    dataset = ray.data.read_parquet(str(cache_path))
                    self.logger.info(f"Loaded dataset from cache: {cache_path}")
                    return dataset
                    
        except Exception as e:
            self.logger.warning(f"Failed to load cached dataset: {e}")
            
        return None

    def load(self, path: str, **kwargs):
        """Load dataset with caching support."""
        raise NotImplementedError("load method is not implemented.")

    def get_ray_dataset(self) -> MaterializedDataset:
        """Get the loaded Ray dataset."""
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")
        return self.ray_dataset

    def lambda_map_chained(
        self, 
        funcs: list, 
        enable_chunk_optimization: bool = True
    ) -> MaterializedDataset:
        """
        Apply multiple functions in sequence with optimizations.
        
        Args:
            funcs: List of functions to apply
            enable_chunk_optimization: Disable combine_chunks for better performance
        """
        if not isinstance(self.ray_dataset, MaterializedDataset):
            raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")
        
        dataset = self.ray_dataset
        
        # Apply Pinterest optimization: avoid unnecessary chunk combination
        for i, func in enumerate(funcs):
            try:
                if enable_chunk_optimization:
                    # Use batch operations when possible for better performance
                    dataset = dataset.map_batches(
                        lambda batch: func(batch),
                        batch_format="numpy",
                        zero_copy_batch=True
                    )
                else:
                    dataset = dataset.map(func)
                    
                self.logger.debug(f"Applied function {i+1}/{len(funcs)}")
                
            except Exception as e:
                self.logger.error(f"Error applying function {i+1}: {e}")
                # Fallback to standard map
                dataset = dataset.map(func)
        
        self.ray_dataset = dataset
        return dataset

    def lambda_map(self, func: Callable) -> MaterializedDataset:
        """Apply function with error handling and retries."""
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if not isinstance(self.ray_dataset, MaterializedDataset):
            raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")
        
        try:
            return self.ray_dataset.map(func)
        except Exception as e:
            self.logger.warning(f"Map operation failed, retrying: {e}")
            # Retry with different strategy
            return self.ray_dataset.map_batches(
                lambda batch: [func(item) for item in batch],
                batch_format="numpy"
            )

    def lambda_map_batch(
        self,
        func: Callable,
        concurrency: int = 2,
        batch_size: int = 1,
        num_cpu: int = 1,
        zero_copy_batch: bool = True,
        batch_format: str = "numpy"
    ) -> MaterializedDataset:
        """
        Optimized batch mapping with zero-copy optimization.
        
        Args:
            func: Function to apply to each batch
            concurrency: Number of concurrent tasks
            batch_size: Number of elements per batch
            num_cpu: CPUs per worker
            zero_copy_batch: Enable zero-copy optimization
            batch_format: Batch format for optimization
        """
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if not isinstance(self.ray_dataset, MaterializedDataset):
            raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")

        try:
            return self.ray_dataset.map_batches(
                func,
                num_cpus=num_cpu,
                batch_size=batch_size,
                concurrency=concurrency,
                zero_copy_batch=zero_copy_batch,
                batch_format=batch_format
            )
        except Exception as e:
            self.logger.warning(f"Optimized batch mapping failed, using fallback: {e}")
            # Fallback without optimizations
            return self.ray_dataset.map_batches(
                func,
                num_cpus=num_cpu,
                batch_size=batch_size,
                concurrency=concurrency
            )

    def lambda_map_batch_gpu(
        self,
        func: Callable,
        concurrency: int = 1,
        batch_size: int = 1,
        num_gpu: int = 1,
        gpu_memory_limit: Optional[float] = None
    ) -> MaterializedDataset:
        """
        GPU batch mapping with memory management.
        
        Args:
            func: Function to apply
            concurrency: Concurrent tasks
            batch_size: Batch size
            num_gpu: GPUs per worker  
            gpu_memory_limit: GPU memory limit ratio (0.0-1.0)
        """
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot allocate GPU resources.")

        if not isinstance(self.ray_dataset, MaterializedDataset):
            raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")

        # Optimize GPU memory usage
        if gpu_memory_limit:
            # Reduce batch size if memory limit is set
            original_batch_size = batch_size
            batch_size = max(1, int(batch_size * gpu_memory_limit))
            if batch_size != original_batch_size:
                self.logger.info(f"Adjusted batch size from {original_batch_size} to {batch_size} for GPU memory limit")

        try:
            return self.ray_dataset.map_batches(
                func,
                num_gpus=num_gpu,
                batch_size=batch_size,
                concurrency=concurrency,
                batch_format="numpy"
            )
        except Exception as e:
            self.logger.error(f"GPU batch mapping failed: {e}")
            # Reduce resources and retry
            return self.ray_dataset.map_batches(
                func,
                num_gpus=max(1, num_gpu // 2),
                batch_size=max(1, batch_size // 2),
                concurrency=max(1, concurrency // 2)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics and performance metrics."""
        if self.ray_dataset is None:
            return {}
            
        try:
            stats = self.ray_dataset.stats()
            return {
                "num_blocks": self.ray_dataset.num_blocks(),
                "size_bytes": self.ray_dataset.size_bytes(),
                "schema": str(self.ray_dataset.schema()),
                "stats": stats
            }
        except Exception as e:
            self.logger.warning(f"Failed to get stats: {e}")
            return {}

    def clear_cache(self):
        """Clear all cached datasets."""
        try:
            for cache_file in self.cache_dir.glob("*.ray_dataset"):
                cache_file.unlink()
            for metadata_file in self.cache_dir.glob("*.metadata"):
                metadata_file.unlink()
            self._dataset_cache.clear()
            self._cache_metadata.clear()
            self.logger.info("Cleared all cached datasets")
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")
