import numpy as np
import ray.data as rd
from typing import Optional, List, Tuple, Dict, Any
import time
import os

from .base import BaseDataLoader


class NumpyDataLoader(BaseDataLoader):
    """
    Optimized NumPy data loader with memory-efficient loading,
    lazy evaluation, and distributed chunk processing.
    """
    
    def __init__(
        self,
        memory_map: bool = True,
        chunk_size_mb: int = 64,
        parallel_loading: bool = True
    ):
        """
        Initialize NumPy data loader with memory optimizations.
        
        Args:
            memory_map: Use memory mapping for large files
            chunk_size_mb: Chunk size in MB for processing
            parallel_loading: Enable parallel chunk loading
        """
        super().__init__()
        self.memory_map = memory_map
        self.chunk_size_mb = chunk_size_mb
        self.parallel_loading = parallel_loading
        self.dataset = None
        self._file_info = {}

    def load(
        self, 
        path: str, 
        mmap_mode: Optional[str] = None,
        optimize_blocks: bool = True,
        cache_dataset: bool = True,
        **kwargs
    ):
        """
        Load NumPy array with memory and performance optimizations.
        
        Args:
            path: Path to numpy file
            mmap_mode: Memory map mode ('r', 'r+', 'w+', 'c')
            optimize_blocks: Optimize block structure
            cache_dataset: Enable caching
            **kwargs: Additional numpy.load arguments
        """
        cache_key = self._generate_cache_key(path, mmap_mode=mmap_mode, **kwargs)
        
        # Try cache first
        if cache_dataset:
            cached_dataset = self._load_cached_dataset(cache_key)
            if cached_dataset is not None:
                self.ray_dataset = cached_dataset
                return

        try:
            self.logger.info(f"Loading NumPy array from: {path}")
            start_time = time.time()
            
            # Get file information
            file_size = os.path.getsize(path)
            self._file_info = {
                'path': path,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024)
            }
            
            # Choose loading strategy based on file size
            if self.memory_map and file_size > 100 * 1024 * 1024:  # 100MB threshold
                self._load_with_memory_mapping(path, mmap_mode, **kwargs)
            else:
                self._load_standard(path, **kwargs)
            
            load_time = time.time() - start_time
            self.logger.info(f"NumPy array loaded in {load_time:.2f}s")
            
            # Convert to Ray dataset with optimizations
            self._convert_to_ray_dataset(optimize_blocks)
            
            # Cache if requested
            if cache_dataset and self.ray_dataset is not None:
                metadata = {
                    'timestamp': time.time(),
                    'path': path,
                    'file_info': self._file_info,
                    'kwargs': kwargs
                }
                self._cache_dataset(cache_key, self.ray_dataset, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to load NumPy array from {path}: {e}")
            raise

    def _load_with_memory_mapping(self, path: str, mmap_mode: Optional[str], **kwargs):
        """Load large files using memory mapping."""
        try:
            mmap_mode = mmap_mode or 'r'
            self.logger.info(f"Using memory mapping with mode: {mmap_mode}")
            
            self.dataset = np.load(path, mmap_mode=mmap_mode, **kwargs)
            self._file_info['loading_method'] = 'memory_mapped'
            self._file_info['mmap_mode'] = mmap_mode
            
        except Exception as e:
            self.logger.warning(f"Memory mapping failed, falling back to standard loading: {e}")
            self._load_standard(path, **kwargs)

    def _load_standard(self, path: str, **kwargs):
        """Standard numpy loading."""
        self.dataset = np.load(path, **kwargs)
        self._file_info['loading_method'] = 'standard'

    def _convert_to_ray_dataset(self, optimize_blocks: bool):
        """Convert numpy array to Ray dataset with optimizations."""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            if optimize_blocks and self.dataset.nbytes > 50 * 1024 * 1024:  # 50MB threshold
                # Calculate optimal number of blocks
                target_block_size = self.chunk_size_mb * 1024 * 1024
                optimal_blocks = max(1, self.dataset.nbytes // target_block_size)
                
                # Limit to reasonable number of blocks
                max_blocks = min(optimal_blocks, 1000)
                
                self.logger.info(f"Creating {max_blocks} blocks for optimal processing")
                self.ray_dataset = rd.from_numpy(self.dataset, override_num_blocks=max_blocks)
                
            else:
                self.ray_dataset = rd.from_numpy(self.dataset)
                
            self._file_info['ray_blocks'] = self.ray_dataset.num_blocks()
            
        except Exception as e:
            self.logger.warning(f"Optimized conversion failed, using standard: {e}")
            self.ray_dataset = rd.from_numpy(self.dataset)

    def load_from_ndarray(
        self, 
        ndarray: np.ndarray,
        copy_data: bool = False,
        optimize_blocks: bool = True
    ):
        """
        Load from existing numpy array with optimization options.
        
        Args:
            ndarray: NumPy array to load
            copy_data: Whether to copy the array (default: use reference)
            optimize_blocks: Optimize block structure
        """
        try:
            self.logger.info(f"Loading from ndarray: shape={ndarray.shape}, dtype={ndarray.dtype}")
            
            # Store array info
            self._file_info = {
                'shape': ndarray.shape,
                'dtype': str(ndarray.dtype),
                'size_bytes': ndarray.nbytes,
                'size_mb': ndarray.nbytes / (1024 * 1024),
                'loading_method': 'from_ndarray'
            }
            
            if copy_data:
                self.dataset = ndarray.copy()
                self._file_info['copied'] = True
            else:
                self.dataset = ndarray
                self._file_info['copied'] = False
            
            # Convert to Ray dataset
            self._convert_to_ray_dataset(optimize_blocks)
            
        except Exception as e:
            self.logger.error(f"Failed to load from ndarray: {e}")
            raise

    def load_multiple_files(
        self, 
        file_paths: List[str],
        concatenate: bool = True,
        axis: int = 0
    ):
        """
        Load and combine multiple NumPy files efficiently.
        
        Args:
            file_paths: List of file paths
            concatenate: Whether to concatenate arrays
            axis: Axis for concatenation
        """
        try:
            self.logger.info(f"Loading {len(file_paths)} NumPy files")
            
            arrays = []
            total_size = 0
            
            for i, path in enumerate(file_paths):
                self.logger.debug(f"Loading file {i+1}/{len(file_paths)}: {path}")
                
                # Check file size for memory mapping decision
                file_size = os.path.getsize(path)
                total_size += file_size
                
                if self.memory_map and file_size > 50 * 1024 * 1024:
                    arr = np.load(path, mmap_mode='r')
                else:
                    arr = np.load(path)
                
                arrays.append(arr)
            
            self._file_info = {
                'num_files': len(file_paths),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'loading_method': 'multiple_files'
            }
            
            if concatenate:
                self.logger.info(f"Concatenating arrays along axis {axis}")
                self.dataset = np.concatenate(arrays, axis=axis)
                self._file_info['concatenated'] = True
                self._file_info['concat_axis'] = axis
            else:
                # Store as list for separate processing
                self.dataset = arrays
                self._file_info['concatenated'] = False
            
            # Convert to Ray dataset
            if concatenate:
                self._convert_to_ray_dataset(True)
            else:
                # Create separate datasets and union them
                ray_datasets = []
                for arr in arrays:
                    ray_datasets.append(rd.from_numpy(arr))
                
                self.ray_dataset = ray_datasets[0]
                for ds in ray_datasets[1:]:
                    self.ray_dataset = self.ray_dataset.union(ds)
                    
        except Exception as e:
            self.logger.error(f"Failed to load multiple files: {e}")
            raise

    def reshape_dataset(self, new_shape: Tuple[int, ...]):
        """
        Reshape the dataset efficiently.
        
        Args:
            new_shape: New shape for the array
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            def reshape_batch(batch):
                # Assuming batch is dict with 'data' key from numpy
                data = batch['data']
                if isinstance(data, list):
                    data = np.array(data)
                return {'data': data.reshape(new_shape)}
            
            self.ray_dataset = self.ray_dataset.map_batches(
                reshape_batch,
                batch_size=100,
                batch_format="dict"
            )
            
            self.logger.info(f"Reshaped dataset to {new_shape}")
            
        except Exception as e:
            self.logger.error(f"Reshape failed: {e}")
            raise

    def normalize_dataset(
        self, 
        method: str = "minmax",
        feature_range: Tuple[float, float] = (0, 1),
        axis: Optional[int] = None
    ):
        """
        Normalize dataset efficiently using batch processing.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust')
            feature_range: Range for minmax scaling
            axis: Axis for normalization
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            if method == "minmax":
                # Calculate global min/max first
                stats = self._calculate_global_stats()
                global_min = stats['min']
                global_max = stats['max']
                
                def minmax_normalize(batch):
                    data = np.array(batch['data'])
                    normalized = (data - global_min) / (global_max - global_min)
                    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
                    return {'data': normalized.tolist()}
                
                self.ray_dataset = self.ray_dataset.map_batches(
                    minmax_normalize,
                    batch_size=1000,
                    batch_format="dict"
                )
                
            elif method == "zscore":
                # Calculate global mean/std
                stats = self._calculate_global_stats()
                global_mean = stats['mean']
                global_std = stats['std']
                
                def zscore_normalize(batch):
                    data = np.array(batch['data'])
                    normalized = (data - global_mean) / (global_std + 1e-8)
                    return {'data': normalized.tolist()}
                
                self.ray_dataset = self.ray_dataset.map_batches(
                    zscore_normalize,
                    batch_size=1000,
                    batch_format="dict"
                )
            
            self.logger.info(f"Applied {method} normalization")
            
        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            raise

    def _calculate_global_stats(self) -> Dict[str, float]:
        """Calculate global statistics for normalization."""
        try:
            # Use Ray's built-in aggregation for efficiency
            def extract_data(batch):
                return np.array(batch['data'])
            
            # Calculate statistics
            stats = {}
            
            # Min/Max
            min_val = self.ray_dataset.map_batches(extract_data).min()
            max_val = self.ray_dataset.map_batches(extract_data).max()
            
            # Mean/Std
            mean_val = self.ray_dataset.map_batches(extract_data).mean()
            std_val = self.ray_dataset.map_batches(extract_data).std()
            
            stats = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val
            }
            
            self.logger.debug(f"Global stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate global stats: {e}")
            # Fallback to approximate stats from first few batches
            sample_data = self.ray_dataset.take(1000)
            data_array = np.array([item['data'] for item in sample_data])
            
            return {
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'mean': float(np.mean(data_array)),
                'std': float(np.std(data_array))
            }

    def split_dataset(
        self, 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[rd.MaterializedDataset, ...]:
        """
        Split dataset into train/val/test sets efficiently.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio  
            test_ratio: Test set ratio
            shuffle: Whether to shuffle before splitting
            seed: Random seed
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        try:
            dataset = self.ray_dataset
            
            if shuffle:
                if seed is not None:
                    dataset = dataset.random_shuffle(seed=seed)
                else:
                    dataset = dataset.random_shuffle()
            
            # Calculate split points
            total_rows = dataset.count()
            train_size = int(total_rows * train_ratio)
            val_size = int(total_rows * val_ratio)
            
            # Split dataset
            train_ds = dataset.limit(train_size)
            remaining = dataset.skip(train_size)
            
            if val_ratio > 0:
                val_ds = remaining.limit(val_size)
                test_ds = remaining.skip(val_size)
                return train_ds, val_ds, test_ds
            else:
                test_ds = remaining
                return train_ds, test_ds
                
        except Exception as e:
            self.logger.error(f"Dataset splitting failed: {e}")
            raise

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {'file_info': self._file_info}
        
        if self.ray_dataset is not None:
            info.update(self.get_stats())
            
            # Add numpy-specific info
            try:
                sample = self.ray_dataset.take(1)[0]
                if 'data' in sample:
                    data_sample = np.array(sample['data'])
                    info['data_info'] = {
                        'sample_shape': data_sample.shape,
                        'sample_dtype': str(data_sample.dtype),
                        'sample_min': float(np.min(data_sample)),
                        'sample_max': float(np.max(data_sample)),
                        'sample_mean': float(np.mean(data_sample))
                    }
            except Exception as e:
                self.logger.warning(f"Failed to get sample data info: {e}")
                
        return info
