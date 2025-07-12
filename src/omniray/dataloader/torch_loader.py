import torch
import ray.data as rd
from typing import Optional, List, Dict, Any, Callable
import time
import gc
import os
from pathlib import Path

from .base import BaseDataLoader


class TorchDataLoader(BaseDataLoader):
    """
    Optimized PyTorch data loader with GPU memory management,
    distributed training support, and efficient tensor operations.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        pin_memory: bool = True,
        enable_cuda_optimization: bool = True,
        memory_fraction: float = 0.8
    ):
        """
        Initialize PyTorch data loader with GPU optimizations.
        
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
            pin_memory: Pin memory for faster GPU transfer
            enable_cuda_optimization: Enable CUDA-specific optimizations
            memory_fraction: Fraction of GPU memory to use
        """
        super().__init__()
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.enable_cuda_optimization = enable_cuda_optimization and torch.cuda.is_available()
        self.memory_fraction = memory_fraction
        
        # Dataset and metadata
        self.dataset = None
        self._tensor_info = {}
        
        # GPU memory management
        if self.enable_cuda_optimization:
            self._setup_cuda_optimization()
        
        self.logger.info(f"Initialized TorchDataLoader on device: {self.device}")

    def _setup_cuda_optimization(self):
        """Setup CUDA optimizations and memory management."""
        try:
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                
                # Enable memory pool for better allocation
                torch.cuda.empty_cache()
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(self.device)
                gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
                
                self._tensor_info['gpu_info'] = {
                    'name': gpu_name,
                    'total_memory_gb': gpu_memory / (1024**3),
                    'memory_fraction': self.memory_fraction,
                    'available_memory_gb': gpu_memory * self.memory_fraction / (1024**3)
                }
                
                self.logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory/(1024**3):.1f}GB")
                
        except Exception as e:
            self.logger.warning(f"CUDA optimization setup failed: {e}")
            self.enable_cuda_optimization = False

    def load(
        self, 
        path: str, 
        map_location: Optional[str] = None,
        weights_only: bool = False,
        optimize_loading: bool = True,
        cache_dataset: bool = True,
        **kwargs
    ):
        """
        Load PyTorch tensor/model with optimizations.
        
        Args:
            path: Path to tensor file
            map_location: Device to load to
            weights_only: Only load weights (security)
            optimize_loading: Apply loading optimizations
            cache_dataset: Enable caching
            **kwargs: Additional torch.load arguments
        """
        cache_key = self._generate_cache_key(
            path, map_location=map_location, weights_only=weights_only, **kwargs
        )
        
        # Try cache first
        if cache_dataset:
            cached_dataset = self._load_cached_dataset(cache_key)
            if cached_dataset is not None:
                self.ray_dataset = cached_dataset
                return

        try:
            self.logger.info(f"Loading PyTorch tensor from: {path}")
            start_time = time.time()
            
            # Get file information
            file_path = Path(path)
            file_size = file_path.stat().st_size
            
            self._tensor_info = {
                'path': path,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024)
            }
            
            # Set loading location
            load_location = map_location or ('cpu' if file_size > 1024**3 else str(self.device))
            
            # Load with optimizations
            load_kwargs = {
                'map_location': load_location,
                'weights_only': weights_only,
                **kwargs
            }
            
            if optimize_loading and file_size > 100 * 1024 * 1024:  # 100MB threshold
                self._load_large_tensor(path, load_kwargs)
            else:
                self.dataset = torch.load(path, **load_kwargs)
                self._tensor_info['loading_method'] = 'standard'
            
            load_time = time.time() - start_time
            self.logger.info(f"Tensor loaded in {load_time:.2f}s")
            
            # Store tensor information
            if isinstance(self.dataset, torch.Tensor):
                self._tensor_info.update({
                    'shape': list(self.dataset.shape),
                    'dtype': str(self.dataset.dtype),
                    'device': str(self.dataset.device),
                    'requires_grad': self.dataset.requires_grad,
                    'memory_mb': self.dataset.numel() * self.dataset.element_size() / (1024 * 1024)
                })
            
            # Convert to Ray dataset
            self._convert_to_ray_dataset()
            
            # Cache if requested
            if cache_dataset and self.ray_dataset is not None:
                metadata = {
                    'timestamp': time.time(),
                    'path': path,
                    'tensor_info': self._tensor_info,
                    'kwargs': kwargs
                }
                self._cache_dataset(cache_key, self.ray_dataset, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch tensor from {path}: {e}")
            raise

    def _load_large_tensor(self, path: str, load_kwargs: Dict):
        """Load large tensors with memory management."""
        try:
            self.logger.info("Loading large tensor with memory optimization")
            
            # Clear GPU cache before loading
            if self.enable_cuda_optimization:
                torch.cuda.empty_cache()
                
            # Load to CPU first for large tensors
            cpu_kwargs = load_kwargs.copy()
            cpu_kwargs['map_location'] = 'cpu'
            
            self.dataset = torch.load(path, **cpu_kwargs)
            self._tensor_info['loading_method'] = 'large_tensor_cpu_first'
            
            # Move to target device in chunks if needed
            if str(self.device) != 'cpu' and isinstance(self.dataset, torch.Tensor):
                self._move_tensor_to_device()
                
        except Exception as e:
            self.logger.warning(f"Large tensor loading failed, using standard: {e}")
            self.dataset = torch.load(path, **load_kwargs)
            self._tensor_info['loading_method'] = 'standard_fallback'

    def _move_tensor_to_device(self):
        """Move tensor to target device efficiently."""
        if not isinstance(self.dataset, torch.Tensor):
            return
            
        try:
            tensor_size_mb = self.dataset.numel() * self.dataset.element_size() / (1024 * 1024)
            
            # If tensor is small enough, move directly
            if tensor_size_mb < 500:  # 500MB threshold
                self.dataset = self.dataset.to(self.device, non_blocking=True)
                self._tensor_info['device_transfer'] = 'direct'
            else:
                # Move in chunks for very large tensors
                self.logger.info("Moving large tensor to device in chunks")
                self._tensor_info['device_transfer'] = 'chunked'
                
                # This is a placeholder - in practice, you might want to keep
                # the tensor on CPU and move chunks as needed during processing
                self.dataset = self.dataset.to(self.device)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.warning("GPU out of memory, keeping tensor on CPU")
                self._tensor_info['device_transfer'] = 'cpu_fallback'
            else:
                raise

    def _convert_to_ray_dataset(self):
        """Convert PyTorch tensor to Ray dataset with optimizations."""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            if isinstance(self.dataset, torch.Tensor):
                # Optimize block size based on tensor characteristics
                tensor_size = self.dataset.numel() * self.dataset.element_size()
                
                if tensor_size > 100 * 1024 * 1024:  # 100MB threshold
                    # Calculate optimal blocks
                    target_block_size = 64 * 1024 * 1024  # 64MB blocks
                    optimal_blocks = max(1, tensor_size // target_block_size)
                    
                    self.ray_dataset = rd.from_torch(
                        self.dataset, 
                        override_num_blocks=min(optimal_blocks, 256)  # Cap at 256 blocks
                    )
                else:
                    self.ray_dataset = rd.from_torch(self.dataset)
                    
                self._tensor_info['ray_conversion'] = {
                    'num_blocks': self.ray_dataset.num_blocks(),
                    'elements_per_block': self.dataset.numel() // self.ray_dataset.num_blocks()
                }
            else:
                # Handle non-tensor objects (like state dicts)
                self.ray_dataset = rd.from_items([self.dataset])
                self._tensor_info['ray_conversion'] = {
                    'type': 'non_tensor',
                    'num_blocks': 1
                }
                
        except Exception as e:
            self.logger.warning(f"Ray conversion failed: {e}")
            # Fallback: convert tensor to numpy then to Ray
            if isinstance(self.dataset, torch.Tensor):
                numpy_array = self.dataset.detach().cpu().numpy()
                self.ray_dataset = rd.from_numpy(numpy_array)

    def load_from_tensor(
        self, 
        tensor: torch.Tensor,
        move_to_device: bool = True,
        optimize_blocks: bool = True
    ):
        """
        Load from existing PyTorch tensor.
        
        Args:
            tensor: Input tensor
            move_to_device: Move to target device
            optimize_blocks: Optimize Ray dataset blocks
        """
        try:
            self.logger.info(f"Loading from tensor: shape={tensor.shape}, dtype={tensor.dtype}")
            
            # Store tensor info
            self._tensor_info = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'requires_grad': tensor.requires_grad,
                'memory_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
                'loading_method': 'from_tensor'
            }
            
            # Move to target device if requested
            if move_to_device and tensor.device != self.device:
                if self.enable_cuda_optimization:
                    torch.cuda.empty_cache()
                    
                self.dataset = tensor.to(self.device, non_blocking=True)
            else:
                self.dataset = tensor
            
            # Convert to Ray dataset
            self._convert_to_ray_dataset()
            
        except Exception as e:
            self.logger.error(f"Failed to load from tensor: {e}")
            raise

    def apply_tensor_transforms(
        self, 
        transforms: List[Callable],
        batch_size: int = 1000,
        use_gpu: bool = True
    ):
        """
        Apply transforms to tensor data efficiently.
        
        Args:
            transforms: List of transform functions
            batch_size: Batch size for processing
            use_gpu: Use GPU for transforms
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            for i, transform in enumerate(transforms):
                self.logger.info(f"Applying transform {i+1}/{len(transforms)}")
                
                def transform_batch(batch):
                    # Convert batch to tensor
                    if isinstance(batch, dict) and 'data' in batch:
                        tensor_data = torch.tensor(batch['data'])
                    else:
                        tensor_data = torch.tensor(batch)
                    
                    # Move to appropriate device
                    if use_gpu and self.enable_cuda_optimization:
                        tensor_data = tensor_data.to(self.device, non_blocking=True)
                    
                    # Apply transform
                    transformed = transform(tensor_data)
                    
                    # Convert back to format expected by Ray
                    if use_gpu:
                        transformed = transformed.cpu()
                    
                    return {'data': transformed.numpy()}
                
                self.ray_dataset = self.ray_dataset.map_batches(
                    transform_batch,
                    batch_size=batch_size,
                    num_gpus=1 if use_gpu and self.enable_cuda_optimization else 0,
                    batch_format="numpy"
                )
                
        except Exception as e:
            self.logger.error(f"Transform application failed: {e}")
            raise

    def create_training_batches(
        self, 
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        pin_memory: Optional[bool] = None
    ):
        """
        Create optimized training batches.
        
        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            drop_last: Drop incomplete last batch
            pin_memory: Pin memory for GPU transfer
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            pin_mem = pin_memory if pin_memory is not None else self.pin_memory
            
            # Shuffle if requested
            if shuffle:
                self.ray_dataset = self.ray_dataset.random_shuffle()
            
            def create_batch(batch_data):
                """Create training batch with optimizations."""
                # Convert to tensors
                if isinstance(batch_data, dict):
                    batch_tensors = {}
                    for key, value in batch_data.items():
                        tensor = torch.tensor(value)
                        if pin_mem:
                            tensor = tensor.pin_memory()
                        batch_tensors[key] = tensor
                    return batch_tensors
                else:
                    tensor = torch.tensor(batch_data)
                    if pin_mem:
                        tensor = tensor.pin_memory()
                    return tensor
            
            # Create batched dataset
            batched_dataset = self.ray_dataset.map_batches(
                create_batch,
                batch_size=batch_size,
                drop_last=drop_last,
                batch_format="numpy"
            )
            
            self.logger.info(f"Created training batches: size={batch_size}, "
                           f"shuffle={shuffle}, pin_memory={pin_mem}")
            
            return batched_dataset
            
        except Exception as e:
            self.logger.error(f"Training batch creation failed: {e}")
            raise

    def tensor_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive tensor statistics."""
        if self.ray_dataset is None:
            return {}
        
        try:
            stats = {}
            
            # Basic statistics
            if isinstance(self.dataset, torch.Tensor):
                tensor = self.dataset
                
                # Move to CPU for statistics calculation if needed
                if tensor.device != torch.device('cpu'):
                    tensor = tensor.cpu()
                
                stats['tensor_stats'] = {
                    'min': float(tensor.min()),
                    'max': float(tensor.max()),
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std()),
                    'sum': float(tensor.sum()),
                    'numel': tensor.numel()
                }
                
                # Distribution statistics
                if tensor.numel() < 1000000:  # Only for reasonable sizes
                    flattened = tensor.flatten()
                    stats['distribution'] = {
                        'median': float(torch.median(flattened)),
                        'q25': float(torch.quantile(flattened, 0.25)),
                        'q75': float(torch.quantile(flattened, 0.75)),
                        'zeros_count': int((flattened == 0).sum()),
                        'nans_count': int(torch.isnan(flattened).sum()),
                        'infs_count': int(torch.isinf(flattened).sum())
                    }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Statistics calculation failed: {e}")
            return {}

    def optimize_for_training(
        self, 
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2
    ):
        """
        Optimize dataset for training workloads.
        
        Args:
            num_workers: Number of worker processes
            prefetch_factor: Prefetch factor for better pipeline
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            # Auto-detect optimal number of workers
            if num_workers is None:
                if self.enable_cuda_optimization:
                    # For GPU training, fewer workers often better
                    num_workers = min(4, os.cpu_count() or 1)
                else:
                    # For CPU training, more workers can help
                    num_workers = min(8, (os.cpu_count() or 1) * 2)
            
            # Configure for training optimization
            self.ray_dataset = self.ray_dataset.map_batches(
                lambda batch: batch,  # Identity function for prefetching
                num_cpus=0.5,  # Lightweight operation
                batch_size=1000,
                prefetch_blocks=prefetch_factor
            )
            
            self.logger.info(f"Optimized for training: workers={num_workers}, "
                           f"prefetch={prefetch_factor}")
            
        except Exception as e:
            self.logger.warning(f"Training optimization failed: {e}")

    def memory_cleanup(self):
        """Clean up GPU memory and cache."""
        try:
            if self.enable_cuda_optimization:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device)
                cached = torch.cuda.memory_reserved(self.device)
                
                self.logger.info(f"GPU memory - Allocated: {allocated/(1024**2):.1f}MB, "
                               f"Cached: {cached/(1024**2):.1f}MB")
                
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def save_processed_dataset(
        self, 
        output_path: str,
        format: str = 'torch'
    ):
        """
        Save processed dataset efficiently.
        
        Args:
            output_path: Output file path
            format: Save format ('torch', 'numpy', 'parquet')
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            self.logger.info(f"Saving dataset to {output_path} in {format} format")
            
            if format == 'torch':
                # Convert back to tensor and save
                if isinstance(self.dataset, torch.Tensor):
                    torch.save(self.dataset, output_path)
                else:
                    # Reconstruct tensor from Ray dataset
                    data = self.ray_dataset.take_all()
                    if isinstance(data[0], dict) and 'data' in data[0]:
                        tensor_data = torch.stack([torch.tensor(item['data']) for item in data])
                    else:
                        tensor_data = torch.tensor(data)
                    torch.save(tensor_data, output_path)
                    
            elif format == 'numpy':
                # Save as numpy
                self.ray_dataset.write_numpy(output_path)
                
            elif format == 'parquet':
                # Save as parquet
                self.ray_dataset.write_parquet(output_path)
                
            else:
                raise ValueError(f"Unsupported save format: {format}")
                
            self.logger.info(f"Dataset saved successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            raise


    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {'tensor_info': self._tensor_info}
        
        if self.ray_dataset is not None:
            info.update(self.get_stats())
            
            # Add tensor-specific statistics
            tensor_stats = self.tensor_statistics()
            if tensor_stats:
                info.update(tensor_stats)
                
        return info
