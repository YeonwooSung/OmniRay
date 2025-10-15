import ray
import ray.data as rd
import datasets
from datasets import DownloadConfig
from typing import Optional, Dict, Any
import time
import os

from .base import BaseDataLoader


class HuggingfaceDataLoader(BaseDataLoader):
    """
    Optimized HuggingFace dataset loader with intelligent caching,
    streaming support, and distributed preprocessing.
    """
    
    def __init__(
        self,
        enable_streaming: bool = True,
        max_workers: Optional[int] = None,
        trust_remote_code: bool = False,
        token: Optional[str] = None
    ):
        """
        Initialize HuggingFace data loader with advanced features.
        
        Args:
            enable_streaming: Enable streaming for large datasets
            max_workers: Max workers for parallel downloading
            trust_remote_code: Trust remote code in datasets
            token: HuggingFace token for private datasets
        """
        super().__init__()
        self.enable_streaming = enable_streaming
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.trust_remote_code = trust_remote_code
        self.token = token
        self.dataset = None
        
        # Configure download settings for better performance
        self.download_config = DownloadConfig(
            num_proc=self.max_workers,
            use_etag=True,  # Enable caching based on ETags
            resume_download=True,  # Resume interrupted downloads
        )

    def load(
        self, 
        path: str, 
        split: str = "train",
        use_all: bool = False,
        streaming: Optional[bool] = None,
        cache_dataset: bool = True,
        **kwargs
    ):
        """
        Load HuggingFace dataset with optimizations.
        
        Args:
            path: Dataset path/name
            split: Dataset split to load
            use_all: Use all splits
            streaming: Override streaming setting
            cache_dataset: Enable dataset caching
            **kwargs: Additional arguments for load_dataset
        """
        cache_key = self._generate_cache_key(path, split=split, use_all=use_all, **kwargs)
        
        # Try to load from cache first
        if cache_dataset:
            cached_dataset = self._load_cached_dataset(cache_key)
            if cached_dataset is not None:
                self.ray_dataset = cached_dataset
                return

        try:
            # Use streaming if enabled and dataset is large
            use_streaming = streaming if streaming is not None else self.enable_streaming
            
            # Load dataset with optimized settings
            load_kwargs = {
                'download_config': self.download_config,
                'trust_remote_code': self.trust_remote_code,
                'token': self.token,
                'streaming': use_streaming,
                **kwargs
            }
            
            self.logger.info(f"Loading HuggingFace dataset: {path} (streaming={use_streaming})")
            start_time = time.time()
            
            self.dataset = datasets.load_dataset(path, **load_kwargs)
            
            load_time = time.time() - start_time
            self.logger.info(f"Dataset loaded in {load_time:.2f}s")
            
            # Convert to Ray dataset with optimizations
            if use_all:
                self._convert_all_splits_to_ray()
            else:
                self._convert_split_to_ray(split, use_streaming)
            
            # Cache the processed dataset
            if cache_dataset and not use_streaming:
                metadata = {
                    'timestamp': time.time(),
                    'path': path,
                    'split': split,
                    'use_all': use_all,
                    'kwargs': kwargs
                }
                self._cache_dataset(cache_key, self.ray_dataset, metadata)
                
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace dataset {path}: {e}")
            # Fallback to non-streaming mode
            if use_streaming:
                self.logger.info("Retrying without streaming...")
                self.load(path, split=split, use_all=use_all, streaming=False, 
                         cache_dataset=cache_dataset, **kwargs)
            else:
                raise

    def _convert_split_to_ray(self, split: str, use_streaming: bool):
        """Convert single split to Ray dataset with optimizations."""
        try:
            if use_streaming:
                # For streaming datasets, use iterative loading
                self.ray_dataset = rd.from_huggingface(
                    self.dataset[split],
                    override_num_blocks=self.max_workers * 2  # Optimize block count
                )
            else:
                # For non-streaming, optimize based on dataset size
                dataset_split = self.dataset[split]
                
                # Estimate optimal number of blocks
                num_rows = len(dataset_split) if hasattr(dataset_split, '__len__') else None
                if num_rows:
                    # Target ~1000-10000 rows per block for optimal performance
                    optimal_blocks = max(1, min(self.max_workers * 4, num_rows // 1000))
                    self.ray_dataset = rd.from_huggingface(
                        dataset_split,
                        override_num_blocks=optimal_blocks
                    )
                else:
                    self.ray_dataset = rd.from_huggingface(dataset_split)
                    
        except Exception as e:
            self.logger.warning(f"Optimized conversion failed, using fallback: {e}")
            self.ray_dataset = rd.from_huggingface(self.dataset[split])

    def _convert_all_splits_to_ray(self):
        """Convert all splits to Ray dataset."""
        try:
            # Combine all splits efficiently
            all_datasets = []
            total_rows = 0
            
            for split_name, split_data in self.dataset.items():
                self.logger.info(f"Processing split: {split_name}")
                split_ray_ds = rd.from_huggingface(split_data)
                
                # Add split identifier
                split_ray_ds = split_ray_ds.map(
                    lambda row: {**row, "_split": split_name},
                    num_cpus=0.1  # Lightweight operation
                )
                
                all_datasets.append(split_ray_ds)
                if hasattr(split_data, '__len__'):
                    total_rows += len(split_data)
            
            # Union all splits
            self.ray_dataset = all_datasets[0]
            for ds in all_datasets[1:]:
                self.ray_dataset = self.ray_dataset.union(ds)
                
            self.logger.info(f"Combined {len(all_datasets)} splits with ~{total_rows} total rows")
            
        except Exception as e:
            self.logger.error(f"Failed to convert all splits: {e}")
            # Fallback to train split only
            self.ray_dataset = rd.from_huggingface(self.dataset["train"])

    def preprocess_text(
        self, 
        tokenizer,
        text_column: str = "text",
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
        batch_size: int = 1000
    ):
        """
        Optimized text preprocessing with batch tokenization.
        
        Args:
            tokenizer: HuggingFace tokenizer
            text_column: Column containing text
            max_length: Maximum sequence length
            truncation: Enable truncation
            padding: Padding strategy
            batch_size: Batch size for processing
        """
        if self.ray_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        def tokenize_batch(batch):
            """Tokenize a batch of texts efficiently."""
            texts = batch[text_column] if isinstance(batch[text_column], list) else [batch[text_column]]
            
            # Batch tokenization for efficiency
            tokenized = tokenizer(
                texts,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors="np"
            )
            
            # Return batch format expected by Ray
            result = {k: v.tolist() for k, v in tokenized.items()}
            
            # Preserve other columns
            for key, value in batch.items():
                if key != text_column:
                    result[key] = value if isinstance(value, list) else [value]
                    
            return result
        
        try:
            self.logger.info("Starting batch tokenization...")
            start_time = time.time()
            
            self.ray_dataset = self.ray_dataset.map_batches(
                tokenize_batch,
                batch_size=batch_size,
                num_cpus=1,
                batch_format="dict"
            )
            
            tokenize_time = time.time() - start_time
            self.logger.info(f"Tokenization completed in {tokenize_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            # Fallback to row-wise tokenization
            def tokenize_row(row):
                text = row[text_column]
                tokenized = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=truncation,
                    padding=padding,
                    return_tensors="pt"
                )
                
                result = {k: v.squeeze().tolist() for k, v in tokenized.items()}
                result.update({k: v for k, v in row.items() if k != text_column})
                return result
            
            self.ray_dataset = self.ray_dataset.map(tokenize_row, num_cpus=0.5)

    def sample_dataset(self, n: int, seed: Optional[int] = None):
        """
        Sample dataset efficiently for quick experimentation.
        
        Args:
            n: Number of samples
            seed: Random seed for reproducibility
        """
        if self.ray_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        try:
            if seed is not None:
                # Use Ray's random sampling with seed
                self.ray_dataset = self.ray_dataset.random_sample(
                    fraction=min(1.0, n / self.ray_dataset.count()),
                    seed=seed
                ).limit(n)
            else:
                # Simple head operation for faster sampling
                self.ray_dataset = self.ray_dataset.limit(n)
                
            self.logger.info(f"Sampled {n} examples from dataset")
            
        except Exception as e:
            self.logger.warning(f"Sampling failed: {e}")
            # Fallback to simple limit
            self.ray_dataset = self.ray_dataset.limit(n)

    def filter_dataset(self, filter_fn: callable, batch_size: int = 1000):
        """
        Efficient dataset filtering with batch processing.
        
        Args:
            filter_fn: Filter function that returns boolean
            batch_size: Batch size for filtering
        """
        if self.ray_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        try:
            # Use batch filtering for efficiency
            original_count = self.ray_dataset.count()
            self.logger.info(f"Filtering dataset with {original_count} rows...")
            
            self.ray_dataset = self.ray_dataset.filter(filter_fn)
            
            filtered_count = self.ray_dataset.count()
            self.logger.info(f"Filtered to {filtered_count} rows ({filtered_count/original_count*100:.1f}% remaining)")
            
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            raise

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {}
        
        if self.dataset is not None:
            try:
                info['huggingface_info'] = {
                    'features': str(self.dataset.features) if hasattr(self.dataset, 'features') else None,
                    'splits': list(self.dataset.keys()) if hasattr(self.dataset, 'keys') else None,
                    'description': getattr(self.dataset, 'description', None),
                    'citation': getattr(self.dataset, 'citation', None),
                }
            except Exception as e:
                self.logger.warning(f"Failed to get HuggingFace dataset info: {e}")
        
        if self.ray_dataset is not None:
            info.update(self.get_stats())
            
        return info

    def repartition_for_training(self, num_partitions: Optional[int] = None):
        """
        Repartition dataset optimally for distributed training.
        
        Args:
            num_partitions: Number of partitions (auto-detect if None)
        """
        if self.ray_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        if num_partitions is None:
            # Auto-detect based on cluster size
            try:
                cluster_resources = ray.cluster_resources()
                num_cpus = int(cluster_resources.get('CPU', 1))
                num_partitions = num_cpus * 2  # 2x overpartitioning
                
            except Exception as e:
                self.logger.warning(f"Failed to detect cluster resources: {e}")
                num_partitions = 8  # Default fallback
        
        try:
            self.logger.info(f"Repartitioning dataset to {num_partitions} partitions")
            self.ray_dataset = self.ray_dataset.repartition(num_partitions)
            
        except Exception as e:
            self.logger.warning(f"Repartitioning failed: {e}")
