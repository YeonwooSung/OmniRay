import pandas as pd
import ray.data as rd
from typing import Optional, List, Dict, Any
import time
from pathlib import Path
import pyarrow.parquet as pq

from .base import BaseDataLoader


class PandasDataLoader(BaseDataLoader):
    """
    Optimized Pandas data loader with format-specific optimizations,
    intelligent chunking, and distributed processing capabilities.
    """
    
    def __init__(
        self,
        chunk_size: int = 10000,
        enable_arrow: bool = True,
        optimize_dtypes: bool = True,
        parallel_reading: bool = True
    ):
        """
        Initialize Pandas data loader with advanced optimizations.
        
        Args:
            chunk_size: Default chunk size for large files
            enable_arrow: Use PyArrow backend for better performance
            optimize_dtypes: Automatically optimize data types
            parallel_reading: Enable parallel file reading
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.enable_arrow = enable_arrow
        self.optimize_dtypes = optimize_dtypes
        self.parallel_reading = parallel_reading
        self.dataset = None
        self._file_info = {}

    def load(
        self, 
        path: str, 
        file_format: Optional[str] = None,
        optimize_blocks: bool = True,
        cache_dataset: bool = True,
        **kwargs
    ):
        """
        Load file with format-specific optimizations.
        
        Args:
            path: File path
            file_format: Override file format detection
            optimize_blocks: Optimize Ray dataset blocks
            cache_dataset: Enable caching
            **kwargs: Format-specific arguments
        """
        cache_key = self._generate_cache_key(path, file_format=file_format, **kwargs)
        
        # Try cache first
        if cache_dataset:
            cached_dataset = self._load_cached_dataset(cache_key)
            if cached_dataset is not None:
                self.ray_dataset = cached_dataset
                return

        try:
            self.logger.info(f"Loading file: {path}")
            start_time = time.time()
            
            # Get file information
            file_path = Path(path)
            file_size = file_path.stat().st_size
            detected_format = file_format or file_path.suffix.lower()
            
            self._file_info = {
                'path': path,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024),
                'format': detected_format
            }
            
            # Choose loading strategy based on file size and format
            if file_size > 100 * 1024 * 1024:  # 100MB threshold
                self._load_large_file(path, detected_format, **kwargs)
            else:
                self._load_standard(path, detected_format, **kwargs)
            
            load_time = time.time() - start_time
            self.logger.info(f"File loaded in {load_time:.2f}s")
            
            # Convert to Ray dataset
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
            self.logger.error(f"Failed to load file {path}: {e}")
            raise

    def _load_large_file(self, path: str, file_format: str, **kwargs):
        """Load large files with chunking and streaming."""
        if file_format == '.csv':
            self._load_large_csv(path, **kwargs)
        elif file_format == '.parquet':
            self._load_large_parquet(path, **kwargs)
        elif file_format in ['.xlsx', '.xls']:
            self._load_large_excel(path, **kwargs)
        else:
            self.logger.warning(f"Unsupported format for large file loading: {file_format}")
            self._load_standard(path, file_format, **kwargs)

    def _load_large_csv(self, path: str, **kwargs):
        """Load large CSV files with optimized chunking."""
        try:
            # Configure optimal CSV reading
            csv_kwargs = {
                'engine': 'c',  # Use C engine for speed
                'low_memory': False,
                'dtype_backend': 'pyarrow' if self.enable_arrow else 'numpy',
                **kwargs
            }
            
            if self.parallel_reading:
                # Use Ray's native CSV reading for parallel processing
                self.logger.info("Using Ray's parallel CSV reading")
                self.ray_dataset = rd.read_csv(
                    path, 
                    override_num_blocks=min(64, max(4, self._file_info['size_mb'] // 50)),
                    **csv_kwargs
                )
                self._file_info['loading_method'] = 'ray_parallel_csv'
                return
            else:
                # Use chunked reading
                self.logger.info(f"Loading CSV in chunks of {self.chunk_size}")
                chunks = []
                
                for chunk in pd.read_csv(path, chunksize=self.chunk_size, **csv_kwargs):
                    if self.optimize_dtypes:
                        chunk = self._optimize_dtypes(chunk)
                    chunks.append(chunk)
                
                self.dataset = pd.concat(chunks, ignore_index=True)
                self._file_info['loading_method'] = 'chunked_csv'
                self._file_info['num_chunks'] = len(chunks)
                
        except Exception as e:
            self.logger.warning(f"Optimized CSV loading failed: {e}")
            self.dataset = pd.read_csv(path, **kwargs)
            self._file_info['loading_method'] = 'standard_csv'

    def _load_large_parquet(self, path: str, **kwargs):
        """Load large Parquet files with PyArrow optimization."""
        try:
            if self.parallel_reading:
                # Use Ray's optimized Parquet reading
                self.logger.info("Using Ray's parallel Parquet reading")
                self.ray_dataset = rd.read_parquet(
                    path,
                    override_num_blocks=min(64, max(4, self._file_info['size_mb'] // 100)),
                    **kwargs
                )
                self._file_info['loading_method'] = 'ray_parallel_parquet'
                return
            else:
                # Use PyArrow for efficient reading
                if self.enable_arrow:
                    self.logger.info("Using PyArrow for Parquet reading")
                    parquet_file = pq.ParquetFile(path)
                    
                    # Read in batches if file is very large
                    if self._file_info['size_mb'] > 500:  # 500MB threshold
                        batches = []
                        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                            df_batch = batch.to_pandas()
                            batches.append(df_batch)
                        self.dataset = pd.concat(batches, ignore_index=True)
                        self._file_info['num_batches'] = len(batches)
                    else:
                        self.dataset = parquet_file.read().to_pandas()
                    
                    self._file_info['loading_method'] = 'pyarrow_parquet'
                else:
                    self.dataset = pd.read_parquet(path, **kwargs)
                    self._file_info['loading_method'] = 'pandas_parquet'
                    
        except Exception as e:
            self.logger.warning(f"Optimized Parquet loading failed: {e}")
            self.dataset = pd.read_parquet(path, **kwargs)
            self._file_info['loading_method'] = 'standard_parquet'

    def _load_large_excel(self, path: str, **kwargs):
        """Load large Excel files with memory optimization."""
        try:
            # Excel files need special handling for large sizes
            self.logger.info("Loading Excel file (large file optimization)")
            
            # Use openpyxl engine for better memory handling
            excel_kwargs = {
                'engine': 'openpyxl',
                **kwargs
            }
            
            # Try to read sheet by sheet if multiple sheets
            excel_file = pd.ExcelFile(path, **excel_kwargs)
            
            if len(excel_file.sheet_names) > 1:
                self.logger.info(f"Found {len(excel_file.sheet_names)} sheets")
                
                # Read each sheet and combine
                dataframes = []
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    df['_sheet_name'] = sheet_name  # Add sheet identifier
                    dataframes.append(df)
                
                self.dataset = pd.concat(dataframes, ignore_index=True)
                self._file_info['num_sheets'] = len(excel_file.sheet_names)
            else:
                self.dataset = pd.read_excel(path, **excel_kwargs)
                self._file_info['num_sheets'] = 1
            
            self._file_info['loading_method'] = 'optimized_excel'
            
        except Exception as e:
            self.logger.warning(f"Optimized Excel loading failed: {e}")
            self.dataset = pd.read_excel(path, **kwargs)
            self._file_info['loading_method'] = 'standard_excel'

    def _load_standard(self, path: str, file_format: str, **kwargs):
        """Standard loading for smaller files."""
        if file_format == '.csv':
            read_kwargs = {
                'dtype_backend': 'pyarrow' if self.enable_arrow else 'numpy',
                **kwargs
            }
            self.dataset = pd.read_csv(path, **read_kwargs)
        elif file_format == '.parquet':
            self.dataset = pd.read_parquet(path, **kwargs)
        elif file_format in ['.xlsx', '.xls']:
            self.dataset = pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        self._file_info['loading_method'] = f'standard_{file_format[1:]}'

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        if not self.optimize_dtypes:
            return df
        
        try:
            optimized = df.copy()
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= 0:  # Unsigned integers
                    if col_max < 255:
                        optimized[col] = df[col].astype('uint8')
                    elif col_max < 65535:
                        optimized[col] = df[col].astype('uint16')
                    elif col_max < 4294967295:
                        optimized[col] = df[col].astype('uint32')
                else:  # Signed integers
                    if col_min > -128 and col_max < 127:
                        optimized[col] = df[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        optimized[col] = df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        optimized[col] = df[col].astype('int32')
            
            # Optimize float columns
            for col in df.select_dtypes(include=['float64']).columns:
                optimized[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns to category where beneficial
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # Less than 50% unique values
                    optimized[col] = df[col].astype('category')
            
            memory_before = df.memory_usage(deep=True).sum()
            memory_after = optimized.memory_usage(deep=True).sum()
            reduction = (memory_before - memory_after) / memory_before * 100
            
            self.logger.info(f"Memory optimization: {reduction:.1f}% reduction")
            self._file_info['memory_optimization'] = {
                'before_mb': memory_before / (1024 * 1024),
                'after_mb': memory_after / (1024 * 1024),
                'reduction_percent': reduction
            }
            
            return optimized
            
        except Exception as e:
            self.logger.warning(f"Dtype optimization failed: {e}")
            return df

    def _convert_to_ray_dataset(self, optimize_blocks: bool):
        """Convert pandas DataFrame to Ray dataset with optimizations."""
        if self.dataset is None and self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        # If already converted via Ray reading, skip
        if self.ray_dataset is not None:
            return
        
        try:
            if optimize_blocks and len(self.dataset) > 10000:
                # Calculate optimal number of blocks
                target_rows_per_block = max(1000, min(10000, len(self.dataset) // 100))
                optimal_blocks = max(1, len(self.dataset) // target_rows_per_block)
                
                self.logger.info(f"Creating {optimal_blocks} blocks with ~{target_rows_per_block} rows each")
                self.ray_dataset = rd.from_pandas(self.dataset, override_num_blocks=optimal_blocks)
                
            else:
                self.ray_dataset = rd.from_pandas(self.dataset)
                
            self._file_info['ray_conversion'] = {
                'num_blocks': self.ray_dataset.num_blocks(),
                'rows_per_block': len(self.dataset) // self.ray_dataset.num_blocks()
            }
            
        except Exception as e:
            self.logger.warning(f"Optimized Ray conversion failed: {e}")
            self.ray_dataset = rd.from_pandas(self.dataset)

    def load_from_dataframe(
        self, 
        dataframe: pd.DataFrame,
        optimize_dtypes: bool = True,
        optimize_blocks: bool = True
    ):
        """
        Load from existing DataFrame with optimizations.
        
        Args:
            dataframe: Input DataFrame
            optimize_dtypes: Apply dtype optimizations
            optimize_blocks: Optimize Ray dataset blocks
        """
        try:
            self.logger.info(f"Loading DataFrame: shape={dataframe.shape}")
            
            # Store DataFrame info
            self._file_info = {
                'shape': dataframe.shape,
                'columns': list(dataframe.columns),
                'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                'memory_mb': dataframe.memory_usage(deep=True).sum() / (1024 * 1024),
                'loading_method': 'from_dataframe'
            }
            
            if optimize_dtypes:
                self.dataset = self._optimize_dtypes(dataframe)
            else:
                self.dataset = dataframe.copy()
            
            # Convert to Ray dataset
            self._convert_to_ray_dataset(optimize_blocks)
            
        except Exception as e:
            self.logger.error(f"Failed to load from DataFrame: {e}")
            raise

    def apply_preprocessing(
        self, 
        preprocessing_steps: List[Dict[str, Any]],
        batch_size: int = 10000
    ):
        """
        Apply preprocessing steps efficiently using batch processing.
        
        Args:
            preprocessing_steps: List of preprocessing configurations
            batch_size: Batch size for processing
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        for i, step in enumerate(preprocessing_steps):
            step_type = step.get('type')
            self.logger.info(f"Applying preprocessing step {i+1}/{len(preprocessing_steps)}: {step_type}")
            
            try:
                if step_type == 'drop_columns':
                    columns_to_drop = step['columns']
                    self.ray_dataset = self.ray_dataset.drop_columns(columns_to_drop)
                    
                elif step_type == 'fill_na':
                    fill_value = step.get('value', 0)
                    method = step.get('method', 'constant')
                    
                    def fill_na_batch(batch_df):
                        if method == 'constant':
                            return batch_df.fillna(fill_value)
                        elif method == 'forward':
                            return batch_df.fillna(method='ffill')
                        elif method == 'backward':
                            return batch_df.fillna(method='bfill')
                        else:
                            return batch_df.fillna(fill_value)
                    
                    self.ray_dataset = self.ray_dataset.map_batches(
                        fill_na_batch,
                        batch_size=batch_size,
                        batch_format="pandas"
                    )
                    
                elif step_type == 'normalize':
                    columns = step.get('columns', [])
                    method = step.get('method', 'minmax')
                    
                    # Calculate statistics first
                    stats = self._calculate_column_stats(columns)
                    
                    def normalize_batch(batch_df):
                        for col in columns:
                            if col in batch_df.columns:
                                if method == 'minmax':
                                    col_min = stats[col]['min']
                                    col_max = stats[col]['max']
                                    batch_df[col] = (batch_df[col] - col_min) / (col_max - col_min)
                                elif method == 'zscore':
                                    col_mean = stats[col]['mean']
                                    col_std = stats[col]['std']
                                    batch_df[col] = (batch_df[col] - col_mean) / col_std
                        return batch_df
                    
                    self.ray_dataset = self.ray_dataset.map_batches(
                        normalize_batch,
                        batch_size=batch_size,
                        batch_format="pandas"
                    )
                    
                elif step_type == 'custom':
                    custom_func = step['function']
                    self.ray_dataset = self.ray_dataset.map_batches(
                        custom_func,
                        batch_size=batch_size,
                        batch_format="pandas"
                    )
                
            except Exception as e:
                self.logger.error(f"Preprocessing step {i+1} failed: {e}")
                raise

    def _calculate_column_stats(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for specified columns."""
        stats = {}
        
        try:
            for col in columns:
                col_stats = {}
                
                # Use Ray's aggregation functions
                col_data = self.ray_dataset.select_columns([col])
                
                col_stats['min'] = col_data.min(col)
                col_stats['max'] = col_data.max(col)
                col_stats['mean'] = col_data.mean(col)
                col_stats['std'] = col_data.std(col)
                
                stats[col] = col_stats
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate stats, using sample-based approach: {e}")
            # Fallback: use sample for estimation
            sample_data = self.ray_dataset.take(1000)
            sample_df = pd.DataFrame(sample_data)
            
            for col in columns:
                if col in sample_df.columns:
                    stats[col] = {
                        'min': float(sample_df[col].min()),
                        'max': float(sample_df[col].max()),
                        'mean': float(sample_df[col].mean()),
                        'std': float(sample_df[col].std())
                    }
        
        return stats

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        info = {'file_info': self._file_info}
        
        if self.ray_dataset is not None:
            info.update(self.get_stats())
            
            # Add pandas-specific info
            try:
                sample = self.ray_dataset.take(5)
                sample_df = pd.DataFrame(sample)
                
                info['pandas_info'] = {
                    'sample_shape': sample_df.shape,
                    'columns': list(sample_df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                    'null_counts': sample_df.isnull().sum().to_dict(),
                    'memory_usage': sample_df.memory_usage(deep=True).to_dict()
                }
            except Exception as e:
                self.logger.warning(f"Failed to get pandas info: {e}")
                
        return info

    def export_to_format(
        self, 
        output_path: str, 
        format: str = 'parquet',
        **kwargs
    ):
        """
        Export dataset to various formats efficiently.
        
        Args:
            output_path: Output file path
            format: Output format ('parquet', 'csv', 'json')
            **kwargs: Format-specific arguments
        """
        if self.ray_dataset is None:
            raise ValueError("No dataset loaded")
        
        try:
            self.logger.info(f"Exporting dataset to {format}: {output_path}")
            
            if format.lower() == 'parquet':
                self.ray_dataset.write_parquet(output_path, **kwargs)
            elif format.lower() == 'csv':
                self.ray_dataset.write_csv(output_path, **kwargs)
            elif format.lower() == 'json':
                self.ray_dataset.write_json(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            self.logger.info(f"Export completed: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise
