import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass
import queue

import ray.data as rd
from ray.data import Dataset

# Kafka dependencies
try:
    from kafka import KafkaProducer, KafkaConsumer, TopicPartition
    from kafka.errors import KafkaError, KafkaTimeoutError, CommitFailedError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Mock classes for graceful degradation
    class KafkaProducer:
        pass
    class KafkaConsumer:
        pass
    class TopicPartition:
        pass
    class KafkaError(Exception):
        pass

from .base import BaseDataLoader


@dataclass
class StreamingMetrics:
    """Container for streaming performance metrics."""
    messages_consumed: int = 0
    messages_produced: int = 0
    bytes_consumed: int = 0
    bytes_produced: int = 0
    processing_latency_ms: float = 0.0
    consumer_lag: int = 0
    error_count: int = 0
    last_offset: Optional[int] = None
    throughput_msg_per_sec: float = 0.0
    start_time: float = 0.0


@dataclass
class KafkaMessage:
    """Container for Kafka message with metadata."""
    topic: str
    partition: int
    offset: int
    key: Optional[bytes]
    value: bytes
    timestamp: int
    headers: Dict[str, bytes]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Ray Data processing."""
        return {
            'topic': self.topic,
            'partition': self.partition,
            'offset': self.offset,
            'key': self.key.decode('utf-8') if self.key else None,
            'value': self.value.decode('utf-8') if self.value else None,
            'timestamp': self.timestamp,
            'headers': {k: v.decode('utf-8') for k, v in self.headers.items()}
        }


class KafkaStreamingDataLoader(BaseDataLoader):
    """
    High-performance Kafka streaming data loader with Ray Data integration.
    
    Features:
    - Real-time data streaming with configurable batching
    - Automatic error recovery and retry mechanisms
    - Backpressure handling and flow control
    - Multiple serialization format support
    - Performance monitoring and metrics collection
    - Distributed processing with Ray
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        consumer_group_id: Optional[str] = None,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False,
        batch_size: int = 1000,
        max_poll_interval_ms: int = 300000,
        session_timeout_ms: int = 30000,
        heartbeat_interval_ms: int = 3000,
        max_poll_records: int = 500,
        fetch_min_bytes: int = 1,
        fetch_max_wait_ms: int = 500,
        buffer_memory: int = 33554432,  # 32MB
        compression_type: str = "gzip",
        enable_monitoring: bool = True,
        monitoring_interval: float = 5.0,
        **kwargs
    ):
        """
        Initialize Kafka streaming data loader.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            consumer_group_id: Consumer group ID (auto-generated if None)
            auto_offset_reset: Offset reset strategy ('earliest', 'latest')
            enable_auto_commit: Enable automatic offset commits
            batch_size: Number of messages per batch for Ray processing
            max_poll_interval_ms: Maximum poll interval
            session_timeout_ms: Session timeout
            heartbeat_interval_ms: Heartbeat interval
            max_poll_records: Maximum records per poll
            fetch_min_bytes: Minimum fetch bytes
            fetch_max_wait_ms: Maximum fetch wait time
            buffer_memory: Producer buffer memory
            compression_type: Compression type for producer
            enable_monitoring: Enable performance monitoring
            monitoring_interval: Monitoring update interval
        """
        super().__init__(**kwargs)
        
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is required for Kafka streaming. "
                            "Install with: pip install kafka-python")
        
        # Kafka configuration
        self.bootstrap_servers = bootstrap_servers
        self.consumer_group_id = consumer_group_id or f"omniray-consumer-{int(time.time())}"
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.batch_size = batch_size
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': self.consumer_group_id,
            'auto_offset_reset': auto_offset_reset,
            'enable_auto_commit': enable_auto_commit,
            'max_poll_interval_ms': max_poll_interval_ms,
            'session_timeout_ms': session_timeout_ms,
            'heartbeat_interval_ms': heartbeat_interval_ms,
            'max_poll_records': max_poll_records,
            'fetch_min_bytes': fetch_min_bytes,
            'fetch_max_wait_ms': fetch_max_wait_ms,
            'value_deserializer': lambda x: x,  # Keep as bytes initially
            'key_deserializer': lambda x: x if x else None,
        }
        
        # Producer configuration
        self.producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'buffer_memory': buffer_memory,
            'compression_type': compression_type,
            'value_serializer': lambda x: x if isinstance(x, bytes) else str(x).encode('utf-8'),
            'key_serializer': lambda x: x if isinstance(x, bytes) else str(x).encode('utf-8') if x else None,
        }
        
        # Streaming state
        self.consumer = None
        self.producer = None
        self.is_streaming = False
        self.streaming_thread = None
        self.message_buffer = deque(maxlen=batch_size * 10)  # 10x batch size buffer
        self.processed_datasets = []
        
        # Serialization handlers
        self.deserializers = {
            'json': self._deserialize_json,
            'avro': self._deserialize_avro,
            'protobuf': self._deserialize_protobuf,
            'bytes': self._deserialize_bytes,
            'string': self._deserialize_string
        }
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.metrics = StreamingMetrics()
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_thread = None
        
        # Error handling
        self.max_retries = 3
        self.retry_backoff_ms = 1000
        self.error_queue = queue.Queue()
        
        # Flow control
        self.max_buffer_size = batch_size * 20
        self.processing_queue = queue.Queue(maxsize=self.max_buffer_size)
        
        self.logger.info(f"Kafka streaming loader initialized for servers: {bootstrap_servers}")

    def connect_consumer(
        self, 
        topics: List[str],
        partition_assignment: Optional[Dict[str, List[int]]] = None
    ):
        """
        Connect Kafka consumer to specified topics.
        
        Args:
            topics: List of topics to subscribe to
            partition_assignment: Manual partition assignment {topic: [partition_list]}
        """
        try:
            self.consumer = KafkaConsumer(**self.consumer_config)
            
            if partition_assignment:
                # Manual partition assignment
                partitions = []
                for topic, partition_list in partition_assignment.items():
                    for partition in partition_list:
                        partitions.append(TopicPartition(topic, partition))
                self.consumer.assign(partitions)
                self.logger.info(f"Manually assigned partitions: {partitions}")
            else:
                # Subscribe to topics
                self.consumer.subscribe(topics)
                self.logger.info(f"Subscribed to topics: {topics}")
            
            # Wait for assignment
            self.consumer.poll(timeout_ms=1000)
            assigned_partitions = self.consumer.assignment()
            self.logger.info(f"Assigned partitions: {assigned_partitions}")
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to connect Kafka consumer: {e}")
            return False

    def connect_producer(self):
        """Connect Kafka producer."""
        try:
            self.producer = KafkaProducer(**self.producer_config)
            self.logger.info("Kafka producer connected successfully")
            return True
            
        except KafkaError as e:
            self.logger.error(f"Failed to connect Kafka producer: {e}")
            return False

    def start_streaming(
        self,
        topics: List[str],
        data_format: str = "json",
        partition_assignment: Optional[Dict[str, List[int]]] = None,
        processing_function: Optional[Callable] = None
    ):
        """
        Start streaming data from Kafka topics.
        
        Args:
            topics: List of Kafka topics to consume
            data_format: Data serialization format ('json', 'avro', 'protobuf', 'bytes', 'string')
            partition_assignment: Manual partition assignment
            processing_function: Optional function to process each message
        """
        if self.is_streaming:
            self.logger.warning("Streaming already started")
            return
        
        # Connect consumer
        if not self.connect_consumer(topics, partition_assignment):
            raise RuntimeError("Failed to connect Kafka consumer")
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self._start_monitoring()
        
        # Initialize metrics
        self.metrics = StreamingMetrics()
        self.metrics.start_time = time.time()
        
        # Start streaming thread
        self.is_streaming = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_loop,
            args=(data_format, processing_function),
            daemon=True
        )
        self.streaming_thread.start()
        
        self.logger.info(f"Started streaming from topics: {topics}")

    def stop_streaming(self, timeout: float = 30.0):
        """
        Stop streaming and cleanup resources.
        
        Args:
            timeout: Timeout for graceful shutdown
        """
        if not self.is_streaming:
            return
        
        self.logger.info("Stopping Kafka streaming...")
        self.is_streaming = False
        
        # Wait for streaming thread to finish
        if self.streaming_thread:
            self.streaming_thread.join(timeout=timeout)
        
        # Stop monitoring
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Close connections
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        
        self.logger.info("Kafka streaming stopped")

    def _streaming_loop(self, data_format: str, processing_function: Optional[Callable]):
        """Main streaming loop running in separate thread."""
        deserializer = self.deserializers.get(data_format, self._deserialize_json)
        batch_messages = []
        last_batch_time = time.time()
        
        while self.is_streaming:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    # Check if we should process partial batch due to timeout
                    if batch_messages and (time.time() - last_batch_time) > 5.0:
                        self._process_message_batch(batch_messages, deserializer, processing_function)
                        batch_messages = []
                        last_batch_time = time.time()
                    continue
                
                # Process messages from all partitions
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Create Kafka message object
                            kafka_msg = KafkaMessage(
                                topic=message.topic,
                                partition=message.partition,
                                offset=message.offset,
                                key=message.key,
                                value=message.value,
                                timestamp=message.timestamp,
                                headers=dict(message.headers) if message.headers else {}
                            )
                            
                            batch_messages.append(kafka_msg)
                            
                            # Update metrics
                            self.metrics.messages_consumed += 1
                            self.metrics.bytes_consumed += len(message.value) if message.value else 0
                            self.metrics.last_offset = message.offset
                            
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
                            self.metrics.error_count += 1
                
                # Process batch when it reaches target size
                if len(batch_messages) >= self.batch_size:
                    self._process_message_batch(batch_messages, deserializer, processing_function)
                    batch_messages = []
                    last_batch_time = time.time()
                
                # Commit offsets if auto-commit is disabled
                if not self.enable_auto_commit:
                    try:
                        self.consumer.commit()
                    except CommitFailedError as e:
                        self.logger.warning(f"Offset commit failed: {e}")
                
            except KafkaError as e:
                self.logger.error(f"Kafka error in streaming loop: {e}")
                self._handle_kafka_error(e)
                
            except Exception as e:
                self.logger.error(f"Unexpected error in streaming loop: {e}")
                self.metrics.error_count += 1
        
        # Process remaining messages
        if batch_messages:
            self._process_message_batch(batch_messages, deserializer, processing_function)

    def _process_message_batch(
        self, 
        messages: List[KafkaMessage], 
        deserializer: Callable,
        processing_function: Optional[Callable]
    ):
        """Process a batch of messages and convert to Ray dataset."""
        try:
            start_time = time.time()
            
            # Deserialize messages
            processed_messages = []
            for msg in messages:
                try:
                    deserialized_data = deserializer(msg.value)
                    
                    # Add metadata
                    message_data = {
                        'data': deserialized_data,
                        'metadata': {
                            'topic': msg.topic,
                            'partition': msg.partition,
                            'offset': msg.offset,
                            'timestamp': msg.timestamp,
                            'key': msg.key.decode('utf-8') if msg.key else None
                        }
                    }
                    
                    # Apply custom processing function if provided
                    if processing_function:
                        message_data = processing_function(message_data)
                    
                    processed_messages.append(message_data)
                    
                except Exception as e:
                    self.logger.error(f"Error deserializing message: {e}")
                    self.metrics.error_count += 1
            
            if processed_messages:
                # Create Ray dataset from processed messages
                ray_dataset = rd.from_items(processed_messages)
                
                # Store for later access
                self.processed_datasets.append({
                    'dataset': ray_dataset,
                    'timestamp': time.time(),
                    'message_count': len(processed_messages),
                    'topics': list(set(msg.topic for msg in messages))
                })
                
                # Keep only recent datasets to manage memory
                if len(self.processed_datasets) > 100:
                    self.processed_datasets.pop(0)
                
                # Update processing latency metric
                processing_time = (time.time() - start_time) * 1000
                self.metrics.processing_latency_ms = processing_time
                
                self.logger.debug(f"Processed batch of {len(messages)} messages in {processing_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error processing message batch: {e}")
            self.metrics.error_count += 1

    def _start_monitoring(self):
        """Start performance monitoring thread."""
        def monitoring_loop():
            while self.is_streaming:
                try:
                    # Calculate throughput
                    if self.metrics.start_time > 0:
                        elapsed_time = time.time() - self.metrics.start_time
                        if elapsed_time > 0:
                            self.metrics.throughput_msg_per_sec = self.metrics.messages_consumed / elapsed_time
                    
                    # Calculate consumer lag (simplified)
                    if self.consumer:
                        try:
                            # Get consumer lag for assigned partitions
                            assigned_partitions = self.consumer.assignment()
                            total_lag = 0
                            
                            for tp in assigned_partitions:
                                # Get high water mark and current position
                                high_water_mark = self.consumer.highwater(tp)
                                current_position = self.consumer.position(tp)
                                
                                if high_water_mark is not None and current_position is not None:
                                    lag = high_water_mark - current_position
                                    total_lag += max(0, lag)
                            
                            self.metrics.consumer_lag = total_lag
                            
                        except Exception as e:
                            self.logger.debug(f"Could not calculate consumer lag: {e}")
                    
                    # Store metrics snapshot
                    metrics_snapshot = StreamingMetrics(
                        messages_consumed=self.metrics.messages_consumed,
                        messages_produced=self.metrics.messages_produced,
                        bytes_consumed=self.metrics.bytes_consumed,
                        bytes_produced=self.metrics.bytes_produced,
                        processing_latency_ms=self.metrics.processing_latency_ms,
                        consumer_lag=self.metrics.consumer_lag,
                        error_count=self.metrics.error_count,
                        last_offset=self.metrics.last_offset,
                        throughput_msg_per_sec=self.metrics.throughput_msg_per_sec,
                        start_time=self.metrics.start_time
                    )
                    
                    self.metrics_history.append(metrics_snapshot)
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _handle_kafka_error(self, error: KafkaError):
        """Handle Kafka errors with retry logic."""
        self.logger.error(f"Kafka error: {error}")
        
        # Implement exponential backoff retry
        retry_count = 0
        while retry_count < self.max_retries and self.is_streaming:
            time.sleep(self.retry_backoff_ms / 1000 * (2 ** retry_count))
            
            try:
                # Try to reconnect consumer
                if self.consumer:
                    self.consumer.close()
                
                # Recreate consumer with same configuration
                self.consumer = KafkaConsumer(**self.consumer_config)
                self.logger.info("Successfully reconnected Kafka consumer")
                break
                
            except KafkaError as e:
                retry_count += 1
                self.logger.warning(f"Retry {retry_count} failed: {e}")
        
        if retry_count >= self.max_retries:
            self.logger.error("Max retries exceeded, stopping streaming")
            self.is_streaming = False

    # Serialization handlers
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize JSON data."""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"JSON deserialization error: {e}")
            return {'raw_data': data.hex(), 'error': str(e)}

    def _deserialize_avro(self, data: bytes) -> Any:
        """Deserialize Avro data (requires fastavro)."""
        try:
            import fastavro
            import io
            
            bytes_reader = io.BytesIO(data)
            reader = fastavro.reader(bytes_reader)
            return next(reader)
            
        except ImportError:
            self.logger.error("fastavro not installed for Avro deserialization")
            return {'raw_data': data.hex(), 'error': 'fastavro not installed'}
        except Exception as e:
            self.logger.error(f"Avro deserialization error: {e}")
            return {'raw_data': data.hex(), 'error': str(e)}

    def _deserialize_protobuf(self, data: bytes) -> Any:
        """Deserialize Protocol Buffers data (requires protobuf)."""
        try:
            # This is a placeholder - actual implementation would need
            # specific protobuf message classes
            return {'raw_data': data.hex(), 'format': 'protobuf'}
            
        except Exception as e:
            self.logger.error(f"Protobuf deserialization error: {e}")
            return {'raw_data': data.hex(), 'error': str(e)}

    def _deserialize_bytes(self, data: bytes) -> Any:
        """Keep data as bytes."""
        return {'data': data.hex(), 'size': len(data)}

    def _deserialize_string(self, data: bytes) -> Any:
        """Deserialize as UTF-8 string."""
        try:
            return {'text': data.decode('utf-8')}
        except UnicodeDecodeError as e:
            return {'raw_data': data.hex(), 'error': str(e)}

    def produce_message(
        self, 
        topic: str, 
        message: Any,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        data_format: str = "json"
    ) -> bool:
        """
        Produce message to Kafka topic.
        
        Args:
            topic: Target topic
            message: Message data
            key: Message key
            partition: Target partition
            headers: Message headers
            data_format: Serialization format
            
        Returns:
            Success status
        """
        if not self.producer:
            if not self.connect_producer():
                return False
        
        try:
            # Serialize message based on format
            if data_format == "json":
                serialized_message = json.dumps(message).encode('utf-8')
            elif data_format == "string":
                serialized_message = str(message).encode('utf-8')
            elif data_format == "bytes":
                serialized_message = message if isinstance(message, bytes) else str(message).encode('utf-8')
            else:
                serialized_message = json.dumps(message).encode('utf-8')
            
            # Prepare headers
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=serialized_message,
                key=key,
                partition=partition,
                headers=kafka_headers
            )
            
            # Wait for result (with timeout)
            result = future.get(timeout=10)
            
            # Update metrics
            self.metrics.messages_produced += 1
            self.metrics.bytes_produced += len(serialized_message)
            
            self.logger.debug(f"Produced message to {topic}: offset {result.offset}")
            return True
            
        except KafkaTimeoutError:
            self.logger.error("Kafka produce timeout")
            return False
        except KafkaError as e:
            self.logger.error(f"Kafka produce error: {e}")
            return False

    def get_latest_dataset(self) -> Optional[Dataset]:
        """Get the most recent processed dataset."""
        if not self.processed_datasets:
            return None
        return self.processed_datasets[-1]['dataset']

    def get_datasets_by_timerange(
        self, 
        start_time: float, 
        end_time: float
    ) -> List[Dataset]:
        """
        Get datasets within specified time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of Ray datasets
        """
        filtered_datasets = []
        for dataset_info in self.processed_datasets:
            if start_time <= dataset_info['timestamp'] <= end_time:
                filtered_datasets.append(dataset_info['dataset'])
        return filtered_datasets

    def merge_recent_datasets(self, count: int = 10) -> Optional[Dataset]:
        """
        Merge recent datasets into a single dataset.
        
        Args:
            count: Number of recent datasets to merge
            
        Returns:
            Merged Ray dataset
        """
        if not self.processed_datasets:
            return None
        
        recent_datasets = [
            info['dataset'] for info in self.processed_datasets[-count:]
        ]
        
        if not recent_datasets:
            return None
        
        # Union all datasets
        merged_dataset = recent_datasets[0]
        for dataset in recent_datasets[1:]:
            merged_dataset = merged_dataset.union(dataset)
        
        return merged_dataset

    def apply_streaming_transformation(
        self,
        transform_function: Callable,
        batch_size: int = 1000,
        window_size: int = 10
    ):
        """
        Apply transformation to streaming data with windowing.
        
        Args:
            transform_function: Function to apply to each batch
            batch_size: Batch size for processing
            window_size: Number of recent datasets to consider
        """
        try:
            # Get recent datasets
            merged_dataset = self.merge_recent_datasets(window_size)
            
            if merged_dataset is None:
                self.logger.warning("No datasets available for transformation")
                return
            
            # Apply transformation
            transformed_dataset = merged_dataset.map_batches(
                transform_function,
                batch_size=batch_size,
                batch_format="dict"
            )
            
            # Store transformed dataset
            self.processed_datasets.append({
                'dataset': transformed_dataset,
                'timestamp': time.time(),
                'message_count': merged_dataset.count(),
                'topics': ['transformed'],
                'transformation_applied': True
            })
            
            self.logger.info("Applied streaming transformation successfully")
            
        except Exception as e:
            self.logger.error(f"Error applying streaming transformation: {e}")

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics."""
        return {
            'current_metrics': {
                'messages_consumed': self.metrics.messages_consumed,
                'messages_produced': self.metrics.messages_produced,
                'bytes_consumed': self.metrics.bytes_consumed,
                'bytes_produced': self.metrics.bytes_produced,
                'processing_latency_ms': self.metrics.processing_latency_ms,
                'consumer_lag': self.metrics.consumer_lag,
                'error_count': self.metrics.error_count,
                'throughput_msg_per_sec': self.metrics.throughput_msg_per_sec,
                'uptime_seconds': time.time() - self.metrics.start_time if self.metrics.start_time > 0 else 0
            },
            'streaming_status': {
                'is_streaming': self.is_streaming,
                'consumer_connected': self.consumer is not None,
                'producer_connected': self.producer is not None,
                'processed_datasets_count': len(self.processed_datasets)
            },
            'configuration': {
                'bootstrap_servers': self.bootstrap_servers,
                'consumer_group_id': self.consumer_group_id,
                'batch_size': self.batch_size,
                'auto_offset_reset': self.auto_offset_reset
            }
        }

    def export_streaming_metrics(self, output_path: str):
        """Export streaming metrics to file."""
        try:
            metrics_data = {
                'export_timestamp': time.time(),
                'current_metrics': self.get_streaming_metrics(),
                'metrics_history': [
                    {
                        'timestamp': time.time(),
                        'messages_consumed': m.messages_consumed,
                        'throughput_msg_per_sec': m.throughput_msg_per_sec,
                        'processing_latency_ms': m.processing_latency_ms,
                        'consumer_lag': m.consumer_lag,
                        'error_count': m.error_count
                    }
                    for m in self.metrics_history
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.info(f"Streaming metrics exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export streaming metrics: {e}")

    def load(self, path: str, **kwargs):
        """
        Load method for compatibility with base class.
        For Kafka loader, use start_streaming() instead.
        """
        raise NotImplementedError(
            "Kafka loader uses start_streaming() instead of load(). "
            "Use start_streaming(topics=['topic1', 'topic2']) to begin streaming."
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_streaming()
