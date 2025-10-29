# STT (Speech-to-Text) Pipeline Guide

OmniRay's STT pipeline provides scalable, Ray-based speech-to-text transcription using Faster Whisper.

## Overview

The STT pipeline uses Ray Data to process audio in chunks, distributing the workload across multiple workers for efficient transcription of long audio files.

## Architecture

```
Audio/Video File
    ↓
[Extract Audio with FFmpeg]
    ↓
Audio Array (float32, 16kHz)
    ↓
[Split into Chunks (e.g., 30s each)]
    ↓
[Ray Data: from_items + flat_map]
    ↓
Ray Dataset of Audio Chunks
    ↓
[Ray: map_batches]
    ↓
[Worker 1]    [Worker 2]    [Worker 3]
Load Model    Load Model    Load Model
    ↓             ↓             ↓
Transcribe    Transcribe    Transcribe
    ↓             ↓             ↓
[Collect Results]
    ↓
Transcription with Timestamps
    ↓
[Export: JSON/TXT/SRT/VTT]
```

## Ray Usage in STT Pipeline

### 1. Audio Chunking (`audio_loader.py`)

```python
# Load audio and create Ray Dataset
ds = ray.data.from_items([{"audio_path": audio_path}])

# Split audio into chunks using flat_map
ds = ds.flat_map(lambda _: self._load_and_chunk_audio())

# Repartition for parallel processing
ds = ds.repartition(num_blocks)
```

**Key Points:**
- Audio is extracted using FFmpeg (supports video files too!)
- Long audio is split into manageable chunks (default: 30 seconds)
- Each chunk has metadata: start_time, end_time, sample_rate

### 2. Distributed Transcription (`pipeline.py`)

```python
# Run transcription using Ray map_batches
results = dataset.map_batches(
    transcribe_fn,
    batch_size=audio_config.batch_size,
    num_gpus=ray_options.get("num_gpus", 0),
)
```

**Benefits:**
- **Parallel Processing**: Multiple chunks transcribed simultaneously
- **GPU Acceleration**: Each worker can use GPU for faster inference
- **Memory Efficient**: Processes in batches, not entire file at once
- **Scalable**: Works on single machine or Ray cluster

## Faster Whisper Configuration

### Model Sizes

| Model | Parameters | VRAM (FP16) | Speed | Accuracy |
|-------|-----------|-------------|-------|----------|
| tiny  | 39M       | ~1GB        | Fast  | Good     |
| base  | 74M       | ~1GB        | Fast  | Better   |
| small | 244M      | ~2GB        | Medium| Good     |
| medium| 769M      | ~5GB        | Slower| Better   |
| large-v3| 1550M   | ~10GB       | Slow  | Best     |

### Compute Types

| Type | Description | GPU Support | Recommended For |
|------|-------------|-------------|-----------------|
| `default` | Auto-select | Yes | General use |
| `int8` | 8-bit quantization | Yes | Fast, less VRAM |
| `float16` | 16-bit precision | Yes | Good balance |
| `float32` | Full precision | Yes/CPU | Best quality |

### Language Support

Faster Whisper supports 90+ languages. Common codes:
- English: `en`
- Korean: `ko`
- Japanese: `ja`
- Chinese: `zh`
- Spanish: `es`
- French: `fr`
- German: `de`

Set `language=None` for automatic detection.

## Usage Examples

### Basic Transcription

```python
from omniray import (
    AudioConfig,
    FasterWhisperConfig,
    STTInferenceConfig,
    STTModelType,
    STTInferencePipeline,
)

config = STTInferenceConfig(
    model_type=STTModelType.FASTER_WHISPER,
    audio_config=AudioConfig(
        audio_path="audio.mp3",
        chunk_length_s=30.0,
        batch_size=8,
    ),
    faster_whisper_config=FasterWhisperConfig(
        model_size="base",
        device="auto",
        language="en",
    ),
)

pipeline = STTInferencePipeline(config)
results = pipeline.run()
text = pipeline.get_full_transcription()
```

### GPU Acceleration

```python
config = STTInferenceConfig(
    model_type=STTModelType.FASTER_WHISPER,
    audio_config=AudioConfig(audio_path="audio.mp3"),
    faster_whisper_config=FasterWhisperConfig(
        model_size="large-v3",
        device="cuda",
        compute_type="float16",  # Faster on GPU
    ),
    ray_options={
        "num_gpus": 1,  # Use 1 GPU
    },
)
```

### Multi-language

```python
# Auto-detect language
config = FasterWhisperConfig(
    model_size="medium",
    language=None,  # Auto-detect
)

# Or specify language
config = FasterWhisperConfig(
    model_size="medium",
    language="ko",  # Korean
)
```

### Translation to English

```python
config = FasterWhisperConfig(
    model_size="medium",
    language="ko",  # Source: Korean
    task="translate",  # Translate to English!
)
```

### Export Formats

```python
pipeline = STTInferencePipeline(config)
results = pipeline.run()

# JSON (full details with timestamps)
pipeline.save_results("transcription.json")

# Plain text
pipeline.save_results("transcription.txt")

# SRT subtitles (for video players)
pipeline.save_results("transcription.srt")

# WebVTT subtitles (for web)
pipeline.save_results("transcription.vtt")
```

## Performance Tips

### 1. Chunk Length

```python
AudioConfig(
    chunk_length_s=30.0,  # Default: good balance
    # chunk_length_s=15.0,  # Shorter: more parallel, more overhead
    # chunk_length_s=60.0,  # Longer: less overhead, less parallel
)
```

### 2. Batch Size

```python
AudioConfig(
    batch_size=8,  # Process 8 chunks at once
    # Increase for more parallelism (if you have resources)
    # Decrease if running out of memory
)
```

### 3. VAD (Voice Activity Detection)

```python
FasterWhisperConfig(
    vad_filter=True,  # Skip silent parts (faster!)
    vad_parameters={
        "threshold": 0.5,  # Voice detection threshold
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 2000,
    }
)
```

### 4. Multi-GPU

```python
# Option 1: Single GPU per worker
ray_options={"num_gpus": 1}

# Option 2: Share GPU across workers
ray_options={"num_gpus": 0.5}  # 2 workers per GPU

# Option 3: Multiple GPUs
ray_options={"num_gpus": 2}  # Use 2 GPUs
```

## Extending to Other STT Models

To add a new STT model (e.g., OpenAI Whisper, Wav2Vec2):

1. **Add model type** in `config/schemas.py`:
```python
class STTModelType(str, Enum):
    FASTER_WHISPER = "faster_whisper"
    OPENAI_WHISPER = "openai_whisper"  # New!
```

2. **Create model wrapper** in `stt/openai_whisper.py`:
```python
from omniray.stt.base import BaseSTTModel

class OpenAIWhisperModel(BaseSTTModel):
    def load_model(self):
        # Load OpenAI Whisper
        pass

    def transcribe(self, audio, sample_rate):
        # Run transcription
        pass
```

3. **Update pipeline** in `stt/pipeline.py`:
```python
if model_type == STTModelType.OPENAI_WHISPER:
    self.model = OpenAIWhisperModel(config)
```

The Ray infrastructure (audio loading, chunking, distributed processing) remains the same!

## Output Format

### JSON Output

```json
[
  {
    "audio": [...],
    "chunk_idx": 0,
    "start_time": 0.0,
    "end_time": 30.0,
    "duration": 30.0,
    "transcription": {
      "text": "This is the transcribed text.",
      "segments": [
        {
          "id": 0,
          "start": 0.0,
          "end": 2.5,
          "text": "This is",
          "words": [...]
        }
      ],
      "language": "en",
      "num_segments": 10
    }
  }
]
```

### SRT Format

```
1
00:00:00,000 --> 00:00:02,500
This is the transcribed text.

2
00:00:02,500 --> 00:00:05,000
More transcription here.
```

## Troubleshooting

### FFmpeg Not Found

Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

### CUDA Out of Memory

1. Use smaller model: `model_size="small"`
2. Use int8 quantization: `compute_type="int8"`
3. Reduce batch size: `batch_size=4`
4. Use CPU: `device="cpu"`

### Slow Performance

1. Use GPU: `device="cuda"`, `num_gpus=1`
2. Use int8: `compute_type="int8"`
3. Enable VAD: `vad_filter=True`
4. Increase batch size: `batch_size=16`
5. Use smaller model for quick drafts: `model_size="tiny"`
