"""Example: STT (Speech-to-Text) using Faster Whisper."""

import logging

import ray

from omniray.config import AudioConfig, FasterWhisperConfig, STTInferenceConfig, STTModelType
from omniray.stt import STTInferencePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run STT inference on an audio/video file using Faster Whisper."""
    # Configure audio input
    audio_config = AudioConfig(
        audio_path="path/to/your/audio.mp3",  # Change this to your audio/video path
        chunk_length_s=30.0,  # Process in 30-second chunks
        batch_size=8,  # Process 8 chunks in parallel
        max_duration_s=None,  # Process entire file (set to a number for testing)
    )

    # Configure Faster Whisper model
    whisper_config = FasterWhisperConfig(
        model_size="base",  # Options: tiny, base, small, medium, large-v2, large-v3
        device="auto",  # Auto-detect GPU/CPU
        compute_type="default",  # Options: int8, float16, float32
        language=None,  # Auto-detect language (or specify: 'en', 'ko', etc.)
        task="transcribe",  # 'transcribe' or 'translate' (to English)
        beam_size=5,
        vad_filter=True,  # Use Voice Activity Detection to filter silence
    )

    # Configure STT pipeline
    stt_config = STTInferenceConfig(
        model_type=STTModelType.FASTER_WHISPER,
        audio_config=audio_config,
        faster_whisper_config=whisper_config,
        ray_options={
            "num_cpus": 4,
            "num_gpus": 0,  # Set to 1 if you have GPU (much faster!)
        },
        output_path="results/transcription.json",
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create and run pipeline
        pipeline = STTInferencePipeline(stt_config)
        logger.info("Starting STT pipeline with Faster Whisper...")

        results = pipeline.run()

        # Get summary
        summary = pipeline.get_summary()
        logger.info(f"Pipeline completed. Summary: {summary}")

        # Get full transcription
        full_text = pipeline.get_full_transcription()
        logger.info("\n" + "=" * 60)
        logger.info("FULL TRANSCRIPTION:")
        logger.info("=" * 60)
        logger.info(full_text)
        logger.info("=" * 60)

        # Display sample results from first chunk
        sample_results = results.take(1)
        if sample_results:
            logger.info("\nSample result from first chunk:")
            result = sample_results[0]
            logger.info(f"  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            logger.info(f"  Text: {result['transcription']['text']}")
            logger.info(f"  Language: {result['transcription']['language']}")
            logger.info(f"  Segments: {result['transcription']['num_segments']}")

        # Save in different formats
        logger.info("\nSaving transcription in multiple formats...")
        pipeline.save_results("results/transcription.txt")  # Plain text
        pipeline.save_results("results/transcription.srt")  # SRT subtitles
        pipeline.save_results("results/transcription.vtt")  # WebVTT subtitles

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
