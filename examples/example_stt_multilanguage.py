"""Example: Multi-language STT with Faster Whisper."""

import logging

import ray

from omniray.config import AudioConfig, FasterWhisperConfig, STTInferenceConfig, STTModelType
from omniray.stt import STTInferencePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, language: str = None, output_prefix: str = "transcription"):
    """Transcribe audio in specified language.

    Args:
        audio_path: Path to audio file
        language: Language code (None for auto-detect)
        output_prefix: Prefix for output files
    """
    # Configure audio input
    audio_config = AudioConfig(
        audio_path=audio_path,
        chunk_length_s=30.0,
        batch_size=8,
    )

    # Configure Faster Whisper with language
    whisper_config = FasterWhisperConfig(
        model_size="medium",  # Larger model for better multi-language support
        device="auto",
        compute_type="float16" if language else "default",  # float16 for better quality
        language=language,  # Specify language or None for auto-detect
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )

    # Configure STT pipeline
    stt_config = STTInferenceConfig(
        model_type=STTModelType.FASTER_WHISPER,
        audio_config=audio_config,
        faster_whisper_config=whisper_config,
        ray_options={"num_cpus": 4, "num_gpus": 1},
        output_path=f"results/{output_prefix}.json",
    )

    # Create and run pipeline
    pipeline = STTInferencePipeline(stt_config)
    logger.info(f"Transcribing {audio_path} (language: {language or 'auto-detect'})...")

    results = pipeline.run()

    # Get summary
    summary = pipeline.get_summary()
    logger.info(f"Detected language: {summary['language']}")
    logger.info(f"Duration: {summary['total_duration_s']:.2f}s")

    # Get full transcription
    full_text = pipeline.get_full_transcription()

    # Save in multiple formats
    pipeline.save_results(f"results/{output_prefix}.txt")
    pipeline.save_results(f"results/{output_prefix}.srt")

    return full_text, summary


def main():
    """Run STT on multiple languages."""
    # Initialize Ray once
    ray.init(ignore_reinit_error=True)

    try:
        # Example 1: Auto-detect language
        logger.info("=" * 60)
        logger.info("Example 1: Auto-detect language")
        logger.info("=" * 60)
        text1, summary1 = transcribe_audio(
            "path/to/multilingual/audio.mp3",
            language=None,  # Auto-detect
            output_prefix="transcription_auto"
        )
        logger.info(f"Detected: {summary1['language']}")
        logger.info(f"Text: {text1[:200]}...")

        # Example 2: Korean transcription
        logger.info("\n" + "=" * 60)
        logger.info("Example 2: Korean transcription")
        logger.info("=" * 60)
        text2, summary2 = transcribe_audio(
            "path/to/korean/audio.mp3",
            language="ko",  # Force Korean
            output_prefix="transcription_korean"
        )
        logger.info(f"Text: {text2[:200]}...")

        # Example 3: Japanese transcription
        logger.info("\n" + "=" * 60)
        logger.info("Example 3: Japanese transcription")
        logger.info("=" * 60)
        text3, summary3 = transcribe_audio(
            "path/to/japanese/audio.mp3",
            language="ja",  # Force Japanese
            output_prefix="transcription_japanese"
        )
        logger.info(f"Text: {text3[:200]}...")

        # Example 4: Translation to English
        logger.info("\n" + "=" * 60)
        logger.info("Example 4: Translate to English")
        logger.info("=" * 60)

        whisper_config = FasterWhisperConfig(
            model_size="medium",
            device="auto",
            compute_type="float16",
            language="ko",  # Source language
            task="translate",  # Translate to English!
            beam_size=5,
            vad_filter=True,
        )

        stt_config = STTInferenceConfig(
            model_type=STTModelType.FASTER_WHISPER,
            audio_config=AudioConfig(
                audio_path="path/to/korean/audio.mp3",
                chunk_length_s=30.0,
            ),
            faster_whisper_config=whisper_config,
            ray_options={"num_cpus": 4, "num_gpus": 1},
            output_path="results/translation_english.json",
        )

        pipeline = STTInferencePipeline(stt_config)
        results = pipeline.run()
        translated_text = pipeline.get_full_transcription()
        logger.info(f"Translation: {translated_text[:200]}...")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
