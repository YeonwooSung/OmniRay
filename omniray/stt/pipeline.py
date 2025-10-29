"""STT inference pipeline orchestrator using Ray."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import ray
from ray.data import Dataset

from omniray.config.schemas import STTInferenceConfig, STTModelType
from omniray.data.audio_loader import load_audio_chunks
from omniray.models.custom import CustomModel
from omniray.stt.faster_whisper import FasterWhisperModel

logger = logging.getLogger(__name__)


class STTInferencePipeline:
    """Ray-based STT (Speech-to-Text) inference pipeline."""

    def __init__(self, config: STTInferenceConfig):
        """Initialize STT inference pipeline.

        Args:
            config: STT inference configuration
        """
        self.config = config
        self.model = None
        self.results = None

    def _create_model(self):
        """Create STT model instance based on configuration."""
        model_type = self.config.model_type

        if model_type == STTModelType.FASTER_WHISPER:
            logger.info("Creating Faster Whisper model")
            self.model = FasterWhisperModel(self.config.faster_whisper_config)

        elif model_type == STTModelType.CUSTOM:
            logger.info("Creating custom STT model")
            if self.config.custom_model_config is None:
                raise ValueError("custom_model_config is required for CUSTOM model type")
            self.model = CustomModel(self.config.custom_model_config)

        else:
            raise ValueError(f"Unsupported STT model type: {model_type}")

        # Load the model
        self.model.load_model()

    def _transcribe_fn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Ray map function for batch transcription.

        Args:
            batch: Batch dictionary containing audio chunk data

        Returns:
            Batch with transcription added
        """
        # This will be called within Ray workers
        # The model needs to be loaded within each worker
        if not hasattr(self, "_worker_model"):
            self._create_model()
            self._worker_model = self.model

        return self._worker_model(batch)

    def run(self) -> Dataset:
        """Run the STT inference pipeline.

        Returns:
            Ray Dataset containing audio chunks with transcriptions

        Example:
            >>> from omniray.config import STTInferenceConfig, AudioConfig, STTModelType
            >>> from omniray.stt.pipeline import STTInferencePipeline
            >>>
            >>> config = STTInferenceConfig(
            ...     model_type=STTModelType.FASTER_WHISPER,
            ...     audio_config=AudioConfig(audio_path="audio.mp3")
            ... )
            >>> pipeline = STTInferencePipeline(config)
            >>> results = pipeline.run()
            >>> print(results.take(1))
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**self.config.ray_options)
            logger.info("Ray initialized")

        # Load audio chunks as Ray Dataset
        logger.info(f"Loading audio: {self.config.audio_config.audio_path}")
        dataset = load_audio_chunks(self.config.audio_config)

        # Create model (this will be serialized to Ray workers)
        self._create_model()

        # Run STT inference using Ray Data
        logger.info("Running STT inference on audio chunks")

        # Use map_batches for efficient batch processing
        results_dataset = dataset.map_batches(
            lambda batch: self._transcribe_fn(batch),
            batch_size=self.config.audio_config.batch_size,
            num_gpus=self.config.ray_options.get("num_gpus", 0),
        )

        self.results = results_dataset

        # Save results if output path is specified
        if self.config.output_path:
            self.save_results(self.config.output_path)

        logger.info("STT inference completed")
        return results_dataset

    def save_results(self, output_path: str) -> None:
        """Save transcription results to file.

        Args:
            output_path: Path to save results (supports .json, .txt, .srt, .vtt)
        """
        if self.results is None:
            raise ValueError("No results to save. Run inference first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        results_list = self.results.take_all()

        if output_path.suffix == ".json":
            # Save as JSON with full details
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2, default=str, ensure_ascii=False)

        elif output_path.suffix == ".txt":
            # Save as plain text (concatenate all transcriptions)
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results_list:
                    f.write(result["transcription"]["text"])
                    f.write("\n")

        elif output_path.suffix == ".srt":
            # Save as SRT subtitle format
            self._save_as_srt(results_list, output_path)

        elif output_path.suffix == ".vtt":
            # Save as WebVTT subtitle format
            self._save_as_vtt(results_list, output_path)

        else:
            raise ValueError(
                f"Unsupported output format: {output_path.suffix}. "
                f"Supported formats: .json, .txt, .srt, .vtt"
            )

        logger.info(f"Results saved to: {output_path}")

    def _save_as_srt(self, results_list: List[Dict[str, Any]], output_path: Path):
        """Save transcription as SRT subtitle format."""
        with open(output_path, "w", encoding="utf-8") as f:
            segment_counter = 1
            for result in results_list:
                transcription = result["transcription"]
                chunk_start = result["start_time"]

                for segment in transcription.get("segments", []):
                    # Calculate absolute timestamps
                    start_time = chunk_start + segment["start"]
                    end_time = chunk_start + segment["end"]

                    # Format timestamps (HH:MM:SS,mmm)
                    start_str = self._format_srt_time(start_time)
                    end_str = self._format_srt_time(end_time)

                    # Write SRT entry
                    f.write(f"{segment_counter}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{segment['text']}\n")
                    f.write("\n")

                    segment_counter += 1

    def _save_as_vtt(self, results_list: List[Dict[str, Any]], output_path: Path):
        """Save transcription as WebVTT subtitle format."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for result in results_list:
                transcription = result["transcription"]
                chunk_start = result["start_time"]

                for segment in transcription.get("segments", []):
                    # Calculate absolute timestamps
                    start_time = chunk_start + segment["start"]
                    end_time = chunk_start + segment["end"]

                    # Format timestamps (HH:MM:SS.mmm)
                    start_str = self._format_vtt_time(start_time)
                    end_str = self._format_vtt_time(end_time)

                    # Write VTT entry
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{segment['text']}\n")
                    f.write("\n")

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format time for WebVTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def get_results(self) -> List[Dict[str, Any]]:
        """Get transcription results as a list.

        Returns:
            List of dictionaries containing audio chunks and transcriptions
        """
        if self.results is None:
            raise ValueError("No results available. Run inference first.")

        return self.results.take_all()

    def get_full_transcription(self) -> str:
        """Get the full transcription as a single text string.

        Returns:
            Complete transcription text
        """
        if self.results is None:
            raise ValueError("No results available. Run inference first.")

        results_list = self.results.take_all()
        texts = [r["transcription"]["text"] for r in results_list]
        return " ".join(texts)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of transcription results.

        Returns:
            Dictionary containing summary statistics
        """
        if self.results is None:
            raise ValueError("No results available. Run inference first.")

        results_list = self.results.take_all()

        total_chunks = len(results_list)
        total_duration = sum(r["duration"] for r in results_list)
        total_segments = sum(
            r["transcription"].get("num_segments", 0) for r in results_list
        )

        # Get language from first result (if available)
        language = None
        if results_list and "transcription" in results_list[0]:
            language = results_list[0]["transcription"].get("language")

        summary = {
            "audio_path": self.config.audio_config.audio_path,
            "model_type": self.config.model_type.value,
            "total_chunks": total_chunks,
            "total_duration_s": total_duration,
            "total_segments": total_segments,
            "language": language,
        }

        return summary
