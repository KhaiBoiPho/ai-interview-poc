"""Audio processing service for MuseTalk inference"""
import soundfile as sf
from app.core.model_loader import model_manager
from app.config import musetalk_config


class AudioService:
    """Handle audio feature extraction for MuseTalk"""

    @staticmethod
    def process_audio_for_inference(audio_path: str, fps: int = None):
        """Process audio và move chunks to GPU ngay"""
        if fps is None:
            fps = musetalk_config.fps

        print(f"🎵 Processing audio: {audio_path}")

        whisper_input_features, librosa_length = (
            model_manager.audio_processor.get_audio_feature(
                audio_path,
                weight_dtype=model_manager.weight_dtype
            )
        )

        whisper_chunks = model_manager.audio_processor.get_whisper_chunk(
            whisper_input_features=whisper_input_features,
            device=model_manager.device,
            weight_dtype=model_manager.weight_dtype,
            whisper=model_manager.whisper,
            librosa_length=librosa_length,
            fps=fps,
            audio_padding_length_left=musetalk_config.audio_padding_length_left,
            audio_padding_length_right=musetalk_config.audio_padding_length_right
        )

        # Chunks đã ở GPU rồi từ get_whisper_chunk
        print(f"   ✓ Extracted {len(whisper_chunks)} audio chunks (on GPU)")
        return whisper_chunks
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Get audio file duration in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        
        with sf.SoundFile(audio_path) as f:
            duration = len(f) / f.samplerate
            
        return duration


# Singleton instance
audio_service = AudioService()