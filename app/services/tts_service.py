"""Text-to-Speech service"""
import torch
import numpy as np
import soundfile as sf
import time
import uuid

from app.core.model_loader import model_manager
from app.config import app_config


class TTSService:
    """Handles text-to-speech synthesis"""

    @staticmethod
    @torch.no_grad()
    def synthesize(text: str, output_path: str = None) -> dict:
        """
        Synthesize speech from Vietnamese text
        
        Args:
            text: Vietnamese text to synthesize
            output_path: Path to save audio file (optional)
            
        Returns:
            dict with audio_path, duration, latency, sample_rate
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        print(f"🎤 TTS: '{text[:50]}...'")

        # Generate output path if not provided
        if output_path is None:
            output_path = app_config.tts_output_dir / f"{uuid.uuid4()}.wav"

        output_path = str(output_path)

        start_time = time.time()

        # Tokenize
        inputs = model_manager.tts_tokenizer(text, return_tensors="pt").to(
            model_manager.device
        )

        # Generate waveform
        with torch.no_grad():
            output = model_manager.tts_model(**inputs)
            waveform = output.waveform

        latency = time.time() -start_time

        # Convert to numpy
        audio = waveform.squeeze().cpu().numpy()

        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # Save to file
        sample_rate = model_manager.tts_model.config.sampling_rate
        sf.write(output_path, audio, sample_rate)

        duration = len(audio) / sample_rate

        print(f"✅ TTS done: {latency:.2f}s latency, {duration:.2f}s audio")
        
        return {
            'audio_path': output_path,
            'duration': duration,
            'latency': latency,
            'sample_rate': sample_rate
        }
    
    @staticmethod
    def synthesize_to_file(text: str, filename: str = None) -> dict:
        """
        Synthesize and save to specific filename
        
        Args:
            text: Text to synthesize
            filename: Output filename (in tts_output_dir)
            
        Returns:
            TTS result dict
        """
        if filename:
            output_path = app_config.tts_output_dir / filename
        else:
            output_path = None

        return TTSService.synthesize(text, output_path)
    

# Singleton instance
tts_service = TTSService()