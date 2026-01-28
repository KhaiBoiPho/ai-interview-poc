# production_pipeline.py - OPTIMIZED VERSION
import os
import torch
from onnx_models import ONNXVAE, ONNXUNet, FasterWhisperProcessor
from onnx.video_preprocessor import VideoPreprocessor
from audio_inference import AudioInference

class ProductionPipeline:
    """
    Production pipeline: preprocess video once, generate with multiple audios
    Optimized for maximum speed
    """
    
    def __init__(self, 
                 onnx_dir="models/onnx",
                 cache_dir="./cache",
                 result_dir="./results",
                 whisper_model_size="base",
                 use_gpu=True,
                 version="v15"):
        
        print("Initializing Production Pipeline...")
        
        # ✅ Set device
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print("\n[1/3] Loading VAE...")
        self.vae = ONNXVAE(model_dir=onnx_dir, use_gpu=use_gpu)
        
        print("\n[2/3] Loading UNet...")
        self.unet = ONNXUNet(model_dir=onnx_dir, use_gpu=use_gpu)
        
        print("\n[3/3] Loading Whisper...")
        # ✅ Use GPU for Whisper if available
        self.whisper_processor = FasterWhisperProcessor(
            model_size=whisper_model_size,
            device="cuda" if (use_gpu and torch.cuda.is_available()) else "cpu",
            compute_type="float16" if (use_gpu and torch.cuda.is_available()) else "int8"
        )
        
        # Initialize components
        self.video_preprocessor = VideoPreprocessor(
            vae=self.vae,
            cache_dir=cache_dir,
            version=version
        )
        
        self.audio_inference = AudioInference(
            vae=self.vae,
            unet=self.unet,
            whisper_processor=self.whisper_processor,
            device=self.device,  # ✅ Pass device
            result_dir=result_dir,
            version=version
        )
        
        # ✅ Warmup ONNX models (giảm latency lần đầu)
        if use_gpu:
            self._warmup_models()
        
        print("\n✓ Pipeline initialized successfully!")
    
    def _warmup_models(self):
        """Warmup ONNX models to reduce first inference latency"""
        import numpy as np
        print("\n[Warmup] Warming up ONNX models...")
        
        try:
            # Dummy VAE encode
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            _ = self.vae.get_latents_for_unet(dummy_image)
            
            # Dummy UNet forward
            dummy_latent = np.random.randn(1, 8, 32, 32).astype(np.float32)
            dummy_audio = np.random.randn(1, 50, 384).astype(np.float32)
            timesteps = np.array([0], dtype=np.int64)
            _ = self.unet.forward(dummy_latent, timesteps, dummy_audio)
            
            # Dummy VAE decode
            dummy_latent_decode = np.random.randn(1, 4, 32, 32).astype(np.float32)
            _ = self.vae.decode_latents(dummy_latent_decode)
            
            print("✓ Warmup completed")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
    
    def preprocess_video(self, video_path, bbox_shift=0, use_cache=True):
        """
        Preprocess video (run once per video)
        
        Args:
            video_path: Path to video file
            bbox_shift: Bounding box shift (v1 only)
            use_cache: Use cached data if available
        
        Returns:
            video_cache: Preprocessed video data
        """
        return self.video_preprocessor.process_video(
            video_path=video_path,
            bbox_shift=bbox_shift,
            use_saved_coord=use_cache
        )
    
    def generate_with_audio(self, 
                           video_cache, 
                           audio_path, 
                           batch_size=16,  # ✅ Default 16
                           audio_padding_length_left=2,
                           audio_padding_length_right=2,
                           output_name=None):
        """
        Generate video with specific audio
        
        Args:
            video_cache: Preprocessed video data
            audio_path: Path to audio file
            batch_size: Batch size (16-32 recommended for GPU)
            output_name: Custom output filename
        
        Returns:
            output_path: Path to generated video
        """
        return self.audio_inference.generate(
            video_cache=video_cache,
            audio_path=audio_path,
            batch_size=batch_size,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
            output_name=output_name
        )
    
    def batch_generate(self, 
                      video_path, 
                      audio_paths, 
                      bbox_shift=0,
                      use_cache=True, 
                      batch_size=16):  # ✅ Default 16
        """
        Generate multiple videos with different audios from single video
        
        Args:
            video_path: Path to video file
            audio_paths: List of audio file paths
            bbox_shift: Bounding box shift (v1 only)
            use_cache: Use cached video data
            batch_size: Batch size (16-32 recommended for GPU)
        
        Returns:
            output_paths: List of generated video paths
        """
        print(f"\n{'='*60}")
        print("BATCH GENERATION")
        print(f"Video: {video_path}")
        print(f"Number of audios: {len(audio_paths)}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Step 1: Preprocess video once
        print("\n" + "="*60)
        print("STEP 1: VIDEO PREPROCESSING (RUN ONCE)")
        print("="*60)
        video_cache = self.preprocess_video(
            video_path, 
            bbox_shift=bbox_shift,
            use_cache=use_cache
        )
        
        # Step 2: Generate with each audio
        print("\n" + "="*60)
        print(f"STEP 2: AUDIO INFERENCE (RUN {len(audio_paths)} TIMES)")
        print("="*60)
        
        output_paths = []
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Processing: {os.path.basename(audio_path)}")
            
            output_path = self.generate_with_audio(
                video_cache=video_cache,
                audio_path=audio_path,
                batch_size=batch_size
            )
            output_paths.append(output_path)
            
            # ✅ Clear GPU cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print("✓ BATCH GENERATION COMPLETED")
        print(f"{'='*60}")
        print(f"Generated {len(output_paths)} videos:")
        for path in output_paths:
            print(f"  - {path}")
        
        return output_paths


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ProductionPipeline(
        onnx_dir="models/onnx",
        cache_dir="./cache",
        result_dir="./results",
        whisper_model_size="base",
        use_gpu=True,
        version="v15"
    )
    
    # Single video, multiple audios
    video_path = "data/video/input.mp4"
    audio_paths = [
        "data/audio/audio1.wav",
        "data/audio/audio2.wav",
        "data/audio/audio3.wav"
    ]
    
    output_paths = pipeline.batch_generate(
        video_path=video_path,
        audio_paths=audio_paths,
        use_cache=True,
        batch_size=16  # Tăng lên 32 nếu GPU mạnh
    )