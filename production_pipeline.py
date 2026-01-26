# production_pipeline.py
import os
from onnx_models import ONNXVAE, ONNXUNet, FasterWhisperProcessor
from video_preprocessor import VideoPreprocessor
from audio_inference import AudioInference

class ProductionPipeline:
    """
    Production pipeline: preprocess video once, generate with multiple audios
    """
    
    def __init__(self, 
                 onnx_dir="models/onnx",
                 cache_dir="./cache",
                 result_dir="./results",
                 whisper_model_size="base",
                 use_gpu=True,
                 version="v15",
                 skip_padding=False):
        
        print("Initializing Production Pipeline...")
        
        # Load models
        print("\n[1/3] Loading VAE...")
        self.vae = ONNXVAE(model_dir=onnx_dir, use_gpu=use_gpu)
        
        print("\n[2/3] Loading UNet...")
        self.unet = ONNXUNet(model_dir=onnx_dir, use_gpu=use_gpu)
        
        print("\n[3/3] Loading Whisper...")
        self.whisper_processor = FasterWhisperProcessor(
            model_size=whisper_model_size,
            device="cuda" if use_gpu else "cpu",
            compute_type="float16" if use_gpu else "int8"
        )
        
        # Initialize components
        self.video_preprocessor = VideoPreprocessor(
            vae=self.vae,
            cache_dir=cache_dir,
            version=version
        )
        
        self.audio_inference = AudioInference(
            unet=self.unet,
            whisper_processor=self.whisper_processor,
            result_dir=result_dir,
            version=version,
            skip_padding=skip_padding
        )
        
        print("\n✓ Pipeline initialized successfully!")
    
    def preprocess_video(self, video_path, use_cache=True):
        """
        Preprocess video (run once per video)
        
        Args:
            video_path: Path to video file
            use_cache: Use cached data if available
        
        Returns:
            video_cache: Preprocessed video data
        """
        return self.video_preprocessor.process_video(
            video_path=video_path,
            use_saved_coord=use_cache
        )
    
    def generate_with_audio(self, video_cache, audio_path, batch_size=16, output_name=None):
        """
        Generate video with specific audio
        
        Args:
            video_cache: Preprocessed video data
            audio_path: Path to audio file
            batch_size: Batch size for inference
            output_name: Custom output filename
        
        Returns:
            output_path: Path to generated video
        """
        return self.audio_inference.generate(
            video_cache=video_cache,
            audio_path=audio_path,
            batch_size=batch_size,
            output_name=output_name
        )
    
    def batch_generate(self, video_path, audio_paths, use_cache=True, batch_size=16):
        """
        Generate multiple videos with different audios from single video
        
        Args:
            video_path: Path to video file
            audio_paths: List of audio file paths
            use_cache: Use cached video data
            batch_size: Batch size for inference
        
        Returns:
            output_paths: List of generated video paths
        """
        print(f"\n{'='*60}")
        print("BATCH GENERATION")
        print(f"Video: {video_path}")
        print(f"Number of audios: {len(audio_paths)}")
        print(f"{'='*60}")
        
        # Step 1: Preprocess video once
        print("\n" + "="*60)
        print("STEP 1: VIDEO PREPROCESSING (RUN ONCE)")
        print("="*60)
        video_cache = self.preprocess_video(video_path, use_cache=use_cache)
        
        # Step 2: Generate with each audio
        print("\n" + "="*60)
        print(f"STEP 2: AUDIO INFERENCE (RUN {len(audio_paths)} TIMES)")
        print("="*60)
        
        output_paths = []
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Processing audio: {os.path.basename(audio_path)}")
            
            output_path = self.generate_with_audio(
                video_cache=video_cache,
                audio_path=audio_path,
                batch_size=batch_size
            )
            output_paths.append(output_path)
        
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
        version="v15",
        skip_padding=False
    )
    
    # Single video, multiple audios
    video_path = "data/video/input.mp4"
    audio_paths = [
        "data/audio/audio.wav"
    ]
    
    output_paths = pipeline.batch_generate(
        video_path=video_path,
        audio_paths=audio_paths,
        use_cache=True,
        batch_size=16
    )