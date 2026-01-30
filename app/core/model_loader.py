import time
import torch
from transformers import WhisperModel, VitsModel, AutoTokenizer

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

from app.config import musetalk_config, tts_config, device
from app.core.cuda_manager import initialize_stream_manager


class ModelManager:
    """Manages all ML models"""
    
    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.weight_dtype = None
        self.timesteps = None
        self.fp = None

        # TTS models
        self.tts_model = None
        self.tts_tokenizer = None
        
    def initialize(self):
        """Initialize all models"""
        print("🚀 Loading all models...")

        self._setup_device()
        self._load_musetalk_models()
        self._enable_optimizations()
        self._load_whisper()
        self._load_audio_processor()
        self._load_face_parsing()
        self._load_tts_models()
        self._warmup()
        
    def _setup_device(self):
        """Setup compute device"""
        self.device = torch.device(
            f"cuda:{device.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print(f"📱 Using device: {self.device}")
        
        # Initialize CUDA stream manager
        initialize_stream_manager(self.device)
        
    def _load_musetalk_models(self):
        """Load MuseTalk VAE, UNet, PE"""
        print("Loading MuseTalk models (VAE, UNet, PE)...")
        
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=musetalk_config.unet_model_path,
            vae_type=musetalk_config.vae_type,
            unet_config=musetalk_config.unet_config,
            device=self.device
        )
        
        # Move to half precision
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        
        # Set timesteps
        self.timesteps = torch.tensor([0], device=self.device)
        self.weight_dtype = self.unet.model.dtype
        
        print(f"Weight dtype: {self.weight_dtype}")
        print("✅ MuseTalk models loaded")
        
    def _enable_optimizations(self):
        """Enable CUDA optimizations"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ CUDA optimizations enabled")
            
        # Compile models if available
        self._compile_models()

    def _load_whisper(self):
        """Load Whisper model"""
        print("Loading Whisper model...")

        self.whisper = WhisperModel.from_pretrained(musetalk_config.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        print("✓ Whisper loaded")

        
    def _compile_models(self):
        """Compile models with torch.compile (PyTorch 2.0+)"""
        if not hasattr(torch, 'compile'):
            print("⚠️  torch.compile not available (PyTorch < 2.0)")
            return
        
        print("🔥 Compiling models with torch.compile...")
        
        try:
            self.vae.vae = torch.compile(self.vae.vae, mode='reduce-overhead')
            print("   ✅ VAE compiled")
        except Exception as e:
            print(f"   ⚠️  VAE compilation failed: {e}")
        
        try:
            self.pe = torch.compile(self.pe, mode='reduce-overhead')
            print("   ✅ PE compiled")
        except Exception as e:
            print(f"   ⚠️  PE compilation failed: {e}")


    def _load_audio_processor(self):
        """Load audio processing"""
        print("Loading audio processor...")
        
        self.audio_processor = AudioProcessor(
            feature_extractor_path=musetalk_config.whisper_dir
        )
        
        print("✅ Audio models loaded")
        
    def _load_tts_models(self):
        """Load TTS models"""
        print("Loading TTS models...")
        
        start_time = time.time()

        self.tts_tokenizer = AutoTokenizer.from_pretrained(tts_config.model_name)
        self.tts_model = VitsModel.from_pretrained(tts_config.model_name).to(self.device)
        self.tts_model.eval()

        load_time = time.time() - start_time

        print(f"   ✓ TTS loaded ({load_time:.2f}s)")
        print(f"   Sample rate: {self.tts_model.config.sampling_rate}Hz")
        
    def _load_face_parsing(self):
        """Load face parsing model"""
        print("Loading face parsing...")

        self.fp = FaceParsing(
            left_cheek_width=musetalk_config.left_cheek_width,
            right_cheek_width=musetalk_config.right_cheek_width
        )
        
        print("✅ Face parsing loaded")
        
    def _warmup(self):
        """Warmup models"""
        print("🔥 Warming up models...")
        
        try:
            dummy_audio = torch.randn(1, 1, 80, 100).to(
                device=self.device, 
                dtype=self.weight_dtype
            )
            
            with torch.no_grad():
                audio_features = self.pe(dummy_audio)
                dummy_latents = torch.randn(1, 4, 32, 32).to(
                    device=self.device, 
                    dtype=self.unet.model.dtype
                )
                _ = self.unet.model(
                    dummy_latents, 
                    self.timesteps, 
                    encoder_hidden_states=audio_features
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            print("✅ Warmup completed")
        except Exception as e:
            print(f"⚠️  Warmup failed (non-critical): {str(e)[:100]}")
    
    def get_optimal_batch_size(self) -> int:
        """Auto-tune batch size based on GPU memory"""
        if not torch.cuda.is_available():
            return musetalk_config.batch_size
            
        gpu_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        
        if gpu_memory_gb >= 24:
            return 32
        elif gpu_memory_gb >= 16:
            return 24
        elif gpu_memory_gb >= 12:
            return 20
        elif gpu_memory_gb >= 8:
            return 16
        else:
            return 12


# Global instance
model_manager = ModelManager()