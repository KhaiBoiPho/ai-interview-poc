import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DeviceConfig:
    """Device config"""
    gpu_id: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class MuseTalkConfig:
    """MuseTalk model configuration"""

    # Model Settings
    version: str = 'v15'
    batch_size: int = 20
    fps: int = 25

    # Audio Settings
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2

    # Avatar Settings
    extra_margin: int = 10
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    bbox_shift: int = 0

    # Model Paths
    vae_type: str = "sd-vae"
    unet_config: str = "./models/musetalk/musetalk.json"
    unet_model_path: str = "./models/musetalk/pytorch_model.bin"
    whisper_dir: str = "./models/whisper"


@dataclass
class TTSConfig:
    """TTS model configuration"""
    model_name: str = "facebook/mms-tts-vie"

@dataclass
class PerformanceSettings:
    # Performance Settings
    use_amp: bool = True
    use_async_encoding: bool = True
    use_gpu_encoding: bool = True
    num_workers: int = 12


@dataclass
class AppConfig:
    """Application configuration"""
    app_name: str = "MuseTalk API"
    host: str = "0.0.0.0"
    port: str = 8000

    # Directory Settings
    upload_dir: Path = Path("./uploads")
    tts_output_dir: Path = Path("./tts_outputs")
    avatar_results_dir: Path = Path("./results/avatars")

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.upload_dir.mkdir(exist_ok=True)
        self.tts_output_dir.mkdir(exist_ok=True)
        self.avatar_results_dir.mkdir(exist_ok=True)


# Global configurations
device = DeviceConfig()
musetalk_config = MuseTalkConfig()
tts_config = TTSConfig()
app_config = AppConfig()
performance_config = PerformanceSettings()
