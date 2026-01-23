from huggingface_hub import hf_hub_download, snapshot_download
import os

CheckpointsDir = "models"
os.makedirs(CheckpointsDir, exist_ok=True)

# Download MuseTalk V1.0 weights
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    allow_patterns=["musetalk/musetalk.json", "musetalk/pytorch_model.bin"],
    local_dir=CheckpointsDir,
    local_dir_use_symlinks=False
)

# Download MuseTalk V1.5 weights
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    allow_patterns=["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
    local_dir=CheckpointsDir,
    local_dir_use_symlinks=False
)

# Download SD VAE weights
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
    local_dir=f"{CheckpointsDir}/sd-vae",
    local_dir_use_symlinks=False
)

# Download Whisper weights
snapshot_download(
    repo_id="openai/whisper-tiny",
    allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    local_dir=f"{CheckpointsDir}/whisper",
    local_dir_use_symlinks=False
)

# Download DWPose weights
snapshot_download(
    repo_id="yzd-v/DWPose",
    allow_patterns=["dw-ll_ucoco_384.pth"],
    local_dir=f"{CheckpointsDir}/dwpose",
    local_dir_use_symlinks=False
)

# Download SyncNet weights
snapshot_download(
    repo_id="ByteDance/LatentSync",
    allow_patterns=["latentsync_syncnet.pt"],
    local_dir=f"{CheckpointsDir}/syncnet",
    local_dir_use_symlinks=False
)

# Download Face Parse Bisent weights (vẫn dùng gdown)
import gdown
os.makedirs(f"{CheckpointsDir}/face-parse-bisent", exist_ok=True)
gdown.download(
    id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
    output=f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
)

import requests
resnet_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
with open(f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth", "wb") as f:
    f.write(requests.get(resnet_url).content)

print("✅ All weights downloaded successfully!")