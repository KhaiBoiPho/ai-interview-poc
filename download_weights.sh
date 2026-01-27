#!/usr/bin/env bash
set -e

MODELS_DIR="models"

echo "📦 Creating models directory..."
mkdir -p $MODELS_DIR

# ✅ REQUIRED: MuseTalk UNet v1.5
echo "⬇️ [1/5] Downloading MuseTalk v1.5..."
python -c "
from huggingface_hub import hf_hub_download
import os

files = [
    'musetalkV15/musetalk.json',
    'musetalkV15/unet.pth'
]

for file in files:
    print(f'Downloading {file}...')
    hf_hub_download(
        repo_id='TMElyralab/MuseTalk',
        filename=file,
        local_dir='$MODELS_DIR',
        local_dir_use_symlinks=False
    )
"

# ✅ REQUIRED: SD VAE
echo "⬇️ [2/5] Downloading SD VAE..."
python -c "
from huggingface_hub import hf_hub_download

files = ['config.json', 'diffusion_pytorch_model.bin']

for file in files:
    print(f'Downloading {file}...')
    hf_hub_download(
        repo_id='stabilityai/sd-vae-ft-mse',
        filename=file,
        local_dir='$MODELS_DIR/sd-vae',
        local_dir_use_symlinks=False
    )
"

# ✅ REQUIRED: Whisper
echo "⬇️ [3/5] Downloading Whisper Tiny..."
python -c "
from huggingface_hub import hf_hub_download

files = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']

for file in files:
    print(f'Downloading {file}...')
    hf_hub_download(
        repo_id='openai/whisper-tiny',
        filename=file,
        local_dir='$MODELS_DIR/whisper',
        local_dir_use_symlinks=False
    )
"

# ✅ REQUIRED: DWPose
echo "⬇️ [4/5] Downloading DWPose..."
mkdir -p $MODELS_DIR/dwpose
python -c "
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='yzd-v/DWPose',
    filename='dw-ll_ucoco_384.pth',
    local_dir='$MODELS_DIR/dwpose',
    local_dir_use_symlinks=False
)
"

# ✅ REQUIRED: Face Parse BiSeNet
echo "⬇️ [5/5] Downloading Face Parse BiSeNet..."
mkdir -p $MODELS_DIR/face-parse-bisent

# Install gdown if needed
pip install -q gdown

# BiSeNet model
gdown --fuzzy https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O $MODELS_DIR/face-parse-bisent/79999_iter.pth

# ResNet18 backbone
wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -O $MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ All required models downloaded successfully!"
echo ""
echo "Downloaded models:"
echo "  📁 $MODELS_DIR/musetalkV15/"
echo "  📁 $MODELS_DIR/sd-vae/"
echo "  📁 $MODELS_DIR/whisper/"
echo "  📁 $MODELS_DIR/dwpose/"
echo "  📁 $MODELS_DIR/face-parse-bisent/"