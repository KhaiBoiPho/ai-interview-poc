#!/usr/bin/env bash
set -e

MODELS_DIR="models"

echo "📦 Creating models directory..."
mkdir -p $MODELS_DIR

echo "[1/5] Downloading MuseTalk v1.5..."
mkdir -p $MODELS_DIR/musetalkV15
huggingface-cli download TMElyralab/MuseTalk \
  musetalkV15/musetalk.json \
  musetalkV15/unet.pth \
  --local-dir $MODELS_DIR \
  --local-dir-use-symlinks False

echo "[2/5] Downloading SD VAE..."
mkdir -p $MODELS_DIR/sd-vae
huggingface-cli download stabilityai/sd-vae-ft-mse \
  config.json \
  diffusion_pytorch_model.bin \
  --local-dir $MODELS_DIR/sd-vae \
  --local-dir-use-symlinks False

echo "[3/5] Downloading Whisper Tiny..."
mkdir -p $MODELS_DIR/whisper
huggingface-cli download openai/whisper-tiny \
  config.json \
  pytorch_model.bin \
  preprocessor_config.json \
  --local-dir $MODELS_DIR/whisper \
  --local-dir-use-symlinks False

echo "[4/5] Downloading DWPose..."
mkdir -p $MODELS_DIR/dwpose
huggingface-cli download yzd-v/DWPose \
  dw-ll_ucoco_384.pth \
  --local-dir $MODELS_DIR/dwpose \
  --local-dir-use-symlinks False

echo "[5/5] Downloading Face Parse BiSeNet..."
mkdir -p $MODELS_DIR/face-parse-bisent

# BiSeNet model
gdown --fuzzy https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
  -O $MODELS_DIR/face-parse-bisent/79999_iter.pth

# ResNet18 backbone
wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -O $MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth

echo "All required models downloaded successfully!"