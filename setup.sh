#!/usr/bin/env bash
set -e

ENV_NAME="khai-env"
PYTHON_VERSION="3.10"
NUMPY_VERSION="1.26.4"
OPENCV_VERSION="4.8.1.78"

echo "============================================"
echo " MuseTalk environment setup: $ENV_NAME"
echo " Python: $PYTHON_VERSION"
echo " NumPy:  $NUMPY_VERSION (HARD PIN)"
echo " OpenCV: $OPENCV_VERSION"
echo " Torch:  2.0.1 + CUDA 11.7"
echo "============================================"

# ------------------------------------------------
# Conda bootstrap
# ------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  MINIFORGE_DIR="$HOME/miniforge3"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ------------------------------------------------
# Env
# ------------------------------------------------
if ! conda env list | grep -q "^$ENV_NAME"; then
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

# ------------------------------------------------
# Tooling (HARD PIN)
# ------------------------------------------------
pip install --upgrade pip wheel
pip install "setuptools==60.2.0"

# ------------------------------------------------
# NumPy / OpenCV (ABI SAFE)
# ------------------------------------------------
pip uninstall -y numpy opencv-python || true
pip install numpy=="$NUMPY_VERSION"
pip install opencv-python=="$OPENCV_VERSION"

# ------------------------------------------------
# PyTorch CUDA 11.7
# ------------------------------------------------
pip uninstall -y torch torchvision torchaudio || true
pip install \
  torch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

# ------------------------------------------------
# OpenMMLab (NO DEPS)
# ------------------------------------------------
pip uninstall -y mmcv mmengine mmpose xtcocotools || true
pip install openmim

mim install mmengine==0.9.1 --no-deps
mim install "mmcv>=2.0.0,<2.2.0" \
  --no-deps \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
pip install mmpose==1.2.0 --no-deps

# ------------------------------------------------
# REQUIRED RUNTIME DEPS (BẮT BUỘC)
# ------------------------------------------------
pip install \
  addict \
  yapf \
  termcolor \
  json-tricks \
  munkres \
  scipy==1.11.4
pip install chumpy --no-build-isolation
# ------------------------------------------------
# xtcocotools (ABI SAFE)
# ------------------------------------------------
pip install xtcocotools==1.14.3 --no-build-isolation
pip install --force-reinstall numpy=="$NUMPY_VERSION"

# ------------------------------------------------
# Verify core stack
# ------------------------------------------------
python - << 'EOF'
import numpy, cv2, torch
import mmcv, mmengine, mmpose
import xtcocotools._mask

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__, torch.version.cuda)
print("MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__)
print("MMPose:", mmpose.__version__)
print("xtcocotools: OK")
print("✅ ABI CLEAN")
EOF

# ------------------------------------------------
# MuseTalk deps
# ------------------------------------------------
pip install \
  "diffusers>=0.20,<0.21" \
  "transformers>=4.30,<4.35" \
  "huggingface-hub>=0.16,<0.18" \
  "accelerate>=0.20,<0.25" \
  librosa==0.10.1 \
  mediapipe==0.10.9 \
  sentencepiece \
  facexlib \
  basicsr \
  einops \
  tqdm \
  fastapi \
  uvicorn \
  python-multipart \
  rich

echo "============================================"
echo " ✅ DONE – ENV IS STABLE"
echo " conda activate $ENV_NAME"
echo " python genavatar_musetalk.py --help"
echo "============================================"
