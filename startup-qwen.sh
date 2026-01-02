#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

echo "========================================="
echo "ComfyUI Qwen Setup - Fixed Version v2"
echo "========================================="

# Set workspace root
ROOT=${ROOT:-/workspace}
cd "$ROOT"

echo "Working directory: $ROOT"

# ============================================
# STEP 1: Install/Update PyTorch (CRITICAL FIX)
# ============================================
echo ""
echo "Step 1: Ensuring PyTorch 2.x compatibility..."

# Install/upgrade to PyTorch 2.x with CUDA 12.1
pip3 install --upgrade pip
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch version
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# ============================================
# STEP 2: Setup ComfyUI (FORCE FRESH CLONE)
# ============================================
echo ""
echo "Step 2: Setting up ComfyUI..."

# Always remove and clone fresh to avoid git issues
echo "Removing old ComfyUI installation (if exists)..."
rm -rf "$ROOT/ComfyUI"

echo "Cloning ComfyUI fresh..."
git clone https://github.com/comfyanonymous/ComfyUI.git "$ROOT/ComfyUI"

cd "$ROOT/ComfyUI"

# Install ComfyUI dependencies
echo "Installing ComfyUI requirements..."
pip3 install -r requirements.txt

# ============================================
# STEP 3: Install ComfyUI Manager (Optional but helpful)
# ============================================
echo ""
echo "Step 3: Installing ComfyUI Manager..."

if [ ! -d "$ROOT/ComfyUI/custom_nodes/ComfyUI-Manager" ]; then
    cd "$ROOT/ComfyUI/custom_nodes"
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd "$ROOT/ComfyUI"
else
    echo "ComfyUI Manager already installed"
fi

# ============================================
# STEP 4: Create Model Directories
# ============================================
echo ""
echo "Step 4: Creating model directories..."

DIFFUSION_DIR="$ROOT/ComfyUI/models/diffusion_models"
TEXT_ENCODER_DIR="$ROOT/ComfyUI/models/text_encoders"
VAE_DIR="$ROOT/ComfyUI/models/vae"
LORA_DIR="$ROOT/ComfyUI/models/loras"

mkdir -p "$DIFFUSION_DIR" "$TEXT_ENCODER_DIR" "$VAE_DIR" "$LORA_DIR"

# ============================================
# STEP 5: Download Qwen Models
# ============================================
echo ""
echo "Step 5: Downloading Qwen models..."

# Install aria2c for faster downloads if not present
if ! command -v aria2c &> /dev/null; then
    echo "Installing aria2c..."
    apt-get update && apt-get install -y aria2
fi

# Model URLs
DIFFUSION_URL="https://huggingface.co/Comfy-Org/Qwen-Image-Edit-2511_mirror/resolve/main/diffusion_models/qwen2vl-2b-ae-v1_1-fp16-img-edit.safetensors"
TEXT_ENCODER_URL="https://huggingface.co/Comfy-Org/Qwen-Image-Edit-2511_mirror/resolve/main/text_encoders/qwen2vl_text_encoder_2b.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Qwen-Image-Edit-2511_mirror/resolve/main/vae/qwen2vl-2b-ae-v1_1-fp16-ae.safetensors"
LORA_URL="https://huggingface.co/Comfy-Org/Qwen-Image-Edit-2511_mirror/resolve/main/loras/qwen2vl-2b-ae-v1_1-lightning-img-edit.safetensors"

# Download main diffusion model
if [ ! -f "$DIFFUSION_DIR/qwen2vl-2b-ae-v1_1-fp16-img-edit.safetensors" ]; then
    echo "Downloading Qwen diffusion model (this may take a while)..."
    aria2c -x 10 -s 10 --dir="$DIFFUSION_DIR" --out="qwen2vl-2b-ae-v1_1-fp16-img-edit.safetensors" "$DIFFUSION_URL"
else
    echo "✓ Diffusion model already exists"
fi

# Download text encoder
if [ ! -f "$TEXT_ENCODER_DIR/qwen2vl_text_encoder_2b.safetensors" ]; then
    echo "Downloading Qwen text encoder..."
    aria2c -x 10 -s 10 --dir="$TEXT_ENCODER_DIR" --out="qwen2vl_text_encoder_2b.safetensors" "$TEXT_ENCODER_URL"
else
    echo "✓ Text encoder already exists"
fi

# Download VAE
if [ ! -f "$VAE_DIR/qwen2vl-2b-ae-v1_1-fp16-ae.safetensors" ]; then
    echo "Downloading Qwen VAE..."
    aria2c -x 10 -s 10 --dir="$VAE_DIR" --out="qwen2vl-2b-ae-v1_1-fp16-ae.safetensors" "$VAE_URL"
else
    echo "✓ VAE already exists"
fi

# Download Lightning LoRA
if [ ! -f "$LORA_DIR/qwen2vl-2b-ae-v1_1-lightning-img-edit.safetensors" ]; then
    echo "Downloading Lightning LoRA..."
    aria2c -x 10 -s 10 --dir="$LORA_DIR" --out="qwen2vl-2b-ae-v1_1-lightning-img-edit.safetensors" "$LORA_URL"
else
    echo "✓ Lightning LoRA already exists"
fi

# ============================================
# STEP 6: Verify Installation
# ============================================
echo ""
echo "Step 6: Verifying installation..."

# Check if models exist
echo "Checking downloaded models:"
ls -lh "$DIFFUSION_DIR"/*.safetensors 2>/dev/null || echo "  Warning: No diffusion models found"
ls -lh "$TEXT_ENCODER_DIR"/*.safetensors 2>/dev/null || echo "  Warning: No text encoders found"
ls -lh "$VAE_DIR"/*.safetensors 2>/dev/null || echo "  Warning: No VAE models found"
ls -lh "$LORA_DIR"/*.safetensors 2>/dev/null || echo "  Warning: No LoRA models found"

# ============================================
# STEP 7: Start ComfyUI
# ============================================
echo ""
echo "========================================="
echo "Setup Complete! Starting ComfyUI..."
echo "========================================="
echo ""
echo "Access ComfyUI at port ${PORT:-8188}"
echo ""

cd "$ROOT/ComfyUI"

# Start ComfyUI with appropriate settings
python3 main.py --listen 0.0.0.0 --port "${PORT:-8188}" --preview-method auto
