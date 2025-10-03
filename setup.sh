#!/bin/bash
###############################################################################
# VibeVoice Voice Cloning App - Complete Setup Script
# One-stop installation for everything you need!
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Header
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  🎙️  VibeVoice Voice Cloning App - Setup Wizard  🎙️   ║"
echo "║                                                        ║"
echo "║  This will install everything you need to start       ║"
echo "║  cloning voices in minutes!                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
log_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_success "Found Python ${PYTHON_VERSION}"

# Check if CUDA is available
log_info "Checking for CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warning "No NVIDIA GPU detected. Will use CPU (slower)."
fi

# Create virtual environment
log_info "Creating virtual environment..."
if [ ! -d "vv_env" ]; then
    python3 -m venv vv_env
    log_success "Virtual environment created: vv_env"
else
    log_warning "Virtual environment already exists, skipping creation"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source vv_env/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA if available)
log_info "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    log_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    log_info "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# Install core dependencies
log_info "Installing core dependencies..."
pip install transformers>=4.51.3
pip install accelerate>=0.43.0
pip install bitsandbytes>=0.43.0
pip install scipy
pip install soundfile
pip install librosa
pip install pydub
pip install tqdm
pip install huggingface-hub

log_success "All Python packages installed!"

# Create directory structure
log_info "Creating directory structure..."
mkdir -p models/vibevoice/tokenizer
mkdir -p outputs
mkdir -p voice_samples
mkdir -p lora_adapters
mkdir -p scripts

log_success "Directory structure created!"

# Check if vvembed exists
if [ ! -d "vvembed" ]; then
    log_error "vvembed directory not found! Please ensure vvembed is in the project directory."
    log_info "You should have the vvembed folder from the VibeVoice ComfyUI nodes."
    exit 1
else
    log_success "vvembed directory found!"
fi

# Download tokenizer
log_info "Checking for Qwen2.5-1.5B tokenizer..."
if [ ! -f "models/vibevoice/tokenizer/tokenizer.json" ]; then
    log_info "Downloading Qwen2.5-1.5B tokenizer (required for all models)..."
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Qwen/Qwen2.5-1.5B \
            tokenizer_config.json vocab.json merges.txt tokenizer.json \
            --local-dir models/vibevoice/tokenizer \
            --local-dir-use-symlinks False
        log_success "Tokenizer downloaded!"
    else
        log_error "huggingface-cli not found. Installing..."
        pip install -U "huggingface_hub[cli]"
        huggingface-cli download Qwen/Qwen2.5-1.5B \
            tokenizer_config.json vocab.json merges.txt tokenizer.json \
            --local-dir models/vibevoice/tokenizer \
            --local-dir-use-symlinks False
        log_success "Tokenizer downloaded!"
    fi
else
    log_success "Tokenizer already downloaded!"
fi

# Ask user which model to download
echo ""
log_info "Which VibeVoice model would you like to download?"
echo ""
echo "  1) VibeVoice-Large-Q8 (11.6GB) - RECOMMENDED"
echo "     Balanced quality/speed with 8-bit quantization"
echo ""
echo "  2) VibeVoice-1.5B (5.4GB) - Fastest"
echo "     Good quality, lowest VRAM usage"
echo ""
echo "  3) VibeVoice-Large (18.7GB) - Best Quality"
echo "     Highest quality, requires more VRAM"
echo ""
echo "  4) VibeVoice-Large-Q4 (6.6GB) - Low VRAM"
echo "     4-bit quantization for minimal memory usage"
echo ""
echo "  5) Skip model download (download later)"
echo ""
read -p "Enter your choice (1-5): " model_choice

case $model_choice in
    1)
        MODEL_ID="FabioSarracino/VibeVoice-Large-Q8"
        MODEL_DIR="VibeVoice-Large-Q8"
        ;;
    2)
        MODEL_ID="microsoft/VibeVoice-1.5B"
        MODEL_DIR="VibeVoice-1.5B"
        ;;
    3)
        MODEL_ID="aoi-ot/VibeVoice-Large"
        MODEL_DIR="VibeVoice-Large"
        ;;
    4)
        MODEL_ID="DevParker/VibeVoice7b-low-vram"
        MODEL_DIR="VibeVoice7b-low-vram"
        ;;
    5)
        log_info "Skipping model download. You can download later using:"
        log_info "huggingface-cli download MODEL_ID --local-dir models/vibevoice/MODEL_NAME --local-dir-use-symlinks False"
        MODEL_ID=""
        ;;
    *)
        log_warning "Invalid choice. Skipping model download."
        MODEL_ID=""
        ;;
esac

# Download selected model
if [ ! -z "$MODEL_ID" ]; then
    log_info "Downloading ${MODEL_ID}..."
    log_warning "This may take a while depending on your internet speed..."
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download ${MODEL_ID} \
            --local-dir models/vibevoice/${MODEL_DIR} \
            --local-dir-use-symlinks False
        log_success "Model downloaded successfully!"
    else
        log_error "huggingface-cli not found!"
        exit 1
    fi
fi

# Create sample voice file placeholder
if [ ! -f "voice_samples/README.txt" ]; then
    cat > voice_samples/README.txt << 'EOF'
# Voice Samples Directory

Place your voice sample files here for voice cloning.

Supported formats:
- .mp3 (recommended)
- .wav
- .flac
- .ogg

Tips for best results:
1. Use clean audio (no background noise)
2. 30-60 seconds of speech is ideal
3. Single speaker only
4. Higher quality = better cloning

Example usage:
python3 vibevoice_standalone_mp3.py \
    --text "Your text here" \
    --voice-sample voice_samples/my_voice.mp3 \
    --output outputs/cloned_speech.mp3
EOF
    log_success "Created voice_samples/README.txt"
fi

# Create example text file
if [ ! -f "scripts/example.txt" ]; then
    cat > scripts/example.txt << 'EOF'
Hello and welcome to VibeVoice!

This is an example script file that demonstrates how to use text files for longer speech generation.

You can write multiple paragraphs, and the system will automatically handle them.

To use this file, run:
python3 vibevoice_standalone_mp3.py --text-file scripts/example.txt --output outputs/example_speech.mp3

Enjoy your voice cloning experience!
EOF
    log_success "Created scripts/example.txt"
fi

# Create activation helper script
cat > activate_vv.sh << 'EOF'
#!/bin/bash
# Quick activation script for VibeVoice environment
source vv_env/bin/activate
echo "✅ VibeVoice environment activated!"
echo "🎙️  Ready to clone voices!"
echo ""
echo "Quick start:"
echo "  python3 vibevoice_standalone_mp3.py --text 'Hello world!' --output speech.mp3"
echo ""
EOF
chmod +x activate_vv.sh

log_success "Created activate_vv.sh helper script"

# Create quick start script
cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick test script to verify everything works
source vv_env/bin/activate

echo "🎙️ Running quick test..."
python3 vibevoice_standalone_mp3.py \
    --text "VibeVoice is working perfectly! Voice cloning activated!" \
    --diffusion-steps 5 \
    --output outputs/quick_test.mp3

if [ $? -eq 0 ]; then
    echo "✅ Test successful! Audio saved to outputs/quick_test.mp3"
else
    echo "❌ Test failed. Please check the error messages above."
fi
EOF
chmod +x quick_test.sh

log_success "Created quick_test.sh"

# Final summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              🎉 SETUP COMPLETE! 🎉                     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
log_success "Everything is installed and ready to go!"
echo ""
echo "📁 Directory Structure:"
echo "   ├── models/vibevoice/        # Downloaded models"
echo "   ├── voice_samples/           # Your voice files"
echo "   ├── outputs/                 # Generated audio"
echo "   ├── scripts/                 # Text scripts"
echo "   └── vv_env/                  # Python environment"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "1. Activate environment:"
echo "   ${GREEN}source activate_vv.sh${NC}"
echo "   (or: source vv_env/bin/activate)"
echo ""
echo "2. Run quick test:"
echo "   ${GREEN}./quick_test.sh${NC}"
echo ""
echo "3. Generate speech:"
echo "   ${GREEN}python3 vibevoice_standalone_mp3.py --text 'Hello world!' --output speech.mp3${NC}"
echo ""
echo "4. Voice cloning:"
echo "   ${GREEN}python3 vibevoice_standalone_mp3.py --text 'Clone me!' --voice-sample voice_samples/my_voice.mp3 --output cloned.mp3${NC}"
echo ""
echo "📚 For more options, run:"
echo "   ${GREEN}python3 vibevoice_standalone_mp3.py --help${NC}"
echo ""
log_info "🎤 Happy voice cloning! 🎤"
echo ""