#!/bin/bash

echo "🤖 Setting up Advanced AI Models for Baby Face Generator"
echo "========================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python 3 and pip3 are available"

# Create virtual environment
echo "🐍 Creating virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (for CPU)
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install advanced requirements
echo "📦 Installing advanced AI dependencies..."
pip install -r requirements-advanced.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Advanced AI dependencies installed successfully!"
    echo ""
    echo "🧪 Testing model loading..."

    # Test the advanced processor
    python3 -c "
try:
    from ai_processor_advanced import BabyFaceGeneratorAdvanced
    generator = BabyFaceGeneratorAdvanced()
    print('✅ Advanced AI processor loaded successfully!')
    print(f'Model: {generator._get_model_name()}')
except Exception as e:
    print(f'❌ Advanced AI processor failed: {e}')
    print('Falling back to simple processor...')
    from ai_processor_simple import BabyFaceGenerator
    generator = BabyFaceGenerator()
    print('✅ Simple processor loaded successfully!')
"

    echo ""
    echo "🚀 To start the development server:"
    echo "   cd backend"
    echo "   source venv/bin/activate"
    echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📋 Available Models:"
    echo "   1. BabyGAN (StyleGAN-based) - Primary model"
    echo "   2. Stable Diffusion - Fallback model"
    echo "   3. Simple OpenCV - Final fallback"
    echo ""
    echo "💡 Note: First run may be slow due to model downloading"

else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "   1. Make sure you have enough disk space (models are large)"
    echo "   2. Try installing PyTorch separately: pip install torch torchvision"
    echo "   3. For GPU support, install CUDA version of PyTorch"
    echo "   4. Check your Python version: python3 --version (should be 3.10+)"
fi
