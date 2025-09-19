#!/bin/bash

echo "🍎 Installing Baby Face Generator for macOS..."
echo "=============================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    echo "   You can install it from: https://www.python.org/downloads/"
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

# Install requirements (macOS-compatible version)
echo "📦 Installing Python dependencies..."
pip install -r requirements-macos.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Backend dependencies installed successfully!"
    echo ""
    echo "🚀 To start the development server:"
    echo "   cd backend"
    echo "   source venv/bin/activate"
    echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "🎉 Installation complete!"
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "   1. Make sure you have Xcode command line tools: xcode-select --install"
    echo "   2. Try installing OpenCV separately: pip install opencv-python-headless"
    echo "   3. If MediaPipe fails, try: pip install mediapipe-silicon (for Apple Silicon)"
    echo "   4. Check your Python version: python3 --version (should be 3.10+)"
fi
