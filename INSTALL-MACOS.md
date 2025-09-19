# 🍎 macOS Installation Guide

This guide will help you install the AI Baby Face Generator on macOS, addressing common compilation issues.

## 🚨 Common Issues on macOS

The main issue you're experiencing is with `insightface` compilation on macOS. This is due to:
- Xcode version compatibility issues
- C++ standard library conflicts
- Missing system dependencies

## 🔧 Solution Options

### Option 1: Simple Installation (Recommended)

Use the simplified version that only requires basic libraries:

```bash
# Install frontend dependencies
npm install

# Install backend dependencies (simple version)
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-simple.txt

# Start the servers
# Terminal 1 (Backend):
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 (Frontend):
npm run dev
```

### Option 2: Full Installation with MediaPipe

If you want the full AI capabilities:

```bash
# Run the macOS installation script
./install-macos.sh
```

### Option 3: Manual Installation

If the scripts don't work, try this step-by-step approach:

#### 1. Install Xcode Command Line Tools
```bash
xcode-select --install
```

#### 2. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 3. Install system dependencies
```bash
brew install cmake
brew install pkg-config
```

#### 4. Create virtual environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

#### 5. Install Python dependencies
```bash
# Try the macOS-compatible version first
pip install -r requirements-macos.txt

# If that fails, use the simple version
pip install -r requirements-simple.txt
```

## 🐛 Troubleshooting

### Issue: "command '/usr/bin/clang++' failed"
**Solution**: This is the Xcode compilation error you're seeing. Use Option 1 (Simple Installation) above.

### Issue: "No module named 'cv2'"
**Solution**:
```bash
pip install opencv-python-headless
```

### Issue: "No module named 'mediapipe'"
**Solution**:
```bash
# For Intel Macs
pip install mediapipe

# For Apple Silicon Macs
pip install mediapipe-silicon
```

### Issue: "Permission denied" when installing packages
**Solution**:
```bash
pip install --user -r requirements-simple.txt
```

### Issue: "Python version not supported"
**Solution**: Make sure you're using Python 3.10+:
```bash
python3 --version
# If not 3.10+, install from https://www.python.org/downloads/
```

## 🚀 Quick Start (Simple Version)

1. **Install dependencies**:
   ```bash
   npm install
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-simple.txt
   ```

2. **Start the application**:
   ```bash
   # Terminal 1 - Backend
   cd backend
   source venv/bin/activate
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

   # Terminal 2 - Frontend
   npm run dev
   ```

3. **Open your browser**: http://localhost:3000

## 📱 Testing the Application

1. Upload two clear face photos (one of each parent)
2. Click "Generate Our Baby!"
3. Wait for processing (should take 5-10 seconds)
4. Download the generated baby face

## 🔍 What Each Version Does

### Simple Version (`ai_processor_simple.py`)
- ✅ Uses only OpenCV (no complex dependencies)
- ✅ Basic face detection with Haar cascades
- ✅ Simple alpha blending of faces
- ✅ Baby-fy transformation (enlarged eyes, softer skin)
- ✅ Works on all macOS versions

### Full Version (`ai_processor.py`)
- ✅ Uses MediaPipe for advanced face detection
- ✅ Precise facial landmark detection
- ✅ More sophisticated face blending
- ✅ Better baby-fy transformation
- ⚠️ Requires more complex installation

## 🎯 Recommended Approach

For most users, I recommend starting with the **Simple Version** because:
- It installs quickly and reliably
- It produces good results
- It works on all macOS versions
- You can always upgrade later

## 📞 Need Help?

If you're still having issues:

1. **Check your Python version**: `python3 --version` (should be 3.10+)
2. **Try the simple version first**: Use `requirements-simple.txt`
3. **Check the logs**: Look at the terminal output for specific error messages
4. **Test with sample images**: Use clear, well-lit face photos

## 🎉 Success!

Once everything is working, you should see:
- Frontend running at http://localhost:3000
- Backend API running at http://localhost:8000
- API documentation at http://localhost:8000/docs

Happy baby face generating! 👶✨
