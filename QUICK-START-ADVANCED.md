# 🚀 Quick Start - Advanced AI Models

This guide will help you set up the advanced AI models for the Baby Face Generator.

## 🎯 Current Status

✅ **Simple Fallback**: Working (OpenCV-based)
⏳ **Advanced Models**: Available for installation
⏳ **BabyGAN**: Ready for setup
⏳ **Stable Diffusion**: Ready for installation

## 🔧 Installation Options

### Option 1: Keep Simple Fallback (Recommended for now)
The simple fallback is already working and provides good results:

```bash
# Already working - no additional setup needed
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Install Stable Diffusion (Better Quality)
```bash
# Install PyTorch and Stable Diffusion
cd backend
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate
```

### Option 3: Full Advanced Setup (Best Quality)
```bash
# Install all advanced models
./setup-advanced-models.sh
```

## 🧪 Testing Your Setup

### Test Current Setup
```bash
cd backend
source venv/bin/activate
python -c "
import main
print('✅ Backend loaded successfully!')
print('Current model:', main.baby_generator._get_model_name() if hasattr(main.baby_generator, '_get_model_name') else 'Simple')
"
```

### Test API Endpoint
```bash
# Start backend
cd backend && source venv/bin/activate && uvicorn main:app --reload &

# Test API
curl http://localhost:8000/health
```

## 📱 Using the Application

1. **Start Backend**: `cd backend && source venv/bin/activate && uvicorn main:app --reload`
2. **Start Frontend**: `npm run dev`
3. **Open Browser**: http://localhost:3000
4. **Upload Photos**: Two clear face photos
5. **Generate Baby**: Click "Generate Our Baby!"

## 🔄 Model Fallback System

The system automatically uses the best available model:

1. **BabyGAN** (if available) - Best quality, StyleGAN-based
2. **Stable Diffusion** (if available) - Good quality, text-to-image
3. **Simple OpenCV** (always available) - Fast, basic blending

## 💡 Performance Tips

### For Development
- Use **Simple Fallback** for fast iteration
- Install **Stable Diffusion** for better quality testing

### For Production
- Use **BabyGAN** for best quality
- Use **Stable Diffusion** for good balance
- Keep **Simple Fallback** as backup

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### "No module named 'diffusers'"
```bash
# Install Stable Diffusion
pip install diffusers transformers accelerate
```

### "BabyGAN not found"
```bash
# Clone BabyGAN repository
git clone https://github.com/tg-bomze/BabyGAN.git backend/babygan
```

## 🎉 Success!

Once everything is working, you should see:

- ✅ Backend running on http://localhost:8000
- ✅ Frontend running on http://localhost:3000
- ✅ API responding to requests
- ✅ Baby face generation working

## 📊 Model Comparison

| Model | Quality | Speed | Setup | Best For |
|-------|---------|-------|-------|----------|
| Simple | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Ready | Development |
| Stable Diffusion | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⏳ Install | Production |
| BabyGAN | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⏳ Setup | Best Quality |

## 🚀 Next Steps

1. **Test the current setup** with simple fallback
2. **Install Stable Diffusion** for better quality
3. **Set up BabyGAN** for best results
4. **Deploy to production** with your preferred model

---

**Current Recommendation**: Start with the simple fallback (already working) and upgrade to advanced models as needed! 🎯
