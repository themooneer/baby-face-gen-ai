# 👶 AI Baby Face Generator

A mobile-friendly web application that generates realistic baby faces by blending two parent photos using AI. Built with React, FastAPI, and computer vision techniques.

![Baby Face Generator](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![React](https://img.shields.io/badge/React-18.2.0-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.2.2-blue)

## ✨ Features

- 📱 **Mobile-First Design**: Responsive UI optimized for phones and tablets
- 🖼️ **Easy Photo Upload**: Drag & drop interface for mom and dad photos
- 🤖 **AI Face Blending**: Advanced face detection and morphing algorithms
- 👶 **Baby-fy Transformation**: Post-processing to make faces look more childlike
- ⚡ **Fast Processing**: Optimized for quick generation
- 💾 **HD Download**: High-resolution image downloads
- 🔒 **Privacy-First**: Photos are automatically deleted after processing

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- Git

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The app will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🛠️ Tech Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for responsive styling
- **React Dropzone** for file uploads
- **Lucide React** for icons
- **Axios** for API calls

### Backend
- **FastAPI** for the API server
- **OpenCV** for image processing
- **InsightFace** for face detection
- **NumPy & SciPy** for mathematical operations
- **Pillow** for image manipulation

## 🧠 AI Processing Pipeline

1. **Face Detection**: Uses InsightFace to detect and extract faces from uploaded photos
2. **Landmark Extraction**: Identifies key facial features and landmarks
3. **Face Blending**: Morphs the two parent faces using landmark-based transformation
4. **Baby-fy Transformation**: Applies post-processing to make the result look more childlike:
   - Enlarges eyes by 15%
   - Softens skin texture
   - Adjusts facial proportions
   - Adds warm color tinting

## 📁 Project Structure

```
baby-face-generator/
├── src/                    # Frontend React app
│   ├── components/         # Reusable UI components
│   ├── App.tsx            # Main app component
│   └── main.tsx           # App entry point
├── backend/               # FastAPI backend
│   ├── main.py           # API server
│   ├── ai_processor.py   # AI processing logic
│   └── requirements.txt  # Python dependencies
├── public/               # Static assets
├── vercel.json          # Vercel deployment config
├── render.yaml          # Render deployment config
└── README.md
```

## 🚀 Deployment

### Frontend (Vercel)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variable: `VITE_API_URL=https://your-backend-url.onrender.com`
4. Deploy!

### Backend (Render)

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Deploy!

### Alternative: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Optional: Replicate API key for advanced models
REPLICATE_API_TOKEN=your_token_here

# Optional: Custom model settings
FACE_DETECTION_CONFIDENCE=0.8
BABY_TRANSFORMATION_STRENGTH=0.7
```

### API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /api/generate-baby` - Generate baby face from two photos

## 🎨 Customization

### Styling
The app uses Tailwind CSS with custom baby-themed colors. Modify `tailwind.config.js` to change the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      baby: {
        pink: '#F8BBD9',
        blue: '#B8E6E6',
        yellow: '#FFF2CC',
        purple: '#E6CCFF',
      }
    }
  }
}
```

### AI Processing
Modify `backend/ai_processor.py` to adjust the baby-fy transformation:

```python
def _babyfy_face(self, face: np.ndarray) -> np.ndarray:
    # Adjust these parameters:
    eye_scale = 1.15        # Eye enlargement factor
    skin_smoothness = 9     # Bilateral filter parameter
    saturation_boost = 1.1  # Color saturation increase
```

## 🧪 Testing

### Frontend Tests
```bash
npm run test
```

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Manual Testing
1. Upload two clear face photos
2. Verify face detection works
3. Check that the generated image looks reasonable
4. Test download functionality
5. Verify mobile responsiveness

## 🐛 Troubleshooting

### Common Issues

**"No face detected" error:**
- Ensure photos have clear, well-lit faces
- Try different angles or lighting
- Check that faces are not too small in the image

**Slow processing:**
- The first run may be slow due to model loading
- Consider using GPU acceleration for production
- Optimize image sizes before upload

**Memory issues:**
- Reduce image sizes in the frontend
- Implement image compression
- Use streaming for large files

## 🔒 Privacy & Security

- **No Data Storage**: All uploaded photos are automatically deleted after processing
- **No User Tracking**: No analytics or user data collection
- **Local Processing**: All AI processing happens on the server, not in the browser
- **HTTPS Only**: All communication is encrypted in production

## 📈 Performance Optimization

- **Image Compression**: Automatic resizing and compression
- **Caching**: Static assets are cached by CDN
- **Lazy Loading**: Components load only when needed
- **Code Splitting**: JavaScript bundles are optimized

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- [OpenCV](https://opencv.org/) for image processing
- [React](https://reactjs.org/) and [FastAPI](https://fastapi.tiangolo.com/) for the framework
- [Tailwind CSS](https://tailwindcss.com/) for styling

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing [GitHub Issues](https://github.com/yourusername/baby-face-generator/issues)
3. Create a new issue with detailed information

---

Made with ❤️ for fun — not a genetic prediction! 👶
