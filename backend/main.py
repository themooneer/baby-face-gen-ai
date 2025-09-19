from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('backend.log')  # File output
    ]
)
logger = logging.getLogger(__name__)
# Try to import the advanced processor with BabyGAN/Stable Diffusion, fallback to simple version
try:
    from ai_processor_advanced import BabyFaceGeneratorAdvanced
    baby_generator = BabyFaceGeneratorAdvanced()
    logger.info("Using advanced AI processor with BabyGAN/Stable Diffusion")
except ImportError as e:
    logger.warning(f"Advanced AI processor not available: {e}")
    logger.info("Falling back to simple processor")
    try:
        from ai_processor import BabyFaceGenerator
        baby_generator = BabyFaceGenerator()
        logger.info("Using full AI processor with MediaPipe")
    except ImportError as e2:
        logger.warning(f"Full AI processor not available: {e2}")
        logger.info("Falling back to simple processor")
        from ai_processor_simple import BabyFaceGenerator
        baby_generator = BabyFaceGenerator()

app = FastAPI(title="AI Baby Face Generator API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI processor is initialized above with fallback logic

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "AI Baby Face Generator API is running!"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/generate-baby")
async def generate_baby(
    mom_photo: UploadFile = File(...),
    dad_photo: UploadFile = File(...)
):
    """
    Generate a baby face from two parent photos
    """
    start_time = time.time()
    logger.info(f"Starting baby face generation - Mom: {mom_photo.filename}, Dad: {dad_photo.filename}")

    # Validate file types
    if not mom_photo.content_type.startswith('image/'):
        logger.error(f"Invalid mom photo type: {mom_photo.content_type}")
        raise HTTPException(status_code=400, detail="Mom's photo must be an image file")

    if not dad_photo.content_type.startswith('image/'):
        logger.error(f"Invalid dad photo type: {dad_photo.content_type}")
        raise HTTPException(status_code=400, detail="Dad's photo must be an image file")

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as mom_temp:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as dad_temp:
            try:
                # Save uploaded files
                mom_content = await mom_photo.read()
                dad_content = await dad_photo.read()

                mom_temp.write(mom_content)
                dad_temp.write(dad_content)

                mom_temp.flush()
                dad_temp.flush()

                # Generate baby face
                logger.info("Generating baby face...")
                result_path = await baby_generator.generate_baby_face(
                    mom_temp.name,
                    dad_temp.name
                )

                processing_time = f"{time.time() - start_time:.1f}s"
                logger.info(f"Baby face generation completed in {processing_time}")

                # In a real deployment, you'd upload to a CDN or cloud storage
                # For now, we'll return a base64 encoded image
                import base64
                with open(result_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_url = f"data:image/png;base64,{image_base64}"

                logger.info("Baby face generation successful")
                return JSONResponse(content={
                    "success": True,
                    "imageUrl": image_url,
                    "processingTime": processing_time
                })

            except Exception as e:
                logger.error(f"Baby face generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating baby face: {str(e)}")

            finally:
                # Clean up temporary files
                try:
                    os.unlink(mom_temp.name)
                    os.unlink(dad_temp.name)
                    if 'result_path' in locals():
                        os.unlink(result_path)
                except:
                    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
