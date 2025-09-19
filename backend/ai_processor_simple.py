import cv2
import numpy as np
from PIL import Image
import os
import tempfile

class BabyFaceGenerator:
    def __init__(self):
        """Initialize the face processing models (simplified version)"""
        print("Simple face processing model loaded (OpenCV only)")

    async def generate_baby_face(self, mom_path: str, dad_path: str) -> str:
        """
        Generate a baby face by blending two parent faces (simplified version)
        """
        try:
            # Load and process images
            mom_img = self._load_and_preprocess_image(mom_path)
            dad_img = self._load_and_preprocess_image(dad_path)

            # Detect faces using simple OpenCV
            mom_face = self._detect_face_simple(mom_img)
            dad_face = self._detect_face_simple(dad_img)

            if mom_face is None:
                raise ValueError("No face detected in mom's photo")
            if dad_face is None:
                raise ValueError("No face detected in dad's photo")

            # Blend the faces
            blended_face = self._blend_faces_simple(mom_face, dad_face)

            # Apply baby-fy transformation
            baby_face = self._babyfy_face(blended_face)

            # Save result
            result_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(result_path, baby_face)

            return result_path

        except Exception as e:
            raise Exception(f"Failed to generate baby face: {str(e)}")

    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _detect_face_simple(self, img: np.ndarray) -> np.ndarray:
        """Simple face detection using OpenCV Haar cascade"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Add padding
        padding = 20
        h_img, w_img = img.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        face = img[y1:y2, x1:x2]
        return face

    def _blend_faces_simple(self, face1: np.ndarray, face2: np.ndarray) -> np.ndarray:
        """Simple face blending using alpha blending"""
        # Resize faces to the same size
        target_size = (512, 512)
        face1_resized = cv2.resize(face1, target_size)
        face2_resized = cv2.resize(face2, target_size)

        # Simple alpha blending
        alpha = 0.5
        blended = cv2.addWeighted(face1_resized, alpha, face2_resized, 1 - alpha, 0)

        return blended

    def _babyfy_face(self, face: np.ndarray) -> np.ndarray:
        """Apply baby-like transformations to make the face look more childlike"""
        # Convert to float for processing
        face_float = face.astype(np.float32) / 255.0

        # 1. Enlarge eyes (simplified - in reality you'd use landmark detection)
        h, w = face_float.shape[:2]

        # Create a mask for the upper part of the face (eyes area)
        eye_mask = np.zeros((h, w), dtype=np.float32)
        eye_region_y = int(h * 0.2)
        eye_region_h = int(h * 0.4)
        eye_mask[eye_region_y:eye_region_y + eye_region_h, :] = 1.0

        # Apply Gaussian blur to soften the mask
        eye_mask = cv2.GaussianBlur(eye_mask, (21, 21), 0)

        # Scale up the eye region slightly
        eye_scale = 1.15
        center_x, center_y = w // 2, eye_region_y + eye_region_h // 2

        # Create transformation matrix for scaling
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, eye_scale)
        face_scaled = cv2.warpAffine(face_float, M, (w, h))

        # Blend the scaled eyes back
        face_float = face_float * (1 - eye_mask[:, :, np.newaxis]) + face_scaled * eye_mask[:, :, np.newaxis]

        # 2. Soften skin (reduce detail and contrast)
        # Apply bilateral filter to smooth skin while preserving edges
        face_soft = cv2.bilateralFilter(face_float, 9, 75, 75)

        # 3. Increase saturation slightly for a more vibrant look
        hsv = cv2.cvtColor(face_soft, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.1  # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        face_soft = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 4. Add a slight warm tint
        face_soft[:, :, 0] = face_soft[:, :, 0] * 1.05  # Slightly more red
        face_soft[:, :, 2] = face_soft[:, :, 2] * 0.95  # Slightly less blue

        # Clip values to valid range
        face_soft = np.clip(face_soft, 0, 1)

        # Convert back to uint8
        result = (face_soft * 255).astype(np.uint8)

        return result
