import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from scipy.spatial.distance import cdist
import os
import tempfile

class BabyFaceGenerator:
    def __init__(self):
        """Initialize the face processing models"""
        try:
            # Initialize MediaPipe for face detection and landmarks
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils

            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("MediaPipe face analysis models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MediaPipe models: {e}")
            self.face_detection = None
            self.face_mesh = None

    async def generate_baby_face(self, mom_path: str, dad_path: str) -> str:
        """
        Generate a baby face by blending two parent faces
        """
        try:
            # Load and process images
            mom_img = self._load_and_preprocess_image(mom_path)
            dad_img = self._load_and_preprocess_image(dad_path)

            # Detect faces
            mom_face = self._detect_face(mom_img)
            dad_face = self._detect_face(dad_img)

            if mom_face is None:
                raise ValueError("No face detected in mom's photo")
            if dad_face is None:
                raise ValueError("No face detected in dad's photo")

            # Extract facial landmarks
            mom_landmarks = self._extract_landmarks(mom_face)
            dad_landmarks = self._extract_landmarks(dad_face)

            # Blend the faces
            blended_face = self._blend_faces(mom_face, dad_face, mom_landmarks, dad_landmarks)

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

    def _detect_face(self, img: np.ndarray) -> np.ndarray:
        """Detect and extract the largest face from an image using MediaPipe"""
        if self.face_detection is None:
            # Fallback: use OpenCV's Haar cascade
            return self._detect_face_opencv(img)

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_img)

            if not results.detections:
                return None

            # Get the largest face
            h, w = img.shape[:2]
            largest_face = None
            largest_area = 0

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                area = width * height

                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, width, height)

            if largest_face is None:
                return None

            x, y, width, height = largest_face

            # Add some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + width + padding)
            y2 = min(h, y + height + padding)

            face = img[y1:y2, x1:x2]
            return face

        except Exception as e:
            print(f"MediaPipe detection failed: {e}")
            return self._detect_face_opencv(img)

    def _detect_face_opencv(self, img: np.ndarray) -> np.ndarray:
        """Fallback face detection using OpenCV"""
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

    def _extract_landmarks(self, face: np.ndarray) -> np.ndarray:
        """Extract facial landmarks using MediaPipe"""
        if self.face_mesh is None:
            # Fallback: use simple grid-based approach
            return self._extract_landmarks_simple(face)

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)

            if not results.multi_face_landmarks:
                return self._extract_landmarks_simple(face)

            # Get the first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = face.shape[:2]

            # Extract key landmarks (MediaPipe has 468 landmarks)
            landmarks = []

            # Left eye center (landmarks 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_x = sum(face_landmarks.landmark[i].x for i in left_eye_landmarks) / len(left_eye_landmarks)
            left_eye_y = sum(face_landmarks.landmark[i].y for i in left_eye_landmarks) / len(left_eye_landmarks)
            landmarks.append([left_eye_x * w, left_eye_y * h])

            # Right eye center (landmarks 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_x = sum(face_landmarks.landmark[i].x for i in right_eye_landmarks) / len(right_eye_landmarks)
            right_eye_y = sum(face_landmarks.landmark[i].y for i in right_eye_landmarks) / len(right_eye_landmarks)
            landmarks.append([right_eye_x * w, right_eye_y * h])

            # Nose tip (landmark 1)
            nose_x = face_landmarks.landmark[1].x * w
            nose_y = face_landmarks.landmark[1].y * h
            landmarks.append([nose_x, nose_y])

            # Left mouth corner (landmark 61)
            left_mouth_x = face_landmarks.landmark[61].x * w
            left_mouth_y = face_landmarks.landmark[61].y * h
            landmarks.append([left_mouth_x, left_mouth_y])

            # Right mouth corner (landmark 291)
            right_mouth_x = face_landmarks.landmark[291].x * w
            right_mouth_y = face_landmarks.landmark[291].y * h
            landmarks.append([right_mouth_x, right_mouth_y])

            # Chin (landmark 175)
            chin_x = face_landmarks.landmark[175].x * w
            chin_y = face_landmarks.landmark[175].y * h
            landmarks.append([chin_x, chin_y])

            return np.array(landmarks)

        except Exception as e:
            print(f"MediaPipe landmark extraction failed: {e}")
            return self._extract_landmarks_simple(face)

    def _extract_landmarks_simple(self, face: np.ndarray) -> np.ndarray:
        """Fallback: Extract facial landmarks using simple grid-based approach"""
        h, w = face.shape[:2]

        # Create a simple landmark grid
        landmarks = np.array([
            [w * 0.2, h * 0.3],  # Left eye
            [w * 0.8, h * 0.3],  # Right eye
            [w * 0.5, h * 0.5],  # Nose
            [w * 0.3, h * 0.7],  # Left mouth corner
            [w * 0.7, h * 0.7],  # Right mouth corner
            [w * 0.5, h * 0.9],  # Chin
        ])

        return landmarks

    def _blend_faces(self, face1: np.ndarray, face2: np.ndarray,
                    landmarks1: np.ndarray, landmarks2: np.ndarray) -> np.ndarray:
        """Blend two faces using landmark-based morphing"""
        # Resize faces to the same size
        target_size = (512, 512)
        face1_resized = cv2.resize(face1, target_size)
        face2_resized = cv2.resize(face2, target_size)

        # Scale landmarks to new size
        scale_x = target_size[0] / face1.shape[1]
        scale_y = target_size[1] / face1.shape[0]
        landmarks1_scaled = landmarks1 * [scale_x, scale_y]

        scale_x = target_size[0] / face2.shape[1]
        scale_y = target_size[1] / face2.shape[0]
        landmarks2_scaled = landmarks2 * [scale_x, scale_y]

        # Create average landmarks
        avg_landmarks = (landmarks1_scaled + landmarks2_scaled) / 2

        # Simple alpha blending for now
        # In a real implementation, you'd use proper morphing
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
