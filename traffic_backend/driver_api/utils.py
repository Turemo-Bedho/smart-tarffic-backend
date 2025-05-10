import os
import django
import cv2
import numpy as np
import json
import logging
from deepface import DeepFace
from mtcnn import MTCNN
from django.core.exceptions import MultipleObjectsReturned

# === Django Setup ===
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "traffic_backend.settings")
django.setup()
from driver_api.models import Driver

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("driver_registration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Constants ===
SIMILARITY_THRESHOLD = 0.65  # ArcFace works best with higher threshold
MIN_FACE_CONFIDENCE = 0.95    # More strict face detection
MODEL_NAME = "ArcFace"  

def initialize_models():
    """Initialize face detection and recognition models"""
    try:
        logger.info("🔄 Initializing MTCNN detector...")
        detector = MTCNN()
        logger.info("✅ Models initialized successfully")
        return detector
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {str(e)}")
        raise

detector = initialize_models()

def process_face(image):
    """
    Complete face processing pipeline:
    1. Load image
    2. Detect face (with padding)
    3. Align and preprocess
    4. Generate embedding
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces with high confidence
        detections = detector.detect_faces(rgb_image)
        if not detections:
            raise ValueError("No faces detected")

        # Get best face detection
        best_face = max(detections, key=lambda x: x['confidence'])
        if best_face['confidence'] < MIN_FACE_CONFIDENCE:
            raise ValueError(f"Low confidence: {best_face['confidence']:.2f}")

        # Add 25% padding
        x, y, w, h = best_face['box']
        padding = int(0.25 * max(w, h))
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(image.shape[1] - x, w + 2*padding), min(image.shape[0] - y, h + 2*padding)

        # Crop and align face
        face = image[y:y+h, x:x+w]
        aligned_face = align_face(face, best_face['keypoints'])

        # Generate embedding using ArcFace
        embedding = DeepFace.represent(
            cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB),
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip'
        )[0]["embedding"]

        # Normalize embedding
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)

        # Save debug images
        # cv2.imwrite("debug_face.jpg", face)
        # cv2.imwrite("debug_aligned.jpg", aligned_face)

        return embedding, aligned_face

    except Exception as e:
        logger.error(f"Face processing failed: {str(e)}")
        raise

def align_face(face, landmarks):
    """Advanced face alignment using eye landmarks"""
    try:
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Calculate rotation angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # Get rotation matrix
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, 
                      (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply affine transformation
        aligned = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]), 
                               flags=cv2.INTER_CUBIC)
        return aligned
    except Exception as e:
        logger.warning(f"Alignment failed, using original: {str(e)}")
        return face