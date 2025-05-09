import cv2
import numpy as np
import json
import logging
from deepface import DeepFace
from mtcnn import MTCNN
from driver_api.models import Driver

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "ArcFace" #

def align_face(face_image, landmarks):
    """Align face using eye landmarks before embedding extraction"""
    try:
        left_eye, right_eye = landmarks.get("left_eye"), landmarks.get("right_eye")
        if not left_eye or not right_eye:
            logger.warning("⚠️ No valid eye landmarks detected. Skipping alignment.")
            return face_image

        dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Get rotation matrix
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

        # Apply transformation
        aligned = cv2.warpAffine(face_image, M, (face_image.shape[1], face_image.shape[0]), flags=cv2.INTER_CUBIC)
        return aligned
    except Exception as e:
        logger.warning(f"⚠️ Alignment failed: {str(e)}")
        return face_image

def get_face_embedding(face_image, landmarks=None):
    """Generate face embedding using ArcFace via DeepFace"""
    try:
        if landmarks:
            face_image = align_face(face_image, landmarks)

        # Convert to RGB format
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Generate embedding using DeepFace (ArcFace model)
        embedding = DeepFace.represent(face_image, model_name=MODEL_NAME, enforce_detection=False, detector_backend="skip")[0]["embedding"]

        # Normalize embedding
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        
        logger.debug(f"📦 Extracted embedding (sample): {embedding[:5]}")
        return embedding
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {str(e)}")
        return None