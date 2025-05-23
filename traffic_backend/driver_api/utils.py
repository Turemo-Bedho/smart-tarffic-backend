import os
import django
import cv2
import numpy as np
import logging
from deepface import DeepFace
from mtcnn import MTCNN

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
        logger.info("[Loading] Initializing MTCNN detector...")
        detector = MTCNN()
        logger.info("[SUCCES] Models initialized successfully")
        return detector
    except Exception as e:
        logger.error(f"[FAILED] Model initialization failed: {str(e)}")
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
        # Resize large images first (maintaining aspect ratio)
        max_dim = 1024  # Adjust based on your needs
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces with high confidence
        detections = detector.detect_faces(rgb_image)
        print("Detections:" * 10)
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
        print("Aligned face:" * 10)
        cv2.imwrite("debug_aligned.jpg", aligned_face)

        # Generate embedding using ArcFace
        embedding = DeepFace.represent(
            cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB),
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip',
        )[0]["embedding"]
        print("Embedding:" * 10)

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



def align_face(face_img, landmarks): # Renamed 'face' to 'face_img' to avoid conflict with 'face' from detections
    """Align face based on eye landmarks with robust error handling."""
    print("Aligning face..."*10)
    try:
        # 1. Check if landmarks for eyes are present
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            logger.warning("Alignment skipped: 'left_eye' or 'right_eye' not in landmarks.")
            return face_img # Return original image if key landmarks are missing

        left_eye_coords = landmarks['left_eye']
        right_eye_coords = landmarks['right_eye']

        # 2. Check if eye coordinates are valid 2-element sequences
        if not (isinstance(left_eye_coords, (tuple, list)) and len(left_eye_coords) == 2 and
                isinstance(right_eye_coords, (tuple, list)) and len(right_eye_coords) == 2):
            logger.warning("Alignment skipped: Eye landmarks are not valid 2-element (x,y) coordinates.")
            return face_img

        # 3. Attempt to convert coordinates to float and check for conversion errors
        try:
            lx, ly = float(left_eye_coords[0]), float(left_eye_coords[1])
            rx, ry = float(right_eye_coords[0]), float(right_eye_coords[1])
        except (ValueError, TypeError) as e_convert:
            logger.warning(f"Alignment skipped: Error converting eye coordinates to float ({str(e_convert)}).")
            return face_img

        # 4. Check for NaN or Inf values in coordinates
        if not (np.isfinite(lx) and np.isfinite(ly) and np.isfinite(rx) and np.isfinite(ry)):
            logger.warning("Alignment skipped: Eye coordinates contain NaN or Inf values.")
            return face_img

        # Calculate rotation angle
        # The -180 in your original code might be an attempt to flip;
        # Let's stick to the standard way first, and adjust if needed.
        dY = ry - ly
        dX = rx - lx
        angle = np.degrees(np.arctan2(dY, dX)) # Standard angle to make the eye-line horizontal

        # Calculate center of the eyes for rotation
        eyes_center_x = (lx + rx) / 2.0
        eyes_center_y = (ly + ry) / 2.0

        # 5. Final check for NaN or Inf in calculated eyes_center
        if not (np.isfinite(eyes_center_x) and np.isfinite(eyes_center_y)):
            logger.warning("Alignment skipped: Calculated eyes_center contains NaN or Inf.")
            return face_img
            
        eyes_center_for_rotation = (eyes_center_x, eyes_center_y)

        # Get rotation matrix
        # OpenCV's getRotationMatrix2D expects center as a tuple of floats.
        M = cv2.getRotationMatrix2D(eyes_center_for_rotation, angle, 1.0)
        
        # Apply affine transformation
        # Ensure output size is (width, height)
        output_width = face_img.shape[1]
        output_height = face_img.shape[0]
        
        aligned_face = cv2.warpAffine(
            face_img, M, (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE # or BORDER_REFLECT, BORDER_CONSTANT etc.
                                          # BORDER_REPLICATE is often a safe choice.
        )
        logger.info("[SUCCESS] Face alignment successful.")
        print("Aligned face: Success" * 10)
        return aligned_face

    except Exception as e:
        # This will now catch any other unexpected errors not handled by specific checks above
        logger.error(f"[ERROR] Alignment failed unexpectedly: {str(e)}. Type: {type(e)}. Using original face.")
        return face_img 