
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
MODEL_NAME = "ArcFace"        # Best performing model

# === Initialize Models ===
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

# === Face Processing Pipeline ===
def process_face(image_path):
    """
    Complete face processing pipeline:
    1. Load image
    2. Detect face (with padding)
    3. Align and preprocess
    4. Generate embedding
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")

        # Convert to RGB for MTCNN
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

# === Database Operations ===
def is_duplicate(embedding):
    """Check if similar face exists in database"""
    try:
        for driver in Driver.objects.exclude(face_embedding__isnull=True):
            try:
                db_embedding = np.array(json.loads(driver.face_embedding))
                similarity = np.dot(embedding, db_embedding)
                
                logger.debug(f"Compared with {driver.name}: similarity={similarity:.4f}")
                
                if similarity > SIMILARITY_THRESHOLD:
                    logger.warning(f"Duplicate detected: {driver.name} (similarity={similarity:.4f})")
                    return True
            except Exception as e:
                logger.warning(f"Error comparing with {driver.name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Duplicate check failed: {str(e)}")
        raise

def save_driver(data, embedding, face_image):
    """Save driver data to database"""
    try:
        # Convert embedding to JSON-serializable format
        embedding_json = json.dumps(embedding.tolist())

        # Create/update driver record
        driver, created = Driver.objects.update_or_create(
            license_number=data['license_number'],
            defaults={
                'name': data['name'],
                'license_status': data.get('license_status', 'active'),
                'violation_history': data.get('violation_history', ''),
                'penalties': float(data.get('penalties', 0.0)),
                'face_embedding': embedding_json
            }
        )
        
        logger.info(f"{'Created' if created else 'Updated'} driver: {driver.name}")
        return True
    except MultipleObjectsReturned:
        logger.error("Multiple drivers with same license number!")
        return False
    except Exception as e:
        logger.error(f"Failed to save driver: {str(e)}")
        return False

# === Main Registration Function ===
def register_driver(image_path, license_number, name, **kwargs):
    """Complete driver registration workflow"""
    try:
        logger.info(f"Starting registration for {name} ({license_number})")
        
        # Validate inputs
        if not os.path.exists(image_path):
            raise ValueError("Image path does not exist")
            
        if not license_number or not name:
            raise ValueError("License number and name are required")
        
        # Process face and generate embedding
        embedding, face_image = process_face(image_path)
        
        # Check for duplicates
        if is_duplicate(embedding):
            raise ValueError("Similar face already exists in database")
        
        # Prepare driver data
        driver_data = {
            'license_number': license_number,
            'name': name,
            **kwargs
        }
        
        # Save to database
        if not save_driver(driver_data, embedding, face_image):
            raise ValueError("Failed to save driver data")
            
        logger.info("✅ Registration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {str(e)}")
        return False

# === Example Usage ===
if __name__ == "__main__":
    registration_success = register_driver(
        image_path=r"/home/lisping/Downloads/Telegram Desktop/3.jpg",
        license_number="DL124370",
        name="Turemo Bedaho",
        license_status="active",
        violation_history="Speeding, March 2024",
        penalties=100.0
    )
    
    print("\nRegistration status:", "SUCCESS" if registration_success else "FAILED")