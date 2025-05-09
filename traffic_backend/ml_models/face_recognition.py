
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

# === Constants ===
SIMILARITY_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.90
MODEL_NAME = "ArcFace" #

# === Initialize Models ===
try:
    logger.info("🔄 Initializing MTCNN detector...")
    detector = MTCNN()
    logger.info("✅ Models initialized successfully!")
except Exception as e:
    logger.error(f"❌ Model initialization failed: {str(e)}")
    raise

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

def compare_with_database(embedding):
    """Compare embedding against stored embeddings in database"""
    if embedding is None:
        return None

    try:
        drivers = Driver.objects.filter(face_embedding__isnull=False).only("license_number", "name", "face_embedding")
        
        best_match = None
        highest_similarity = -1
        
        for driver in drivers:
            try:
                stored_embedding = np.array(json.loads(driver.face_embedding))
                similarity = np.dot(embedding, stored_embedding)

                logger.debug(f"🔍 Compared with {driver.name}, Similarity={similarity:.4f}")

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = driver
            except Exception as e:
                logger.warning(f"⚠️ Error comparing with {driver.name}: {str(e)}")
                continue

        if best_match and highest_similarity > SIMILARITY_THRESHOLD:
            logger.info(f"✅ Match found: {best_match.license_number} (similarity={highest_similarity:.4f})")
            return best_match.license_number
        else:
            logger.info(f"❌ No match found (best similarity={highest_similarity:.4f})")
            return None
    except Exception as e:
        logger.error(f"❌ Database comparison failed: {str(e)}")
        return None

def recognize_face(image):
    """Face recognition pipeline using live images from Flutter"""
    try:
        if image is None:
            logger.error("❌ Null image provided")
            return None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)

        if not detections:
            logger.info("❌ No faces detected")
            return None, image
        
        matched_license = None
        output_image = image.copy()

        for face in detections:
            if face["confidence"] < CONFIDENCE_THRESHOLD:
                continue

            x, y, w, h = face["box"]
            padding = int(0.2 * max(w, h))
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(image.shape[1] - x, w + 2 * padding), min(image.shape[0] - y, h + 2 * padding)
            
            face_crop = image[y:y+h, x:x+w]

            # Generate embedding
            embedding = get_face_embedding(face_crop, face.get("keypoints"))
            if embedding is None:
                continue

            # Compare with database
            license_number = compare_with_database(embedding)
            
            # Annotate image
            color = (0, 255, 0) if license_number else (0, 0, 255)
            label = license_number if license_number else "Unknown"
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if license_number:
                matched_license = license_number

        return matched_license, output_image

    except Exception as e:
        logger.error(f"❌ Face recognition failed: {str(e)}")
        return None, image
