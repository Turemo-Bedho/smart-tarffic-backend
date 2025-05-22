
import os
import logging
import cv2
import numpy as np
import json
from deepface import DeepFace
from mtcnn import MTCNN
from driver_api.models import Driver

# === Constants ===
SIMILARITY_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.90
MODEL_NAME = "ArcFace" #

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.FileHandler("driver_registration.log"),
    #     logging.StreamHandler()
    # ]
)
logger = logging.getLogger(__name__)





# === Initialize Models ===
try:
    detector = MTCNN()
except Exception as e:
    raise




def align_face(face_img, landmarks): # Renamed 'face' to 'face_img' to avoid conflict with 'face' from detections
    """Align face based on eye landmarks with robust error handling."""
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
        return aligned_face

    except Exception as e:
        logger.error(f"[ERROR] Alignment failed unexpectedly: {str(e)}. Type: {type(e)}. Using original face.")
        return face_img 



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
        
        return embedding
    except Exception as e:
        return None

def compare_with_database(embedding):
    """Compare embedding against stored embeddings in database"""
    if embedding is None:
        return None

    try:
        drivers = Driver.objects.all().only("id", "embedding", 'license_number')
        
        best_match = None
        highest_similarity = -1
        
        for driver in drivers:
            try:
                stored_embedding = np.array(json.loads(driver.embedding))
                similarity = np.dot(embedding, stored_embedding)


                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = driver
            except Exception as e:
                continue

        if best_match and highest_similarity > SIMILARITY_THRESHOLD:
            return best_match.license_number
        else:
            return None
    except Exception as e:
        return None

def recognize_face_one(image):
    """Face recognition pipeline using live images from Flutter"""
    try:
        if image is None:
            return None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)

        if not detections:
            return None, image
        
        matched_license = None
        output_image = image.copy()
        
        face = detections[0]
        
        if face["confidence"] < CONFIDENCE_THRESHOLD:
            raise Exception("Low confidence in face detection please insert a clear image")
        
        x, y, w, h = face["box"]
        padding = int(0.2 * max(w, h))
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(image.shape[1] - x, w + 2 * padding), min(image.shape[0] - y, h + 2 * padding)
        
        face_crop = image[y:y+h, x:x+w]

        # Generate embedding
        embedding = get_face_embedding(face_crop, face.get("keypoints"))
        
        if embedding is None:
            raise Exception("Failed to generate face embedding")

        # Compare with database
        license_number = compare_with_database(embedding)
        # Annotate image
        color = (0, 255, 0) if license_number else (0, 0, 255)
        label = str(license_number) if license_number else "Unknown"
       
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
       
        return license_number, output_image

    except Exception as e:
        print(e)
        raise Exception(f"Face recognition failed")
