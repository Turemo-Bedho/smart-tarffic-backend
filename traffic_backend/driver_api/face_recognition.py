
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

# === Initialize Models ===
try:
    detector = MTCNN()
except Exception as e:
    raise


def align_face(face, landmarks):
    """Align face based on eye landmarks."""
    try:
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        # Ensure numeric values
        left_eye = (float(left_eye[0]), float(left_eye[1]))
        right_eye = (float(right_eye[0]), float(right_eye[1]))

        # Calculate rotation angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate center
       
        eyes_center = (
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0,
        )

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply transformation
        aligned = cv2.warpAffine(
            face, M, (face.shape[1], face.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
        
        return aligned

    except Exception as e:
        print("❌ Error during face alignment:", e)
        return face


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
