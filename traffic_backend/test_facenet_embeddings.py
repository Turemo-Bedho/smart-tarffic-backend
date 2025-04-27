
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from numpy.linalg import norm

# ===== CONFIG =====
IMAGE1_PATH = r"C:\Users\hello\Desktop\smartTrafficFaceRecogntion\backend\traffic_backend\images\face2_African_man.jpg"
IMAGE2_PATH = r"C:\Users\hello\Desktop\smartTrafficFaceRecogntion\backend\traffic_backend\images\handsome-adult-male-posing.jpg"
MODEL_NAME = "ArcFace"  # Best options: "ArcFace", "Facenet", "VGG-Face"

# ===== IMPROVED FACE DETECTION =====
def detect_faces(img):
    """Detect faces using MTCNN with proper error handling"""
    detector = MTCNN()
    try:
        faces = detector.detect_faces(img)
        if not faces:
            raise ValueError("No faces detected")
        return faces[0]['box']  # Returns (x, y, w, h)
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")

# ===== ENHANCED FACE PREPROCESSING =====
def preprocess_face(img_path):
    """Complete face processing pipeline"""
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        # Detect face
        x, y, w, h = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Add 20% padding around the face
        padding = int(0.2 * max(w, h))
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2*padding), min(img.shape[0] - y, h + 2*padding)
        
        # Crop and resize while maintaining aspect ratio
        face = img[y:y+h, x:x+w]
        return face
    except Exception as e:
        raise ValueError(f"Preprocessing failed for {img_path}: {str(e)}")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🔄 Loading and preprocessing images...")
    try:
        face1 = preprocess_face(IMAGE1_PATH)
        face2 = preprocess_face(IMAGE2_PATH)
        cv2.imwrite("face1_aligned.jpg", face1)
        cv2.imwrite("face2_aligned.jpg", face2)
        print("✅ Face alignment successful")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit()

    print(f"\n🧠 Generating embeddings using {MODEL_NAME}...")
    try:
        # Convert to RGB (DeepFace expects RGB)
        embedding1 = DeepFace.represent(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB), 
                                      model_name=MODEL_NAME, 
                                      enforce_detection=False,
                                      detector_backend='skip')[0]["embedding"]
        embedding2 = DeepFace.represent(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB), 
                                      model_name=MODEL_NAME,
                                      enforce_detection=False,
                                      detector_backend='skip')[0]["embedding"]
    except Exception as e:
        print(f"❌ Embedding generation failed: {str(e)}")
        exit()

    # Calculate similarity metrics
    def cosine_similarity(a, b):
        a_norm, b_norm = norm(a), norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)

    distance = norm(np.array(embedding1) - np.array(embedding2))
    similarity = cosine_similarity(embedding1, embedding2)

    # ===== RESULTS =====
    print("\n🔍 Results:")
    print(f"Embedding 1 (first 5): {np.round(embedding1[:5], 5)}")
    print(f"Embedding 2 (first 5): {np.round(embedding2[:5], 5)}")
    print(f"\nEuclidean Distance: {distance:.4f}")
    print(f"Cosine Similarity: {similarity:.4f}")

    # ===== INTERPRETATION =====
    print("\n🧐 Interpretation:")
    if similarity > 0.68:  # Optimal threshold for ArcFace
        print("🔴 POTENTIAL MATCH (Similarity > 0.68)")
    elif similarity > 0.5:
        print("🟡 POSSIBLE MATCH (0.5 < Similarity ≤ 0.68)")
    else:
        print("🟢 DIFFERENT FACES (Similarity ≤ 0.5)")

    print("\n💾 Saved aligned faces: face1_aligned.jpg, face2_aligned.jpg")