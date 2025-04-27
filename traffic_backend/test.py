import tensorflow as tf

# Path to the model
MODEL_PATH = "C:/Users/hello/Desktop/smartTrafficFaceRecogntion/backend/traffic_backend/ml_models/facenet_keras.h5"

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")