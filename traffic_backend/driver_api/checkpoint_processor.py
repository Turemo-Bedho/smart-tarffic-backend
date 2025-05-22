# import cv2
# import os
# import base64
# from datetime import datetime
# from django.conf import settings
# import numpy as np
# import logging

# logger = logging.getLogger(__name__)

# class CheckpointProcessor:
#     def __init__(self):
#         from face_recognition import recognize_face_one
#         self.face_recognizer = recognize_face_one
#         self.plate_recognizer = PlateRecognizer()
#         logger.info("Checkpoint processor initialized")

#     def process_frame(self, frame):
#         try:
#             # Face recognition
#             license_number, annotated_frame = self.face_recognizer(frame)
            
#             # Plate recognition only if face not recognized
#             plates = []
#             if not license_number:
#                 plates = self.plate_recognizer.process_image(frame)
                
#                 if plates:
#                     self.save_unrecognized_data(frame, plates)
            
#             # Convert annotated image to base64 for Postman response
#             _, buffer = cv2.imencode('.jpg', annotated_frame)
#             annotated_image = base64.b64encode(buffer).decode('utf-8')
            
#             return {
#                 'success': True,
#                 'face_recognition': license_number,
#                 'plates': plates,
#                 'annotated_image': annotated_image
#             }
#         except Exception as e:
#             logger.error(f"Processing error: {str(e)}")
#             return {
#                 'success': False,
#                 'error': str(e)
#             }

#     def save_unrecognized_data(self, frame, plates):
#         try:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             save_dir = os.path.join(settings.MEDIA_ROOT, 'unrecognized', timestamp)
#             os.makedirs(save_dir, exist_ok=True)
            
#             # Save full frame
#             frame_path = os.path.join(save_dir, 'frame.jpg')
#             cv2.imwrite(frame_path, frame)
            
#             # Save plate data
#             plate_data = []
#             for i, plate in enumerate(plates):
#                 plate_path = os.path.join(save_dir, f'plate_{i}.txt')
#                 with open(plate_path, 'w') as f:
#                     f.write(plate['text'])
#                 plate_data.append({
#                     'text': plate['text'],
#                     'confidence': plate['confidence']
#                 })
            
#             logger.info(f"Saved unrecognized driver data to {save_dir}")
#             return plate_data
#         except Exception as e:
#             logger.error(f"Data saving error: {str(e)}")
#             return None