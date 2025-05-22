



# import cv2
# import easyocr
# import torch
# import numpy as np
# import logging
# import os
# import re
# import time
# from typing import List, Dict, Optional
# logger = logging.getLogger(__name__)

# class EthiopianPlateRecognizer:
#     def __init__(self, debug_mode=False):
#         """Ethiopian License Plate Recognizer with refined detection and debugging"""
#         self.debug_mode = debug_mode
#         print(f"🔍 Debug Mode Enabled: {self.debug_mode}")
#         self.debug_dir = "./debug_images"
#         os.makedirs(self.debug_dir, exist_ok=True)
        
#         try:
#             # Initialize YOLOv5 with refined parameters
#             self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#             print(f"🚀 YOLO Model Loaded Successfully: {self.model}")
#             self.model.conf = 0.4
#             self.model.iou = 0.45
#             self.model.classes = [2, 5, 7]  # Cars, buses, trucks
            
#             # Initialize EasyOCR with English model
#             self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./ocr_models')
#             logger.info("Plate recognition initialized successfully")
#         except Exception as e:
#             logger.error(f"Initialization failed: {str(e)}")
#             raise RuntimeError("Plate recognition system initialization failed")

#     def _debug_save(self, image: np.ndarray, stage: str):
#         """Save debug images for troubleshooting"""
#         if self.debug_mode:
#             timestamp = int(time.time() * 1000)
#             path = os.path.join(self.debug_dir, f"{stage}_{timestamp}.jpg")
#             success = cv2.imwrite(path, image)
#             if success:
#                 print(f"✅ Debug image saved successfully: {path}")
#             else:
#                 print(f"❌ Failed to save debug image: {path}")

#     def detect_plate_regions(self, image: np.ndarray) -> List[Dict]:
#         """Detect vehicle plates using YOLOv5"""
#         try:
#             results = self.model(image)
#             print(f"🛑 YOLO Raw Predictions: {results.xyxy[0]}")
#             plates = []

#             for detection in results.xyxy[0]:
#                 x1, y1, x2, y2, conf, _ = detection.tolist()
#                 if conf > 0.4:
#                     # Ensure plate is tightly cropped
#                     plate_height = int((y2 - y1) * 0.3)  # Focus strictly on plate
#                     plate_y = y1 + int((y2 - y1) * 0.6)

#                     width_expansion = int((x2 - x1) * 0.05)  # Small width expansion
#                     x1 = max(0, x1 - width_expansion)
#                     x2 = min(image.shape[1], x2 + width_expansion)

#                     plates.append({
#                         'bbox': [x1, plate_y, x2, plate_y + plate_height],
#                         'confidence': float(conf)
#                     })
            
#             # Save debug image of detected plates
#             if self.debug_mode and plates:
#                 debug_img = image.copy()
#                 for plate in plates:
#                     x1, y1, x2, y2 = map(int, plate['bbox'])
#                     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 self._debug_save(debug_img, "00_plate_detections")

#             return plates
#         except Exception as e:
#             logger.error(f"Plate detection failed: {str(e)}")
#             return []

#     def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
#         """Optimize plate image for OCR"""
#         gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#         self._debug_save(gray, "01_gray")

#         # Improve contrast before thresholding
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#         self._debug_save(enhanced, "02_clahe_enhanced")

#         # Apply Gaussian Blur to reduce noise
#         blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#         self._debug_save(blurred, "03_blur")

#         # Use Otsu’s thresholding for better segmentation
#         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         self._debug_save(thresh, "04_threshold")

#         return thresh

   
#     def recognize_plate_text(self, plate_img: np.ndarray) -> Optional[str]:
#         """Extract plate text using OCR and ensure proper assembly"""

#         try:
#             processed = self._preprocess_plate(plate_img)

#             # Perform OCR with improved parameters
#             results = self.reader.readtext(
#                 processed,
#                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\u1200-\u137F',
#                 text_threshold=0.7,
#                 width_ths=0.6,
#                 decoder='beamsearch'
#             )

#             # Debug OCR output
#             detected_text_list = []
#             for res in results:
#                 text, confidence = res[1], res[2]
#                 print(f"🔎 OCR detected: {text}, Confidence: {confidence}")

#                 # Only add text if confidence is above 0.5
#                 if confidence >= 0.45:
#                     detected_text_list.append(text)

#             # Merge detected characters into one plate string
#             detected_text = " ".join(detected_text_list).strip().upper()
#             print(f"🚗 OCR Full Plate Text: {detected_text}")

#             # Validate Ethiopian plate format
#             # clean_text = ''.join(c for c in detected_text if c.isalnum()).upper()
#             # if re.match(r'^[A-Za-z]{2,3}\d{3,4}$|^\d[A-Za-z]{2,3}\d{2,3}$|^\d{2}[A-Za-z]{2}\d{2,3}$', clean_text):
#             #     return clean_text
#             return detected_text if detected_text else None
#         except Exception as e:
#             print(f"❌ OCR failed: {str(e)}")
#             return None


#     def process_image(self, image: np.ndarray) -> List[Dict]:
#         """Complete plate recognition pipeline"""
#         plates = self.detect_plate_regions(image)
#         results = []

#         for i, plate in enumerate(plates):
#             x1, y1, x2, y2 = map(int, plate['bbox'])
#             plate_img = image[y1:y2, x1:x2]

#             # Save cropped plate for debugging
#             if self.debug_mode:
#                 self._debug_save(plate_img, f"05_plate_crop_{i}")

#             plate_text = self.recognize_plate_text(plate_img)

#             if plate_text:
#                 results.append({
#                     'text': plate_text,
#                     'bbox': plate['bbox'],
#                     'confidence': plate['confidence']
#                 })

#                 # Debug recognized plate
#                 if self.debug_mode:
#                     debug_img = plate_img.copy()
#                     cv2.putText(debug_img, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     self._debug_save(debug_img, f"06_recognized_{i}")

#         return results









# import cv2
# import easyocr
# import torch
# from ultralytics import YOLO  # Import YOLOv8
# import numpy as np
# import logging
# import os
# import re
# import time
# from typing import List, Dict, Optional

# logger = logging.getLogger(__name__)

# class EthiopianPlateRecognizer:
#     def __init__(self, debug_mode=False):
#         """Ethiopian License Plate Recognizer with YOLOv8 integration"""
#         self.debug_mode = debug_mode
#         print(f"🔍 Debug Mode Enabled: {self.debug_mode}")
#         self.debug_dir = "./debug_images"
#         os.makedirs(self.debug_dir, exist_ok=True)
        
#         try:
#             # Load the trained YOLOv8 model
#             model_path = "yolo_model/best.pt"  # Ensure this path is correct
#             self.model = YOLO(model_path)  # Load custom YOLOv8 model
#             print(f"🚀 YOLOv8 Model Loaded Successfully: {model_path}")

#             # Initialize EasyOCR with Ethiopian language support
#             self.reader = easyocr.Reader(['en','amh'], gpu=False, model_storage_directory='./ocr_models')
#             logger.info("Plate recognition initialized successfully")
#         except Exception as e:
#             logger.error(f"Initialization failed: {str(e)}")
#             raise RuntimeError("Plate recognition system initialization failed")

#     def _debug_save(self, image: np.ndarray, stage: str):
#         """Save debug images for troubleshooting"""
#         if self.debug_mode:
#             timestamp = int(time.time() * 1000)
#             path = os.path.join(self.debug_dir, f"{stage}_{timestamp}.jpg")
#             success = cv2.imwrite(path, image)
#             if success:
#                 print(f"✅ Debug image saved successfully: {path}")
#             else:
#                 print(f"❌ Failed to save debug image: {path}")

#     def detect_plate_regions(self, image: np.ndarray) -> List[Dict]:
#         """Detect vehicle plates using YOLOv8"""
#         try:
#             results = self.model(image)  # Run YOLOv8 inference
#             detections = results[0].boxes.data  # YOLOv8 stores predictions differently

#             print(f"🛑 YOLOv8 Raw Predictions: {detections}")  # Debug output

#             plates = []
#             for detection in detections:
#                 x1, y1, x2, y2, conf, class_id = detection.tolist()
#                 if conf > 0.4:  # Adjust confidence threshold if needed
#                     plates.append({
#                         'bbox': [int(x1), int(y1), int(x2), int(y2)],
#                         'confidence': float(conf)
#                     })
            
#             # Save debug image of detected plates
#             if self.debug_mode and plates:
#                 debug_img = image.copy()
#                 for plate in plates:
#                     x1, y1, x2, y2 = plate['bbox']
#                     cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 self._debug_save(debug_img, "00_plate_detections")

#             return plates
#         except Exception as e:
#             logger.error(f"Plate detection failed: {str(e)}")
#             return []

#     def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
#         """Optimize plate image for OCR"""
#         gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#         self._debug_save(gray, "01_gray")

#         # Improve contrast before thresholding
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#         self._debug_save(enhanced, "02_clahe_enhanced")

#         # Apply Gaussian Blur to reduce noise
#         blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
#         self._debug_save(blurred, "03_blur")

#         # Use Otsu’s thresholding for better segmentation
#         _, thresh = cv2.adaptiveThreshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         self._debug_save(thresh, "04_threshold")

#         return thresh

#     def recognize_plate_text(self, plate_img: np.ndarray) -> Optional[str]:
#         """Extract plate text using OCR and ensure proper assembly"""
#         try:
#             processed = self._preprocess_plate(plate_img)

#             # Perform OCR with improved parameters
#             results = self.reader.readtext(
#                 processed,
#                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\u1200-\u137F',
#                 text_threshold=0.8,
#                 width_ths=0.6,
#                 decoder='beamsearch'
#             )

#             # Debug OCR output
#             detected_text_list = []
#             for res in results:
#                 text, confidence = res[1], res[2]
#                 print(f"🔎 OCR detected: {text}, Confidence: {confidence}")

#                 # Only add text if confidence is above 0.5
#                 if confidence >= 0.35:
#                     detected_text_list.append(text)

#             # Merge detected characters into one plate string
#             detected_text = " ".join(detected_text_list).strip().upper()
#             print(f"🚗 OCR Full Plate Text: {detected_text}")

#             return detected_text if detected_text else None
#         except Exception as e:
#             print(f"❌ OCR failed: {str(e)}")
#             return None

#     def process_image(self, image: np.ndarray) -> List[Dict]:
#         """Complete plate recognition pipeline"""
#         plates = self.detect_plate_regions(image)
#         results = []

#         for i, plate in enumerate(plates):
#             x1, y1, x2, y2 = plate['bbox']
#             plate_img = image[y1:y2, x1:x2]

#             # Save cropped plate for debugging
#             if self.debug_mode:
#                 self._debug_save(plate_img, f"05_plate_crop_{i}")

#             plate_text = self.recognize_plate_text(plate_img)

#             if plate_text:
#                 results.append({
#                     'text': plate_text,
#                     'bbox': plate['bbox'],
#                     'confidence': plate['confidence']
#                 })

#                 # Debug recognized plate
#                 if self.debug_mode:
#                     debug_img = plate_img.copy()
#                     cv2.putText(debug_img, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     self._debug_save(debug_img, f"06_recognized_{i}")

#         return results


import cv2
import easyocr
import torch
from ultralytics import YOLO
import numpy as np
import logging
import os
import time
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class EthiopianPlateRecognizer:
    def __init__(self, debug_mode=False):
        """Ethiopian License Plate Recognizer with YOLOv8 integration"""
        self.debug_mode = debug_mode
        print(f"🔍 Debug Mode Enabled: {self.debug_mode}")
        self.debug_dir = "./debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        try:
            # Load the trained YOLOv8 model
            model_path = "yolo_model/best.pt"
            self.model = YOLO(model_path)
            print(f"🚀 YOLOv8 Model Loaded Successfully: {model_path}")

            # Initialize EasyOCR with English and custom Amharic characters
            # Note: EasyOCR doesn't support 'amh' directly, so we use allowlist for Amharic characters
            self.reader = easyocr.Reader(['en'], gpu=False)  # English only base
            logger.info("Plate recognition initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError("Plate recognition system initialization failed")

    def _debug_save(self, image: np.ndarray, stage: str):
        """Save debug images for troubleshooting"""
        if self.debug_mode:
            timestamp = int(time.time() * 1000)
            path = os.path.join(self.debug_dir, f"{stage}_{timestamp}.jpg")
            success = cv2.imwrite(path, image)
            if success:
                print(f"✅ Debug image saved successfully: {path}")
            else:
                print(f"❌ Failed to save debug image: {path}")

    def detect_plate_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicle plates using YOLOv8"""
        try:
            results = self.model(image)
            detections = results[0].boxes.data

            print(f"🛑 YOLOv8 Raw Predictions: {detections}")

            plates = []
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection.tolist()
                if conf > 0.4:
                    plates.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
            
            if self.debug_mode and plates:
                debug_img = image.copy()
                for plate in plates:
                    x1, y1, x2, y2 = plate['bbox']
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self._debug_save(debug_img, "00_plate_detections")

            return plates
        except Exception as e:
            logger.error(f"Plate detection failed: {str(e)}")
            return []

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Optimize plate image for OCR with Ethiopian characters"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        self._debug_save(gray, "01_gray")

        # Enhance contrast specifically for Ethiopian plates
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        self._debug_save(enhanced, "02_clahe_enhanced")

        # Denoising for Amharic characters
        # blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # self._debug_save(blurred, "03_blur")

        # Thresholding with Ethiopian character consideration
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._debug_save(thresh, "04_threshold")

        return thresh

    def recognize_plate_text(self, plate_img: np.ndarray) -> Optional[str]:
        """Extract plate text with support for Ethiopian characters"""
        try:
            processed = self._preprocess_plate(plate_img)

            # Define Amharic Unicode ranges (Ethiopic block)
            ethiopic_chars = ''.join([chr(c) for c in range(0x1200, 0x137F)])
            custom_allowlist = f'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{ethiopic_chars}'

            # Perform OCR with custom settings for Ethiopian plates
            results = self.reader.readtext(
                processed,
                allowlist=custom_allowlist,
                text_threshold=0.7,  # Slightly lower for Amharic
                width_ths=0.7,       # More tolerant for connected characters
                decoder='beamsearch',
                batch_size=10,
                detail=1
            )

            # Process results with confidence filtering
            detected_text_list = []
            for res in results:
                text, confidence = res[1], res[2]
                print(f"🔎 OCR detected: {text}, Confidence: {confidence}")

                if confidence >= 0.3:  # Lower threshold for Amharic
                    detected_text_list.append(text)

            # Merge and clean the text
            detected_text = " ".join(detected_text_list).strip()
            
            # Post-processing for Ethiopian plates
            detected_text = self._postprocess_text(detected_text)
            print(f"🚗 OCR Full Plate Text: {detected_text}")

            return detected_text if detected_text else None
        except Exception as e:
            print(f"❌ OCR failed: {str(e)}")
            return None

    def _postprocess_text(self, text: str) -> str:
        """Clean and format recognized text for Ethiopian plates"""
        # Remove common OCR artifacts
        text = re.sub(r'[^A-Z0-9\u1200-\u137F\s]', '', text, flags=re.UNICODE)
        
        # Normalize spacing
        text = ' '.join(text.split())
        
        # Convert to uppercase (for English parts)
        text = text.upper()
        
        return text

    def process_image(self, image: np.ndarray) -> List[Dict]:
        """Complete plate recognition pipeline"""
        plates = self.detect_plate_regions(image)
        results = []

        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate['bbox']
            plate_img = image[y1:y2, x1:x2]

            if self.debug_mode:
                self._debug_save(plate_img, f"05_plate_crop_{i}")

            plate_text = self.recognize_plate_text(plate_img)

            if plate_text:
                results.append({
                    'text': plate_text,
                    'bbox': plate['bbox'],
                    'confidence': plate['confidence']
                })

                if self.debug_mode:
                    debug_img = plate_img.copy()
                    cv2.putText(debug_img, plate_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self._debug_save(debug_img, f"06_recognized_{i}")

        return results