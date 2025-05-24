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

            return plate_img