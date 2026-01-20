import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

# --- Configuration ---
# Using YOLOv11 as it outperformed v12 in your benchmarks
MODEL_PATH = 'models/yolov11_best.pt'
CONFIDENCE_THRESHOLD = 0.5

# Tesseract Config: Whitelist alphanumeric to reduce errors
OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

class LicensePlateRecognizer:
    def __init__(self, model_path):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def preprocess_for_ocr(self, plate_crop):
        """
        Applies computer vision techniques to improve OCR accuracy
        on real-world images (lighting/angles).
        """
        # 1. Grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur to remove noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. Adaptive Thresholding (Crucial for varied lighting)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def process_frame(self, frame):
        # 1. Detect Plates
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                # Get Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop the detected plate
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.size == 0: continue

                # 2. Preprocess & OCR
                processed_plate = self.preprocess_for_ocr(plate_crop)
                text = pytesseract.image_to_string(processed_plate, config=OCR_CONFIG)
                
                # Clean result (remove newlines/spaces)
                clean_text = text.strip().replace(" ", "")

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": clean_text,
                    "conf": conf
                })

        return frame, detections

if __name__ == "__main__":
    # Simulate processing
    alpr = LicensePlateRecognizer(MODEL_PATH)
    
    img = cv2.imread("test_car.jpg")
    if img:
        _, results = alpr.process_frame(img)
        for r in results:
            print(f"Detected: {r['text']} (Conf: {r['conf']:.2f})")
