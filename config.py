"""
Configuration file for Number Plate Recognition System
contains all tunable parameters for detection and OCR
"""

import cv2

# Camera Settings
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Image Processing Parameters
GAUSSIAN_BLUR_KERNEL = (5, 5)
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200
MORPH_KERNEL_SIZE = (3, 3)

# Plate Detection Parameters (Classical)
# These are used as a fallback or inside the YOLO car detection
MIN_PLATE_AREA = 500  
MAX_PLATE_AREA = 250000  
# RESTORED: Strict ratio to ignore single letters like 'D'
MIN_ASPECT_RATIO = 1.5  
MAX_ASPECT_RATIO = 7.0  
MIN_RECT_SIMILARITY = 0.5

# Indian Number Plate Regex Patterns
# Format: XX00XX0000 or XX00X0000
INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',  # Standard format
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',    # Single letter variant
]

# OCR Settings
TESSERACT_CONFIG = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Stability Parameters
CONSECUTIVE_FRAMES_REQUIRED = 3
OCR_EVERY_N_FRAMES = 5

# Visual Settings
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# Output Settings
SAVE_DETECTED_PLATES = True
OUTPUT_DIR = "detected_plates"
LOG_DETECTIONS = True

# YOLO Settings (Deep Learning)
USE_YOLO = True
# If you have the specific plate model, set this to "license_plate_detector.pt"
YOLO_MODEL_PATH = "yolov8n.pt" 
YOLO_CONFIDENCE = 0.5
# 2=Car, 3=Motorcycle, 5=Bus, 7=Truck in COCO dataset
YOLO_VEHICLE_CLASSES = [2, 3, 5, 7] 
