"""
Detector Module
Supports both Classical CV and YOLO Deep Learning
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import (
    USE_YOLO,
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    YOLO_VEHICLE_CLASSES,
    MIN_PLATE_AREA,
    MAX_PLATE_AREA,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO
)
from utils import is_valid_contour

class PlateDetector:
    """
    Hybrid Detector:
    1. Case A: User has Plate-Specific YOLO Model -> Detect Plates directly
    2. Case B: User has Standard YOLO Model -> Detect Cars -> Detect Plates inside Cars (CV)
    3. Case C: No YOLO -> Use Classical CV on full frame
    """
    
    def __init__(self):
        self.debug_mode = False
        self.yolo_model = None
        self.is_plate_model = False
        
        if USE_YOLO:
            print(f"[INFO] Loading YOLO model: {YOLO_MODEL_PATH}...")
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                
                # Check if this is a plate text model or generic car model
                # Standard COCO has 'car' (2) but not 'license_plate'
                names = self.yolo_model.names
                print(f"[INFO] Model classes: {names}")
                
                # Heuristic: Check for 'plate' in class names
                self.is_plate_model = any('plate' in str(name).lower() for name in names.values())
                
                if self.is_plate_model:
                    print("[INFO] Mode: Direct Plate Detection (High Accuracy)")
                else:
                    print("[INFO] Mode: Hybrid (Car Detection + Classical Plate Search)")
                    
            except Exception as e:
                print(f"[ERROR] Failed to load YOLO: {e}")
                print("Falling back to Classical CV")

    def detect_plate(self, frame, edged=None):
        """
        Detect plates using the best available method
        """
        # Method 1: Direct Plate Detection (Best)
        if self.yolo_model and self.is_plate_model:
            return self._detect_yolo_plates(frame)
            
        # Method 2: Hybrid (Car -> CV)
        if self.yolo_model:
            return self._detect_hybrid(frame, edged)
            
        # Method 3: Classical Only
        return self._detect_classical(frame, edged)

    def _detect_yolo_plates(self, frame):
        """Use YOLO to find plates directly"""
        results = self.yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        candidates = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                candidates.append((x,y,w,h))
        
        return candidates

    def _detect_hybrid(self, frame, edged=None):
        """
        Step 1: Detect Cars
        Step 2: Look for plates ONLY inside car regions
        """
        candidates = []
        
        # Detect vehicles
        results = self.yolo_model(frame, conf=YOLO_CONFIDENCE, classes=YOLO_VEHICLE_CLASSES, verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy, cw, ch = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Draw car box (debug)
                # cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 1)
                
                # Extract Car ROI
                car_roi = frame[cy:cy+ch, cx:cx+cw]
                if car_roi.size == 0: continue
                
                # Run Classical Detection INSIDE the car
                # We need edge detection for the car ROI
                if edged is not None:
                    car_edged = edged[cy:cy+ch, cx:cx+cw]
                    
                    # Find contours in this small region
                    plates = self._detect_classical(car_roi, car_edged, offset=(cx, cy))
                    candidates.extend(plates)
        
        # Fallback: If no cars found, try full frame? 
        # Usually better to rely on cars, but for safety:
        if not candidates:
             candidates = self._detect_classical(frame, edged)
             
        return candidates

    def _detect_classical(self, frame, edged, offset=(0,0)):
        """
        Original OpenCV contour-based detection
        offset: (x,y) to add to coordinates (used when detecting inside ROI)
        """
        if edged is None:
             return []
             
        ox, oy = offset
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        candidates = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area Check
            if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect Ratio Check
            aspect = w / float(h)
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                continue
            
            # Add valid candidate (adjusted for offset)
            candidates.append((x + ox, y + oy, w, h))
            
        return candidates

    def extract_plate_roi(self, frame, bbox, padding=5):
        x, y, w, h = bbox
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + padding*2)
        h = min(frame.shape[0] - y, h + padding*2)
        return frame[y:y+h, x:x+w]
