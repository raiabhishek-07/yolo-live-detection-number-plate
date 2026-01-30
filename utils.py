"""
Utility functions for the Number Plate Recognition System
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime
from config import (
    INDIAN_PLATE_PATTERNS,
    OUTPUT_DIR,
    SAVE_DETECTED_PLATES,
    LOG_DETECTIONS
)


def validate_plate_text(text):
    """
    Validate if the extracted text matches Indian number plate format
    
    Args:
        text (str): OCR extracted text
        
    Returns:
        tuple: (is_valid, cleaned_text)
    """
    # SIMPLIFIED LOGIC: Accept any text that looks like a plate (User Request)
    cleaned = text.strip().upper()
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    
    # ------------------------------------------------------
    # SMART CORRECTION FOR INDIAN PLATES
    # ------------------------------------------------------
    if len(cleaned) > 1:
        chars = list(cleaned)
        
        # 1. State Code (First 2 chars) should be LETTERS
        # M is often read as W, 1 as I
        state_corrections = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '8':'B', 'W':'M'}
        if len(chars) > 0 and chars[0] in state_corrections: chars[0] = state_corrections[chars[0]]
        if len(chars) > 1 and chars[1] in state_corrections: chars[1] = state_corrections[chars[1]]

        # 2. Number Code (Next 2 chars) should be NUMBERS
        # I->1, O->0, Z->2
        digit_corrections = {'I':'1', 'L':'1', 'T':'1', 'O':'0', 'Q':'0', 'Z':'2', 'S':'5', 'B':'8', 'A':'4', 'G':'6'}
        # (Assuming format MH 12 ...)
        if len(chars) > 3:
            if chars[2] in digit_corrections: chars[2] = digit_corrections[chars[2]]
            if chars[3] in digit_corrections: chars[3] = digit_corrections[chars[3]]

        cleaned = "".join(chars)
    # ------------------------------------------------------

    if len(cleaned) > 3:
         return True, cleaned
            
    # Original strict logic (Commented out for now)
    # for pattern in INDIAN_PLATE_PATTERNS: ...
    
    return False, cleaned


def is_valid_contour(contour, frame_area):
    """
    Check if a contour could be a number plate based on geometric properties
    
    Args:
        contour: OpenCV contour
        frame_area: Total area of the frame
        
    Returns:
        bool: True if contour matches plate characteristics
    """
    from config import (
        MIN_PLATE_AREA,
        MAX_PLATE_AREA,
        MIN_ASPECT_RATIO,
        MAX_ASPECT_RATIO,
        MIN_RECT_SIMILARITY
    )
    
    # Get contour area
    area = cv2.contourArea(contour)
    
    # Filter by area
    if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
        # print(f"Debug: Rejected by area {area}")
        return False
    
    # Don't consider if area is too large relative to frame
    if area > frame_area * 0.3:
        # print("Debug: Rejected by relative area")
        return False
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check aspect ratio
    aspect_ratio = w / float(h) if h > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        # print(f"Debug: Rejected by Aspect Ratio {aspect_ratio:.2f}")
        return False
    
    # Check rectangularity (how similar to a rectangle)
    rect_area = w * h
    rectangularity = area / float(rect_area) if rect_area > 0 else 0
    
    if rectangularity < MIN_RECT_SIMILARITY:
        # print(f"Debug: Rejected by Rectangularity {rectangularity:.2f}")
        return False
    
    return True


def save_plate_image(plate_img, plate_text):
    """
    Save detected plate image to disk
    
    Args:
        plate_img: Cropped plate image
        plate_text: Recognized text
    """
    if not SAVE_DETECTED_PLATES:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{plate_text}_{timestamp}.jpg"
    
    # Save image
    cv2.imwrite(filename, plate_img)
    
    return filename


def log_detection(plate_text, confidence=None):
    """
    Log plate detection to file
    
    Args:
        plate_text: Recognized plate number
        confidence: OCR confidence (optional)
    """
    if not LOG_DETECTIONS:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Plate: {plate_text}"
    
    if confidence:
        log_entry += f" | Confidence: {confidence:.2f}%"
    
    log_entry += "\n"
    
    # Append to log file
    with open("detections.log", "a") as f:
        f.write(log_entry)


def warp_plate(plate_img):
    """
    Straighten a slanted plate image using perspective transform
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return plate_img
    
    # Find largest quadrilateral
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    if screen_cnt is None: return plate_img
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = screen_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # Width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    # Warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))
    
    return warped

def enhance_plate_image(plate_img):
    """
    Preprocess plate image for OCR
    """
    # Try to straighten the plate first
    try:
        # Only warp if image is large enough
        if plate_img.shape[0] > 50 and plate_img.shape[1] > 100:
            warped = warp_plate(plate_img)
            # Check if warp result is valid
            if warped.shape[0] > 20 and warped.shape[1] > 50:
                plate_img = warped
    except:
        pass # Fallback
        
    # Resize to standard size (improves OCR)
    plate_img = cv2.resize(plate_img, (400, 100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter (preserves edges while removing noise)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding (Pure Black & White)
    thresh = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


def draw_plate_info(frame, bbox, plate_text, is_valid):
    """
    Draw bounding box and plate text on frame
    
    Args:
        frame: Video frame
        bbox: Bounding box coordinates (x, y, w, h)
        plate_text: Recognized text
        is_valid: Whether the plate is valid
        
    Returns:
        Annotated frame
    """
    from config import (
        BBOX_COLOR,
        BBOX_THICKNESS,
        TEXT_COLOR,
        TEXT_FONT,
        TEXT_SCALE,
        TEXT_THICKNESS
    )
    
    x, y, w, h = bbox
    
    # Choose color based on validity
    color = BBOX_COLOR if is_valid else (0, 0, 255)  # Red if invalid
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BBOX_THICKNESS)
    
    # Add label background
    label = f"Plate: {plate_text}" if is_valid else f"Invalid: {plate_text}"
    (label_width, label_height), _ = cv2.getTextSize(
        label,
        TEXT_FONT,
        TEXT_SCALE,
        TEXT_THICKNESS
    )
    
    # Draw filled rectangle for text background
    cv2.rectangle(
        frame,
        (x, y - label_height - 10),
        (x + label_width, y),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        TEXT_FONT,
        TEXT_SCALE,
        (0, 0, 0),  # Black text
        TEXT_THICKNESS
    )
    
    return frame


class PlateHistory:
    """
    Maintains history of detected plates to avoid flickering
    """
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.current_plate = None
        self.confidence_count = 0
    
    def add_detection(self, plate_text):
        """Add a plate detection"""
        from config import CONSECUTIVE_FRAMES_REQUIRED
        
        # If same as current, increase confidence
        if plate_text == self.current_plate:
            self.confidence_count += 1
        else:
            self.current_plate = plate_text
            self.confidence_count = 1
        
        # If confident enough, add to permanent history
        if self.confidence_count >= CONSECUTIVE_FRAMES_REQUIRED:
            if plate_text not in self.history:
                self.history.append(plate_text)
                
                # Maintain max history size
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Log the detection
                log_detection(plate_text)
                
                return True  # New plate confirmed
        
        return False  # Not yet confirmed
    
    def get_stable_plate(self):
        """Get the current stable plate"""
        from config import CONSECUTIVE_FRAMES_REQUIRED
        
        if self.confidence_count >= CONSECUTIVE_FRAMES_REQUIRED:
            return self.current_plate
        return None
    
    def reset(self):
        """Reset the current tracking"""
        self.current_plate = None
        self.confidence_count = 0
