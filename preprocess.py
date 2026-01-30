"""
Image preprocessing module for Number Plate Detection
Converts raw camera frames into edge-detected images suitable for contour detection
"""

import cv2
import numpy as np
from config import (
    GAUSSIAN_BLUR_KERNEL,
    CANNY_THRESHOLD_1,
    CANNY_THRESHOLD_2,
    MORPH_KERNEL_SIZE
)


def preprocess_frame(frame):
    """
    Preprocess frame for plate detection
    
    Pipeline:
    1. Convert to grayscale
    2. Apply Gaussian blur (noise reduction)
    3. Edge detection (Canny)
    4. Morphological operations (close gaps)
    
    Args:
        frame: BGR image from camera
        
    Returns:
        edged: Binary edge image
        gray: Grayscale image (for OCR later)
    """
    # Step 1: Convert to grayscale
    # Why? Plates are detected by edges, not color
    # Single channel is faster to process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian Blur
    # Why? Removes high-frequency noise that creates false edges
    # Kernel size (5,5) is optimal for vehicle plates
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    
    # Step 3: Canny Edge Detection
    # Why? Number plates have high edge density
    # Threshold 1: 100, Threshold 2: 200 (standard values)
    edged = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    
    # Step 4: Morphological Closing
    # Why? Connects broken edges, fills small gaps
    # This makes plate contours more complete
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    return closed, gray


def adaptive_preprocess(frame, use_clahe=False):
    """
    Advanced preprocessing with adaptive techniques for difficult lighting
    
    Args:
        frame: Input BGR image
        use_clahe: Whether to use CLAHE (for low light conditions)
        
    Returns:
        edged: Edge detected image
        gray: Grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if use_clahe:
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        # Improves contrast in low-light conditions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Bilateral filter: Preserves edges while smoothing
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive thresholding for varying lighting
    thresh = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Edge detection on filtered image
    edged = cv2.Canny(filtered, 30, 200)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    return edged, gray


def get_visualization(frame, edged):
    """
    Create visualization showing preprocessing steps
    
    Args:
        frame: Original BGR frame
        edged: Edge detected image
        
    Returns:
        Combined visualization image
    """
    # Convert edged to BGR for concatenation
    edged_bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    
    # Resize for display
    height = frame.shape[0] // 2
    width = frame.shape[1] // 2
    
    frame_small = cv2.resize(frame, (width, height))
    edged_small = cv2.resize(edged_bgr, (width, height))
    
    # Add labels
    cv2.putText(frame_small, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(edged_small, "Edge Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([frame_small, edged_small])
    
    return combined
