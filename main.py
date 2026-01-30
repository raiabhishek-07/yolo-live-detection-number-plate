"""
Main Application - Live Number Plate Recognition System
Complete pipeline from camera to OCR output
"""

import cv2
import time
import sys
from detector import PlateDetector
from ocr import PlateOCR
from preprocess import preprocess_frame
from utils import (
    PlateHistory,
    draw_plate_info,
    save_plate_image
)
from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    OCR_EVERY_N_FRAMES
)


class NumberPlateRecognitionSystem:
    """
    Complete ANPR System
    Handles live video capture, detection, and OCR
    """
    
    def __init__(self, source=0, tesseract_path=None):
        """
        Initialize the system
        
        Args:
            source: Camera index (0 for default) or video file path
            tesseract_path: Path to tesseract executable (Windows only)
        """
        # Initialize components
        self.detector = PlateDetector()
        self.ocr = PlateOCR(tesseract_path=tesseract_path)
        self.history = PlateHistory()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
        # Current detection
        self.current_plate = None
        self.current_bbox = None
        
        print("[INFO] Number Plate Recognition System Initialized")
        print(f"[INFO] Camera: {source}")
        print(f"[INFO] Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print("[INFO] Press 'q' to quit, 's' to save current plate, 'd' for debug mode")
    
    def run(self):
        """
        Main processing loop
        """
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera")
            return
        
        debug_mode = False
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            self.frame_count += 1
            
            # Process frame
            result_frame = self._process_frame(frame, debug_mode)
            
            # Calculate FPS
            self.processing_time = time.time() - start_time
            self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
            
            # Display FPS and info
            self._draw_info_overlay(result_frame)
            
            # Show frame
            cv2.imshow("Number Plate Recognition", result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[INFO] Exiting...")
                break
            elif key == ord('s'):
                self._save_current_detection(frame)
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug mode: {debug_mode}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"[INFO] Total frames processed: {self.frame_count}")
        print(f"[INFO] Plates detected: {len(self.history.history)}")
        print(f"[INFO] Detected plates: {self.history.history}")
    
    def _process_frame(self, frame, debug_mode=False):
        """
        Process a single frame through the pipeline
        
        Pipeline:
        1. Preprocess (edge detection)
        2. Detect plate candidates
        3. Run OCR (every N frames)
        4. Validate & display
        
        Args:
            frame: Input frame
            debug_mode: Show preprocessing steps
            
        Returns:
            Annotated frame
        """
        # Clone for annotation
        display_frame = frame.copy()
        
        # Step 1: Preprocess
        edged, gray = preprocess_frame(frame)
        
        # Step 2: Detect plates
        plate_candidates = self.detector.detect_plate(frame, edged)
        
        # Step 3: Process plates
        if plate_candidates:
            # Get best candidate (largest)
            best_bbox = plate_candidates[0]
            x, y, w, h = best_bbox
            
            # VISUAL FEEDBACK: Draw Blue Box around candidate immediately
            # This lets the user see we found the plate, even if OCR is slow or fails
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract plate ROI
            plate_roi = self.detector.extract_plate_roi(frame, best_bbox)
            
            # Run OCR (not every frame for performance)
            if self.frame_count % OCR_EVERY_N_FRAMES == 0:
                # Extract text
                plate_text, is_valid, confidence = self.ocr.extract_text_multiple_methods(plate_roi)
                
                if is_valid:
                    # Update history
                    is_new = self.history.add_detection(plate_text)
                    
                    if is_new:
                        print(f"[DETECTION] New plate: {plate_text} (Confidence: {confidence:.1f}%)")
                        save_plate_image(plate_roi, plate_text)
                    
                    self.current_plate = plate_text
                    self.current_bbox = best_bbox
                else:
                    # Invalid plate, BUT show it anyway for debugging (in Orange)
                    if plate_text and len(plate_text) > 3:
                        print(f"[RAW OCR] Rejected: {plate_text}")
                        # Draw raw text in orange
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(display_frame, f"Raw: {plate_text}", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Draw CONFIRMED detection (Green)
            # Re-validate the stored plate if we didn't run OCR this frame
            from utils import validate_plate_text
            if self.current_plate:
                 is_valid_stored, _ = validate_plate_text(self.current_plate)
            else:
                 is_valid_stored = False

            if self.current_plate and self.current_bbox and is_valid_stored:
                # Get stable plate or current
                stable_plate = self.history.get_stable_plate()
                display_text = stable_plate if stable_plate else self.current_plate
                
                # Check validity
                from utils import validate_plate_text
                is_valid, _ = validate_plate_text(display_text)
                
                # Draw on frame
                draw_plate_info(
                    display_frame,
                    self.current_bbox,
                    display_text,
                    is_valid
                )
        
        # Debug mode: Show preprocessing
        if debug_mode:
            from preprocess import get_visualization
            viz = get_visualization(frame, edged)
            cv2.imshow("Debug: Preprocessing", viz)
        
        return display_frame
    
    def _draw_info_overlay(self, frame):
        """
        Draw system information on frame
        
        Args:
            frame: Frame to annotate
        """
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame count
        count_text = f"Frames: {self.frame_count}"
        cv2.putText(frame, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detected plates count
        plates_text = f"Plates: {len(self.history.history)}"
        cv2.putText(frame, plates_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        help_text = "Q:Quit | S:Save | D:Debug"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_current_detection(self, frame):
        """
        Manually save current plate detection
        
        Args:
            frame: Current frame
        """
        if self.current_bbox and self.current_plate:
            plate_roi = self.detector.extract_plate_roi(frame, self.current_bbox)
            filename = save_plate_image(plate_roi, self.current_plate)
            print(f"[SAVED] Plate saved: {filename}")
        else:
            print("[WARNING] No plate detected to save")


def main():
    """
    Entry point
    """
    print("=" * 60)
    print(" Number Plate Recognition System")
    print(" Using: OpenCV + Tesseract OCR")
    print(" No Deep Learning Required")
    print("=" * 60)
    print()
    
    # Check arguments
    source = CAMERA_INDEX
    tesseract_path = None
    
    if len(sys.argv) > 1:
        # Video file or camera index provided
        try:
            source = int(sys.argv[1])
        except ValueError:
            source = sys.argv[1]  # Assume video file path
    
    # Automatic Tesseract Detection for Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\AppData\Local\Tesseract-OCR\tesseract.exe'
    ]
    
    tesseract_path = None
    import os
    
    # Check if 'tesseract' is in PATH
    import shutil
    if shutil.which('tesseract'):
        print("[INFO] Tesseract found in system PATH")
    else:
        # Check common directories
        for path in possible_paths:
            if os.path.exists(path):
                tesseract_path = path
                print(f"[INFO] Auto-detected Tesseract at: {path}")
                break
        
        if tesseract_path is None:
            print("\n[WARNING] Tesseract executable not found!")
            print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("Or if installed, set the path in main.py manually.\n")
    
    # Initialize and run system
    try:
        system = NumberPlateRecognitionSystem(
            source=source,
            tesseract_path=tesseract_path
        )
        system.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
