"""
OCR Module for Number Plate Recognition
Uses Tesseract OCR to extract text from plate images
"""

import cv2
import numpy as np
import pytesseract
import threading
import signal
from utils import enhance_plate_image, validate_plate_text
from config import TESSERACT_CONFIG


class PlateOCR:
    """
    OCR engine for extracting text from number plates
    """
    
    def __init__(self, tesseract_path=None, use_easyocr=True):
        """
        Initialize OCR engine
        
        Args:
            tesseract_path: Path to tesseract executable (Windows)
                          Set to None for Linux/Mac (auto-detected)
            use_easyocr: Whether to attempt EasyOCR initialization (default: True)
        """
        # Set Tesseract path for Windows
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR reader (optional, with timeout)
        self.reader = None
        self.use_easyocr = use_easyocr
        
        if use_easyocr:
            self._initialize_easyocr()
    
    def _initialize_easyocr(self):
        """
        Initialize EasyOCR Reader with timeout handling
        """
        try:
            import easyocr
            print("[INFO] Initializing EasyOCR Reader (this takes a moment)...")
            
            # Initialize in a thread with timeout
            reader_holder = [None]
            exception_holder = [None]
            
            def init_reader():
                try:
                    reader_holder[0] = easyocr.Reader(['en'], gpu=False, verbose=False)
                except Exception as e:
                    exception_holder[0] = e
            
            thread = threading.Thread(target=init_reader, daemon=True)
            thread.start()
            thread.join(timeout=120)  # 2-minute timeout
            
            if exception_holder[0]:
                raise exception_holder[0]
            
            if reader_holder[0] is None:
                print("[WARNING] EasyOCR initialization timed out (>120 seconds). Using Tesseract only.")
                self.reader = None
            else:
                self.reader = reader_holder[0]
                print("[INFO] EasyOCR Reader initialized successfully")
                
        except ImportError:
            print("[WARNING] EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None
        except Exception as e:
            print(f"[WARNING] EasyOCR initialization failed: {e}")
            print("[WARNING] Falling back to Tesseract OCR only")
            self.reader = None
    
    def extract_text(self, plate_img):
        """
        Extract text from plate image using Tesseract OCR
        
        Args:
            plate_img: Cropped plate image (BGR)
            
        Returns:
            tuple: (text, is_valid, confidence)
        """
        # Step 1: Enhance image for OCR
        enhanced = enhance_plate_image(plate_img)
        
        # Step 2: Run OCR
        try:
            # Extract text with configuration
            raw_text = pytesseract.image_to_string(
                enhanced,
                config=TESSERACT_CONFIG
            )
            
            # Get confidence data (optional)
            ocr_data = pytesseract.image_to_data(
                enhanced,
                config=TESSERACT_CONFIG,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", False, 0
        
        # Step 3: Validate extracted text
        is_valid, cleaned_text = validate_plate_text(raw_text)
        
        return cleaned_text, is_valid, avg_confidence
    
    def extract_text_multiple_methods(self, plate_img):
        """
        Try multiple OCR methods and return best result
        
        Methods:
        1. Standard enhancement + OCR
        2. Inverted image + OCR (for dark plates)
        3. Additional thresholding variations
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            Best OCR result
        """
        results = []
        
        # Method 0: EasyOCR (Best)
        text0, valid0, conf0 = self.extract_text_easyocr(plate_img)
        if valid0:
            return text0, valid0, conf0  # If valid, just return immediately
        if text0: # If invalid but found text, store it
            results.append((text0, valid0, conf0))
            
        # Method 1: Standard (Tesseract)
        text1, valid1, conf1 = self.extract_text(plate_img)
        if valid1:
            results.append((text1, valid1, conf1))
        
        # Method 2: Try with inverted image (handles dark plates)
        enhanced = enhance_plate_image(plate_img)
        inverted = cv2.bitwise_not(enhanced)
        
        try:
            text2 = pytesseract.image_to_string(inverted, config=TESSERACT_CONFIG)
            valid2, cleaned2 = validate_plate_text(text2)
            if valid2:
                results.append((cleaned2, valid2, 85.0))
        except:
            pass
        
        # Method 3: Try with Otsu's thresholding
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            text3 = pytesseract.image_to_string(otsu, config=TESSERACT_CONFIG)
            valid3, cleaned3 = validate_plate_text(text3)
            if valid3:
                results.append((cleaned3, valid3, 80.0))
        except:
            pass
        
        # Return best result (highest confidence)
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            return results[0]
        
        # No valid result found
        return "", False, 0
    
    def extract_with_visualization(self, plate_img):
        """
        Extract text and provide visualization of preprocessing steps
        
        Args:
            plate_img: Plate image
            
        Returns:
            tuple: (text, is_valid, confidence, vis_image)
        """
        # Enhance
        enhanced = enhance_plate_image(plate_img)
        
        # OCR
        text, is_valid, conf = self.extract_text(plate_img)
        
        # Create visualization
        # Resize for display
        plate_display = cv2.resize(plate_img, (400, 100))
        enhanced_display = cv2.resize(enhanced, (400, 100))
        enhanced_display = cv2.cvtColor(enhanced_display, cv2.COLOR_GRAY2BGR)
        
        # Stack vertically
        vis = np.vstack([plate_display, enhanced_display])
        
        # Add text overlay
        label = f"OCR: {text} ({conf:.1f}%)"
        cv2.putText(vis, label, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return text, is_valid, conf, vis


    
    def extract_text_easyocr(self, plate_img):
        """
        Use EasyOCR (Deep Learning) - Much better accuracy
        """
        try:
            if self.reader is None:
                # EasyOCR not available, fall back to Tesseract
                return self.extract_text(plate_img)
            
            results = self.reader.readtext(plate_img)
            
            # EasyOCR returns list of (bbox, text, prob)
            full_text = ""
            max_conf = 0.0
            
            for (bbox, text, prob) in results:
                full_text += text + " "
                if prob > max_conf: max_conf = prob
                
            full_text = full_text.strip().upper()
            from utils import validate_plate_text
            is_valid, cleaned = validate_plate_text(full_text)
            
            if not is_valid and len(cleaned) > 2:
                # Fallback, just return cleaned text even if invalid
                return cleaned, False, max_conf * 100
                
            return cleaned, is_valid, max_conf * 100
            
        except Exception as e:
            print(f"[WARNING] EasyOCR extraction failed: {e}")
            return self.extract_text(plate_img)
