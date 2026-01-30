"""
Test Script - Process Images Instead of Live Camera
Useful for testing without webcam or with sample images
"""

import cv2
import sys
import os
from detector import PlateDetector
from ocr import PlateOCR
from preprocess import preprocess_frame
from utils import draw_plate_info, validate_plate_text, save_plate_image


def process_image(image_path, save_output=True):
    """
    Process a single image for plate detection
    
    Args:
        image_path: Path to image file
        save_output: Whether to save annotated result
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print('='*60)
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    
    print(f"[INFO] Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Initialize components
    detector = PlateDetector()
    # detector.debug_mode = True  # Enable debug prints
    ocr = PlateOCR()
    
    # Preprocess
    print("[STEP 1] Preprocessing...")
    edged, gray = preprocess_frame(frame)
    
    # Detect plates
    print("[STEP 2] Detecting plates...")
    plate_candidates = detector.detect_plate(frame, edged)
    
    print(f"[INFO] Found {len(plate_candidates)} plate candidates")
    
    # Process each candidate
    result_frame = frame.copy()
    detected_plates = []
    
    for idx, bbox in enumerate(plate_candidates[:3]):  # Top 3 candidates
        print(f"\n[CANDIDATE {idx+1}]")
        x, y, w, h = bbox
        print(f"  Location: ({x}, {y}), Size: {w}x{h}")
        
        # Extract ROI
        plate_roi = detector.extract_plate_roi(frame, bbox)
        
        # Run OCR
        print(f"[STEP 3] Running OCR...")
        plate_text, is_valid, confidence = ocr.extract_text_multiple_methods(plate_roi)
        
        print(f"  Raw OCR: {plate_text}")
        print(f"  Valid: {is_valid}")
        print(f"  Confidence: {confidence:.1f}%")
        
        if is_valid:
            detected_plates.append(plate_text)
            
            # Save plate
            saved_path = save_plate_image(plate_roi, plate_text)
            print(f"  ✅ Saved: {saved_path}")
        
        # Draw on result
        draw_plate_info(result_frame, bbox, plate_text, is_valid)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print('='*60)
    print(f"Total candidates: {len(plate_candidates)}")
    print(f"Valid plates detected: {len(detected_plates)}")
    if detected_plates:
        print(f"Plates: {', '.join(detected_plates)}")
    print('='*60)
    
    # Save result
    if save_output:
        output_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result_frame)
        print(f"\n[SAVED] Result: {output_path}")
    
    # Display
    # Resize for display if too large
    max_height = 800
    if result_frame.shape[0] > max_height:
        scale = max_height / result_frame.shape[0]
        new_width = int(result_frame.shape[1] * scale)
        result_frame = cv2.resize(result_frame, (new_width, max_height))
    
    cv2.imshow("Result", result_frame)
    cv2.imshow("Edge Detection", edged)
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_folder(folder_path):
    """
    Process all images in a folder
    
    Args:
        folder_path: Path to folder containing images
    """
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all image files
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"[ERROR] No images found in {folder_path}")
        return
    
    print(f"[INFO] Found {len(image_files)} images")
    
    # Process each image
    for img_path in image_files:
        process_image(img_path, save_output=True)
        print("\n" + "="*60 + "\n")


def download_sample_images():
    """
    Create a guide to get sample images for testing
    """
    print("""
    📸 Sample Images for Testing
    ============================
    
    To test this system, you need sample images of vehicles with number plates.
    
    Option 1: Use Your Own Images
    ------------------------------
    - Take photos of vehicles (ensure plates are visible)
    - Save them in a folder
    - Run: python test_images.py /path/to/folder
    
    Option 2: Download Samples
    --------------------------
    Search for "Indian vehicle number plate dataset" or use:
    
    1. Kaggle: https://www.kaggle.com/datasets
       Search: "vehicle number plate"
    
    2. GitHub: Search for "ANPR dataset"
    
    3. Create your own:
       - Use your phone camera
       - Take clear, well-lit photos
       - Ensure plate is visible and in focus
    
    Image Requirements:
    -------------------
    ✅ Clear, well-lit images
    ✅ Plate clearly visible
    ✅ Resolution: 640x480 minimum
    ✅ Formats: JPG, PNG, BMP
    
    ❌ Avoid blurry images
    ❌ Avoid extreme angles
    ❌ Avoid very low resolution
    """)


def main():
    """
    Entry point for image testing
    """
    print("=" * 60)
    print(" Number Plate Recognition - Image Testing")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python test_images.py image.jpg")
        print("  Folder:        python test_images.py /path/to/folder")
        print("  Help:          python test_images.py --help")
        print()
        
        # Ask user
        choice = input("Enter image path or folder path (or 'help' for sample info): ").strip()
        
        if choice.lower() == 'help':
            download_sample_images()
            return
        
        if not choice:
            print("[ERROR] No path provided")
            return
        
        path = choice
    else:
        path = sys.argv[1]
        
        if path == '--help':
            download_sample_images()
            return
    
    # Check if path exists
    if not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        return
    
    # Process
    if os.path.isfile(path):
        process_image(path)
    elif os.path.isdir(path):
        process_folder(path)
    else:
        print(f"[ERROR] Invalid path: {path}")


if __name__ == "__main__":
    main()
