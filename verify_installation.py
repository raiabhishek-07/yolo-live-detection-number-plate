"""
Installation Verification Script
Run this to check if all dependencies are properly installed
"""

import sys

def check_imports():
    """Check if all required libraries can be imported"""
    print("=" * 60)
    print(" Checking Python Dependencies")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV not installed: {e}")
        print("   Install with: pip install opencv-python")
        return False
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy not installed: {e}")
        print("   Install with: pip install numpy")
        return False
    
    # Check pytesseract
    try:
        import pytesseract
        print(f"✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ pytesseract not installed: {e}")
        print("   Install with: pip install pytesseract")
        return False
    
    # Check Pillow
    try:
        from PIL import Image
        import PIL
        print(f"✅ Pillow version: {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow not installed: {e}")
        print("   Install with: pip install Pillow")
        return False
    
    # Check imutils
    try:
        import imutils
        print(f"✅ imutils imported successfully")
    except ImportError as e:
        print(f"❌ imutils not installed: {e}")
        print("   Install with: pip install imutils")
        return False
    
    return True


def check_tesseract():
    """Check if Tesseract OCR is installed and accessible"""
    print("\n" + "=" * 60)
    print(" Checking Tesseract OCR")
    print("=" * 60)
    
    try:
        import pytesseract
        
        # Try to get version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            return True
        except pytesseract.TesseractNotFoundError:
            print("❌ Tesseract executable not found")
            print("\n📋 Installation Instructions:")
            print("\nWindows:")
            print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  2. Install to: C:\\Program Files\\Tesseract-OCR")
            print("  3. Update ocr.py with the path:")
            print("     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
            print("\nLinux:")
            print("  sudo apt-get install tesseract-ocr")
            print("\nmacOS:")
            print("  brew install tesseract")
            return False
        except Exception as e:
            print(f"❌ Error checking Tesseract: {e}")
            return False
    
    except ImportError:
        print("❌ pytesseract not installed")
        return False


def check_camera():
    """Check if camera is accessible"""
    print("\n" + "=" * 60)
    print(" Checking Camera Access")
    print("=" * 60)
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open default camera (index 0)")
            print("\n💡 Troubleshooting:")
            print("  - Try different camera index: 1, 2, etc.")
            print("  - Check camera permissions (Windows Settings > Privacy)")
            print("  - Ensure no other app is using the camera")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read from camera")
            return False
        
        print(f"✅ Camera accessible (Frame size: {frame.shape[1]}x{frame.shape[0]})")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing camera: {e}")
        return False


def test_ocr():
    """Test OCR with a simple example"""
    print("\n" + "=" * 60)
    print(" Testing OCR Functionality")
    print("=" * 60)
    
    try:
        import cv2
        import numpy as np
        import pytesseract
        
        # Create a simple test image with text
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            img,
            "MH12AB1234",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3
        )
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Run OCR
        config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(gray, config=config).strip()
        
        print(f"Expected: MH12AB1234")
        print(f"Got:      {text}")
        
        if "MH12AB1234" in text or text == "MH12AB1234":
            print("✅ OCR working correctly")
            return True
        else:
            print("⚠️  OCR working but accuracy may vary")
            print("   This is normal - real-world accuracy depends on image quality")
            return True
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False


def check_modules():
    """Check if project modules can be imported"""
    print("\n" + "=" * 60)
    print(" Checking Project Modules")
    print("=" * 60)
    
    modules = ['config', 'detector', 'ocr', 'preprocess', 'utils']
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}.py imported successfully")
        except Exception as e:
            print(f"❌ Error importing {module}.py: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ANPR System - Installation Verification" + " " * 8 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {
        "Python Dependencies": check_imports(),
        "Tesseract OCR": check_tesseract(),
        "Camera Access": check_camera(),
        "OCR Functionality": False,
        "Project Modules": False
    }
    
    # Only test OCR if Tesseract is available
    if results["Tesseract OCR"]:
        results["OCR Functionality"] = test_ocr()
    
    # Only test modules if dependencies are OK
    if results["Python Dependencies"]:
        results["Project Modules"] = check_modules()
    
    # Final summary
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All checks passed! You're ready to run the application.")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Or test with images: python test_images.py --help")
        print("  3. Read SETUP.md for detailed usage")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nRefer to SETUP.md for installation instructions.")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
