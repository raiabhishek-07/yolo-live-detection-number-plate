# 🚀 Quick Setup Guide

## Prerequisites Installation

### 1. Install Tesseract OCR

#### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (tesseract-ocr-w64-setup-v5.x.x.exe)
3. During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)
4. Add to PATH or update `main.py` and `ocr.py` with the path

**Update in main.py (line ~150):**
```python
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Update in ocr.py (line ~20):**
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

### 2. Verify Tesseract Installation

```bash
tesseract --version
```

Should output something like:
```
tesseract 5.3.0
  leptonica-1.82.0
```

---

## Python Environment Setup

### Option 1: Using pip (Recommended)

```bash
# Navigate to project directory
cd plate_ocr

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n anpr python=3.9

# Activate environment
conda activate anpr

# Install dependencies
pip install -r requirements.txt
```

---

## Verify Installation

Run this test script:

```python
# test_installation.py
import cv2
import numpy as np
import pytesseract

print("✓ OpenCV version:", cv2.__version__)
print("✓ NumPy version:", np.__version__)

try:
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract version: {version}")
except Exception as e:
    print(f"✗ Tesseract error: {e}")
    print("  → Please set tesseract path in ocr.py")

print("\n✅ All dependencies installed successfully!")
```

Run with:
```bash
python test_installation.py
```

---

## Running the Application

### 1. Live Camera Mode

```bash
python main.py
```

### 2. Specific Camera

```bash
python main.py 1  # Use camera index 1
```

### 3. Video File

```bash
python main.py path/to/video.mp4
```

### 4. Image Testing

```bash
# Single image
python test_images.py image.jpg

# Folder of images
python test_images.py path/to/images/

# Help
python test_images.py --help
```

---

## Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current detected plate |
| `d` | Toggle debug mode (shows preprocessing steps) |

---

## Troubleshooting

### Issue: "Tesseract not found"

**Solution 1:** Add to PATH
```bash
# Windows (PowerShell as Administrator)
$env:Path += ";C:\Program Files\Tesseract-OCR"

# Or permanently via System Properties > Environment Variables
```

**Solution 2:** Set path in code
```python
# In ocr.py, line ~20
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# In main.py, line ~150
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue: "Camera not opening"

**Solutions:**
1. Try different camera index:
   ```bash
   python main.py 0
   python main.py 1
   python main.py 2
   ```

2. Check camera permissions (Windows Settings > Privacy > Camera)

3. Test with another application (Windows Camera app)

### Issue: "No plates detected"

**Solutions:**
1. Ensure good lighting
2. Plate should be clearly visible and in focus
3. Adjust detection parameters in `config.py`:
   ```python
   MIN_PLATE_AREA = 300  # Lower threshold
   MAX_ASPECT_RATIO = 6.0  # Wider range
   ```

### Issue: "Low FPS / Slow performance"

**Solutions:**
1. Reduce resolution in `config.py`:
   ```python
   FRAME_WIDTH = 640
   FRAME_HEIGHT = 480
   ```

2. Increase OCR interval:
   ```python
   OCR_EVERY_N_FRAMES = 10  # Run OCR less frequently
   ```

3. Close other applications

### Issue: "Import errors"

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or individually
pip install opencv-python
pip install pytesseract
pip install numpy
pip install Pillow
pip install imutils
```

---

## Testing Checklist

- [ ] Tesseract installed and accessible
- [ ] Python dependencies installed
- [ ] Camera accessible
- [ ] Application launches without errors
- [ ] Can see video feed
- [ ] Detects plates (green box)
- [ ] Extracts text correctly
- [ ] Saves images to `detected_plates/`
- [ ] Creates `detections.log`

---

## Next Steps

1. **Read the code:** Start with `main.py`, understand the flow
2. **Test with images:** Use `test_images.py` before live camera
3. **Experiment:** Modify parameters in `config.py`
4. **Understand algorithms:** Read `TECHNICAL_DOCS.md`
5. **Prepare for viva:** Review interview questions in docs

---

## Project Structure

```
plate_ocr/
│
├── main.py              # ← Start here
├── test_images.py       # Image testing
├── detector.py          # Plate detection
├── ocr.py              # Text extraction
├── preprocess.py       # Image enhancement
├── utils.py            # Helper functions
├── config.py           # All parameters
│
├── requirements.txt    
├── README.md           # User guide
├── TECHNICAL_DOCS.md   # Deep dive
├── SETUP.md           # This file
│
└── detected_plates/    # Auto-created
```

---

## Getting Sample Images

### Option 1: Create Your Own
- Use phone camera
- Take clear photos of vehicles
- Ensure number plate is visible

### Option 2: Download Dataset
- Search "Indian vehicle number plate dataset"
- Kaggle: https://www.kaggle.com/datasets
- GitHub: Search "ANPR dataset"

### Image Requirements
- ✅ Clear, well-lit
- ✅ Plate visible and in focus
- ✅ Resolution: 640×480 minimum
- ✅ Format: JPG, PNG, BMP

---

## Support

If you encounter issues:

1. Check **Troubleshooting** section above
2. Verify all prerequisites are installed
3. Read error messages carefully
4. Check `config.py` parameters

---

**Ready to go! Run `python main.py` to start. 🎉**
