# 🚀 Getting Started - Quick Guide

Welcome to the **Automatic Number Plate Recognition (ANPR) System**!

This is your **5-minute quick start guide** to get the system running.

---

## 📋 What You Need

1. **Python 3.8+** installed
2. **Tesseract OCR** installed
3. **Webcam** (or test images)

---

## ⚡ Quick Start (3 Steps)

### Step 1: Verify Installation

```bash
cd plate_ocr
python verify_installation.py
```

**Expected Output:**
```
✅ PASS  Python Dependencies
✅ PASS  Tesseract OCR
✅ PASS  Camera Access
✅ PASS  OCR Functionality
✅ PASS  Project Modules

🎉 All checks passed!
```

If you see ❌ errors, read [SETUP.md](SETUP.md) for detailed instructions.

---

### Step 2: Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

**Install Tesseract:**
- **Windows:** Download from [here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux:** `sudo apt-get install tesseract-ocr`
- **Mac:** `brew install tesseract`

---

### Step 3: Run the System

```bash
python main.py
```

**You should see:**
- Live camera feed
- Green boxes around detected plates
- Plate numbers displayed when recognized
- FPS counter

**Controls:**
- Press `Q` to quit
- Press `S` to save current plate
- Press `D` for debug mode

---

## 📚 Documentation Overview

We have comprehensive documentation. Here's what to read and when:

| Document | When to Read | What It Contains |
|----------|--------------|------------------|
| **[README.md](README.md)** | First | Overview, features, basic usage |
| **[SETUP.md](SETUP.md)** | If installation issues | Detailed setup, troubleshooting |
| **[SUMMARY.md](SUMMARY.md)** | To understand system | Architecture, design decisions |
| **[TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)** | For deep learning | Algorithms, math, interview prep |
| **[Getting Started](GETTING_STARTED.md)** | You're here! | Quick start guide |

---

## 🎯 Common Tasks

### Task: Test with Images (No Camera Needed)

```bash
# Single image
python test_images.py path/to/image.jpg

# Folder of images
python test_images.py path/to/folder/

# Get help
python test_images.py --help
```

### Task: Adjust Detection Sensitivity

Edit `config.py`:

```python
# Detect smaller plates
MIN_PLATE_AREA = 300  # Lower value

# Wider aspect ratio range
MAX_ASPECT_RATIO = 6.0  # Higher value
```

### Task: Improve Performance (Higher FPS)

Edit `config.py`:

```python
# Lower resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Less frequent OCR
OCR_EVERY_N_FRAMES = 10
```

### Task: Improve Accuracy

Edit `config.py`:

```python
# More frequent OCR
OCR_EVERY_N_FRAMES = 3

# More confirmation frames
CONSECUTIVE_FRAMES_REQUIRED = 5
```

### Task: Save All Detections

Already enabled by default!

Check:
- `detected_plates/` folder for images
- `detections.log` for history

---

## 🐛 Troubleshooting

### "Tesseract not found"

**Windows:**
```python
# Edit ocr.py, line ~20
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux/Mac:**
```bash
which tesseract  # Should show path
```

### "Camera not opening"

Try different camera index:
```bash
python main.py 0
python main.py 1
python main.py 2
```

### "No plates detected"

1. Ensure good lighting
2. Plate should be clearly visible
3. Lower `MIN_PLATE_AREA` in config.py

### "Low FPS"

1. Lower resolution in config.py
2. Increase `OCR_EVERY_N_FRAMES`
3. Close other applications

---

## 🎓 Learning Path

### Beginner (Just want to run it)

1. ✅ Run `verify_installation.py`
2. ✅ Run `main.py`
3. ✅ Test with `test_images.py`
4. ✅ Read [README.md](README.md)

**Time:** 30 minutes

---

### Intermediate (Want to understand how it works)

1. ✅ Complete Beginner path
2. ✅ Read [SUMMARY.md](SUMMARY.md)
3. ✅ Read code in this order:
   - `config.py` (parameters)
   - `preprocess.py` (image processing)
   - `detector.py` (plate detection)
   - `ocr.py` (text extraction)
   - `utils.py` (helpers)
   - `main.py` (putting it together)
4. ✅ Experiment with parameters
5. ✅ Try modifying the code

**Time:** 3-4 hours

---

### Advanced (Want to master it)

1. ✅ Complete Intermediate path
2. ✅ Read [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) thoroughly
3. ✅ Understand algorithms:
   - Canny edge detection
   - Contour analysis
   - Morphological operations
   - Tesseract OCR internals
4. ✅ Prepare answers for all interview questions
5. ✅ Try enhancements:
   - Add perspective transform
   - Implement plate tracking
   - Create REST API
   - Build GUI with Tkinter

**Time:** 1-2 days

---

## 📊 Project Structure (Quick Reference)

```
plate_ocr/
│
├── 🎯 CORE CODE
│   ├── main.py              # Main application - START HERE
│   ├── detector.py          # Finds plates in frame
│   ├── ocr.py              # Extracts text from plates
│   ├── preprocess.py       # Image enhancement
│   ├── utils.py            # Helper functions
│   └── config.py           # All settings - MODIFY HERE
│
├── 🧪 TESTING
│   ├── verify_installation.py   # Check setup
│   └── test_images.py           # Test with images
│
├── 📚 DOCS
│   ├── GETTING_STARTED.md  # This file
│   ├── README.md           # Overview
│   ├── SETUP.md            # Installation
│   ├── SUMMARY.md          # Architecture
│   └── TECHNICAL_DOCS.md   # Deep dive
│
└── 📦 DEPENDENCIES
    └── requirements.txt
```

---

## 🎯 Next Steps After Running

### Option 1: Understand the Code
→ Read [SUMMARY.md](SUMMARY.md) for architecture
→ Then dive into code files

### Option 2: Prepare for Interviews
→ Read [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)
→ Practice explaining algorithms
→ Review interview questions

### Option 3: Extend the System
Ideas:
- Add database to store detections
- Create web interface (Flask/FastAPI)
- Build mobile app (Kivy)
- Integrate with YOLO for better detection
- Add vehicle tracking

### Option 4: Create a Portfolio Project
- Document your implementation
- Create demo video
- Host on GitHub
- Write a blog post
- Present in college/interviews

---

## 💡 Pro Tips

1. **Start with test_images.py**
   - Easier to debug than live camera
   - Can test with different images
   - Consistent results

2. **Read the code in order**
   - config.py → preprocess.py → detector.py → ocr.py → main.py
   - Each builds on the previous

3. **Experiment with parameters**
   - All in config.py
   - Try different values
   - Observe the effect

4. **Use debug mode**
   - Press 'D' during live run
   - Shows preprocessing steps
   - Helps understand what's happening

5. **Check the logs**
   - detections.log has all detections
   - detected_plates/ has all images
   - Good for analyzing performance

---

## 🏆 Success Criteria

You've successfully completed the setup when you can:

- [x] Run `verify_installation.py` with all ✅
- [x] Run `main.py` and see live video feed
- [x] See green boxes around number plates
- [x] See plate text displayed correctly
- [x] Find saved images in `detected_plates/`
- [x] Find detection history in `detections.log`

---

## 📞 Need Help?

1. **Check documentation:**
   - [SETUP.md](SETUP.md) for installation issues
   - [README.md](README.md) for usage questions
   - [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) for algorithm details

2. **Debug:**
   - Run `verify_installation.py`
   - Use `test_images.py` instead of camera
   - Enable debug mode (press 'D')

3. **Modify:**
   - All parameters in `config.py`
   - Well-commented code
   - Modular structure

---

## ⚡ TL;DR - Absolute Minimum to Get Running

```bash
# 1. Install Tesseract (Windows - download installer)
# https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install Python packages
pip install opencv-python numpy pytesseract Pillow imutils

# 3. Run verification
python verify_installation.py

# 4. Run the app
python main.py

# Press 'Q' to quit
```

**That's it! 🎉**

---

## 🎓 For College Projects / Viva

**Key Points to Mention:**

1. **Why Classical CV?**
   - Fast, interpretable, no training needed
   - Good for learning fundamentals

2. **How Detection Works?**
   - Edge detection → Contour finding → Geometric filtering
   - Filter by area, aspect ratio, rectangularity

3. **How OCR Works?**
   - Tesseract OCR with optimized configuration
   - Multiple preprocessing methods for robustness

4. **Accuracy?**
   - 85-90% in good conditions
   - Can be improved with deep learning

**Read [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) for detailed interview preparation!**

---

**Happy Coding! 🚀**

Need more details? Check the other documentation files!
