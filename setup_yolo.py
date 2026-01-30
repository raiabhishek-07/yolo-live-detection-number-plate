import os
import requests
import sys

MODEL_URL = "https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8/raw/main/model/license_plate_detector.pt"
MODEL_NAME = "license_plate_detector.pt"

def download_model():
    print(f"Downloading License Plate YOLOv8 Model...")
    print(f"URL: {MODEL_URL}")
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        
        with open(MODEL_NAME, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote += len(data)
                f.write(data)
                
                # Progress bar
                if total_size > 0:
                    percent = wrote / total_size * 100
                    sys.stdout.write(f"\rProgress: {percent:.1f}%")
                    sys.stdout.flush()
        
        print("\n\n✅ Model downloaded successfully!")
        print(f"Saved as: {os.path.abspath(MODEL_NAME)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Please manually download the file from the URL above")
        print(f"and save it as '{MODEL_NAME}' in this folder.")
        return False

if __name__ == "__main__":
    if os.path.exists(MODEL_NAME):
        print(f"✅ Model '{MODEL_NAME}' already exists.")
    else:
        download_model()
