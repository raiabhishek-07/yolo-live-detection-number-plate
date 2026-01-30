from huggingface_hub import hf_hub_download
import shutil
import os

def download_plate_model():
    # Get token from user input
    print("🔑 Enter your Hugging Face API Key (starts with hf_...):")
    token = input(">> ").strip()
    
    if not token:
        print("❌ No token provided.")
        return False

    print("⏳ Downloading License Plate Model (AZIIIIIIIIZ/License-plate-detection)...")
    try:
        # Download the model file from Hugging Face
        model_path = hf_hub_download(
            repo_id="AZIIIIIIIIZ/License-plate-detection",
            filename="best.pt",
            token=token
        )
        
        # Move it to our folder
        target_path = "license_plate_detector.pt"
        shutil.copy(model_path, target_path)
        
        print(f"\n✅ SUCCESS! Model saved as: {target_path}")
        print("You can now run 'python main.py'")
        return True
        
    except Exception as e:
        print(f"\n❌ Download Failed: {e}")
        return False

if __name__ == "__main__":
    download_plate_model()
