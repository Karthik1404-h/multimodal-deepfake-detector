import kagglehub
import shutil
from pathlib import Path

def install_lightweight_dataset():
    print("Authenticating with Kaggle...")
    kagglehub.login()
    
    print("Downloading Small Image Dataset (< 1GB)...")
    dataset_path = kagglehub.dataset_download('manjilkarki/deepfake-and-real-images')
    
    # We are putting this directly into the processed folder
    target_dir = Path("data/processed/train")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Moving images to {target_dir}...")
    
    # The Kaggle dataset has a specific folder structure: Dataset/Train/Real and Dataset/Train/Fake
    source_train = Path(dataset_path) / "Dataset" / "Train"
    
    for class_name in ["Real", "Fake"]:
        src = source_train / class_name
        dest = target_dir / class_name.lower()
        
        if src.exists():
            print(f"Copying {class_name} images...")
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            print(f"Warning: Could not find {src}")
            
    print("\nSuccess! The images are correctly placed.")
    print("You can now run train_spatial.py directly!")

if __name__ == "__main__":
    install_lightweight_dataset()