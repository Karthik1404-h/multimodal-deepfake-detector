import os
from huggingface_hub import snapshot_download

print("1/2: Downloading REAL videos (UniDataPro/face-anti-spoofing)...")
real_dir = snapshot_download(
    repo_id="UniDataPro/face-anti-spoofing",
    repo_type="dataset",
    local_dir="data/raw/face_anti_spoofing"
)
print(f" -> Real videos saved to: {real_dir}\n")

print("2/2: Downloading FAKE videos (UniDataPro/deepfake-videos-dataset)...")
fake_dir = snapshot_download(
    repo_id="UniDataPro/deepfake-videos-dataset",
    repo_type="dataset",
    local_dir="data/raw/deepfake_videos"
)
print(f" -> Fake videos saved to: {fake_dir}\n")

print("All downloads completed successfully!")