import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.preprocessing.face_extractor import FaceExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Inference")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_faces_from_video(video_path: Path, face_extractor: FaceExtractor, frame_stride: int = 5):
    """Samples video frames and extracts cropped face images."""
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    faces = []
    timestamps = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            # Call extract_all_faces which returns a list of cropped faces
            cropped_faces = face_extractor.extract_all_faces(frame)
            
            if cropped_faces and len(cropped_faces) > 0:
                # Use the primary/largest detected face in the frame
                cropped_face = cropped_faces[0]
                
                # Convert BGR (OpenCV) to RGB (PIL)
                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                faces.append(Image.fromarray(face_rgb))
                timestamps.append(round(frame_idx / fps, 2))

        frame_idx += 1

    cap.release()
    return faces, timestamps

def predict_video(video_path: Path, model, face_extractor, transform, device, frame_stride=5, top_k_ratio=0.3):
    """Runs inference across extracted faces and aggregates predictions."""
    faces, timestamps = extract_faces_from_video(video_path, face_extractor, frame_stride=frame_stride)

    if not faces:
        return {
            "video": video_path.name,
            "status": "FAILED",
            "error": "No faces detected in video."
        }

    # Prepare batch tensor
    face_tensors = torch.stack([transform(face) for face in faces]).to(device)

    with torch.no_grad():
        logits = model(face_tensors)
        # Class 1 = Fake, Class 0 = Real
        fake_probs = torch.sigmoid(logits).cpu().numpy().ravel()

    # Aggregation Strategy:
    # 1. Simple Mean probability
    mean_score = float(np.mean(fake_probs))
    
    # 2. Top-K Worst Frame Pooling (robust against partial-video deepfakes)
    k = max(1, int(len(fake_probs) * top_k_ratio))
    top_k_score = float(np.mean(np.sort(fake_probs)[-k:]))

    # Decision rule: use top-k to catch short temporal glitches
    decision_score = top_k_score
    prediction = "FAKE" if decision_score >= 0.50 else "REAL"
    confidence = decision_score if prediction == "FAKE" else (1.0 - decision_score)

    return {
        "video": video_path.name,
        "status": "SUCCESS",
        "verdict": prediction,
        "confidence_pct": round(confidence * 100, 2),
        "mean_fake_prob": round(mean_score, 4),
        "top_k_fake_prob": round(top_k_score, 4),
        "frames_analyzed": len(faces),
        "frame_details": [
            {"time_sec": t, "fake_prob": round(float(p), 4)}
            for t, p in zip(timestamps, fake_probs)
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Test videos for Deepfake classification")
    parser.add_argument("--source", type=str, required=True, help="Path to a video file (.mp4) or directory of videos")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--frame-stride", type=int, default=5, help="Sample every N-th frame")
    parser.add_argument("--output-dir", type=str, default="models/reports/inference", help="Directory to save prediction logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model
    model = SpatialForensicClassifier(pretrained=False, freeze_backbone=False)
    if Path(args.checkpoint).exists():
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {args.checkpoint}")
    else:
        logger.error(f"Checkpoint not found at {args.checkpoint}")
        return

    model.to(device)
    model.eval()

   # Initialize Transform
    transform = get_transforms()

    # Resolve Video Paths
    source_path = Path(args.source)
    video_paths = [source_path] if source_path.is_file() else list(source_path.glob("*.mp4"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    
    # 🚨 FIX: Use FaceExtractor as a context manager 🚨
    with FaceExtractor() as face_extractor:
        for vid_path in video_paths:
            logger.info(f"Analyzing {vid_path.name}...")
            res = predict_video(vid_path, model, face_extractor, transform, device, frame_stride=args.frame_stride)
            results.append(res)
            
            if res["status"] == "SUCCESS":
                logger.info(f"Result for {res['video']}: [{res['verdict']}] with {res['confidence_pct']}% confidence ({res['frames_analyzed']} faces analyzed)")
            else:
                logger.warning(f"Result for {res['video']}: {res['error']}")

    # Save JSON Report
    report_file = output_dir / "inference_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"\nInference complete. Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()