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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_faces = face_extractor.extract_all_faces(frame_rgb, is_bgr=False)
            
            if cropped_faces and len(cropped_faces) > 0:
                cropped_face = cropped_faces[0]
                faces.append(Image.fromarray(cropped_face))
                timestamps.append(round(frame_idx / fps, 2))

        frame_idx += 1

    cap.release()
    return faces, timestamps

def predict_video(video_path: Path, model, face_extractor, transform, device, frame_stride: int = 5, top_k_ratio: float = 0.20):
    """Runs inference across extracted faces and aggregates predictions using weighted hybrid scoring."""
    faces, timestamps = extract_faces_from_video(video_path, face_extractor, frame_stride=frame_stride)

    if not faces:
        return {
            "video": video_path.name,
            "status": "FAILED",
            "verdict": "ERROR",
            "is_fake": None,
            "error": "No faces detected in video."
        }

    # Prepare batch tensor
    face_tensors = torch.stack([transform(face) for face in faces]).to(device)

    with torch.no_grad():
        logits = model(face_tensors)
        # Class 1 = Fake, Class 0 = Real
        fake_probs = torch.sigmoid(logits).cpu().numpy().ravel()

    # 1. Mean frame probability across all processed frames
    mean_frame_prob = float(np.mean(fake_probs))

    # 2. Top-K mean probability (average of top 20% highest fake probabilities)
    k = max(1, int(len(fake_probs) * top_k_ratio))
    top_k_mean_prob = float(np.mean(np.sort(fake_probs)[-k:]))

    # 3. Hybrid Final Score: 60% global mean + 40% top-k spike pooling
    final_score = (0.6 * mean_frame_prob) + (0.4 * top_k_mean_prob)

    # 4. Classification Thresholds:
    #    >= 0.65: FAKE
    #    >= 0.45 and < 0.65: UNCERTAIN
    #    < 0.45: REAL
    if final_score >= 0.65:
        verdict = "FAKE"
        is_fake = True
        confidence_pct = round(final_score * 100, 2)
    elif final_score >= 0.45:
        verdict = "UNCERTAIN"
        is_fake = None  # Serializes as null in JSON
        confidence_pct = round(final_score * 100, 2)
    else:
        verdict = "REAL"
        is_fake = False
        confidence_pct = round((1.0 - final_score) * 100, 2)

    return {
        "video": video_path.name,
        "status": "SUCCESS",
        "verdict": verdict,
        "is_fake": is_fake,
        "confidence_pct": confidence_pct,
        "final_score": round(final_score, 4),
        "mean_frame_prob": round(mean_frame_prob, 4),
        "top_k_mean_prob": round(top_k_mean_prob, 4),
        "mean_fake_prob": round(mean_frame_prob, 4),
        "top_k_fake_prob": round(top_k_mean_prob, 4),
        "frames_analyzed": len(faces),
        "frame_details": [
            {"time_sec": t, "fake_prob": round(float(p), 4)}
            for t, p in zip(timestamps, fake_probs)
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Test videos for Deepfake classification")
    parser.add_argument("--source", type=str, required=True, help="Path to a video file (.mp4) or directory of videos")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_spatial_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--frame-stride", type=int, default=5, help="Sample every N-th frame")
    parser.add_argument("--output-dir", type=str, default="models/reports/inference", help="Directory to save prediction logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Model
    model = SpatialForensicClassifier(pretrained=False, freeze_backbone=False)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists() and Path("models/best_model.pth").exists():
        checkpoint_path = Path("models/best_model.pth")

    if checkpoint_path.exists():
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {checkpoint_path}")
    else:
        logger.error(f"Checkpoint not found at {args.checkpoint}")
        return

    model.to(device)
    model.eval()

    # Initialize Transform
    transform = get_transforms()

    # Resolve Video Paths
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    source_path = Path(args.source)
    if source_path.is_file():
        video_paths = [source_path]
    elif source_path.is_dir():
        video_paths = sorted([p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in video_extensions])
    else:
        video_paths = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with FaceExtractor() as face_extractor:
        for vid_path in video_paths:
            logger.info(f"Analyzing {vid_path.name}...")
            res = predict_video(vid_path, model, face_extractor, transform, device, frame_stride=args.frame_stride)
            results.append(res)
            
            if res["status"] == "SUCCESS":
                if res["verdict"] == "UNCERTAIN":
                    logger.warning(
                        f"Result for {res['video']}: [UNCERTAIN] (is_fake: null) | "
                        f"Final Score: {res['final_score']} (Mean: {res['mean_frame_prob']}, "
                        f"Top-20%: {res['top_k_mean_prob']}) across {res['frames_analyzed']} frames "
                        f"- Inconclusive/Borderline artifacts"
                    )
                else:
                    logger.info(
                        f"Result for {res['video']}: [{res['verdict']}] (is_fake: {res['is_fake']}) with "
                        f"{res['confidence_pct']}% confidence | Final Score: {res['final_score']} "
                        f"({res['frames_analyzed']} faces analyzed)"
                    )
            else:
                logger.error(f"Result for {res['video']}: {res['error']}")

    # Save JSON Report
    report_file = output_dir / "inference_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"\nInference complete. Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()