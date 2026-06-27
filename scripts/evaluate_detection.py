import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluate_detection import evaluate_detector

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Paths
    DATA_YAML = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd" / "data.yaml"
    BEST_MODEL = PROJECT_ROOT / "outputs" / "checkpoints" / "detection_best.pt"
    
    # If the copy in outputs doesn't exist, look in the runs folder
    if not BEST_MODEL.exists():
        BEST_MODEL = PROJECT_ROOT / "runs" / "detect" / "blood_detection" / "detection_run" / "weights" / "best.pt"
    
    if not BEST_MODEL.exists():
        print(f"[ERROR] Model not found at {BEST_MODEL}. Please run training first.")
        sys.exit(1)
    
    print("[START] Evaluating the detection model...")
    print(f"[INFO] Using model: {BEST_MODEL}")
    print(f"[INFO] Using data: {DATA_YAML}")
    
    # Run evaluation
    metrics = evaluate_detector(
        model_weights_path=BEST_MODEL,
        data_yaml_path=DATA_YAML,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device="cpu",
    )
