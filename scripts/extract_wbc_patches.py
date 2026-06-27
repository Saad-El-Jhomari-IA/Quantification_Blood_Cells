# scripts/05_extract_wbc_patches.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.extract_cells import extract_wbc_from_split_file

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd"

    SPLIT_PATH = DATA_ROOT / "splits" / "val.txt"
    MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "detection_best.pt"

    if not MODEL_PATH.exists():
        MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "blood_detection" / "detection_run" / "weights" / "best.pt"

    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_cells"

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}. Please train first.")
        sys.exit(1)

    if not SPLIT_PATH.exists():
        print(f"[ERROR] Split file not found at {SPLIT_PATH}. Run 01_convert_bccd_to_yolo.py first.")
        sys.exit(1)

    print("[START] Extracting WBC patches from validation images...")
    print(f"[INFO] Using model: {MODEL_PATH}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    extract_wbc_from_split_file(
        split_path=SPLIT_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25,
    )