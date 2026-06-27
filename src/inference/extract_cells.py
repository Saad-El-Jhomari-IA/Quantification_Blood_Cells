# src/inference/extract_cells.py

from pathlib import Path
from typing import List
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def extract_wbc_patches(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float = 0.25,
    target_class_id: int = 0,
    padding: int = 50,
) -> List[Path]:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    img = cv2.imread(str(image_path))
    if img is None:
        return saved_paths

    h, w = img.shape[:2]
    results = model(image_path, conf=conf_threshold)
    boxes = results[0].boxes

    if boxes is None:
        return saved_paths

    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        if cls_id != target_class_id:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        patch_filename = f"{image_path.stem}_cell_{idx}.jpg"
        patch_path = output_dir / patch_filename
        cv2.imwrite(str(patch_path), patch)
        saved_paths.append(patch_path)

    return saved_paths


def extract_wbc_from_split_file(
    split_path: Path,
    model_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> int:
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, 'r') as f:
        image_paths = [Path(line.strip()) for line in f.readlines() if line.strip()]

    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    total_patches = 0
    for img_path in tqdm(image_paths, desc="Extracting WBC patches"):
        if not img_path.exists():
            print(f"[WARNING] Image not found: {img_path}")
            continue
        patches = extract_wbc_patches(
            image_path=img_path,
            model=model,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
        )
        total_patches += len(patches)

    print(f"[SUCCESS] Extracted {total_patches} WBC patches from {len(image_paths)} validation images.")
    return total_patches