import sys
from pathlib import Path
import random

# Add the project root to Python path to allow imports
sys.path.append(str(Path(_file_).parent.parent))

from src.data.preprocessing import parse_voc_xml, convert_voc_to_yolo


def convert_all_annotations(
    images_dir: Path, labels_dir: Path, split_ratio: float = 0.8
) -> None:
    """
    Convert all XML files in labels_dir to YOLO .txt format and create train/val splits.

    Args:
        images_dir: Path to directory containing images.
        labels_dir: Path to directory containing XML annotations.
        split_ratio: Ratio of training data (default 0.8).
    """
    # Collect all XML files
    xml_files = list(labels_dir.glob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found in {labels_dir}")
        return

    print(f"[INFO] Found {len(xml_files)} annotation files.")

    # Shuffle and split
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * split_ratio)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]

    # Write train.txt and val.txt (containing absolute paths to images)
    output_split_dir = labels_dir.parent / "splits"
    output_split_dir.mkdir(parents=True, exist_ok=True)

    for file_list, split_name in [(train_files, "train"), (val_files, "val")]:
        with open(output_split_dir / f"{split_name}.txt", "w") as f:
            for xml_path in file_list:
                # Get the corresponding image name
                img_filename = xml_path.stem + ".jpg"
                img_path = images_dir / img_filename
                if img_path.exists():
                    f.write(str(img_path.resolve()) + "\n")
                else:
                    print(f"[WARNING] Image {img_path} not found for {xml_path.name}")

    # Convert each XML to YOLO .txt
    for xml_path in xml_files:
        try:
            filename, width, height, objects = parse_voc_xml(xml_path)

            # Output .txt path (same name as image, placed in labels_dir)
            txt_path = labels_dir / f"{xml_path.stem}.txt"

            with open(txt_path, "w") as f:
                for obj in objects:
                    yolo_bbox = convert_voc_to_yolo(obj["bbox"], width, height)
                    # Format: class_id center_x center_y width height
                    line = f"{obj['class_id']} " + " ".join(f"{v:.6f}" for v in yolo_bbox)
                    f.write(line + "\n")

            print(f"[SUCCESS] Converted {xml_path.name} -> {txt_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed to convert {xml_path.name}: {e}")

    print("\n[INFO] Conversion complete!")
    print(f"[INFO] Train images: {len(train_files)}, Val images: {len(val_files)}")
    print(f"[INFO] Splits saved in: {output_split_dir}")


if _name_ == "_main_":
    # Define paths (adapt if your structure is slightly different)
    PROJECT_ROOT = Path(_file_).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd"

    IMAGES_DIR = DATA_ROOT / "images"
    LABELS_DIR = DATA_ROOT / "labels"

    print(f"[START] Converting BCCD dataset...")
    convert_all_annotations(IMAGES_DIR, LABELS_DIR, split_ratio=0.8)