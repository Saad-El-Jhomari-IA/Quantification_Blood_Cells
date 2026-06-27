import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import generate_yolo_data_yaml

if __name__ == "__main__":
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd"
    
    IMAGES_DIR = DATA_ROOT / "images"
    LABELS_DIR = DATA_ROOT / "labels"
    SPLITS_DIR = DATA_ROOT / "splits"
    
    # Les fichiers générés à l'étape 1
    TRAIN_TXT = SPLITS_DIR / "train.txt"
    VAL_TXT = SPLITS_DIR / "val.txt"
    
    # Les classes du dataset BCCD (l'ordre est crucial !)
    # On met les IDs exactement comme on les a définis dans la conversion : WBC=0, RBC=1, Platelets=2
    CLASS_NAMES = ['WBC', 'RBC', 'Platelets']
    
    # Où sauvegarder le data.yaml
    OUTPUT_YAML = DATA_ROOT / "data.yaml"
    
    print("[START] Generating data.yaml for YOLOv8...")
    
    # Vérifier que les fichiers de split existent
    if not TRAIN_TXT.exists() or not VAL_TXT.exists():
        print("[ERROR] train.txt or val.txt not found. Please run 01_convert_bccd_to_yolo.py first.")
        sys.exit(1)
    
    generate_yolo_data_yaml(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        train_list_path=TRAIN_TXT,
        val_list_path=VAL_TXT,
        class_names=CLASS_NAMES,
        output_path=OUTPUT_YAML,
    )
    
    print(f"[INFO] data.yaml content:")
    with open(OUTPUT_YAML, 'r') as f:
        print(f.read())