# scripts/03_train_detection.py

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train_detection import train_yolo_detector, load_params_from_yaml

if __name__ == "__main__":
    # Define paths
    PROJECT_ROOT = Path(__file__).parent.parent
    PARAMS_PATH = PROJECT_ROOT / "config" / "params.yaml"
    DATA_YAML_PATH = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd" / "data.yaml"
    
    print("[START] Initiating YOLOv8 detection training...")
    
    # Check if data.yaml exists
    if not DATA_YAML_PATH.exists():
        print(f"[ERROR] data.yaml not found at {DATA_YAML_PATH}. Please run 02_create_data_yaml.py first.")
        sys.exit(1)
    
    # Load parameters
    if not PARAMS_PATH.exists():
        print(f"[WARNING] params.yaml not found at {PARAMS_PATH}. Using default parameters.")
        params = {}
    else:
        params = load_params_from_yaml(PARAMS_PATH)
        print(f"[INFO] Loaded parameters: {params}")
    
    # Extract params with defaults
    model_size = params.get('model_size', 'yolov8n.pt')
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 16)
    image_size = params.get('image_size', 640)
    learning_rate = params.get('learning_rate', 0.01)
    momentum = params.get('momentum', 0.937)
    weight_decay = params.get('weight_decay', 0.0005)
    device = params.get('device', 0)
    workers = params.get('workers', 4)
    project_name = params.get('project_name', 'blood_detection')
    
    print(f"[INFO] Training configuration:")
    print(f"  - Model: {model_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {image_size}")
    print(f"  - Device: {device}")
    print("=" * 50)
    
    # Launch training
    try:
        best_model_path = train_yolo_detector(
            data_yaml_path=DATA_YAML_PATH,
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            workers=workers,
            project_name=project_name,
        )
        print(f"\n[SUCCESS] Detection training complete! Best model: {best_model_path}")
        print("[NEXT] You can now evaluate the model using scripts/04_evaluate_detection.py")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)