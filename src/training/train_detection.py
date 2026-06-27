import yaml
from pathlib import Path
from typing import Union
from ultralytics import YOLO


def train_yolo_detector(
    data_yaml_path: Union[str, Path],
    model_size: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    learning_rate: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    device: Union[int, str] = 0,
    workers: int = 4,
    project_name: str = "blood_detection",
) -> str:
    
    # Load a pretrained YOLO model
    model = YOLO(model_size)

    # Start training
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        lr0=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        workers=workers,
        project=project_name,
        name="detection_run",       # Subfolder under project
        exist_ok=True,             # Overwrite existing run
        pretrained=True,
        seed=42,                   # For reproducibility
    )

    
    best_model_path = Path(project_name) / "detection_run" / "weights" / "best.pt"
    
    print(f"[SUCCESS] Training completed! Best model saved at: {best_model_path}")
    
    # Also save a copy to our structured 'outputs/checkpoints/' folder
    output_checkpoint_dir = Path("outputs") / "checkpoints"
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    final_destination = output_checkpoint_dir / "detection_best.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_destination)
        print(f"[SUCCESS] Model copied to: {final_destination}")
    
    return str(final_destination)


def load_params_from_yaml(yaml_path: Path) -> dict:
    """Load detection parameters from the central params.yaml file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('detection', {})