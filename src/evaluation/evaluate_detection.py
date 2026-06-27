# src/evaluation/evaluate_detection.py
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO


def evaluate_detector(
    model_weights_path: Path,
    data_yaml_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cpu"  # Use 'cpu' for CPU or '0' for GPU,
) -> Dict[str, Any]:
    
    # Load the trained model
    model = YOLO(str(model_weights_path))

    # Run validation
    results = model.val(
        data=str(data_yaml_path),
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        split='val',  # Use the validation set
        plots=True,   # Generate the plots (confusion matrix, etc.)
    )

    # Extract metrics
    metrics = {
        "mAP50": results.box.map50,
        "mAP50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP50      : {metrics['mAP50']:.4f}")
    print(f"mAP50-95   : {metrics['mAP50_95']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print("=" * 50)

    return metrics