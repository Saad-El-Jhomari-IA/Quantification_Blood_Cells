# scripts/08_evaluate_classification.py

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.data.classification_dataset import WBCCellDataset
from src.models.classification.efficientnet import EfficientNetWBCClassifier

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
    TEST_CSV = SPLIT_DIR / "classification_test.csv"
    MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "classification_best.pt"
    
    if not TEST_CSV.exists():
        print("[ERROR] classification_test.csv not found. Run 06_prepare_classification_data.py first.")
        sys.exit(1)
    
    if not MODEL_PATH.exists():
        print("[ERROR] Model not found. Run 07_train_classification.py first.")
        sys.exit(1)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load Test Dataset (NO augmentation)
    test_dataset = WBCCellDataset(TEST_CSV, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"[INFO] Test samples: {len(test_dataset)}")
    
    # Load Model
    model = EfficientNetWBCClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    class_names = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET (UNSEEN DATA)")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("=" * 60)
    
    # Save results to JSON for the notebook
    results_dir = PROJECT_ROOT / "outputs" / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }
    with open(results_dir / "classification_test_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[SUCCESS] Results saved to: {results_dir / 'classification_test_results.json'}")