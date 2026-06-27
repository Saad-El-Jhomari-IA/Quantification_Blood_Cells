import sys
from pathlib import Path
from turtle import pd
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.data.classification_dataset import WBCCellDataset
from src.models.classification.efficientnet import EfficientNetWBCClassifier
from src.training.train_classification import train_classification_model, compute_class_weights

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    PARAMS_PATH = PROJECT_ROOT / "config" / "params.yaml"
    SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
    
    TRAIN_CSV = SPLIT_DIR / "classification_train.csv"
    VAL_CSV = SPLIT_DIR / "classification_val.csv"
    
    # Check prerequisites
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        print("[ERROR] CSV files not found. Run 06_prepare_classification_data.py first.")
        sys.exit(1)
    
    # Load parameters
    with open(PARAMS_PATH, 'r') as f:
        config = yaml.safe_load(f)
    params = config.get('classification', {})
    
    # Extract params with defaults
    batch_size = params.get('batch_size', 32)
    epochs = params.get('epochs', 30)
    lr = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 0.0001)
    scheduler_factor = params.get('scheduler_factor', 0.1)
    scheduler_patience = params.get('scheduler_patience', 3)
    num_classes = params.get('num_classes', 4)
    device = torch.device(f"cuda:{params.get('device', 0)}" if torch.cuda.is_available() else "cpu")
    save_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    
    print("[START] Initializing classification training...")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Epochs: {epochs}")
    
    # Create datasets and dataloaders
    train_dataset = WBCCellDataset(TRAIN_CSV, is_train=True)
    val_dataset = WBCCellDataset(VAL_CSV, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Compute class weights for imbalance
    class_weights = compute_class_weights(train_loader)
    print(f"[INFO] Class weights: {class_weights.tolist()}")
    
    # Create model
    model = EfficientNetWBCClassifier(num_classes=num_classes, pretrained=True)
    print(f"[INFO] Model: EfficientNet-B0 (pretrained=True)")
    
    # Train
    results = train_classification_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        class_weights=class_weights,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        device=device,
        save_dir=save_dir,
    )
    
    # Save training history to CSV for the notebook
    history_df = pd.DataFrame(results['history'])
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(log_dir / "classification_history.csv", index=False)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"[INFO] Best model saved at: {save_dir / 'classification_best.pt'}")
    print(f"[INFO] Training history saved at: {log_dir / 'classification_history.csv'}")
    print("[NEXT] Open notebooks/07_monitor_classification_training.ipynb to visualize results.")