# src/training/train_classification.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import json


def train_classification_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    class_weights: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    scheduler_factor: float = 0.1,
    scheduler_patience: int = 3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir: Path = Path("outputs/checkpoints"),
) -> Dict[str, Any]:
    
    save_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
    )

    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        # Scheduler step
        scheduler.step(val_loss)

        # Log history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_dir / "classification_best.pt")
            print(f"  *** Best model saved with Val Acc: {val_acc:.4f} ***")

        print("-" * 50)

    # Load the best model state for final evaluation
    model.load_state_dict(best_model_state)

    # Final evaluation on validation set
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    # Classification report
    class_names = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    report = classification_report(final_labels, final_preds, target_names=class_names, output_dict=True)
    confusion = confusion_matrix(final_labels, final_preds)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 50)
    print(classification_report(final_labels, final_preds, target_names=class_names))
    print("=" * 50)

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'classification_report': report,
        'confusion_matrix': confusion.tolist()
    }


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    class_counts = np.bincount(all_labels)
    total = len(all_labels)
    num_classes = len(class_counts)
    
    # Weight = total_samples / (num_classes * class_count)
    weights = total / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)