# scripts/09_evaluate_test_set.py

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import numpy as np

# Ajout du chemin racine pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.classification_dataset import WBCCellDataset
from src.models.classification.efficientnet import EfficientNetWBCClassifier

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # --- 1. Chemins ---
    SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
    TEST_CSV = SPLIT_DIR / "classification_test.csv"   # Le CSV généré par 06_prepare_classification_data.py
    MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "classification_best.pt"
    
    # --- 2. Vérifications ---
    if not TEST_CSV.exists():
        print(f"[ERROR] Fichier CSV de test introuvable : {TEST_CSV}")
        print("[INFO] Assurez-vous d'avoir exécuté 06_prepare_classification_data.py")
        sys.exit(1)
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Modèle introuvable : {MODEL_PATH}")
        print("[INFO] Assurez-vous d'avoir exécuté 07_train_classification.py")
        sys.exit(1)
    
    # --- 3. Périphérique (GPU si disponible) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Périphérique utilisé : {device}")
    
    # --- 4. Chargement du Dataset de Test ---
    # On utilise le même Dataset que pour l'entraînement, mais avec is_train=False
    # (donc pas d'augmentation, juste le resize et la normalisation)
    test_dataset = WBCCellDataset(TEST_CSV, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"[INFO] Nombre d'images dans le Test Set : {len(test_dataset)}")
    
    # --- 5. Chargement du Modèle Entraîné ---
    model = EfficientNetWBCClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Modèle chargé avec succès : {MODEL_PATH}")
    
    # --- 6. Inférence sur le Test Set ---
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- 7. Calcul des Métriques ---
    accuracy = accuracy_score(all_labels, all_preds)
    class_names = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    
    # Classification Report détaillé
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # --- 8. Affichage dans le terminal ---
    print("\n" + "=" * 60)
    print("📊  ÉVALUATION FINALE SUR LE TEST SET (DONNÉES INÉDITES)")
    print("=" * 60)
    print(f"\n✅ ACCURACY GLOBALE : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📋 CLASSIFICATION REPORT :")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("-" * 60)
    
    print("\n📈 MATRICE DE CONFUSION :")
    print(conf_matrix)
    print("=" * 60)
    
    # --- 9. Sauvegarde des résultats dans un fichier JSON (pour le notebook) ---
    results_dir = PROJECT_ROOT / "outputs" / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "class_names": class_names
    }
    
    with open(results_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Résultats sauvegardés dans : {results_dir / 'test_results.json'}")
    print("\n[SUCCESS] Évaluation terminée !")