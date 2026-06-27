# scripts/10_classify_extracted_cells.py

import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classification.efficientnet import EfficientNetWBCClassifier

def classify_extracted_cells():
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # --- 1. Chemins ---
    INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_cells"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "classified_cells"
    MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "classification_best.pt"
    CSV_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "extracted_cells_predictions.csv"
    
    # --- 2. Vérifications ---
    if not INPUT_DIR.exists():
        print(f"[ERROR] Dossier d'entrée introuvable : {INPUT_DIR}")
        print("[INFO] Lancez d'abord scripts/05_extract_wbc_patches.py")
        return
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Modèle introuvable : {MODEL_PATH}")
        print("[INFO] Lancez d'abord scripts/07_train_classification.py")
        return
    
    # --- 3. Périphérique ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Périphérique : {device}")
    
    # --- 4. Chargement du modèle ---
    model = EfficientNetWBCClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[INFO] Modèle chargé : {MODEL_PATH}")
    
    # --- 5. Transformations (comme pour la validation) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- 6. Classes ---
    class_names = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    
    # --- 7. Création du dossier de sortie ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 8. Liste des images ---
    image_files = list(INPUT_DIR.glob("*.jpg"))
    print(f"[INFO] {len(image_files)} cellules à classifier.")
    
    predictions = []
    
    # --- 9. Boucle d'inférence ---
    for img_path in tqdm(image_files, desc="Classification des cellules"):
        # Charger l'image
        img = Image.open(img_path).convert('RGB')
        original_img = img.copy()  # On garde une copie pour écrire dessus
        
        # Transformer et inférer
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        pred_class = class_names[pred.item()]
        confidence = conf.item()
        
        # --- 10. Sauvegarder l'image avec le texte dessus ---
        # On utilise PIL pour écrire le texte
        draw = ImageDraw.Draw(original_img)
        try:
            # Essayer une police par défaut, sinon utiliser une police basique
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Rectangle de fond pour que le texte soit lisible
        text = f"{pred_class} ({confidence:.2f})"
        # On écrit le texte en haut à gauche
        draw.rectangle([(0, 0), (300, 30)], fill=(0, 0, 0, 180))  # Fond noir semi-transparent
        draw.text((10, 5), text, fill=(0, 255, 0), font=font)
        
        # Sauvegarder l'image classifiée
        output_filename = f"{pred_class}_{img_path.name}"
        output_path = OUTPUT_DIR / output_filename
        original_img.save(output_path)
        
        # Stocker les résultats pour le CSV
        predictions.append({
            "original_filename": img_path.name,
            "predicted_class": pred_class,
            "confidence": confidence,
            "output_path": str(output_path)
        })
    
    # --- 11. Sauvegarder le CSV ---
    df = pd.DataFrame(predictions)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT, index=False)
    
    print(f"\n[SUCCESS] {len(predictions)} images classifiées et sauvegardées dans :")
    print(f"  - Dossier : {OUTPUT_DIR}")
    print(f"  - Rapport CSV : {CSV_OUTPUT}")
    print("\n👉 Ouvrez ce dossier pour voir toutes les images avec leur classe écrite en vert.")

if __name__ == "__main__":
    classify_extracted_cells()