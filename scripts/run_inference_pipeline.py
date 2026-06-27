# scripts/11_run_inference_pipeline.py

import sys
from pathlib import Path
import cv2
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.pipeline import load_pipeline

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # --- 1. Configuration de l'image d'entrée ---
    # Par défaut, on prend la première image de validation de BCCD pour tester
    DEFAULT_IMAGE = PROJECT_ROOT / "data" / "raw" / "detection" / "bccd" / "images" / "BloodImage_00000.jpg"
    
    # Tu peux changer ici pour tester une autre image
    IMAGE_PATH = DEFAULT_IMAGE
    
    if not IMAGE_PATH.exists():
        print(f"[ERROR] Image introuvable : {IMAGE_PATH}")
        sys.exit(1)
    
    # --- 2. Chargement du pipeline ---
    print("=" * 60)
    print("🏥 PIPELINE COMPLET - DÉTECTION + CLASSIFICATION")
    print("=" * 60)
    pipeline = load_pipeline()
    
    # --- 3. Exécution ---
    print(f"\n[INFO] Traitement de l'image : {IMAGE_PATH.name}")
    result = pipeline.predict_on_image(IMAGE_PATH)
    
    # --- 4. Sauvegarde de l'image annotée ---
    output_img_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_img_path = output_img_dir / f"result_{IMAGE_PATH.stem}.jpg"
    cv2.imwrite(str(output_img_path), result['annotated_image'])
    print(f"[SUCCESS] Image annotée sauvegardée : {output_img_path}")
    
    # --- 5. Affichage du rapport ---
    report = result['report']
    print("\n" + "=" * 60)
    print("📊 RAPPORT D'ANALYSE DES GLOBULES BLANCS")
    print("=" * 60)
    print(f"Total WBC détectés : {report['total_wbc']}")
    print("\nDétail par type :")
    for cls, count in report['counts'].items():
        print(f"  - {cls:>12} : {count}")
    
    print(f"\n📈 Ratio Neutrophiles / Lymphocytes : {report['ratio']:.2f}")
    print(f"💡 Interprétation : {report['interpretation']}")
    print("=" * 60)
    
    # Sauvegarde du rapport en JSON
    report_path = PROJECT_ROOT / "outputs" / "reports" / f"report_{IMAGE_PATH.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Rapport sauvegardé : {report_path}")