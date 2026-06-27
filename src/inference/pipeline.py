
# src/inference/pipeline.py 

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from ultralytics import YOLO

from src.models.classification.efficientnet import EfficientNetWBCClassifier


class WBCInferencePipeline:
  
    
    def __init__(
        self,
        detection_weights: Path,
        classification_weights: Path,
        device: str = None,
        conf_threshold: float = 0.9,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.conf_threshold = 0.5
        self.WBC_CLASS_ID = 0  # Dans BCCD, 0 = WBC, 1 = RBC, 2 = Platelets

        
        print(f"[PIPELINE] Chargement de YOLO depuis {detection_weights}...")
        self.detection_model = YOLO(str(detection_weights))

        
        print(f"[PIPELINE] Chargement d'EfficientNet depuis {classification_weights}...")
        self.class_model = EfficientNetWBCClassifier(num_classes=4, pretrained=False)
        self.class_model.load_state_dict(torch.load(classification_weights, map_location=self.device))
        self.class_model = self.class_model.to(self.device)
        self.class_model.eval()

        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
        self.class_names = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
        self.color_map = {
            'NEUTROPHIL': (0, 255, 0),    # Vert
            'MONOCYTE': (255, 255, 0),    # Cyan
            'LYMPHOCYTE': (255, 0, 0),    # Rouge
            'EOSINOPHIL': (0, 0, 255)     # Bleu
        }
        print("[PIPELINE] Pipeline prêt !")

    def predict_on_image(self, image_path: Path, padding: int = 5) -> Dict[str, Any]:
        """
        Exécute le pipeline complet sur une image.
        """
       
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

       
        results = self.detection_model(img_rgb, conf=self.conf_threshold)
        boxes = results[0].boxes
        
        per_cell = []         
        counts = {name: 0 for name in self.class_names}
        wbc_detected = 0
        rbc_detected = 0

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                
                
                if cls_id != self.WBC_CLASS_ID:
                    rbc_detected += 1  #
                    continue  
                
                wbc_detected += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                
                cell_crop = img_rgb[y1:y2, x1:x2]
                if cell_crop.size == 0:
                    continue

                
                pil_img = Image.fromarray(cell_crop)
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.class_model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, dim=1)
                
                pred_class = self.class_names[pred.item()]
                conf_score = confidence.item()

                
                per_cell.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': pred_class,
                    'confidence': conf_score
                })
                counts[pred_class] += 1

        
        print(f"[FILTRE] YOLO a détecté {wbc_detected + rbc_detected} cellules au total.")
        print(f"[FILTRE]   - {wbc_detected} globules BLANCS (WBC) → envoyés au classifieur.")
        print(f"[FILTRE]   - {rbc_detected} globules ROUGES (RBC) → IGNORÉS (non-classifiés).")

       
        total_wbc = sum(counts.values())
        ratio = 0.0
        if counts['LYMPHOCYTE'] > 0:
            ratio = counts['NEUTROPHIL'] / counts['LYMPHOCYTE']
        elif total_wbc > 0:
            ratio = float('inf')

        
        if ratio == float('inf'):
            interpretation = "Aucun Lymphocyte détecté. Vérifier l'échantillon."
        elif ratio > 3.0:
            interpretation = "Ratio N/L ÉLEVÉ (>3.0) : Suggère une infection bactérienne ou un stress inflammatoire."
        elif ratio < 1.0:
            interpretation = "Ratio N/L BAS (<1.0) : Peut suggérer une infection virale ou une immuno-déficience."
        else:
            interpretation = "Ratio N/L NORMAL (1.0 - 3.0) : Absence de signe inflammatoire majeur."

        
        annotated_img = img_bgr.copy()
        for cell in per_cell:
            x1, y1, x2, y2 = cell['bbox']
            cls_name = cell['class']
            conf = cell['confidence']
            color = self.color_map.get(cls_name, (255, 255, 255))
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        
        report = {
            'total_wbc': total_wbc,
            'counts': counts,
            'ratio': ratio if ratio != float('inf') else 0.0,
            'interpretation': interpretation
        }

        return {
            'annotated_image': annotated_img,
            'report': report,
            'per_cell_predictions': per_cell
        }


def load_pipeline(
    detection_weights: Path = None,
    classification_weights: Path = None,
) -> WBCInferencePipeline:
    """Charge le pipeline avec les chemins par défaut (fonctionne depuis n'importe où)."""
    project_root = Path(__file__).parent.parent.parent
    
    if detection_weights is None:
        detection_weights = project_root / "outputs" / "checkpoints" / "detection_best.pt"
        if not detection_weights.exists():
            detection_weights = project_root / "runs" / "detect" / "blood_detection" / "detection_run" / "weights" / "best.pt"
    
    if classification_weights is None:
        classification_weights = project_root / "outputs" / "checkpoints" / "classification_best.pt"
    
    return WBCInferencePipeline(detection_weights, classification_weights)