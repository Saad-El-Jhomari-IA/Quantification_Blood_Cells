import sys
from pathlib import Path
import base64
import cv2
import json
import numpy as np
import tempfile  # <-- NOUVEAU : Pour gérer les chemins temporaires
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.inference.pipeline import load_pipeline


app = FastAPI(title="WBC Detection & Classification API", version="1.0")


print("[API] Démarrage du serveur FastAPI...")
print("[API] Chargement des modèles (cela peut prendre quelques secondes)...")
PIPELINE = load_pipeline()
PIPELINE.conf_threshold = 0.78
print("[API] Modèles chargés avec succès !")

@app.get("/")
def read_root():
    return {"message": "WBC API is running. Use POST /predict to analyze an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Image invalide"})


        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / file.filename
        
        
        cv2.imwrite(str(temp_path), img)
        print(f"[API] Image sauvegardée dans : {temp_path}")  

        
        result = PIPELINE.predict_on_image(temp_path)

       
        annotated_img = result['annotated_image']
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        
        temp_path.unlink(missing_ok=True)

        
        return JSONResponse(content={
            "status": "success",
            "image": img_base64,
            "report": result['report'],
            "per_cell": result['per_cell_predictions']
        })

    except Exception as e:
        
        print(f"[API] ERREUR : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)