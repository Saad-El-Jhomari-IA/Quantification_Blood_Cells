
import requests
import base64
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from pathlib import Path

API_URL = "http://localhost:8001/predict"

def index(request):
    context = {
        'image': None,
        'report': None,
        'error': None
    }

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        
       
        file_path = default_storage.save(f"tmp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        full_path = Path(default_storage.location) / file_path

        try:
           
            with open(full_path, 'rb') as f:
                response = requests.post(API_URL, files={'file': f})
            
            if response.status_code == 200:
                data = response.json()
                
               
                img_data = base64.b64decode(data['image'])
                
                context['image'] = base64.b64encode(img_data).decode('utf-8')
                context['report'] = data['report']
            else:
                context['error'] = f"Erreur de l'API : {response.text}"

        except Exception as e:
            context['error'] = f"Impossible de contacter l'API IA. Vérifiez que le serveur FastAPI tourne sur le port 8001. Erreur : {e}"

        finally:
            # Nettoyer le fichier temporaire
            full_path.unlink(missing_ok=True)

    return render(request, 'analyzer/index.html', context)
