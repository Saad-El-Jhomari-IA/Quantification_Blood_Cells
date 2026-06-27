# import sys
# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split

# sys.path.append(str(Path(__file__).parent.parent))

# if __name__ == "__main__":
#     PROJECT_ROOT = Path(__file__).parent.parent
    
#     # Chemin vers les 4 classes (corrigé avec ta structure)
#     DATA_RAW = PROJECT_ROOT / "data" / "raw" / "classification" / "PBC_dataset" / "wbc_resized"
    
#     CLASS_NAMES = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    
#     # 1. Construire la liste de toutes les images
#     data_list = []
#     for label_name in CLASS_NAMES:
#         class_dir = DATA_RAW / label_name
#         if not class_dir.exists():
#             print(f"[ERROR] Dossier non trouvé : {class_dir}")
#             sys.exit(1)
#         for img_path in class_dir.glob("*.jpg"):
#             data_list.append({"image_path": str(img_path), "label": label_name})
    
#     df = pd.DataFrame(data_list)
#     print(f"[INFO] Total images trouvées : {len(df)}")
#     print(df['label'].value_counts())
    
#     # 2. Split STRATIFIE (pour garder la même proportion dans train et val)
#     # On prend 80% train, 20% val
#     train_df, val_df = train_test_split(
#         df, 
#         test_size=0.2, 
#         stratify=df['label'], 
#         random_state=42
#     )
    
#     # 3. Sauvegarder les CSV dans data/splits/
#     SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
#     SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
#     train_df.to_csv(SPLIT_DIR / "classification_train.csv", index=False)
#     val_df.to_csv(SPLIT_DIR / "classification_val.csv", index=False)
    
#     print(f"[SUCCESS] Train: {len(train_df)} images, Val: {len(val_df)} images")
#     print(f"[SUCCESS] Fichiers sauvegardés dans {SPLIT_DIR}")
# scripts/06_prepare_classification_data.py

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    
    DATA_RAW = PROJECT_ROOT / "data" / "raw" / "classification" / "PBC_dataset" / "wbc_resized"
    CLASS_NAMES = ['NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE', 'EOSINOPHIL']
    
    # 1. Construire la liste de toutes les images
    data_list = []
    for label_name in CLASS_NAMES:
        class_dir = DATA_RAW / label_name
        if not class_dir.exists():
            print(f"[ERROR] Dossier non trouvé : {class_dir}")
            sys.exit(1)
        for img_path in class_dir.glob("*.jpg"):
            data_list.append({"image_path": str(img_path), "label": label_name})
    
    df = pd.DataFrame(data_list)
    print(f"[INFO] Total images trouvées : {len(df)}")
    print(df['label'].value_counts())
    
    # 2. Split STRATIFIE en 3 ensembles : 80% Train, 10% Val, 10% Test
    # Étape 1 : Séparer Train (80%) et Temp (20% = Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2,           # 20% pour la validation + test
        stratify=df['label'], 
        random_state=42
    )
    
    # Étape 2 : Séparer Temp en Val (50% de 20% = 10%) et Test (50% de 20% = 10%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5,           # La moitié de temp = 10% du total
        stratify=temp_df['label'], 
        random_state=42
    )
    
    # 3. Sauvegarder les 3 CSV dans data/splits/
    SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(SPLIT_DIR / "classification_train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "classification_val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "classification_test.csv", index=False)
    
    print(f"[SUCCESS] Train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"[SUCCESS] Val: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"[SUCCESS] Test: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    print(f"[SUCCESS] Fichiers sauvegardés dans {SPLIT_DIR}")