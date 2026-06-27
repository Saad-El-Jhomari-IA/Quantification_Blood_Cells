import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class WBCCellDataset(Dataset):
   
    
    def __init__(self, csv_path: Path, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        
        
        self.class_to_idx = {
            'NEUTROPHIL': 0,
            'MONOCYTE': 1,
            'LYMPHOCYTE': 2,
            'EOSINOPHIL': 3
        }
        
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),      
                transforms.RandomHorizontalFlip(),  
                transforms.RandomRotation(15),      
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(               
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),      
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        label_str = row['label']
        
        
        image = Image.open(img_path).convert('RGB')
        
        
        image = self.transform(image)
        
        label = self.class_to_idx[label_str]
        return image, label