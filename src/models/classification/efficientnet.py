import torch
import torch.nn as nn
from torchvision import models


class EfficientNetWBCClassifier(nn.Module):
   

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)