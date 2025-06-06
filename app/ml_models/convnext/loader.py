import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import os

OUT_CLASSES = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_convnext_model(out_classes):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, out_classes)
    )
    return model
    
model = get_convnext_model(OUT_CLASSES).to(device)
model_path = os.path.join(os.path.dirname(__file__), "model_convnext_88.pth")
model.load_state_dict(torch.load(model_path, map_location='cpu'))

def load_convnext_model():
    classes = {
        "0": "Combination",
        "1": "Dry",
        "2": "Normal",
        "3": "Oily",
    }
    return model, classes
