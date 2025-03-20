import torch
import clip
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define the mapping of images to "buy" or "sell"
buy_sell_mapping = {
    "AT.jpg": "buy", "CAH.jpg": "buy", "DBW.jpg": "buy", "DT.jpg": "sell", "DTM.jpg": "sell", "FF.jpg": "buy",
    "BB.jpg": "buy", "BT.jpg": "sell", "TB.jpg": "buy", "TT.jpg": "sell", "RB.jpg": "buy", "RT.jpg": "sell", "HAS.jpg": "sell", "IHAS.jpg": "buy",
    "FP.jpg": "buy", "FW.jpg": "buy", "ICAH.jpg": "sell", "RF.jpg": "sell", "RP.jpg": "sell", "RW.jpg": "sell"
}

# Load ResNet model
def load_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Fix for warning
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    model.eval()
    return model


# Load CLIP model
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

# Extract features using ResNet
def extract_features_resnet(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()

# Extract features using CLIP
def extract_features_clip(model, preprocess, device, image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()
