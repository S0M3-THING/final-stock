import torch
import clip
from torchvision import models, transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

buy_sell_mapping = {
    "AT.jpg": "buy", "CAH.jpg": "buy", "DBW.jpg": "buy", "DT.jpg": "sell", "DTM.jpg": "sell", "FF.jpg": "buy",
    "BB.jpg": "buy", "BT.jpg": "sell", "TB.jpg": "buy", "TT.jpg": "sell", "RB.jpg": "buy", "RT.jpg": "sell", "HAS.jpg": "sell", "IHAS.jpg": "buy",
    "FP.jpg": "buy", "FW.jpg": "buy", "ICAH.jpg": "sell", "RF.jpg": "sell", "RP.jpg": "sell", "RW.jpg": "sell"
}

def load_resnet():
    model_path = hf_hub_download(
        repo_id="AkashS08/resnet50trendfinder",
        filename="resnet50.pth",
        cache_dir="./model_cache"
    )
    model = models.resnet50()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device

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

def extract_features_clip(model, preprocess, device, image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()

