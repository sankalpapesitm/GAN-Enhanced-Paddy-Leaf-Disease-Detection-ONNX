import os, re, sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, r"c:\Users\sanka\OneDrive\Desktop\GAN Enhanced Paddy Leaf Disease Detection\backend")
# pyrefly: ignore[missing-import]
from app import model, CLASSES, eval_transform

def extract_botanical_signals(image_pil):
    im_scaled = image_pil.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(im_scaled, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    im_hsv = np.array(im_scaled.convert("HSV"), dtype=np.float32)
    hue, sat, val = im_hsv[:, :, 0], im_hsv[:, :, 1], im_hsv[:, :, 2]
    
    # Leaf surface pixels
    leaf = (val > 35) & ((g > b * 0.95) | (r > b * 0.95) | (g > 45))
    n_leaf = max(int(np.count_nonzero(leaf)), 1)
    
    # Hispa: White/translucent longitudinal feeding scrapes
    hispa_scrapes = leaf & (
        ((val > 110) & (sat < 95) & (g > 80) & (r > 80)) |
        ((r > 125) & (g > 125) & (b > 75) & (np.abs(r - g) < 35))
    )
    scrape_ratio = float(np.count_nonzero(hispa_scrapes)) / n_leaf
    
    # Brown Spot: Discrete dark reddish-brown circular spots
    brown_spots = leaf & (
        (r > g * 1.15) & (r > b * 1.25) & (r > 50) & (r < 175) & (g > 30) & (sat > 40)
    )
    brown_ratio = float(np.count_nonzero(brown_spots)) / n_leaf
    
    # Healthy: Pure chlorophyll green
    healthy_green = leaf & (
        (hue >= 22) & (hue <= 100) & (sat >= 30) & (g > b * 1.10) & (g >= r * 0.90) & (2 * g - r - b > 15)
    )
    green_ratio = float(np.count_nonzero(healthy_green)) / n_leaf
    
    return {
        "scrape_ratio": scrape_ratio,
        "brown_ratio": brown_ratio,
        "green_ratio": green_ratio
    }

print("=== HISPA SAMPLES EVALUATION ===")
cdir = r"E:\Segregation final\Hispa"
for fn in ["000041_b91.JPG", "000042_t91.JPG", "000057_b170.JPG", "000011_b56.JPG"]:
    fp = os.path.join(cdir, fn)
    pil_im = Image.open(fp).convert("RGB")
    im_tensor = eval_transform(pil_im).unsqueeze(0)
    with torch.no_grad():
        out = model(im_tensor, im_tensor)
        probs = F.softmax(out, dim=1)[0].numpy()
    sig = extract_botanical_signals(pil_im)
    sc = sig["scrape_ratio"] * 100
    br = sig["brown_ratio"] * 100
    print(f"{fn:<16} | NN Probs: Brown={probs[0]*100:.1f}%, Hispa={probs[2]*100:.1f}% | Scrapes={sc:.1f}%, BrownSpots={br:.2f}%")
