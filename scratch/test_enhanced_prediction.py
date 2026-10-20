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
    
    # Segment leaf surface from dark backdrop
    leaf = (val > 35) & ((g > b * 0.95) | (r > b * 0.95) | (g > 45))
    n_leaf = max(int(np.count_nonzero(leaf)), 1)
    
    # 1. Hispa feeding trails: whitish/translucent scrapes along leaf venation
    hispa_scrapes = leaf & (
        ((val > 110) & (sat < 95) & (g > 80) & (r > 80)) |
        ((r > 125) & (g > 125) & (b > 75) & (np.abs(r - g) < 35))
    )
    scrape_ratio = float(np.count_nonzero(hispa_scrapes)) / n_leaf
    
    # 2. Brown Spot: discrete dark reddish-brown circular spots (R > G > B with distinct contrast)
    brown_spots = leaf & (
        (r > g * 1.15) & (r > b * 1.25) & (r > 55) & (r < 175) & (g > 30) & (sat > 40)
    )
    brown_ratio = float(np.count_nonzero(brown_spots)) / n_leaf
    
    # 3. Healthy: pure chlorophyll green
    healthy_green = leaf & (
        (hue >= 22) & (hue <= 100) & (sat >= 30) & (g > b * 1.10) & (g >= r * 0.90) & (2 * g - r - b > 15)
    )
    green_ratio = float(np.count_nonzero(healthy_green)) / n_leaf
    
    return scrape_ratio, brown_ratio, green_ratio

def is_top_filename(fn):
    fn_lower = fn.lower()
    return bool(re.search(r"[_-](?:t\d+|t1|t|top|front|adaxial)([_-]|\.|$)", fn_lower))

def is_bottom_filename(fn):
    fn_lower = fn.lower()
    return bool(re.search(r"[_-](?:b\d+|d\d+|b1|d1|b|d|bottom|back|abaxial)([_-]|\.|$)", fn_lower))

def predict_single_view_enhanced(image_pil):
    im_tensor = eval_transform(image_pil).unsqueeze(0)
    # Test-time horizontal flip
    im_flipped = eval_transform(image_pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
    
    with torch.no_grad():
        out1 = model(im_tensor, im_tensor)
        out2 = model(im_flipped, im_flipped)
        raw_logits = (out1[0] + out2[0]) / 2.0
        logits = raw_logits.cpu().numpy()
        
    sc, br, gr = extract_botanical_signals(image_pil)
    
    # Botanical calibration for Hispa vs Brown Spot:
    # If leaf has prominent scraping trails and lack of discrete circular brown spots:
    if (sc > 0.10 and br < 0.005) or (sc > 0.18 and sc > br * 4.0):
        boost = 1.8 * min(sc / 0.20, 2.5)
        logits[2] += boost  # Boost Hispa
        logits[0] -= boost * 0.6  # Suppress Brown spot
    elif br > 0.010 and sc < 0.15:
        boost = 1.5 * min(br / 0.02, 2.0)
        logits[0] += boost  # Boost Brown spot
        logits[2] -= boost * 0.5
        
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], float(probs[pred_idx]), probs

def predict_dual_view_enhanced(front_pil, back_pil, front_fn="", back_fn=""):
    # Check if user inverted front/back in upload
    if is_bottom_filename(front_fn) and is_top_filename(back_fn):
        top_pil, bottom_pil = back_pil, front_pil
    else:
        top_pil, bottom_pil = front_pil, back_pil
        
    top_t = eval_transform(top_pil).unsqueeze(0)
    bot_t = eval_transform(bottom_pil).unsqueeze(0)
    
    with torch.no_grad():
        out_canonical = model(top_t, bot_t)
        out_flipped = model(bot_t, top_t)
        # Weighted canonical + symmetric combination
        logits = (0.75 * out_canonical[0] + 0.25 * out_flipped[0]).cpu().numpy()
        
    sc1, br1, gr1 = extract_botanical_signals(top_pil)
    sc2, br2, gr2 = extract_botanical_signals(bottom_pil)
    sc = max(sc1, sc2)
    br = max(br1, br2)
    
    if (sc > 0.10 and br < 0.005) or (sc > 0.18 and sc > br * 4.0):
        boost = 1.8 * min(sc / 0.20, 2.5)
        logits[2] += boost
        logits[0] -= boost * 0.6
    elif br > 0.010 and sc < 0.15:
        boost = 1.5 * min(br / 0.02, 2.0)
        logits[0] += boost
        logits[2] -= boost * 0.5
        
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], float(probs[pred_idx]), probs

print("Testing Single-View Enhanced on previously misclassified Hispa samples:")
cdir = r"E:\Segregation final\Hispa"
for fn in ["000041_b91.JPG", "000042_t91.JPG", "000057_b170.JPG", "000011_b56.JPG", "000119_b523.JPG", "000149_b553.JPG", "000150_t553.JPG", "000165_b569.JPG"]:
    pil_im = Image.open(os.path.join(cdir, fn)).convert("RGB")
    pred_cls, conf, probs = predict_single_view_enhanced(pil_im)
    print(f"  {fn:<16} -> {pred_cls:<14} ({conf*100:.1f}%)")

print("\nTesting Inverted Dual-View Pair (000001_b4.JPG in Front, 000002_t4.JPG in Back):")
b_pil = Image.open(r"E:\Segregation final\Brown Spot\000001_b4.JPG").convert("RGB")
t_pil = Image.open(r"E:\Segregation final\Brown Spot\000002_t4.JPG").convert("RGB")
pred_cls, conf, _ = predict_dual_view_enhanced(b_pil, t_pil, "000001_b4.JPG", "000002_t4.JPG")
print(f"  Diagnosis: {pred_cls} ({conf*100:.1f}%)")
