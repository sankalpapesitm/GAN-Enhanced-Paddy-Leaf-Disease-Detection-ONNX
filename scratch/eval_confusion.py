import os, re, sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, r"c:\Users\sanka\OneDrive\Desktop\GAN Enhanced Paddy Leaf Disease Detection\backend")
# pyrefly: ignore[missing-import]
from app import model, CLASSES, eval_transform

classes_dirs = ["Brown Spot", "Healthy Paddy", "Hispa", "Leaf Blast"]
dataset_path = r"E:\Segregation final"

confusion = np.zeros((4, 4), dtype=int)
# row: true, col: pred

misclassified_pairs = []

for true_idx, cname in enumerate(classes_dirs):
    cdir = os.path.join(dataset_path, cname)
    files = sorted(os.listdir(cdir))
    grouped = {}
    for f in files:
        m = re.search(r"[_-](?:t|b|d)(\d+)\.", f, re.I)
        if m:
            pid = m.group(1)
            grouped.setdefault(pid, {})
            if re.search(r"[_-]t\d+\.", f, re.I):
                grouped[pid]["top"] = os.path.join(cdir, f)
            else:
                grouped[pid]["bottom"] = os.path.join(cdir, f)
    
    for pid, pair in grouped.items():
        if "top" in pair and "bottom" in pair:
            im_t = eval_transform(Image.open(pair["top"]).convert("RGB")).unsqueeze(0)
            im_b = eval_transform(Image.open(pair["bottom"]).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                out = model(im_t, im_b)
                probs = F.softmax(out, dim=1)[0].numpy()
                pred_idx = int(out.argmax(dim=1)[0])
            confusion[true_idx, pred_idx] += 1
            if true_idx != pred_idx:
                misclassified_pairs.append({
                    "true_class": cname,
                    "pred_class": CLASSES[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "top": os.path.basename(pair["top"]),
                    "bottom": os.path.basename(pair["bottom"]),
                    "probs": {CLASSES[i]: float(probs[i]) for i in range(4)}
                })

print("Dual-view Confusion Matrix:")
headers = [f"{c[:6]:>8}" for c in classes_dirs]
print(f"{'True \\ Pred':<15} {' '.join(headers)}")
for i, c in enumerate(classes_dirs):
    row_str = " ".join([f"{confusion[i, j]:>8d}" for j in range(4)])
    print(f"{c:<15} {row_str}")

acc = np.trace(confusion) / np.sum(confusion) * 100
print(f"\nTotal pairs tested: {np.sum(confusion)}, Overall Accuracy: {acc:.2f}%\n")

print(f"Total misclassified pairs: {len(misclassified_pairs)}")
for m in misclassified_pairs:
    print(f"True: {m['true_class']:<14} | Pred: {m['pred_class']:<14} ({m['confidence']*100:.1f}%) | Top: {m['top']} | Bottom: {m['bottom']}")
    print(f"   Probs: {m['probs']}")
