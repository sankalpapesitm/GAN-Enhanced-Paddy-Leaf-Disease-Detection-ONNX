import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
PTH_PATH = BASE_DIR / "backend" / "models" / "best_dual_efficientnet_b0.pth"
ONNX_BACKEND_PATH = BASE_DIR / "backend" / "models" / "best_dual_efficientnet_b0.onnx"
FRONTEND_MODELS_DIR = BASE_DIR / "frontend" / "models"
ONNX_FRONTEND_PATH = FRONTEND_MODELS_DIR / "best_dual_efficientnet_b0.onnx"

CLASSES = [
    "Brown spot",
    "Healthy Paddy",
    "Hispa",
    "Leaf Blast"
]

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class DualViewEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        base_model = efficientnet_b0(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        backbone_features = base_model.classifier[1].in_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(backbone_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, img_top, img_bottom):
        feat_top = self.extract_features(img_top)
        feat_bottom = self.extract_features(img_bottom)
        fused = torch.cat([feat_top, feat_bottom], dim=1)
        logits = self.classifier(fused)
        return logits

def main():
    print(f"Loading checkpoint: {PTH_PATH}")
    if not PTH_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {PTH_PATH}")

    net = DualViewEfficientNetB0(num_classes=len(CLASSES))
    checkpoint = torch.load(str(PTH_PATH), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(clean_sd, strict=True)
    net.eval()
    print("State dict loaded successfully into DualViewEfficientNetB0.")

    dummy_top = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    dummy_bottom = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    print("Exporting to ONNX...")
    torch.onnx.export(
        net,
        (dummy_top, dummy_bottom),
        str(ONNX_BACKEND_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["img_top", "img_bottom"],
        output_names=["logits"],
        dynamic_axes={
            "img_top": {0: "batch_size"},
            "img_bottom": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"ONNX exported to {ONNX_BACKEND_PATH} (size: {ONNX_BACKEND_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Verify ONNX model with onnx package
    onnx_model = onnx.load(str(ONNX_BACKEND_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker passed successfully!")

    # Copy to frontend/models
    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(ONNX_BACKEND_PATH), str(ONNX_FRONTEND_PATH))
    print(f"Copied ONNX model to {ONNX_FRONTEND_PATH}")

    # Validate numerical parity using actual sample image
    sample_top_path = BASE_DIR / "frontend" / "assets" / "samples" / "sample_01_t1.jpg"
    sample_bot_path = BASE_DIR / "frontend" / "assets" / "samples" / "sample_01_b1.jpg"

    if sample_top_path.exists() and sample_bot_path.exists():
        im_top = Image.open(sample_top_path).convert("RGB")
        im_bot = Image.open(sample_bot_path).convert("RGB")
        t_top = eval_transform(im_top).unsqueeze(0)
        t_bot = eval_transform(im_bot).unsqueeze(0)

        # PyTorch inference
        with torch.inference_mode():
            pyt_logits = net(t_top, t_bot).numpy()

        # ONNX Runtime inference
        ort_sess = ort.InferenceSession(str(ONNX_BACKEND_PATH), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "img_top": t_top.numpy(),
            "img_bottom": t_bot.numpy()
        }
        ort_logits = ort_sess.run(["logits"], ort_inputs)[0]

        max_diff = np.max(np.abs(pyt_logits - ort_logits))
        print(f"PyTorch logits:     {pyt_logits}")
        print(f"ONNXRuntime logits: {ort_logits}")
        print(f"Maximum absolute difference: {max_diff:.8f}")
        assert max_diff < 1e-4, f"Discrepancy too high: {max_diff}"
        print("VERIFICATION SUCCESS: PyTorch and ONNX Runtime outputs match identically!")
    else:
        print("Sample images not found for numerical test.")

if __name__ == "__main__":
    main()
