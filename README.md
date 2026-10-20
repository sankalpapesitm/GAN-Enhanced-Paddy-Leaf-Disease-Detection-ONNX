# PaddyScan AI • ONNX Runtime Web Dual-View Paddy Leaf Disease Diagnostic System

An agricultural diagnostic web platform powered by a trained **Dual-View Siamese EfficientNet-B0** model (`best_dual_efficientnet_b0.onnx`) running **100% on the user's device CPU via WebAssembly (WASM)**.

Designed for universal deployment across **laptops, desktops, Android phones, and iPhones**, hosted reliably on **Render** with zero GPU requirements and zero server-side image processing.

---

## 🌾 Architecture & Highlights

- **100% On-Device CPU Inference**: All AI image predictions execute directly inside the user's browser using **ONNX Runtime Web with WebAssembly (WASM)**. No uploaded image is ever sent to Flask, Render, or any remote server.
- **Zero Retraining Required**: Converted the existing trained checkpoint `best_dual_efficientnet_b0.pth` directly to ONNX format (`best_dual_efficientnet_b0.onnx`, 20.37 MB) with strict numerical parity (< 0.00004 logit deviation).
- **Universal Device CPU Support**: Supports Android (Chrome, Samsung Internet), iOS (Safari, Chrome), Windows, macOS, and Linux without WebGPU, WebGL, or CUDA dependencies.
- **Siamese 2560D Feature Fusion**: Extracts 1280 features from the Front View (T1) and 1280 features from the Back View (D1), fusing them into a 2560-dimensional representation.
- **Native Front-Only Support (T1)**: Evaluates adaxial leaf surfaces via Siamese feature extraction fallback with horizontal-flip Test-Time Augmentation (TTA).
- **Client-Side Botanical Validation & Calibration**: Performs real-time foliar color segmentation (chlorophyll vegetative tissue, chlorotic margins, necrotic lesions) and uncertainty calibration between Brown Spot and Hispa.
- **Folder-Based Batch Processing**: Identifies leaf pairs by specimen ID, manages pairing modes (Auto, Strict Dual, Top-Only), generates real-time Chart.js distribution charts, and exports results to CSV.
- **Ultra-Lightweight Render Hosting**: The Flask server only hosts static frontend assets and the ONNX model files. No heavy PyTorch runtime required on Render!

---

## 📁 Project Structure

```
PaddyDiseaseApp/
├── frontend/
│   ├── index.html              # Responsive agricultural dashboard
│   ├── css/
│   │   └── style.css           # Premium Emerald & Mint agricultural design system
│   ├── js/
│   │   ├── script.js           # Browser ONNX WASM engine, canvas preprocessing & UI controller
│   │   └── ort/                # Self-hosted ONNX Runtime Web universal WASM CPU binaries
│   │       ├── ort.min.js
│   │       ├── ort-wasm.wasm
│   │       └── ort-wasm-simd.wasm
│   ├── models/
│   │   └── best_dual_efficientnet_b0.onnx # Converted ONNX model (20.37 MB)
│   └── assets/
│       └── samples/            # Verified demo leaf images for testing
├── backend/
│   ├── app.py                  # Lightweight static asset & model file server
│   ├── requirements.txt        # Flask, Flask-Cors, gunicorn (fast Render build)
│   ├── test_api.py             # Hosting & MIME-type validation script
│   └── models/
│       ├── best_dual_efficientnet_b0.pth  # Original trained PyTorch checkpoint
│       └── best_dual_efficientnet_b0.onnx # Converted ONNX model
├── convert_to_onnx.py          # PyTorch-to-ONNX conversion & numerical verification script
└── README.md                   # System documentation
```

---

## 🎯 Supported Disease Classes

| Index | Class Name | Severity | Pathogen / Cause |
|---|---|---|---|
| **0** | `Brown spot` | Moderate to High | *Bipolaris oryzae* (*Cochliobolus miyabeanus*) |
| **1** | `Healthy Paddy` | None (Normal) | Physiologically balanced leaf |
| **2** | `Hispa` | Moderate to Severe | *Dicladispa armigera* (Leaf miner beetle) |
| **3** | `Leaf Blast` | Critical / Severe | *Magnaporthe oryzae* (*Pyricularia oryzae*) |

---

## 🚀 Running Locally

### Step 1: Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

### Step 2: Start the Web Server
```powershell
python backend/app.py
```

### Step 3: Open in Browser
Open your browser on PC or mobile phone:
```
http://localhost:5000
```
On mobile phones connected to the same Wi-Fi, use the LAN URL displayed in the terminal (e.g. `http://192.168.x.x:5000`).

---

## 🌐 Deploying to Render

1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your repository.
3. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `gunicorn --chdir backend app:app -b 0.0.0.0:$PORT`
4. Deploy! The build completes in seconds because no heavy GPU or PyTorch binaries are needed on the server.
