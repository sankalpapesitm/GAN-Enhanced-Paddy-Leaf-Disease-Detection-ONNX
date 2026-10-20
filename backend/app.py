import os
import io
import mimetypes
import logging
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register critical MIME types for Render & local deployment
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

# ==============================================================================
# 1. CONSTANTS & METADATA (Preserved from trained model architecture)
# ==============================================================================

CLASSES = [
    "Brown spot",
    "Healthy Paddy",
    "Hispa",
    "Leaf Blast"
]

DISEASE_METADATA = {
    "Healthy Paddy": {
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None (Plant is physiologically healthy)",
        "symptoms": "Leaves display vibrant, uniform green coloration without necrotic lesions, discolored margins, or feeding trails.",
        "management": "Maintain standard water leveling (2-5 cm during vegetative stage), balanced NPK fertilization, and periodic field scouting.",
        "color_theme": "#16a34a"
    },
    "Brown spot": {
        "status": "Diseased",
        "severity": "Moderate to High",
        "pathogen": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "symptoms": "Small, circular to oval dark brown spots with yellow halos across leaf surfaces. Severe infection leads to seedling blight and kernel discoloration.",
        "management": "Apply balanced soil fertilization especially potassium (K) and silicon. Seed treatment with Carbendazim (2g/kg) or foliar spray of Mancozeb / Propiconazole.",
        "color_theme": "#d97706"
    },
    "Hispa": {
        "status": "Diseased",
        "severity": "Moderate to Severe",
        "pathogen": "Dicladispa armigera (Coleoptera: Chrysomelidae)",
        "symptoms": "Adult beetles scrape upper leaf epidermis creating characteristic white parallel streaks; grubs mine inside leaves causing blotch blister mines and withered tips.",
        "management": "Clip leaf tips before transplanting to destroy eggs. Sweep net collection in early morning. Foliar spray of Chlorpyrifos 20% EC (2.5 ml/L) or Quinalphos 25% EC.",
        "color_theme": "#ca8a04"
    },
    "Leaf Blast": {
        "status": "Diseased",
        "severity": "Critical / Severe",
        "pathogen": "Magnaporthe oryzae (Pyricularia oryzae)",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with gray-whitish centers and dark reddish-brown margins. Rapid coalescence leads to complete leaf blighting.",
        "management": "Avoid excessive nitrogen application. Spray Tricyclazole 75% WP (0.6 g/L) or Isoprothiolane 40% EC (1.5 ml/L) at initial lesion detection. Ensure good water drainage.",
        "color_theme": "#dc2626"
    }
}

# Diagnostic Metrics
FRONT_VIEW_MODEL_METRICS = {
    "accuracy": 88.4,
    "precision": 88.9,
    "recall": 87.6,
    "f1_score": 88.2
}

FRONT_VIEW_DISEASE_METRICS = {
    "Healthy Paddy": {"accuracy": 92.1, "precision": 91.5, "recall": 91.8, "f1_score": 91.6},
    "Brown spot": {"accuracy": 87.5, "precision": 88.2, "recall": 86.4, "f1_score": 87.3},
    "Hispa": {"accuracy": 88.9, "precision": 88.4, "recall": 89.2, "f1_score": 88.8},
    "Leaf Blast": {"accuracy": 88.2, "precision": 86.8, "recall": 89.5, "f1_score": 88.1}
}

DUAL_VIEW_MODEL_METRICS = {
    "accuracy": 94.2,
    "precision": 94.6,
    "recall": 93.8,
    "f1_score": 94.2
}

DUAL_VIEW_DISEASE_METRICS = {
    "Healthy Paddy": {"accuracy": 97.4, "precision": 96.7, "recall": 96.7, "f1_score": 96.7},
    "Brown spot": {"accuracy": 93.8, "precision": 95.7, "recall": 91.2, "f1_score": 93.4},
    "Hispa": {"accuracy": 94.5, "precision": 93.8, "recall": 95.2, "f1_score": 94.5},
    "Leaf Blast": {"accuracy": 94.8, "precision": 92.5, "recall": 96.0, "f1_score": 94.2}
}

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ONNX_MODEL_FILE = "best_dual_efficientnet_b0.onnx"
FRONTEND_ONNX_PATH = FRONTEND_DIR / "models" / ONNX_MODEL_FILE
BACKEND_ONNX_PATH = BASE_DIR / "models" / ONNX_MODEL_FILE

# Verify ONNX model availability
def get_onnx_status():
    if FRONTEND_ONNX_PATH.exists():
        size_mb = FRONTEND_ONNX_PATH.stat().st_size / (1024 * 1024)
        return True, str(FRONTEND_ONNX_PATH), size_mb
    if BACKEND_ONNX_PATH.exists():
        size_mb = BACKEND_ONNX_PATH.stat().st_size / (1024 * 1024)
        return True, str(BACKEND_ONNX_PATH), size_mb
    return False, None, 0.0

is_onnx_ready, onnx_loc, onnx_size = get_onnx_status()
logger.info(f"ONNX Model Status: Ready={is_onnx_ready}, Location={onnx_loc}, Size={onnx_size:.2f} MB")

# ==============================================================================
# 2. FLASK APPLICATION & STATIC ASSET HOSTING (Optimized for Render)
# ==============================================================================

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

@app.after_request
def apply_static_headers(response):
    """Ensure appropriate caching and security headers for static assets."""
    # Permit SharedArrayBuffer when available without blocking CDN resources
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    return response

@app.route("/")
def serve_index():
    """Serves the frontend application dashboard."""
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({
        "message": "PaddyScan AI is running (ONNX Runtime Web WASM CPU).",
        "docs": "Endpoints available at /api/health and /api/model-info"
    })

@app.route("/<path:path>")
def serve_static(path):
    """Serves static files (.html, .css, .js, .wasm, .onnx, assets) with accurate MIME types."""
    target_file = FRONTEND_DIR / path
    if target_file.exists():
        mimetype = None
        if path.endswith(".wasm"):
            mimetype = "application/wasm"
        elif path.endswith(".onnx"):
            mimetype = "application/octet-stream"
        elif path.endswith(".js") or path.endswith(".mjs"):
            mimetype = "text/javascript"
        return send_from_directory(str(FRONTEND_DIR), path, mimetype=mimetype)
    # If not found, fallback to index.html for SPA routing
    return send_from_directory(str(FRONTEND_DIR), "index.html")

# ==============================================================================
# 3. API HEALTH & METADATA ENDPOINTS (Zero Server-Side Image Compute)
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint confirming on-device WASM CPU inference architecture."""
    ready, loc, size = get_onnx_status()
    return jsonify({
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "inference_engine": "ONNX Runtime Web (WebAssembly WASM CPU)",
        "inference_location": "Client Device Browser (Laptop, Desktop, Android, iPhone)",
        "server_side_inference": False,
        "gpu_required": False,
        "onnx_model_file": ONNX_MODEL_FILE,
        "onnx_model_size_mb": round(size, 2),
        "onnx_static_url": f"/models/{ONNX_MODEL_FILE}",
        "device": "User Device CPU (WASM)",
        "device_name": "On-Device CPU (WASM SIMD/Single-Thread)",
        "classes": CLASSES,
        "total_parameters": 5321856,
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB"
    })

@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Returns technical specifications of the Dual-View Siamese EfficientNet-B0 model."""
    ready, loc, size = get_onnx_status()
    return jsonify({
        "status": "success",
        "loaded": ready,
        "checkpoint": ONNX_MODEL_FILE,
        "model_name": "Dual-View Siamese EfficientNet-B0 + cGAN Augmented Pipeline",
        "gan_augmentation": "Conditional Generative Adversarial Network (cGAN)",
        "inference_engine": "ONNX Runtime Web (WASM CPU)",
        "device": "Browser WASM CPU (Laptop, Android, iPhone, Desktop)",
        "device_name": "User Device CPU (WASM)",
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "total_parameters": 5321856,
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "feature_fusion": "Dual 1280-dim concatenated feature vectors (2560-dim)",
        "front_only_support": True,
        "front_only_mechanism": "Trained Siamese single-view projection (img_bottom = img_top fallback from author's training code)",
        "metrics": DUAL_VIEW_MODEL_METRICS,
        "class_metrics": DUAL_VIEW_DISEASE_METRICS,
        "front_metrics": FRONT_VIEW_MODEL_METRICS,
        "disease_reference": DISEASE_METADATA
    })

@app.route("/api/network-info", methods=["GET"])
def api_network_info():
    """Returns local network IP addresses for phone pairing if running locally."""
    ips = []
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip and not primary_ip.startswith("127."):
            ips.append(primary_ip)
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    primary = ips[0] if ips else "127.0.0.1"
    return jsonify({
        "status": "success",
        "primary_ip": primary,
        "port": port,
        "all_ips": ips,
        "mobile_url": f"http://{primary}:{port}",
        "local_url": f"http://127.0.0.1:{port}"
    })

@app.route("/api/predict/front-view", methods=["POST"])
@app.route("/api/predict/dual-view", methods=["POST"])
@app.route("/api/predict/batch", methods=["POST"])
def redirect_to_browser_inference():
    """Safety endpoint: informs that inference runs 100% in user browser on CPU."""
    return jsonify({
        "status": "info",
        "message": "AI prediction runs directly on the user's device CPU using ONNX Runtime Web. Uploading images to the server is not required.",
        "inference_mode": "Client-Side Browser WASM CPU"
    }), 200

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))

    logger.info(f"Starting PaddyScan AI Web Server on http://0.0.0.0:{port}")
    print("\n" + "=" * 66)
    print("[PADDYSCAN AI SERVER RUNNING - ONNX RUNTIME WEB HOST]")
    print(f"-> Local Access (On PC):     http://localhost:{port}")
    print(f"-> Architecture:             100% On-Device CPU Inference (WASM)")
    print(f"-> Devices Supported:        Laptops, Desktops, Android & iPhones")
    print(f"-> Server Role:              Static Host for Render / LAN")
    print("=" * 66 + "\n", flush=True)

    app.run(host=host, port=port, debug=False)
