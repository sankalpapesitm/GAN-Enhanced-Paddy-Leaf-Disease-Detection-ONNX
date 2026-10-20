import os
import urllib.request
from pathlib import Path

ORT_VERSION = "1.18.0"
BASE_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
TARGET_DIR = Path(__file__).resolve().parent / "frontend" / "js" / "ort"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "ort.min.js",
    "ort-wasm.wasm",
    "ort-wasm-simd.wasm",
    "ort-wasm-simd-threaded.wasm"
]

def main():
    print(f"Downloading ONNX Runtime Web v{ORT_VERSION} (universal WASM CPU builds) to {TARGET_DIR}...")
    for filename in FILES:
        url = BASE_URL + filename
        target_path = TARGET_DIR / filename
        try:
            print(f"Fetching {url} -> {filename} ...")
            urllib.request.urlretrieve(url, target_path)
            print(f"Downloaded {filename}: {target_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    print("Universal ONNX Runtime Web asset setup complete!")

if __name__ == "__main__":
    main()
