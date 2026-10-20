import urllib.request
import json

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing /api/health ...")
    with urllib.request.urlopen(f"{base_url}/api/health") as resp:
        health_data = json.loads(resp.read().decode())
        print("  Status:", health_data.get("status"))
        print("  Inference Engine:", health_data.get("inference_engine"))
        print("  Inference Location:", health_data.get("inference_location"))
        print("  ONNX Model Size:", health_data.get("onnx_model_size_mb"), "MB")
        assert health_data.get("status") == "healthy"
        assert health_data.get("server_side_inference") is False
    
    print("Testing /api/model-info ...")
    with urllib.request.urlopen(f"{base_url}/api/model-info") as resp:
        model_info = json.loads(resp.read().decode())
        print("  Model:", model_info.get("model_name"))
        print("  Engine:", model_info.get("inference_engine"))
        print("  Classes:", model_info.get("classes"))
        assert len(model_info.get("classes", [])) == 4
    
    print("Testing ONNX static model serving ...")
    req = urllib.request.Request(f"{base_url}/models/best_dual_efficientnet_b0.onnx")
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type")
        length = resp.headers.get("Content-Length")
        print(f"  Content-Type: {content_type}, Length: {int(length)/(1024*1024):.2f} MB")
        assert resp.status == 200
        assert int(length) > 10 * 1024 * 1024
    
    print("Testing WASM binary serving ...")
    req = urllib.request.Request(f"{base_url}/js/ort/ort-wasm-simd.wasm")
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type")
        length = resp.headers.get("Content-Length")
        print(f"  ort-wasm-simd.wasm -> Content-Type: {content_type}, Length: {int(length)/(1024*1024):.2f} MB")
        assert resp.status == 200
        assert "application/wasm" in content_type

    print("\nALL BACKEND HOSTING CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
