import urllib.request
import json
import os

def post_batch(url, file_paths, mode="auto"):
    boundary = "----WebKitFormBoundaryBatchTest123"
    body = bytearray()
    body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}\r\n".encode("utf-8"))
    for fp in file_paths:
        fn = os.path.basename(fp)
        with open(fp, "rb") as f:
            data = f.read()
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{fn}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

hispa_dir = r"E:\Segregation final\Hispa"
test_files = [os.path.join(hispa_dir, fn) for fn in ["000041_b91.JPG", "000042_t91.JPG", "000057_b170.JPG", "000058_t170.JPG", "000011_b56.JPG", "000012_t56.JPG"]]

print("=== Batch Results (auto mode) ===")
status, res = post_batch("http://127.0.0.1:5000/api/predict/batch", test_files, mode="auto")
for r in res.get("results", []):
    print(" ", r.get("sample_id"), "| mode:", r.get("mode"), "| status:", r.get("status"), "| pred:", r.get("predicted_class"), "| conf:", r.get("confidence"))

print("\n=== Batch Results (front mode) ===")
status, res = post_batch("http://127.0.0.1:5000/api/predict/batch", test_files, mode="front")
for r in res.get("results", []):
    print(" ", r.get("sample_id"), "| mode:", r.get("mode"), "| status:", r.get("status"), "| pred:", r.get("predicted_class"), "| conf:", r.get("confidence"))
