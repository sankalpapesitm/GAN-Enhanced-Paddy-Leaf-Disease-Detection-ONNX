import urllib.request
import json
import os

def post_batch(url, file_paths, mode="front"):
    boundary = "----WebKitFormBoundaryTopOnlyBatchTest"
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

base = r"E:\Segregation final"
test_cases = {
    "Hispa": [os.path.join(base, "Hispa", "000042_t91.JPG"), os.path.join(base, "Hispa", "000058_t170.JPG")],
    "Brown Spot": [os.path.join(base, "Brown Spot", "000006_t24.JPG"), os.path.join(base, "Brown Spot", "000008_t25.JPG")],
    "Healthy Paddy": [os.path.join(base, "Healthy Paddy", "000002_t7.JPG"), os.path.join(base, "Healthy Paddy", "000004_t8.JPG")],
    "Leaf Blast": [os.path.join(base, "Leaf Blast", "000002_t1.JPG"), os.path.join(base, "Leaf Blast", "000004_t2.JPG")]
}

print("=== TESTING TOP-ONLY BATCH MODE ACROSS ALL 4 CLASSES ===")
for true_class, file_list in test_cases.items():
    status, res = post_batch("http://127.0.0.1:5000/api/predict/batch", file_list, mode="front")
    for r in res.get("results", []):
        sid = r.get("sample_id")
        pred = r.get("predicted_class")
        conf = r.get("confidence")
        ok = "[OK]" if pred.lower().replace(" ", "") == true_class.lower().replace(" ", "") else "[MISMATCH]"
        print(f"  {ok} {sid:<10} | True: {true_class:<14} | Pred: {pred:<14} ({conf:5.1f}%)")
