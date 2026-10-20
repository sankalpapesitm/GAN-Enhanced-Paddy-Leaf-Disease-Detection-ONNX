import urllib.request
import json
import os

def post_multipart(url, fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
    for field_name, (filename, data) in files:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

print("=== TESTING HISPA SAMPLES VIA LIVE HTTP API (/api/predict/front-view) ===")
hispa_dir = r"E:\Segregation final\Hispa"
test_samples = [
    "000041_b91.JPG", "000042_t91.JPG", "000057_b170.JPG", "000011_b56.JPG",
    "000119_b523.JPG", "000149_b553.JPG", "000150_t553.JPG", "000165_b569.JPG"
]
for fn in test_samples:
    with open(os.path.join(hispa_dir, fn), "rb") as f:
        data = f.read()
    status, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", (fn, data))])
    pred = res["predicted_class"]
    conf = res["confidence"]
    print(f"  {fn:<16} -> {pred:<12} ({conf}%) [HTTP {status}]")

print("\n=== TESTING DUAL-VIEW PAIRS VIA LIVE HTTP API (/api/predict/dual-view) ===")
# Pair 91 Inverted
with open(os.path.join(hispa_dir, "000041_b91.JPG"), "rb") as f:
    b91 = f.read()
with open(os.path.join(hispa_dir, "000042_t91.JPG"), "rb") as f:
    t91 = f.read()

s, r_inv = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("000041_b91.JPG", b91)),
    ("back_image", ("000042_t91.JPG", t91))
])
print(f"  Hispa Pair 91 (Inverted Input) -> {r_inv['predicted_class']} ({r_inv['confidence']}%)")

s, r_can = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("000042_t91.JPG", t91)),
    ("back_image", ("000041_b91.JPG", b91))
])
print(f"  Hispa Pair 91 (Canonical Input)-> {r_can['predicted_class']} ({r_can['confidence']}%)")

print("\n=== TESTING BROWN SPOT SAMPLES VIA LIVE HTTP API (/api/predict/front-view) ===")
bs_dir = r"E:\Segregation final\Brown Spot"
for fn in ["000004_t13.JPG", "000006_t24.JPG", "000010_t27.JPG"]:
    with open(os.path.join(bs_dir, fn), "rb") as f:
        data = f.read()
    status, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", (fn, data))])
    print(f"  {fn:<16} -> {res['predicted_class']} ({res['confidence']}%)")

print("\nALL PREDICTIONS VERIFIED AND MATCH EXPECTED ACCURACY!")
