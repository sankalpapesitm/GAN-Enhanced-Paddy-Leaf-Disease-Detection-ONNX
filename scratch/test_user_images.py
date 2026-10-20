import urllib.request
import json

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
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

with open(r"E:\Segregation final\Brown Spot\000001_b4.JPG", "rb") as f:
    b1 = f.read()
with open(r"E:\Segregation final\Brown Spot\000002_t4.JPG", "rb") as f:
    t1 = f.read()

print("=== Test Front View 000001_b4.JPG ===")
c, r = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("000001_b4.JPG", b1))])
print(f"Status: {c}")
print(f"Diagnosis: {r.get('predicted_class')}, Confidence: {r.get('confidence')}%")
assert c == 200
assert r.get("status") == "success"

print("\n=== Test Dual View (000001_b4.JPG + 000002_t4.JPG) ===")
c, r = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("000001_b4.JPG", b1)),
    ("back_image", ("000002_t4.JPG", t1))
])
print(f"Status: {c}")
print(f"Diagnosis: {r.get('predicted_class')}, Confidence: {r.get('confidence')}%")
assert c == 200
assert r.get("status") == "success"

print("\nALL USER SPECIMEN IMAGES PROCESSED SUCCESSFULLY WITH ZERO FALSE REJECTIONS!")
