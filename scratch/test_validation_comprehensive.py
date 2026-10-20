import urllib.request
import urllib.error
import json
import io
from pathlib import Path
from PIL import Image
import numpy as np

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

def make_solid_image_bytes(color):
    img = Image.new("RGB", (224, 224), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def make_noise_image_bytes():
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

samples_dir = Path("frontend/assets/samples")
with open(samples_dir / "sample_01_t1.jpg", "rb") as f:
    valid_t1 = f.read()
with open(samples_dir / "sample_01_b1.jpg", "rb") as f:
    valid_b1 = f.read()

blue_bytes = make_solid_image_bytes((30, 100, 240))
white_bytes = make_solid_image_bytes((255, 255, 255))
noise_bytes = make_noise_image_bytes()
corrupted_bytes = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"

print("--- TEST 1: Valid Paddy Leaf (Front-View) ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("sample_01_t1.jpg", valid_t1))])
print(f"Status: {code}")
print(f"Result: Disease='{res.get('predicted_class')}', Confidence={res.get('confidence')}%\n")
assert code == 200
assert res.get("predicted_class") is not None

print("--- TEST 2: Valid Paddy Leaf (Dual-View) ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("sample_01_t1.jpg", valid_t1)),
    ("back_image", ("sample_01_b1.jpg", valid_b1))
])
print(f"Status: {code}")
print(f"Result: Disease='{res.get('predicted_class')}', Confidence={res.get('confidence')}%\n")
assert code == 200
assert res.get("predicted_class") is not None

print("--- TEST 3: Invalid Non-Leaf (Blue Sky / Solid Blue) Front-View ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("sky.jpg", blue_bytes))])
print(f"Status: {code}")
print(f"Message: {res.get('message')}")
print(f"Error Type: {res.get('error_type')}\n")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")
assert "predicted_class" not in res

print("--- TEST 4: Invalid Non-Leaf (Blank White Image) Front-View ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("blank.jpg", white_bytes))])
print(f"Status: {code}")
print(f"Message: {res.get('message')}")
print(f"Error Type: {res.get('error_type')}\n")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")
assert "predicted_class" not in res

print("--- TEST 5: Invalid Random Noise Front-View ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("noise.jpg", noise_bytes))])
print(f"Status: {code}")
print(f"Message: {res.get('message')}")
print(f"Error Type: {res.get('error_type')}\n")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")
assert "predicted_class" not in res

print("--- TEST 6: Corrupted File Front-View ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("corrupted.jpg", corrupted_bytes))])
print(f"Status: {code}")
print(f"Message: {res.get('message')}")
print(f"Error Type: {res.get('error_type')}\n")
assert code == 400
assert "Corrupted or invalid image" in res.get("message")

print("--- TEST 7: Dual-View with Invalid Back Image ---")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("sample_01_t1.jpg", valid_t1)),
    ("back_image", ("blue_back.jpg", blue_bytes))
])
print(f"Status: {code}")
print(f"Message: {res.get('message')}")
print(f"Failed View: {res.get('failed_view')}\n")
assert code == 400
assert "Invalid image (Back View): Please upload a paddy leaf image for disease detection." in res.get("message")
assert res.get("failed_view") == "back"
assert "predicted_class" not in res

print("ALL 7 TESTS PASSED PERFECTLY!")
