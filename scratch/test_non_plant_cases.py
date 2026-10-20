import urllib.request
import urllib.error
import json
import io
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

# Test with a synthetic red object (e.g. car or apple)
red_img = Image.new("RGB", (224, 224), (200, 30, 30))
buf = io.BytesIO()
red_img.save(buf, format="JPEG")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("red_apple.jpg", buf.getvalue()))])
print(f"Red image test: Status={code}, Message='{res.get('message')}', ErrorType='{res.get('error_type')}'")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")

# Test with a synthetic skin/portrait tone
skin_img = Image.new("RGB", (224, 224), (230, 180, 150))
buf = io.BytesIO()
skin_img.save(buf, format="JPEG")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("face_or_hand.jpg", buf.getvalue()))])
print(f"Skin image test: Status={code}, Message='{res.get('message')}', ErrorType='{res.get('error_type')}'")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")

# Test with a synthetic dark blue / denim fabric
fabric_img = Image.new("RGB", (224, 224), (25, 45, 90))
buf = io.BytesIO()
fabric_img.save(buf, format="JPEG")
code, res = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("image", ("fabric.jpg", buf.getvalue()))])
print(f"Fabric image test: Status={code}, Message='{res.get('message')}', ErrorType='{res.get('error_type')}'")
assert code == 400
assert "Invalid image: Please upload a paddy leaf image for disease detection." in res.get("message")

print("All non-plant synthetic tests rejected with exact requested message!")
