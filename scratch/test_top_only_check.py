import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"

def test_top_only_batch():
    samples_dir = Path("frontend/assets/samples")
    files_to_test = [
        "sample_01_t1.jpg",
        "sample_01_b1.jpg",
        "sample_02_t2.jpg",
        "sample_03_t3.jpg"
    ]
    
    # 1. Test mode = "front" (Top-Only Evaluation)
    multipart_files = []
    for fn in files_to_test:
        fp = samples_dir / fn
        if fp.exists():
            multipart_files.append(("files", (fn, open(fp, "rb"), "image/jpeg")))
            
    print(f"Uploading {len(multipart_files)} files in mode='front'...")
    res = requests.post(f"{BASE_URL}/api/predict/batch", files=multipart_files, data={"mode": "front"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    
    print("\n--- Summary for Top-Only Mode ---")
    print(json.dumps(data["summary"], indent=2))
    
    print("\n--- Results for Top-Only Mode ---")
    for r in data["results"]:
        print(f"Sample: {r['sample_id']} | Mode: {r['mode']} | Front: {r['front_filename']} | Back: {r['back_filename']} | Pred: {r.get('predicted_class')} ({r.get('confidence')}%)")
        assert r["back_filename"] is None, f"Expected back_filename to be None for Top-Only, got {r['back_filename']}"
        assert "Top-Only" in r["mode"], f"Expected mode to contain Top-Only, got {r['mode']}"
        assert r["status"] == "success", f"Expected success status, got {r['status']}"
        
    print("\nTop-Only batch verification PASSED successfully!")

if __name__ == "__main__":
    test_top_only_batch()
