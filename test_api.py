import warnings
warnings.filterwarnings("error")

from app import app
client = app.test_client()

# --- Test 1: health ---
r = client.get("/api/health")
assert r.status_code == 200
print("PASS  /api/health           ", r.get_json())

# --- Test 2: metadata ---
r = client.get("/api/metadata")
assert r.status_code == 200
d = r.get_json()
assert d["success"]
keys = list(d["data"].keys())
print("PASS  /api/metadata          keys:", keys[:4])

# --- Test 3: valid prediction (good student) ---
r = client.post("/api/predict", json={
    "studytime": 2, "attendance_pct": 85,
    "G1": 12, "failures": 0, "absences": 6
})
assert r.status_code == 200
d = r.get_json()
assert d["success"]
data = d["data"]
print(f"PASS  /api/predict (good)   G3={data['predicted_g3']}  "
      f"{data['pass_fail_label']}  prob={data['pass_probability']}")
assert 0 <= data["predicted_g3"] <= 20
assert data["pass_fail_label"] in ("Pass", "Fail")
assert 0 <= data["pass_probability"] <= 1
assert len(data["explanation"]) > 0
assert len(data["feature_importance_reg"]) == 5

# --- Test 4: valid prediction (struggling student) ---
r = client.post("/api/predict", json={
    "studytime": 1, "attendance_pct": 40,
    "G1": 4, "failures": 3, "absences": 30
})
assert r.status_code == 200
d = r.get_json()
assert d["success"]
data = d["data"]
print(f"PASS  /api/predict (weak)   G3={data['predicted_g3']}  "
      f"{data['pass_fail_label']}  prob={data['pass_probability']}")
assert data["pass_fail_label"] == "Fail", f"Expected Fail, got {data['pass_fail_label']}"

# --- Test 5: validation error (studytime out of range) ---
r = client.post("/api/predict", json={"studytime": 99, "G1": 5})
assert r.status_code == 422
d = r.get_json()
assert not d["success"]
print("PASS  /api/predict (422)     errors:", d["errors"])

# --- Test 6: empty body ---
r = client.post("/api/predict", json={})
assert r.status_code == 400
d = r.get_json()
assert not d["success"]
print("PASS  /api/predict (empty)   errors:", d["errors"])

# --- Test 7: plot route ---
r = client.get("/api/plots/confusion_matrix.png")
assert r.status_code == 200
assert "image/png" in r.content_type
print("PASS  /api/plots/confusion_matrix.png  size:", len(r.data), "bytes")

# --- Test 8: path traversal blocked ---
r = client.get("/api/plots/../app.py")
assert r.status_code in (400, 404)
print("PASS  path traversal blocked  status:", r.status_code)

# --- Test 9: wrong method ---
r = client.get("/api/predict")
assert r.status_code == 405
print("PASS  wrong method (405)     ", r.get_json())

print()
print("=" * 45)
print("  All 9 API tests passed successfully.")
print("=" * 45)
