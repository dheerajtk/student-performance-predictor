"""Integration test for the full web app (Flask test client)."""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from app import app
c = app.test_client()

# 1. Homepage
r = c.get("/")
assert r.status_code == 200
assert b"Student Performance Predictor" in r.data
assert b"predict-form" in r.data
print("PASS  GET /                     HTML + form present")

# 2. CSS
r = c.get("/static/css/style.css")
assert r.status_code == 200
assert b"font-family" in r.data
print("PASS  GET /static/css/style.css")

# 3. JS
r = c.get("/static/js/app.js")
assert r.status_code == 200
assert b"buildPlotsGrid" in r.data
print("PASS  GET /static/js/app.js")

# 4. Full prediction roundtrip — good student
r = c.post("/api/predict", json={
    "studytime": 3, "attendance_pct": 92,
    "G1": 15, "failures": 0, "absences": 3
})
d = r.get_json()
assert d["success"], d
data = d["data"]
assert data["pass_fail_label"] == "Pass"
assert 0 <= data["predicted_g3"] <= 20
assert len(data["explanation"]) >= 3
assert len(data["feature_importance_reg"]) == 5
print(f"PASS  POST /api/predict  G3={data['predicted_g3']}  {data['grade_label']}")
print(f"        pass_prob={data['pass_probability']}  confidence={data['confidence_label']}")
print(f"        AUC={data['model_clf_metrics']['auc']}  R2={data['model_reg_metrics']['r2']}")

# 5. All 7 EDA plots
plots = [
    "plots/01_target_distribution.png",
    "plots/02_passfail_pie.png",
    "plots/03_feature_distributions.png",
    "plots/04_correlation_heatmap.png",
    "plots/05_studytime_vs_g3.png",
    "plots/06_attendance_vs_g3.png",
    "plots/07_failures_vs_g3.png",
]
for p in plots:
    r = c.get(f"/api/plots/{p}")
    assert r.status_code == 200, f"{p} -> {r.status_code}"
print("PASS  All 7 EDA plot routes  200 OK")

# 6. Client-side validation bypassed — server still validates
r = c.post("/api/predict", json={"studytime": 5, "attendance_pct": 110, "G1": -1, "failures": 0, "absences": 0})
d = r.get_json()
assert not d["success"]
assert r.status_code == 422
print("PASS  Server validation catches out-of-range values")

print()
print("=" * 48)
print("  All integration tests passed.")
print("=" * 48)
