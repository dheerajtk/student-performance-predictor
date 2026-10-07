# Student Performance Predictor

> Predict a student's **final grade (0–20)** and **Pass/Fail status** from five pre-exam inputs.  
> Stack: Python · Flask · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Joblib

---

## Quick Start

```bash
# 1. Enter the project folder
cd student-performance-predictor

# 2. Install dependencies  (Python 3.10+ required)
pip install -r requirements.txt

# 3. Place student-mat.csv in data/
#    Download from: https://archive.ics.uci.edu/dataset/320/student+performance

# 4. Train all models  (~5 seconds)
python -m src.train

# 5. (Optional) Run EDA to regenerate plots
python -m src.eda

# 6. Start the web app
python app.py
#    Open: http://127.0.0.1:5000
```

---

## Dataset

**UCI Student Performance — Math** (`student-mat.csv`)  
Source: <https://archive.ics.uci.edu/dataset/320/student+performance>  
395 students · 33 original attributes · Portuguese secondary school

### Features used

| Input field | Source column | Range | Description |
|---|---|---|---|
| Study Time | `studytime` | 1–4 | Weekly study-time category |
| Attendance % | *derived* | 0–100 | `100 − absences × 2.5` (see note) |
| Previous Marks | `G1` | 0–20 | First-period grade |
| Previous Failures | `failures` | 0–4 | Past class failures |
| Absences | `absences` | 0–93 | Recorded absences |

**Target:** `G3` — Final grade (0–20).  Pass if G3 ≥ 10.

### Attendance note
The UCI dataset has **no attendance percentage column**.  
`attendance_pct` is derived from `absences` using:  `attendance_pct = max(0, 100 − absences × 2.5)`.  
This maps 0 absences → 100 % and 40+ absences → 0 %. Both values are sent to the model.  
This derivation is documented here, in the UI info note, and in `src/load_data.py`.

### Target leakage prevention
`G2` (second-period grade) is **excluded**. It correlates ≈ 0.90 with G3 and  
would represent near-future knowledge unavailable at prediction time.  
Only `G1` (first-period grade, "Previous Marks") is used.

---

## Models

### Regression  (predicting G3, 0–20)

| Model | RMSE | MAE | R² | CV-R² |
|---|---|---|---|---|
| Linear Regression | 2.849 | 2.142 | 0.596 | 0.658 |
| Ridge Regression (L2) | 2.845 | 2.133 | 0.597 | 0.660 |
| **Polynomial Ridge (degree=2)** ✅ | **2.716** | **2.024** | **0.633** | 0.580 |

### Classification  (predicting Pass/Fail)

| Model | Accuracy | F1 | AUC | CV-Acc |
|---|---|---|---|---|
| **Logistic Regression (C=1)** ✅ | **82.3 %** | **0.860** | **0.905** | **85.8 %** |
| Logistic Regression (C=0.1) | 82.3 % | 0.863 | 0.898 | 85.1 % |
| Polynomial Logistic (deg=2) | 83.5 % | 0.876 | 0.897 | 83.9 % |

> Best regressor selected by highest test-set R². Best classifier by highest AUC.

---

## Project Structure

```
student-performance-predictor/
├── data/
│   └── student-mat.csv          # UCI dataset (not committed, download separately)
├── models/
│   ├── scaler.pkl               # Fitted StandardScaler
│   ├── best_regressor.pkl       # Polynomial Ridge
│   ├── best_classifier.pkl      # Logistic Regression (C=1)
│   ├── model_metadata.json      # All metrics, model names, feature list
│   ├── confusion_matrix.png
│   ├── feature_importance_reg.png
│   ├── feature_importance_clf.png
│   └── plots/                   # 7 EDA plots (PNG)
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── src/
│   ├── __init__.py
│   ├── load_data.py             # Dataset loading + feature engineering
│   ├── eda.py                   # EDA + 7 visualisation plots
│   ├── train.py                 # Full training pipeline
│   └── predict.py               # Prediction + explanation logic
├── app.py                       # Flask application (6 routes)
├── test_api.py                  # 9 Flask test-client unit tests
├── test_integration.py          # Full integration test (HTML + API + plots)
├── requirements.txt
└── README.md
```

---

## API Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check `{"status":"ok"}` |
| `GET` | `/api/metadata` | Model names + all metrics (JSON) |
| `POST` | `/api/predict` | Predict from JSON body (see below) |
| `GET` | `/api/plots/<file>` | Serve PNG plots |

### POST /api/predict

**Request body:**
```json
{
  "studytime":      2,
  "attendance_pct": 85,
  "G1":             12,
  "failures":       0,
  "absences":       6
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predicted_g3":      12.3,
    "predicted_g3_int":  12,
    "grade_label":       "Sufficient (10-14)",
    "pass_fail":         1,
    "pass_fail_label":   "Pass",
    "pass_probability":  0.9371,
    "fail_probability":  0.0629,
    "confidence_label":  "Very High",
    "explanation":       ["Your previous mark of 12.0/20 is average..."],
    "feature_importance_reg": [{"feature": "Previous Marks (G1)", "importance": 0.72}, ...],
    "feature_importance_clf": [...],
    "model_reg_name":    "Polynomial Ridge (degree=2)",
    "model_clf_name":    "Logistic Regression (C=1)",
    "model_reg_metrics": {"rmse": 2.716, "mae": 2.024, "r2": 0.633, ...},
    "model_clf_metrics": {"accuracy": 0.823, "f1": 0.860, "auc": 0.905, ...}
  }
}
```

---

## Running Tests

```bash
# Unit tests (9 API tests)
python test_api.py

# Integration tests (HTML + API + all 7 plots)
python test_integration.py
```

---

## License

Dataset: UCI Machine Learning Repository (CC BY 4.0)  
Code: MIT
