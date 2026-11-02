# Task 4 — Cloud Deployment, CI/CD & Technical Report

## 1. Project

Predictive Maintenance System for Industrial Rotating Equipment.

The existing project contains a FastAPI REST API and Streamlit monitoring dashboard. The FastAPI service exposes `/`, `/health`, `/model/info`, `/predict`, and `/predict/batch` endpoints.

## 2. Cloud Deployment

### Recommended platform: Render

The repository includes `render.yaml`.

Render build command:

```bash
pip install -r requirements.txt
```

Render start command:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
/health
```

After deployment, the public API URL can be tested with:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
```

Do not replace `YOUR-RENDER-SERVICE` in this document until the actual Render URL is available.

## 3. CI/CD

GitHub Actions workflow:

```text
.github/workflows/python-check.yml
```

The workflow runs on pushes and pull requests to `main` or `master`.

It:
1. Checks out the repository.
2. Installs Python 3.11.
3. Installs `requirements.txt`.
4. Compiles the Python source files.
5. Runs `test_api.py` when that file exists.

Render is configured for automatic deployment, so a successful repository update can trigger a new cloud deployment.

## 4. API Architecture

```text
Browser / Plant Operator
          |
          v
Streamlit Dashboard
          |
          | HTTP
          v
FastAPI REST API
          |
          v
Physical Feature Engineering
          |
          v
XGBoost / Random Forest Model
          |
          v
Prediction + Risk + Recommendation
```

## 5. Existing Model Performance Benchmark

The project's Task 2 held-out evaluation reports the following results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9938 | 0.3333 | 0.5000 | 0.4000 | 0.9954 |
| XGBoost | 0.9958 | 0.5000 | 0.8333 | 0.6250 | 0.9967 |

XGBoost is documented as the best-performing production model.

The held-out test contained 1,440 samples. XGBoost detected 5 of 6 failure events.

## 6. Runtime API Benchmark

The repository includes `benchmark_api.py`.

Run locally:

```bash
python benchmark_api.py
```

Run against the deployed Render service:

### Windows PowerShell

```powershell
$env:API_URL="https://YOUR-RENDER-SERVICE.onrender.com"
python benchmark_api.py
```

### Command Prompt

```cmd
set API_URL=https://YOUR-RENDER-SERVICE.onrender.com
python benchmark_api.py
```

The script reports minimum, mean, median, and maximum HTTP response time for:

- `/`
- `/health`
- `/model/info`

These latency values must be recorded from an actual deployment run; they are intentionally not pre-filled.

## 7. Deployment Verification Checklist

- [ ] GitHub repository contains the application source files.
- [ ] `api.py` is present at repository root.
- [ ] `models/xgboost_model.pkl` is present.
- [ ] `models/random_forest_model.pkl` is present.
- [ ] `requirements.txt` is present.
- [ ] `render.yaml` is present.
- [ ] Render deployment succeeds.
- [ ] `/health` returns a healthy response.
- [ ] `/docs` opens Swagger UI.
- [ ] `/predict` accepts a valid telemetry request.
- [ ] `benchmark_api.py` completes against the public URL.
- [ ] GitHub Actions CI passes.

## 8. Important Note

The deployment configuration assumes the existing project structure documented in the main README. This file does not invent cloud latency measurements. Runtime latency must be measured after the actual service is deployed.
