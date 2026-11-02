# Predictive Maintenance System for Industrial Rotating Equipment

An end-to-end machine-learning project that predicts equipment degradation and anomalies in industrial rotating equipment using multi-sensor time-series telemetry.

---

## Internship Task Coverage

### Task 1 — Telemetry Data Ingestion & Exploratory Analysis
- Sensor time-series telemetry inspection across 8 rotating machines (7,200 hourly readings)
- Missing-value handling and imputation
- Extreme class-imbalance analysis (0.44% raw failure event rate)
- Sensor correlation matrix and cross-sensor dependencies
- Failure-mode profiling (Bearing Wear, Overheating, Lubrication Issue, Rotor Imbalance)
- 48-hour forward failure target engineering (`failure_next_48h`)
- Baseline Random Forest modeling with chronological 80/20 train/test split
- Streamlit interactive monitoring dashboard (`app.py`)

---

### Task 2 — Feature Engineering & Anomaly Detection Modeling

#### 1. Objective
Advance the predictive maintenance capability by synthesizing physically grounded mechanical and thermodynamic operational features, training supervised anomaly/failure classification algorithms (Random Forest and XGBoost), and rigorously validating performance using stratified 5-fold cross-validation and held-out test evaluation under extreme class imbalance.

#### 2. Feature Engineering
Raw sensor channels are augmented with two domain-specific physical interaction metrics:

- **Mechanical Power (`Mechanical_Power`)**:
  - *Physical Basis*: In rotating equipment (pumps, compressors, and motor drives), mechanical power is the product of shaft torque ($\tau$) and angular velocity ($\omega$):
    $$P_{\text{mech}} = \tau \times \omega$$
    where $\omega = \frac{2\pi \times \text{rpm}}{60}\text{ [rad/s]}$ and torque load factor is normalized as $\frac{\text{load\_pct}}{100}$.
  - *Implemented Formula*:
    $$\text{Mechanical\_Power} = \left(\frac{\text{load\_pct}}{100}\right) \times \left(\frac{2\pi \times \text{rpm}}{60}\right) = \frac{\text{load\_pct} \times \text{rpm} \times \pi}{3000}$$
  - *Statistics*: Min = 63.2110, Max = 127.5750, Mean = 92.9991.

- **Thermal Dissipation (`Thermal_Dissipation`)**:
  - *Physical Basis*: In electric induction motors, internal heat generation is dominated by stator and winding ohmic losses ($I^2 R$) and friction. Under healthy cooling, heat is efficiently dissipated, maintaining an equilibrium temperature proportional to current. When cooling fins clog, airflow is obstructed, or lubrication degrades, heat accumulates and temperature spikes relative to current draw.
  - *Implemented Formula*:
    $$\text{Thermal\_Dissipation} = \frac{\text{temperature\_c}}{\text{motor\_current\_a}}\quad [^\circ\text{C}/\text{A}]$$
  - *Statistics*: Min = 6.8147, Max = 8.8080, Mean = 7.7340.

#### 3. Classification Models
Evaluated against the ground-truth anomaly/failure target (`failure`, 32 failure events across 7,200 hours):

- **Random Forest Classifier**:
  - 250 trees, unconstrained depth, `min_samples_leaf=1`, `class_weight='balanced'`, `random_state=42`.
  - Non-linear decision boundary bagging with out-of-bag variance reduction.
- **XGBoost Classifier**:
  - 150 boosting rounds, `max_depth=6`, `learning_rate=0.08`, `scale_pos_weight=220.54`, `eval_metric='logloss'`, `random_state=42`.
  - Gradient-boosted trees with weighted loss penalties for minority class anomalies.

#### 4. Stratified 5-Fold Cross-Validation

| Model | CV Mean Accuracy | CV Mean Precision | CV Mean Recall | CV Mean F1-Score | CV Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9944 (±0.0013) | 0.4069 (±0.0908) | 0.4333 (±0.2402) | 0.3849 (±0.1187) | 0.9933 (±0.0037) |
| **XGBoost** | **0.9942 (±0.0014)** | **0.4050 (±0.1122)** | **0.5286 (±0.2121)** | **0.4372 (±0.1063)** | **0.9625 (±0.0655)** |

#### 5. Held-Out Test Set Evaluation (1,440 samples, 20% Stratified Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9938 | 0.3333 | 0.5000 | 0.4000 | 0.9954 |
| **XGBoost (Best Model)** | **0.9958** | **0.5000** | **0.8333** | **0.6250** | **0.9967** |

- **Confusion Matrices (Test Set)**:
  - Random Forest: TN = 1,428, FP = 6, FN = 3, **TP = 3** (out of 6 test failures)
  - XGBoost: TN = 1,429, FP = 5, FN = 1, **TP = 5** (out of 6 test failures, **83.33% detection rate**)

#### 6. Feature Importance Analysis
Both models confirmed that the engineered features actively contribute to prediction alongside mechanical vibration and acoustics:

- **XGBoost**:
  1. `vibration_mm_s` (89.95%)
  2. `acoustic_db` (3.15%)
  3. `pressure_bar` (1.76%)
  4. `Mechanical_Power` (1.02%)
  5. `Thermal_Dissipation` (1.00%)
- **Random Forest**:
  1. `vibration_mm_s` (36.34%)
  2. `temperature_c` (30.53%)
  3. `acoustic_db` (14.02%)
  4. `motor_current_a` (10.53%)
  5. `pressure_bar` (5.97%)

#### 7. Anomaly & Failure Analysis
- Across the held-out test partition (1,440 hours), XGBoost captured **5 out of 6 failure events (83.33% sensitivity)** with only 5 false alarms out of 1,434 normal hours (0.35% false alarm rate).
- Across the full 7,200-hour telemetry lifecycle, XGBoost successfully flagged **31 out of 32 failure events (96.88% total coverage)**.
- **Best-Performing Model**: **XGBoost Classifier** is designated as the primary production model due to its significantly higher F1-score (`0.6250` vs `0.4000`), superior failure recall (`83.33%` vs `50.00%`), and higher ROC-AUC (`0.9967`).

#### 8. Generated Deliverables
- Dataset: `data/engineered_features.csv`
- Models: `models/random_forest_model.pkl`, `models/xgboost_model.pkl`
- Metrics & Analysis:
  - `data/feature_engineering_report.txt`
  - `data/cross_validation_results.csv`
  - `data/final_model_performance.csv`
  - `data/model_comparison_summary.csv`
  - `data/random_forest_feature_importance.csv`
  - `data/xgboost_feature_importance.csv`
  - `data/anomaly_detection_summary.csv`
  - `data/task2_final_report.txt`
- Publication Plots:
  - `data/plots/model_performance_comparison.png`
  - `data/plots/random_forest_confusion_matrix.png`
  - `data/plots/xgboost_confusion_matrix.png`
  - `data/plots/random_forest_feature_importance.png`
  - `data/plots/xgboost_feature_importance.png`

---

### Task 3 — FastAPI REST API & Health Monitoring Dashboard

#### 1. Objective & Architecture
Productionize the trained predictive-maintenance modeling pipeline by wrapping it in a high-throughput, low-latency asynchronous FastAPI REST API backend, documented with interactive OpenAPI/Swagger schemas, and pairing it with a Streamlit health monitoring dashboard for real-time condition monitoring, failure probability calibration, and operational risk mitigation.

```text
  [Browser / Plant Operator]
              │
              ▼
  [Streamlit Health Dashboard (dashboard.py :8501)]
              │
              ▼ HTTP REST POST /predict (JSON Telemetry)
  [FastAPI High-Throughput API Gateway (api.py :8000)]
              │
              ├─► Dynamic Physical Feature Derivation (P_mech & R_th)
              │
              ▼ 9-Feature Ingestion Vector
  [Task 2 Production Model: models/xgboost_model.pkl]
              │
              ├─► Anomaly Classification (0 = Normal, 1 = Anomaly)
              ├─► Calibrated Failure Probability (%)
              ├─► Operational Severity (Low, Medium, High, Critical)
              └─► Automated Engineering Maintenance Guidance
              │
              ▼ HTTP 200 OK
  [Live Telemetry Dashboard Status Display]
```

#### 2. REST API Endpoints Overview (`api.py`)

| Endpoint | Method | Purpose & Description | Output Schema |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | Root service descriptor, version, status, and navigation links. | `{"status": "online", "version": "1.0.0", ...}` |
| `/health` | `GET` | Live readiness diagnostics, uptime, and loaded model verification. | `HealthResponse` (`status="healthy"`, `available_models=["xgboost", "random_forest"]`) |
| `/model/info` | `GET` | Architecture specifications, Task 2 test metrics, and required features list. | `ModelInfoResponse` (Accuracy, Precision, Recall, F1, ROC-AUC) |
| `/predict` | `POST` | Single telemetry reading evaluation with automatic physical feature engineering. | `PredictionResponse` (prediction, label, probability, confidence, risk level, recommendation) |
| `/predict/batch` | `POST` | High-throughput batch inference for concurrent sensor streams or time windows. | `BatchPredictionResponse` (total samples, anomaly count, anomaly rate, predictions list) |
| `/docs` | `GET` | Interactive Swagger UI API documentation and testing workbench. | HTML / Swagger Bundle |
| `/openapi.json` | `GET` | Raw OpenAPI 3.1.0 compliant schema definition. | JSON Schema |

#### 3. Automatic Dynamic Feature Engineering
If raw sensor inputs are supplied without the Task 2 physical interaction features, `api.py` automatically computes them on-the-fly using exact physical kinematics and thermodynamics formulations:
- **Mechanical Power**: $P_{\text{mech}} = \left(\frac{\text{load\_pct}}{100}\right) \times \left(\frac{2\pi \times \text{rpm}}{60}\right) = \frac{\text{load\_pct} \times \text{rpm} \times \pi}{3000}$
- **Thermal Dissipation**: $R_{\text{th}} = \frac{\text{temperature\_c}}{\text{motor\_current\_a}}\quad [^\circ\text{C}/\text{A}]$

#### 4. Streamlit Health Monitoring Dashboard (`dashboard.py`)
- **Real-Time Telemetry Sliders**: Ingestion controls for all 7 primary sensor channels (`vibration_mm_s`, `temperature_c`, `pressure_bar`, `motor_current_a`, `acoustic_db`, `load_pct`, `rpm`).
- **Live Physical Interaction Preview**: Real-time display of calculated $P_{\text{mech}}$ and $R_{\text{th}}$ with optional manual override for engineering stress tests.
- **Model Switcher**: Dynamic runtime selection between XGBoost (Task 2 champion) and Random Forest.
- **Visual Alert Status**: Instant visual feedback (`Condition: Normal ✅` vs `Condition: FAILURE 🚨`), calibrated failure probability gauge, and operational risk classification (`Low`, `Medium`, `High`, `Critical`).
- **Quick Preset Telemetry**: 1-click loading of known healthy baseline and failing operating states.

#### 5. Verification & Test Results
- Full automated test suite verified via `test_api.py` and `verify_task3_e2e.py`.
- **Pre-flight Syntax**: 100% clean compilation across all modules (`py_compile`).
- **Dependencies**: All 13 production dependencies verified.
- **End-to-End Test Status**: **TASK 3 STATUS = PASS** (Documented in `data/task3_test_report.txt`).

---

## Project Structure

```text
predictive-maintenance-system/
├── api.py                                  # FastAPI REST API backend service (Task 3)
├── app.py                                  # Baseline Streamlit monitoring dashboard (Task 1)
├── dashboard.py                            # Streamlit Health Monitoring Dashboard (Task 3)
├── feature_engineering.py                  # Mechanical & Thermal feature pipeline (Task 2)
├── generate_data.py                        # Synthetic telemetry generator (Task 1)
├── run_task2_pipeline.py                   # Master evaluation & plotting pipeline (Task 2)
├── test_api.py                             # FastAPI automated unit test suite (Task 3)
├── train_anomaly_models.py                 # RF & XGBoost classification trainer (Task 2)
├── train_model.py                          # Baseline ML pipeline & 48h predictor (Task 1)
├── verify_task3_e2e.py                     # Master Task 3 end-to-end verification suite (Task 3)
├── requirements.txt                        # Production package dependencies
├── README.md                               # Complete engineering documentation
├── .gitignore                              # Git exclusion rules (excludes py_embed/, caches)
├── .github/
│   └── workflows/
│       └── python-check.yml                # GitHub Actions CI workflow
├── data/
│   ├── sensor_data.csv                     # Raw sensor telemetry (Task 1)
│   ├── engineered_features.csv             # Augmented feature dataset (Task 2)
│   ├── feature_engineering_report.txt      # Feature engineering documentation
│   ├── cross_validation_results.csv        # 5-fold cross-validation metrics
│   ├── final_model_performance.csv         # Held-out test performance table
│   ├── model_comparison_summary.csv        # RF vs XGBoost summary table
│   ├── random_forest_feature_importance.csv# RF feature importance rankings
│   ├── xgboost_feature_importance.csv      # XGBoost feature importance rankings
│   ├── anomaly_detection_summary.csv       # Anomaly & detection rate statistics
│   ├── task2_final_report.txt              # Comprehensive Task 2 technical report
│   ├── task3_test_report.txt              # End-to-end Task 3 verification report
│   └── plots/
│       ├── model_performance_comparison.png
│       ├── random_forest_confusion_matrix.png
│       ├── xgboost_confusion_matrix.png
│       ├── random_forest_feature_importance.png
│       └── xgboost_feature_importance.png
└── models/
    ├── model.joblib                        # Task 1 baseline model
    ├── metrics.json                        # Task 1 baseline evaluation metrics
    ├── random_forest_model.pkl             # Task 2 Random Forest classifier
    └── xgboost_model.pkl                   # Task 2 XGBoost classifier (Task 3 primary engine)
```

---

## Execution Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Task 1 — Baseline Ingestion & Streamlit App
```bash
python generate_data.py
python train_model.py
streamlit run app.py
```

### 3. Task 2 — Feature Engineering & Anomaly Detection Pipeline
```bash
# Run feature engineering
python feature_engineering.py

# Train Random Forest and XGBoost classifiers
python train_anomaly_models.py

# Or run the complete evaluation, cross-validation, and plotting suite:
python run_task2_pipeline.py
```

### 4. Task 3 — FastAPI Backend & Streamlit Dashboard

#### A. Start the FastAPI REST API Gateway
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI documentation: `http://127.0.0.1:8000/docs`
- Alternative ReDoc documentation: `http://127.0.0.1:8000/redoc`
- System health diagnostic check: `http://127.0.0.1:8000/health`

#### B. Launch the Streamlit Health Monitoring Dashboard
```bash
streamlit run dashboard.py
```
- Local dashboard interface: `http://localhost:8501`

#### C. Run Automated API & End-to-End Verification
```bash
# Run FastAPI unit and integration tests
python test_api.py

# Run complete multi-tier end-to-end verification
python verify_task3_e2e.py
```

---

## Manual GitHub Upload Guide (Browser Drag & Drop)

When uploading this project to GitHub using the **GitHub Web UI (Add file → Upload files)**:

1. **Directories to Upload**:
   - `.github/` (drag and drop the entire folder)
   - `data/` (drag and drop the entire folder, includes all CSVs, TXT reports, and `plots/`)
   - `models/` (drag and drop the entire folder, includes `.pkl` and `.joblib` model binaries)

2. **Root Files to Upload**:
   - `api.py`
   - `app.py`
   - `dashboard.py`
   - `feature_engineering.py`
   - `generate_data.py`
   - `README.md`
   - `requirements.txt`
   - `run_task2_pipeline.py`
   - `test_api.py`
   - `train_anomaly_models.py`
   - `train_model.py`
   - `verify_task3_e2e.py`
   - `.gitignore`

3. **Important — What to EXCLUDE from Upload**:
   - **DO NOT UPLOAD `py_embed/`**: This is a local Windows-embedded Python distribution and must never be committed to GitHub (it is already properly ignored in `.gitignore`).
   - **DO NOT UPLOAD `__pycache__/`** or any temporary `.log` / `.tmp` files.

---

## Suggested GitHub Repository Description

> End-to-end industrial predictive maintenance system for rotating equipment: physical feature engineering (Mechanical Power & Thermal Dissipation), anomaly detection with Random Forest & XGBoost, and real-time Streamlit telemetry monitoring.
