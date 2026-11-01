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

## Project Structure

```text
predictive-maintenance-system/
├── app.py                                  # Streamlit monitoring dashboard (Task 1)
├── train_model.py                          # Baseline ML pipeline & 48h predictor (Task 1)
├── generate_data.py                        # Synthetic telemetry generator (Task 1)
├── feature_engineering.py                  # Mechanical & Thermal feature pipeline (Task 2)
├── train_anomaly_models.py                 # RF & XGBoost classification trainer (Task 2)
├── run_task2_pipeline.py                   # Master end-to-end evaluation & plotting script (Task 2)
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation & task reports
├── .gitignore                              # Git exclusion rules
├── .github/
│   └── workflows/
│       └── python-check.yml                # CI syntax and execution workflow
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
│   ├── task2_final_report.txt              # Comprehensive 16-section technical report
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
    └── xgboost_model.pkl                   # Task 2 XGBoost classifier
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

# Or run the complete evaluation, cross-validation, and plotting suite in one step:
python run_task2_pipeline.py
```

---

## Suggested GitHub Repository Description

> End-to-end industrial predictive maintenance system for rotating equipment: physical feature engineering (Mechanical Power & Thermal Dissipation), anomaly detection with Random Forest & XGBoost, and real-time Streamlit telemetry monitoring.
