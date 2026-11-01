"""
Task 2 - Step 3: Classification Model Training
Predictive Maintenance System for Industrial Rotating Equipment

Models trained:
1. Random Forest Classifier
2. XGBoost Classifier

Uses: data/engineered_features.csv
Saves:
- models/random_forest_model.pkl
- models/xgboost_model.pkl
- data/model_performance_initial.csv
- data/model_training_report.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path("data/engineered_features.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
PERF_CSV_PATH = Path("data/model_performance_initial.csv")
REPORT_PATH = Path("data/model_training_report.txt")

FEATURES = [
    "vibration_mm_s",
    "temperature_c",
    "pressure_bar",
    "motor_current_a",
    "acoustic_db",
    "load_pct",
    "rpm",
    "Mechanical_Power",
    "Thermal_Dissipation",
]
TARGET_COL = "failure"

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Verify target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    print(f"Features: {FEATURES}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Handle missing values if any remain
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

    # Train / Test split with stratification and reproducible random_state
    RANDOM_STATE = 42
    TEST_SIZE = 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    n_train_total = len(y_train)
    n_test_total = len(y_test)
    n_train_pos = int(y_train.sum())
    n_test_pos = int(y_test.sum())

    print(f"Train set: {n_train_total} samples ({n_train_pos} failures, {n_train_pos/n_train_total:.3%})")
    print(f"Test set:  {n_test_total} samples ({n_test_pos} failures, {n_test_pos/n_test_total:.3%})")

    # 1. Train Random Forest Classifier
    print("\n--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_cm = confusion_matrix(y_test, rf_pred)

    print(f"Random Forest Accuracy:  {rf_acc:.4f}")
    print(f"Random Forest Precision: {rf_prec:.4f}")
    print(f"Random Forest Recall:    {rf_rec:.4f}")
    print(f"Random Forest F1-score:  {rf_f1:.4f}")
    print(f"Random Forest ROC-AUC:   {rf_auc:.4f}")

    # 2. Train XGBoost Classifier
    print("\n--- Training XGBoost Classifier ---")
    scale_pos = (n_train_total - n_train_pos) / max(n_train_pos, 1)

    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    print(f"XGBoost Accuracy:  {xgb_acc:.4f}")
    print(f"XGBoost Precision: {xgb_prec:.4f}")
    print(f"XGBoost Recall:    {xgb_rec:.4f}")
    print(f"XGBoost F1-score:  {xgb_f1:.4f}")
    print(f"XGBoost ROC-AUC:   {xgb_auc:.4f}")

    # Save models as .pkl files
    print("\nSaving trained models...")
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Saved Random Forest model: {RF_MODEL_PATH}")

    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"Saved XGBoost model: {XGB_MODEL_PATH}")

    # Save initial model performance comparison CSV
    perf_df = pd.DataFrame([
        {
            "Model": "Random Forest Classifier",
            "Accuracy": round(float(rf_acc), 4),
            "Precision": round(float(rf_prec), 4),
            "Recall": round(float(rf_rec), 4),
            "F1_Score": round(float(rf_f1), 4),
            "ROC_AUC": round(float(rf_auc), 4),
        },
        {
            "Model": "XGBoost Classifier",
            "Accuracy": round(float(xgb_acc), 4),
            "Precision": round(float(xgb_prec), 4),
            "Recall": round(float(xgb_rec), 4),
            "F1_Score": round(float(xgb_f1), 4),
            "ROC_AUC": round(float(xgb_auc), 4),
        },
    ])
    perf_df.to_csv(PERF_CSV_PATH, index=False)
    print(f"Saved performance comparison table: {PERF_CSV_PATH}")

    # Create model training report
    report_text = f"""================================================================================
MODEL TRAINING REPORT (TASK 2 - STEP 3)
================================================================================
Project: Predictive Maintenance System for Industrial Rotating Equipment
Task: Task 2 — Feature Engineering & Classification Modeling (Step 3: Initial Training)
Data Source: {DATA_PATH}
Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. TARGET COLUMN USED:
--------------------------------------------------------------------------------
- Target Name: '{TARGET_COL}'
- Semantic Meaning: Existing binary anomaly / machine failure indicator.
    * 0 = Normal machinery operating state
    * 1 = Failure event
- Class Breakdown (Total {len(df)} samples):
    * Class 0 (Normal):  {(y == 0).sum():,} rows ({(y == 0).mean():.2%})
    * Class 1 (Failure): {(y == 1).sum():,} rows ({(y == 1).mean():.2%})
- Target Authenticity: 100% existing ground-truth target; no artificial labels created.

2. FEATURES USED (9 NUMERICAL FEATURES):
--------------------------------------------------------------------------------
 1. vibration_mm_s       (Raw sensor: vibration velocity)
 2. temperature_c        (Raw sensor: operating temperature in °C)
 3. pressure_bar         (Raw sensor: operating fluid/gas pressure in bar)
 4. motor_current_a      (Raw sensor: motor electrical current in Amperes)
 5. acoustic_db          (Raw sensor: acoustic noise in decibels)
 6. load_pct             (Raw sensor: mechanical operating load %)
 7. rpm                  (Raw sensor: shaft rotational speed in RPM)
 8. Mechanical_Power     (Engineered: P_mech = (load_pct/100) * (2*pi*rpm/60))
 9. Thermal_Dissipation  (Engineered: R_th = temperature_c / motor_current_a)

Excluded metadata / non-predictive columns:
- 'timestamp': Chronological observation index
- 'machine_id': Equipment grouping identifier
- 'failure_mode': Multi-class diagnostic label available only post-failure

3. TRAIN / TEST SPLIT DETAILS:
--------------------------------------------------------------------------------
- Split Strategy: Stratified train-test split
- Train / Test Ratio: 80% Train ({n_train_total} samples) / 20% Test ({n_test_total} samples)
- Reproducible Random State: {RANDOM_STATE}
- Stratification: Enabled on target '{TARGET_COL}' to guarantee proportional representation
  of the rare minority failure events across partitions.
    * Train failures: {n_train_pos} ({(n_train_pos/n_train_total):.3%})
    * Test failures:  {n_test_pos} ({(n_test_pos/n_test_total):.3%})

4. RANDOM FOREST CLASSIFIER RESULTS:
--------------------------------------------------------------------------------
- Configuration:
    * Estimators: 250
    * Max Depth: None (unconstrained)
    * Min Samples Leaf: 1
    * Class Weighting: 'balanced'
    * Random State: {RANDOM_STATE}
- Metrics:
    * Accuracy:   {rf_acc:.4f}
    * Precision:  {rf_prec:.4f}
    * Recall:     {rf_rec:.4f}
    * F1-score:   {rf_f1:.4f}
    * ROC-AUC:    {rf_auc:.4f}
- Confusion Matrix:
    [TN={rf_cm[0, 0]}, FP={rf_cm[0, 1]}]
    [FN={rf_cm[1, 0]}, TP={rf_cm[1, 1]}]
- Classification Report:
{classification_report(y_test, rf_pred, target_names=['Normal (0)', 'Failure (1)'], digits=4, zero_division=0)}

5. XGBOOST CLASSIFIER RESULTS:
--------------------------------------------------------------------------------
- Configuration:
    * Estimators: 150
    * Max Depth: 6
    * Learning Rate: 0.08
    * Scale Pos Weight: {scale_pos:.2f} (handling class imbalance)
    * Eval Metric: 'logloss'
    * Random State: {RANDOM_STATE}
- Metrics:
    * Accuracy:   {xgb_acc:.4f}
    * Precision:  {xgb_prec:.4f}
    * Recall:     {xgb_rec:.4f}
    * F1-score:   {xgb_f1:.4f}
    * ROC-AUC:    {xgb_auc:.4f}
- Confusion Matrix:
    [TN={xgb_cm[0, 0]}, FP={xgb_cm[0, 1]}]
    [FN={xgb_cm[1, 0]}, TP={xgb_cm[1, 1]}]
- Classification Report:
{classification_report(y_test, xgb_pred, target_names=['Normal (0)', 'Failure (1)'], digits=4, zero_division=0)}

6. INITIAL MODEL COMPARISON SUMMARY:
--------------------------------------------------------------------------------
{perf_df.to_string(index=False)}

7. MODEL FILE LOCATIONS:
--------------------------------------------------------------------------------
- Random Forest Model: models/random_forest_model.pkl
- XGBoost Model:       models/xgboost_model.pkl
- Comparison Table:    data/model_performance_initial.csv
- Training Report:     data/model_training_report.txt

Task 1 files status: Unmodified and preserved.
================================================================================
"""
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Saved model training report: {REPORT_PATH}")

if __name__ == "__main__":
    main()

