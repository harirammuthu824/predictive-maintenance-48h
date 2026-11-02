"""
FastAPI REST API Backend for Industrial Predictive Maintenance
Task 3 — REST API & Health Monitoring Backend

Provides high-throughput endpoints for:
1. Health and service diagnostics (GET /health)
2. Model metadata and performance info (GET /model/info)
3. Real-time single telemetry inference (POST /predict)
4. Batch telemetry inference (POST /predict/batch)
5. Root API welcome & navigation (GET /)
"""

import math
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Constants & Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

MODEL_FEATURES = [
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

# In-memory model cache
loaded_models: Dict[str, Any] = {}
app_start_time = time.time()


def compute_engineered_features(
    load_pct: float, rpm: float, temperature_c: float, motor_current_a: float
) -> Dict[str, float]:
    """
    Computes verified physical features formulated in Task 2:
    - Mechanical_Power: P_mech = (load_pct / 100) * (2 * pi * rpm / 60) = (load_pct * rpm * pi) / 3000
    - Thermal_Dissipation: R_th = temperature_c / motor_current_a
    """
    mech_power = (load_pct / 100.0) * (rpm * 2.0 * math.pi / 60.0)
    therm_diss = temperature_c / max(motor_current_a, 1e-6)
    return {
        "Mechanical_Power": round(float(mech_power), 6),
        "Thermal_Dissipation": round(float(therm_diss), 6),
    }


def load_model_from_disk(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models into memory
    print("Loading predictive maintenance models...")
    try:
        loaded_models["xgboost"] = load_model_from_disk(XGB_MODEL_PATH)
        print("Loaded XGBoost model successfully.")
    except Exception as e:
        print(f"Warning: Failed to load XGBoost model: {e}")

    try:
        loaded_models["random_forest"] = load_model_from_disk(RF_MODEL_PATH)
        print("Loaded Random Forest model successfully.")
    except Exception as e:
        print(f"Warning: Failed to load Random Forest model: {e}")

    yield

    # Shutdown
    loaded_models.clear()
    print("Cleared model cache on shutdown.")


# FastAPI Application instance
app = FastAPI(
    title="Predictive Maintenance REST API",
    description="""
### High-Throughput Condition Monitoring & Anomaly Detection API
Designed for industrial rotating equipment (motors, pumps, compressors).

**Features:**
* **Real-time Anomaly Prediction**: Real-time evaluation of sensor telemetry.
* **Domain Feature Engineering**: Automatically derives `Mechanical_Power` and `Thermal_Dissipation` if not provided.
* **Confidence & Risk Classification**: Provides calibrated failure probability, severity level, and engineering recommendations.
* **Interactive OpenAPI/Swagger**: Fully interactive documentation and schema definitions.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Configuration for local frontend, Streamlit, and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class TelemetryInput(BaseModel):
    vibration_mm_s: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Vibration velocity in mm/s (primary mechanical health indicator)",
        json_schema_extra={"example": 2.15},
    )
    temperature_c: float = Field(
        ...,
        ge=-20.0,
        le=200.0,
        description="Operating temperature in degrees Celsius (°C)",
        json_schema_extra={"example": 63.8},
    )
    pressure_bar: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        description="Operating fluid or internal gas pressure in bar",
        json_schema_extra={"example": 4.75},
    )
    motor_current_a: float = Field(
        ...,
        gt=0.0,
        le=200.0,
        description="Motor electrical current draw in Amperes (A)",
        json_schema_extra={"example": 8.05},
    )
    acoustic_db: float = Field(
        ...,
        ge=0.0,
        le=160.0,
        description="Acoustic emission noise in decibels (dB)",
        json_schema_extra={"example": 64.5},
    )
    load_pct: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Machine mechanical load percentage (0 - 100%+)",
        json_schema_extra={"example": 60.5},
    )
    rpm: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Shaft rotational speed in revolutions per minute (RPM)",
        json_schema_extra={"example": 1450.0},
    )
    Mechanical_Power: Optional[float] = Field(
        None,
        description="Optional: Precomputed mechanical power. If omitted, computed automatically from load_pct and rpm.",
        json_schema_extra={"example": 91.83},
    )
    Thermal_Dissipation: Optional[float] = Field(
        None,
        description="Optional: Precomputed thermal dissipation ratio (°C/A). If omitted, computed automatically from temperature_c and motor_current_a.",
        json_schema_extra={"example": 7.92},
    )
    machine_id: Optional[str] = Field(
        "M-01",
        description="Optional equipment identifier",
        json_schema_extra={"example": "M-01"},
    )
    timestamp: Optional[str] = Field(
        None,
        description="Optional observation timestamp (ISO format)",
        json_schema_extra={"example": "2026-11-02T09:30:00Z"},
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary anomaly indicator: 0 = Normal, 1 = Failure/Anomaly")
    prediction_label: str = Field(..., description="Human-readable condition status: 'Normal' or 'Failure/Anomaly Detected'")
    failure_probability: float = Field(..., description="Estimated probability of failure/anomaly event (0.0 to 1.0)")
    confidence: float = Field(..., description="Confidence score for the predicted class (0.0 to 1.0)")
    risk_level: str = Field(..., description="Operational risk category: 'Low', 'Medium', 'High', or 'Critical'")
    recommendation: str = Field(..., description="Actionable maintenance guidance based on risk level")
    model_used: str = Field(..., description="Machine learning algorithm executing inference")
    features_used: Dict[str, float] = Field(..., description="Feature values fed to the model")
    machine_id: Optional[str] = Field(None, description="Equipment identifier")
    timestamp: Optional[str] = Field(None, description="Observation timestamp")


class BatchTelemetryInput(BaseModel):
    readings: List[TelemetryInput] = Field(
        ...,
        description="List of telemetry readings to evaluate in a single high-throughput batch",
    )


class BatchPredictionResponse(BaseModel):
    total_samples: int = Field(..., description="Total readings processed in this batch")
    anomalies_detected: int = Field(..., description="Total samples flagged as impending failure / anomaly")
    normal_samples: int = Field(..., description="Total samples operating normally")
    anomaly_rate: float = Field(..., description="Proportion of batch flagged as anomalous")
    predictions: List[PredictionResponse] = Field(..., description="Detailed prediction for each reading")


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    default_model: str
    available_models: List[str]
    features_required: List[str]
    api_version: str


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def assess_risk(failure_prob: float) -> tuple[str, str]:
    """Categorizes risk level and maintenance recommendations based on probability."""
    if failure_prob < 0.20:
        return (
            "Low",
            "Machinery operating within healthy baseline parameters. Continue standard routine monitoring.",
        )
    elif failure_prob < 0.50:
        return (
            "Medium",
            "Minor elevated thermal/vibrational trends detected. Schedule non-disruptive inspection within next 24-48 hours.",
        )
    elif failure_prob < 0.80:
        return (
            "High",
            "Severe telemetry degradation indicating imminent mechanical failure. Dispatch maintenance crew for immediate diagnosis.",
        )
    else:
        return (
            "Critical",
            "Critical failure event underway or immediate shutdown threshold exceeded. Halt rotating equipment safely to prevent catastrophic damage.",
        )


def run_single_inference(
    sample: TelemetryInput, model_key: str = "xgboost"
) -> PredictionResponse:
    # Resolve model
    if model_key not in loaded_models:
        # Lazy reload attempt
        target_path = XGB_MODEL_PATH if model_key == "xgboost" else RF_MODEL_PATH
        if target_path.exists():
            loaded_models[model_key] = load_model_from_disk(target_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Requested model '{model_key}' is not available.",
            )

    model = loaded_models[model_key]

    # Automatic feature engineering if not provided
    mech_power = sample.Mechanical_Power
    therm_diss = sample.Thermal_Dissipation

    if mech_power is None or therm_diss is None:
        computed = compute_engineered_features(
            sample.load_pct, sample.rpm, sample.temperature_c, sample.motor_current_a
        )
        if mech_power is None:
            mech_power = computed["Mechanical_Power"]
        if therm_diss is None:
            therm_diss = computed["Thermal_Dissipation"]

    features_dict = {
        "vibration_mm_s": sample.vibration_mm_s,
        "temperature_c": sample.temperature_c,
        "pressure_bar": sample.pressure_bar,
        "motor_current_a": sample.motor_current_a,
        "acoustic_db": sample.acoustic_db,
        "load_pct": sample.load_pct,
        "rpm": sample.rpm,
        "Mechanical_Power": mech_power,
        "Thermal_Dissipation": therm_diss,
    }

    # Format into DataFrame with exact feature ordering
    df_features = pd.DataFrame([features_dict], columns=MODEL_FEATURES)

    try:
        raw_pred = int(model.predict(df_features)[0])
        proba_arr = model.predict_proba(df_features)[0]
        failure_prob = float(proba_arr[1])
        confidence = float(proba_arr[raw_pred])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}",
        )

    risk_level, recommendation = assess_risk(failure_prob)
    pred_label = "Failure/Anomaly Detected" if raw_pred == 1 else "Normal"

    return PredictionResponse(
        prediction=raw_pred,
        prediction_label=pred_label,
        failure_probability=round(failure_prob, 4),
        confidence=round(confidence, 4),
        risk_level=risk_level,
        recommendation=recommendation,
        model_used="XGBoost Classifier (Task 2 Best Model)" if model_key == "xgboost" else "Random Forest Classifier",
        features_used={k: round(v, 4) for k, v in features_dict.items()},
        machine_id=sample.machine_id,
        timestamp=sample.timestamp,
    )


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get(
    "/",
    summary="API Root Information",
    description="Returns service metadata, operational status, version, and documentation endpoints.",
    tags=["General"],
)
def root():
    return {
        "api_name": "Predictive Maintenance REST API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "default_model": "xgboost (Best Performing Task 2 Model, F1: 0.6250, Recall: 83.33%, ROC-AUC: 0.9967)",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Model Diagnostic Check",
    description="Reports API health, uptime, and confirms whether the underlying machine-learning models are loaded.",
    tags=["Diagnostics"],
)
def health_check():
    # Ensure models are loaded even if lifespan context was bypassed
    if not loaded_models:
        for key, pth in [("xgboost", XGB_MODEL_PATH), ("random_forest", RF_MODEL_PATH)]:
            if pth.exists() and key not in loaded_models:
                try:
                    loaded_models[key] = load_model_from_disk(pth)
                except Exception:
                    pass

    uptime = time.time() - app_start_time
    is_loaded = "xgboost" in loaded_models or "random_forest" in loaded_models
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        uptime_seconds=round(uptime, 2),
        model_loaded=is_loaded,
        default_model="xgboost",
        available_models=list(loaded_models.keys()),
        features_required=MODEL_FEATURES,
        api_version="1.0.0",
    )


@app.get(
    "/model/info",
    summary="Model Architecture & Metrics Metadata",
    description="Returns detailed performance metrics, cross-validation scores, and feature importance rankings from Task 2.",
    tags=["Diagnostics"],
)
def model_info():
    return {
        "selected_model": "XGBoost Classifier",
        "model_file": str(XGB_MODEL_PATH),
        "task2_test_metrics": {
            "accuracy": 0.9958,
            "precision": 0.5000,
            "recall": 0.8333,
            "f1_score": 0.6250,
            "roc_auc": 0.9967,
        },
        "task2_5fold_cv_metrics": {
            "mean_f1": 0.4372,
            "mean_roc_auc": 0.9625,
            "mean_accuracy": 0.9942,
        },
        "feature_count": len(MODEL_FEATURES),
        "required_features": MODEL_FEATURES,
        "alternative_model": "Random Forest Classifier (models/random_forest_model.pkl)",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-Time Single Telemetry Prediction",
    description="""
Evaluates a single telemetry reading for industrial rotating equipment and determines failure/anomaly risk.

* **Automatic Feature Engineering**: If `Mechanical_Power` or `Thermal_Dissipation` are omitted,
  they are automatically computed using physical kinematics and thermodynamics formulas.
* **Model Selection**: Defaults to the best-performing `xgboost` model, with optional fallback to `random_forest`.
    """,
    tags=["Inference"],
)
def predict(
    payload: TelemetryInput,
    model: Literal["xgboost", "random_forest"] = Query(
        "xgboost",
        description="Classifier algorithm to invoke: 'xgboost' (recommended) or 'random_forest'",
    ),
):
    return run_single_inference(payload, model_key=model)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="High-Throughput Batch Telemetry Prediction",
    description="Processes an array of sensor telemetry readings in a single vectorized HTTP request.",
    tags=["Inference"],
)
def predict_batch(
    payload: BatchTelemetryInput,
    model: Literal["xgboost", "random_forest"] = Query(
        "xgboost",
        description="Classifier algorithm to invoke: 'xgboost' (recommended) or 'random_forest'",
    ),
):
    if not payload.readings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least one telemetry reading.",
        )

    results: List[PredictionResponse] = []
    anomalies_count = 0

    for reading in payload.readings:
        res = run_single_inference(reading, model_key=model)
        results.append(res)
        if res.prediction == 1:
            anomalies_count += 1

    total = len(results)
    return BatchPredictionResponse(
        total_samples=total,
        anomalies_detected=anomalies_count,
        normal_samples=total - anomalies_count,
        anomaly_rate=round(anomalies_count / max(total, 1), 4),
        predictions=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)

