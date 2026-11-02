"""
Automated Test Suite for Predictive Maintenance FastAPI Backend (Task 3)
Verifies:
1. Root endpoint (GET /)
2. Health & Model check (GET /health)
3. Model metadata endpoint (GET /model/info)
4. Real-time prediction (POST /predict) - Normal telemetry
5. Real-time prediction (POST /predict) - Degraded anomaly telemetry
6. Real-time prediction with alternate model (Random Forest)
7. Batch inference (POST /predict/batch)
8. Input validation & error handling (HTTP 422)
9. Swagger & OpenAPI schema documentation (GET /docs, GET /openapi.json)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from api import app


def test_root_endpoint(client: TestClient):
    print("Testing GET / ...", flush=True)
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "documentation" in data
    print(f"  [PASS] Root endpoint returned: {data['api_name']} (version {data['version']})", flush=True)


def test_health_endpoint(client: TestClient):
    print("\nTesting GET /health ...", flush=True)
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "xgboost" in data["available_models"]
    assert len(data["features_required"]) == 9
    print(
        f"  [PASS] Health check passed: status={data['status']}, uptime={data['uptime_seconds']}s, models={data['available_models']}",
        flush=True,
    )


def test_model_info_endpoint(client: TestClient):
    print("\nTesting GET /model/info ...", flush=True)
    response = client.get("/model/info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["selected_model"] == "XGBoost Classifier"
    assert data["task2_test_metrics"]["accuracy"] == 0.9958
    assert data["task2_test_metrics"]["recall"] == 0.8333
    assert data["task2_test_metrics"]["roc_auc"] == 0.9967
    print(
        f"  [PASS] Model info verified: {data['selected_model']}, F1={data['task2_test_metrics']['f1_score']}",
        flush=True,
    )


def test_predict_normal_telemetry(client: TestClient):
    print("\nTesting POST /predict (Normal Telemetry, auto feature calculation) ...", flush=True)
    payload = {
        "vibration_mm_s": 2.05,
        "temperature_c": 62.5,
        "pressure_bar": 4.75,
        "motor_current_a": 7.9,
        "acoustic_db": 63.0,
        "load_pct": 58.0,
        "rpm": 1450.0,
        "machine_id": "M-01",
        "timestamp": "2026-11-02T09:30:00Z",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["prediction"] == 0
    assert data["prediction_label"] == "Normal"
    assert data["risk_level"] == "Low"
    assert data["failure_probability"] < 0.10
    assert "Mechanical_Power" in data["features_used"]
    assert "Thermal_Dissipation" in data["features_used"]
    print(
        f"  [PASS] Normal prediction verified: label={data['prediction_label']}, prob={data['failure_probability']}, risk={data['risk_level']}",
        flush=True,
    )


def test_predict_degraded_telemetry(client: TestClient):
    print("\nTesting POST /predict (Degraded Anomaly Telemetry) ...", flush=True)
    payload = {
        "vibration_mm_s": 5.5,
        "temperature_c": 82.0,
        "pressure_bar": 3.9,
        "motor_current_a": 10.8,
        "acoustic_db": 86.0,
        "load_pct": 70.0,
        "rpm": 1420.0,
        "machine_id": "M-02",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["prediction"] == 1
    assert data["prediction_label"] == "Failure/Anomaly Detected"
    assert data["risk_level"] in ["High", "Critical"]
    assert data["failure_probability"] > 0.80
    print(
        f"  [PASS] Anomaly prediction verified: label={data['prediction_label']}, prob={data['failure_probability']}, risk={data['risk_level']}",
        flush=True,
    )


def test_predict_random_forest_option(client: TestClient):
    print("\nTesting POST /predict with ?model=random_forest ...", flush=True)
    payload = {
        "vibration_mm_s": 2.10,
        "temperature_c": 63.0,
        "pressure_bar": 4.80,
        "motor_current_a": 8.0,
        "acoustic_db": 64.0,
        "load_pct": 60.0,
        "rpm": 1450.0,
    }
    response = client.post("/predict?model=random_forest", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "Random Forest" in data["model_used"]
    assert data["prediction"] == 0
    print(f"  [PASS] Random Forest fallback verified: model={data['model_used']}", flush=True)


def test_predict_batch(client: TestClient):
    print("\nTesting POST /predict/batch ...", flush=True)
    payload = {
        "readings": [
            {
                "vibration_mm_s": 2.05,
                "temperature_c": 62.5,
                "pressure_bar": 4.75,
                "motor_current_a": 7.9,
                "acoustic_db": 63.0,
                "load_pct": 58.0,
                "rpm": 1450.0,
                "machine_id": "M-01",
            },
            {
                "vibration_mm_s": 5.6,
                "temperature_c": 83.5,
                "pressure_bar": 3.8,
                "motor_current_a": 11.0,
                "acoustic_db": 87.0,
                "load_pct": 72.0,
                "rpm": 1410.0,
                "machine_id": "M-02",
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["total_samples"] == 2
    assert data["anomalies_detected"] == 1
    assert data["normal_samples"] == 1
    assert data["anomaly_rate"] == 0.5
    assert len(data["predictions"]) == 2
    print(
        f"  [PASS] Batch prediction verified: total={data['total_samples']}, anomalies={data['anomalies_detected']}",
        flush=True,
    )


def test_input_validation(client: TestClient):
    print("\nTesting Input Validation Error Handling (HTTP 422) ...", flush=True)
    # Invalid: negative vibration
    bad_payload = {
        "vibration_mm_s": -5.0,
        "temperature_c": 60.0,
        "pressure_bar": 4.5,
        "motor_current_a": 8.0,
        "acoustic_db": 60.0,
        "load_pct": 50.0,
        "rpm": 1450.0,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("  [PASS] Validation caught negative vibration properly with status 422.", flush=True)


def test_swagger_and_openapi_docs(client: TestClient):
    print("\nTesting OpenAPI & Swagger Schema (/openapi.json & /docs) ...", flush=True)
    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200, f"Expected 200, got {r_openapi.status_code}"
    schema = r_openapi.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/model/info" in schema["paths"]
    assert schema["info"]["title"] == "Predictive Maintenance REST API"

    r_docs = client.get("/docs")
    assert r_docs.status_code == 200, f"Expected 200, got {r_docs.status_code}"
    print("  [PASS] OpenAPI schema and Swagger documentation UI confirmed working.", flush=True)


def main():
    print("================================================================================", flush=True)
    print("STARTING FASTAPI BACKEND TEST SUITE", flush=True)
    print("================================================================================", flush=True)
    with TestClient(app) as client:
        test_root_endpoint(client)
        test_health_endpoint(client)
        test_model_info_endpoint(client)
        test_predict_normal_telemetry(client)
        test_predict_degraded_telemetry(client)
        test_predict_random_forest_option(client)
        test_predict_batch(client)
        test_input_validation(client)
        test_swagger_and_openapi_docs(client)
    print("\n================================================================================", flush=True)
    print("ALL API TESTS PASSED SUCCESSFULLY! (100% SUCCESS)", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
