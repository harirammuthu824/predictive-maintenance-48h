"""
Predictive Maintenance Health Monitoring Dashboard (Task 3 Step 3)
Streamlit Frontend connected to FastAPI REST API backend

Connects to FastAPI at http://127.0.0.1:8000 (configurable)
Provides real-time machine telemetry input, prediction trigger, and health diagnostics.
"""

import math
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance Health Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean professional presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-anomaly {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR: API CONNECTION & SETTINGS
# ==============================================================================

st.sidebar.header("🔌 API Connection")

api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value="http://127.0.0.1:8000",
    help="URL of the running FastAPI predictive maintenance service",
)

model_choice = st.sidebar.selectbox(
    "Classifier Model",
    options=["xgboost", "random_forest"],
    index=0,
    format_func=lambda x: "XGBoost (Recommended - Best Model)" if x == "xgboost" else "Random Forest",
    help="Select which machine-learning classifier to invoke for predictions.",
)

# Function to test API Health
def check_api_health(url: str):
    try:
        res = requests.get(f"{url}/health", timeout=2.5)
        if res.status_code == 200:
            return True, res.json()
        return False, {"error": f"HTTP {res.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

api_is_online, health_data = check_api_health(api_base_url)

if api_is_online:
    st.sidebar.success("🟢 API Status: Connected")
    st.sidebar.caption(f"**Uptime:** {health_data.get('uptime_seconds', 0):.1f}s | **Version:** {health_data.get('api_version', '1.0.0')}")
    st.sidebar.caption(f"**Loaded Models:** {', '.join(health_data.get('available_models', []))}")
else:
    st.sidebar.error("🔴 API Status: Offline")
    st.sidebar.caption(
        "Make sure FastAPI is running locally:\n"
        "`uvicorn api:app --host 127.0.0.1 --port 8000`"
    )

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Quick Presets:**
    Click to load typical operating conditions:
    """
)

# Preset telemetry states for demonstration
if st.sidebar.button("Load Healthy Machine Preset"):
    st.session_state["vibration"] = 2.05
    st.session_state["temp"] = 62.5
    st.session_state["pressure"] = 4.75
    st.session_state["current"] = 8.00
    st.session_state["acoustic"] = 63.5
    st.session_state["load"] = 58.0
    st.session_state["rpm"] = 1450.0

if st.sidebar.button("Load Failing Machine Preset"):
    st.session_state["vibration"] = 5.65
    st.session_state["temp"] = 83.2
    st.session_state["pressure"] = 3.85
    st.session_state["current"] = 10.95
    st.session_state["acoustic"] = 86.5
    st.session_state["load"] = 72.0
    st.session_state["rpm"] = 1410.0


# ==============================================================================
# MAIN PAGE HEADER & API HEALTH SECTION
# ==============================================================================

st.markdown('<div class="main-title">Predictive Maintenance Health Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time condition monitoring, physical feature engineering & anomaly prediction via FastAPI</div>', unsafe_allow_html=True)

# SECTION: API Health Overview
st.subheader("1. System & API Health")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if api_is_online:
        st.metric("API Backend Status", "Healthy 🟢")
    else:
        st.metric("API Backend Status", "Offline 🔴")

with c2:
    if api_is_online and health_data.get("model_loaded"):
        st.metric("Active Model", model_choice.upper(), "Loaded 🟢")
    else:
        st.metric("Active Model", model_choice.upper(), "Unavailable 🔴")

with c3:
    if api_is_online:
        st.metric("Service Uptime", f"{health_data.get('uptime_seconds', 0):.0f} sec")
    else:
        st.metric("Service Uptime", "N/A")

with c4:
    if api_is_online:
        st.metric("Required Features", f"{len(health_data.get('features_required', []))} Features")
    else:
        st.metric("Required Features", "9 Features")

if not api_is_online:
    st.warning(
        """
        ⚠️ **FastAPI backend is currently offline or unreachable at the configured URL.**
        
        To start the API service, open a terminal in the project directory and run:
        ```bash
        uvicorn api:app --host 127.0.0.1 --port 8000
        ```
        Then refresh this page or verify the connection in the sidebar.
        """
    )

st.divider()

# ==============================================================================
# SECTION: MACHINE INPUT PARAMETERS
# ==============================================================================

st.subheader("2. Machine Input Parameters")
st.caption("Enter current sensor readings from rotating equipment telemetry:")

col_left, col_right = st.columns(2)

with col_left:
    machine_id = st.selectbox(
        "Machine Identifier",
        options=["M-01", "M-02", "M-03", "M-04", "M-05", "M-06", "M-07", "M-08"],
        index=0,
    )
    vibration_val = st.slider(
        "Vibration Velocity (mm/s)",
        min_value=0.5,
        max_value=10.0,
        value=float(st.session_state.get("vibration", 2.10)),
        step=0.05,
        help="Normal baseline is ~1.8 to 2.5 mm/s. Readings above 3.5 mm/s indicate mechanical wear or imbalance.",
    )
    temp_val = st.slider(
        "Operating Temperature (°C)",
        min_value=30.0,
        max_value=120.0,
        value=float(st.session_state.get("temp", 63.0)),
        step=0.5,
        help="Normal operating temperature is ~58°C to 68°C. Higher temperatures indicate cooling degradation or friction.",
    )
    pressure_val = st.slider(
        "System Pressure (bar)",
        min_value=1.0,
        max_value=10.0,
        value=float(st.session_state.get("pressure", 4.75)),
        step=0.05,
        help="Normal operating baseline is ~4.5 to 5.0 bar.",
    )
    current_val = st.slider(
        "Motor Electrical Current (A)",
        min_value=1.0,
        max_value=25.0,
        value=float(st.session_state.get("current", 8.00)),
        step=0.1,
        help="Normal operating baseline is ~7.5 to 8.5 A. Escalated current reflects excessive mechanical resistance.",
    )

with col_right:
    acoustic_val = st.slider(
        "Acoustic Noise (dB)",
        min_value=40.0,
        max_value=120.0,
        value=float(st.session_state.get("acoustic", 64.0)),
        step=0.5,
        help="Normal baseline is ~60 to 70 dB. Values above 75 dB indicate bearing degradation or cavitation.",
    )
    load_val = st.slider(
        "Mechanical Load Percentage (%)",
        min_value=20.0,
        max_value=100.0,
        value=float(st.session_state.get("load", 60.0)),
        step=0.5,
        help="Operational load percentage on the motor shaft.",
    )
    rpm_val = st.slider(
        "Rotational Speed (RPM)",
        min_value=800.0,
        max_value=2000.0,
        value=float(st.session_state.get("rpm", 1450.0)),
        step=5.0,
        help="Rotational shaft speed in RPM (synchronous design ~1450-1500 RPM).",
    )

    # Live physical feature calculation preview
    calc_power = (load_val / 100.0) * (rpm_val * 2.0 * math.pi / 60.0)
    calc_thermal = temp_val / max(current_val, 1e-4)

    st.markdown("##### 🔬 Engineered Physical Features (Calculated Live)")
    sub_c1, sub_c2 = st.columns(2)
    sub_c1.metric(
        "Mechanical Power (P_mech)",
        f"{calc_power:.2f}",
        help="Formulated in Task 2: P_mech = (load_pct/100) * (2*pi*rpm/60)",
    )
    sub_c2.metric(
        "Thermal Dissipation (R_th)",
        f"{calc_thermal:.2f} °C/A",
        help="Formulated in Task 2: R_th = temperature_c / motor_current_a",
    )

# Optional override toggle
with st.expander("Advanced: Manual Override for Engineered Features"):
    use_manual_override = st.checkbox("Manually override calculated features", value=False)
    if use_manual_override:
        manual_power = st.number_input("Mechanical Power Override", value=float(calc_power))
        manual_thermal = st.number_input("Thermal Dissipation Override", value=float(calc_thermal))
    else:
        manual_power = None
        manual_thermal = None

st.divider()

# ==============================================================================
# SECTION: PREDICTION RESULT
# ==============================================================================

st.subheader("3. Anomaly Prediction Result")

predict_clicked = st.button("🚀 Run Anomaly Prediction", type="primary", use_container_width=True)

if predict_clicked:
    if not api_is_online:
        st.error(
            "❌ **Prediction Failed**: Could not establish connection to the FastAPI service at "
            f"`{api_base_url}`. Please start the API server and try again."
        )
    else:
        payload = {
            "vibration_mm_s": vibration_val,
            "temperature_c": temp_val,
            "pressure_bar": pressure_val,
            "motor_current_a": current_val,
            "acoustic_db": acoustic_val,
            "load_pct": load_val,
            "rpm": rpm_val,
            "machine_id": machine_id,
        }
        if use_manual_override:
            payload["Mechanical_Power"] = manual_power
            payload["Thermal_Dissipation"] = manual_thermal

        with st.spinner("Calling FastAPI `/predict` endpoint..."):
            try:
                response = requests.post(
                    f"{api_base_url}/predict?model={model_choice}",
                    json=payload,
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()

                    # Result Display
                    pred_code = data["prediction"]
                    pred_label = data["prediction_label"]
                    prob = data["failure_probability"]
                    conf = data["confidence"]
                    risk_level = data["risk_level"]
                    recommendation = data["recommendation"]
                    model_used = data["model_used"]

                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                    with res_col1:
                        if pred_code == 0:
                            st.success(f"### Condition: Normal ✅")
                        else:
                            st.error(f"### Condition: FAILURE 🚨")

                    with res_col2:
                        st.metric("Failure Probability", f"{prob * 100:.2f}%")

                    with res_col3:
                        st.metric("Prediction Confidence", f"{conf * 100:.2f}%")

                    with res_col4:
                        if risk_level in ["High", "Critical"]:
                            st.metric("Operational Risk", risk_level, delta="-Degraded", delta_color="inverse")
                        else:
                            st.metric("Operational Risk", risk_level, delta="Normal", delta_color="normal")

                    # Maintenance Guidance Alert
                    if pred_code == 0:
                        st.info(f"**Maintenance Guidance:** {recommendation}")
                    else:
                        st.error(f"**URGENT Maintenance Alert:** {recommendation}")

                    with st.expander("View Full API Response Details & Telemetry Used"):
                        st.json(data)

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to communicate with prediction API: {str(e)}")

st.divider()

# Footer Information
st.caption(
    "Predictive Maintenance System — Task 3 REST API & Dashboard | "
    "Connected Model: XGBoost / Random Forest (Task 2) | "
    f"Backend Endpoint: `{api_base_url}/docs`"
)

