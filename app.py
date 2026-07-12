import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------------------------------
# Predictive Maintenance of Industrial Machinery — Streamlit Frontend
# Loads a locally trained model (model.pkl) — no IBM Cloud deployment needed.
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

# ----------------------------------------------------------------------------
# Styling — industrial control-panel look: dark steel background, amber
# accent (like a machine status light), monospace readouts for numbers.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #12161c 0%, #0d1014 100%);
    }
    .header-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 24px;
        margin-bottom: 6px;
        background: #171c23;
        border: 1px solid #262d38;
        border-left: 5px solid #f2a900;
        border-radius: 6px;
    }
    .header-wrap h1 {
        font-size: 26px;
        margin: 0;
        color: #f5f6f7;
        letter-spacing: 0.3px;
    }
    .header-wrap p {
        margin: 2px 0 0 0;
        color: #9aa4b2;
        font-size: 14px;
    }
    .badge {
        display: inline-block;
        background: #1f2732;
        color: #f2a900;
        border: 1px solid #3a4453;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 12px;
        font-family: monospace;
        letter-spacing: 0.5px;
    }
    .panel-label {
        color: #7d8a9c;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin: 18px 0 10px 0;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        font-family: "Courier New", monospace;
        color: #f2a900 !important;
    }
    div[data-testid="column"] {
        background: #171c23;
        border: 1px solid #232b36;
        border-radius: 6px;
        padding: 10px 14px 4px 14px;
    }
    .stButton>button {
        background: #f2a900;
        color: #12161c;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 6px;
        height: 3em;
    }
    .stButton>button:hover {
        background: #ffc22e;
        color: #12161c;
    }
    .result-card {
        padding: 18px 22px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 20px;
        font-weight: 700;
    }
    .result-ok {
        background: #10261a;
        border: 1px solid #1f6b3f;
        color: #4ade80;
    }
    .result-warn {
        background: #2a1a12;
        border: 1px solid #8a4a1c;
        color: #f2a900;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-wrap">
    <div style="font-size:34px;">🛠️</div>
    <div>
        <h1>Predictive Maintenance of Industrial Machinery</h1>
        <p>Live sensor console · Random Forest + SMOTE · <span class="badge">MODEL: LOCAL</span></p>
    </div>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "model.pkl"

# ----------------------------------------------------------------------------
# Load the trained model once (cached across reruns)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        f"⚠️ Could not find `{MODEL_PATH}` in this folder. "
        f"Run `train_model.py` first to create it, then place the resulting "
        f"`model.pkl` next to `app.py`."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sensor readings — single horizontal row of compact panels
# ----------------------------------------------------------------------------
st.markdown('<div class="panel-label">Sensor Readings</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
with c1:
    udi = st.number_input("UDI", min_value=1, value=1, step=1)
with c2:
    product_id = st.text_input("Product ID", value="M14860")
with c3:
    machine_type = st.selectbox("Type", ["L", "M", "H"])
with c4:
    air_temp = st.number_input("Air temp [K]", value=298.1, format="%.1f")
with c5:
    process_temp = st.number_input("Process temp [K]", value=308.6, format="%.1f")
with c6:
    rot_speed = st.number_input("Rot. speed [rpm]", value=1551, step=1)
with c7:
    torque = st.number_input("Torque [Nm]", value=42.8, format="%.1f")
with c8:
    tool_wear = st.number_input("Tool wear [min]", value=0, step=1)

target = st.select_slider("Target (0 = No Failure expected, 1 = Failure expected)", options=[0, 1])

st.write("")
predict_clicked = st.button("🔍  RUN PREDICTION", use_container_width=True)

# ----------------------------------------------------------------------------
# Predict + results
# ----------------------------------------------------------------------------
if predict_clicked:
    try:
        input_df = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Target": target,
        }])

        prediction = model.predict(input_df)[0]
        card_class = "result-ok" if prediction == "No Failure" else "result-warn"
        icon = "✅" if prediction == "No Failure" else "⚠️"
        st.markdown(
            f'<div class="result-card {card_class}">{icon}&nbsp;&nbsp;Predicted Failure Type: {prediction}</div>',
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            proba_df = pd.DataFrame({"Failure Type": classes, "Probability": proba}) \
                         .sort_values("Probability", ascending=True) \
                         .set_index("Failure Type")

            st.markdown('<div class="panel-label">Probability Distribution</div>', unsafe_allow_html=True)
            # Native horizontal bar chart — no extra dependency needed.
            st.bar_chart(proba_df, horizontal=True, color="#f2a900")

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

st.divider()
st.caption("Model: Random Forest Classifier + SMOTE · Trained locally / via IBM Watson Studio Lite · "
           "Dataset: Kaggle Machine Predictive Maintenance Classification")