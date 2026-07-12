# import streamlit as st
# import requests
# import os

# # ----------------------------------------------------------------------------
# # Predictive Maintenance of Industrial Machinery — Streamlit Frontend
# # Calls the deployed IBM Watson Machine Learning scoring endpoint.
# # ----------------------------------------------------------------------------

# st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="centered")

# st.title("🛠️ Predictive Maintenance of Industrial Machinery")
# st.caption("Enter live sensor readings below to predict the type of machine failure. "
#            "Powered by a Random Forest model deployed on IBM Watson Machine Learning.")

# # ----------------------------------------------------------------------------
# # Sidebar — IBM Cloud credentials (kept out of source code, entered at runtime)
# # ----------------------------------------------------------------------------
# st.sidebar.header("🔑 IBM Cloud Credentials")
# api_key = st.sidebar.text_input("IBM_Cloud_key", type="password", value=os.getenv("IBM_CLOUD_API_KEY", "_dyE9b4VPZDHylbaqFx-hrauuttSA7phAfba6KWkJ58p"))
# deployment_url = st.sidebar.text_input(
#     "WML Scoring Endpoint URL",
#     value="https://us-south.ml.cloud.ibm.com/ml/v4/deployments/019f460bff52-77bd-bcec-5b8fe9744f4e/predictions?version=2021-05-01"
# )

# st.sidebar.info(
#     "Your API key is never stored — it's only used in-memory for this session "
#     "to fetch a temporary IAM token."
# )

# # ----------------------------------------------------------------------------
# # Main form — sensor inputs
# # ----------------------------------------------------------------------------
# st.subheader("Sensor Readings")

# col1, col2 = st.columns(2)
# with col1:
#     udi = st.number_input("UDI", min_value=1, value=1, step=1)
#     product_id = st.text_input("Product ID", value="M14860")
#     machine_type = st.selectbox("Type", ["L", "M", "H"])
#     air_temp = st.number_input("Air temperature [K]", value=298.1, format="%.1f")
#     process_temp = st.number_input("Process temperature [K]", value=308.6, format="%.1f")

# with col2:
#     rot_speed = st.number_input("Rotational speed [rpm]", value=1551, step=1)
#     torque = st.number_input("Torque [Nm]", value=42.8, format="%.1f")
#     tool_wear = st.number_input("Tool wear [min]", value=0, step=1)
#     target = st.selectbox("Target (0 = No Failure expected, 1 = Failure expected)", [0, 1])

# st.divider()

# # ----------------------------------------------------------------------------
# # Predict button
# # ----------------------------------------------------------------------------
# if st.button("🔍 Predict Failure Type", use_container_width=True):
#     if not api_key:
#         st.error("Please enter your IBM Cloud API key in the sidebar first.")
#     else:
#         with st.spinner("Getting IAM token and calling the deployed model..."):
#             try:
#                 # Step 1: exchange API key for a short-lived IAM access token
#                 token_response = requests.post(
#                     "https://iam.cloud.ibm.com/identity/token",
#                     data={
#                         "apikey": api_key,
#                         "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
#                     },
#                     headers={"Content-Type": "application/x-www-form-urlencoded"}
#                 )
#                 token_response.raise_for_status()
#                 iam_token = token_response.json()["access_token"]

#                 # Step 2: build the scoring payload matching the model's expected fields
#                 payload = {
#                     "input_data": [
#                         {
#                             "fields": [
#                                 "UDI", "Product ID", "Type",
#                                 "Air temperature [K]", "Process temperature [K]",
#                                 "Rotational speed [rpm]", "Torque [Nm]",
#                                 "Tool wear [min]", "Target"
#                             ],
#                             "values": [[
#                                 udi, product_id, machine_type,
#                                 air_temp, process_temp,
#                                 rot_speed, torque,
#                                 tool_wear, target
#                             ]]
#                         }
#                     ]
#                 }

#                 headers = {
#                     "Authorization": f"Bearer {iam_token}",
#                     "Content-Type": "application/json"
#                 }

#                 # Step 3: call the deployed WML endpoint
#                 response = requests.post(deployment_url, json=payload, headers=headers)
#                 response.raise_for_status()
#                 result = response.json()

#                 prediction = result["predictions"][0]["values"][0][0]
#                 probability = result["predictions"][0]["values"][0][1]

#                 st.success(f"### Predicted Failure Type: **{prediction}**")
#                 with st.expander("See raw probability distribution"):
#                     st.write(probability)

#             except requests.exceptions.HTTPError as e:
#                 st.error(f"Request failed: {e}\n\nCheck your API key and endpoint URL.")
#             except Exception as e:
#                 st.error(f"Something went wrong: {e}")

# st.divider()
# st.caption("Model: Random Forest Classifier + SMOTE · Trained & deployed on IBM Watson Machine Learning · "
#            "Dataset: Kaggle Machine Predictive Maintenance Classification")






import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
 
DATA_PATH = "predictive_maintenance.csv"
MODEL_OUT = "model.pkl"
 
# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
 
# Feature columns used by the app (kept in sync with app.py's input_df)
feature_cols = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Target",
]
label_col = "Failure Type"  # the multi-class target we're predicting
 
X = df[feature_cols]
y = df[label_col]
 
# ----------------------------------------------------------------------------
# 2. Train / test split
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# ----------------------------------------------------------------------------
# 3. Preprocessing + model pipeline
#    - One-hot encode the categorical "Type" column
#    - SMOTE to balance rare failure classes
#    - Random Forest classifier
# ----------------------------------------------------------------------------
categorical_cols = ["Type"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]
 
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)
 
pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight=None
    )),
])
 
# ----------------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------------
pipeline.fit(X_train, y_train)
 
# ----------------------------------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
print("Classification report on held-out test set:\n")
print(classification_report(y_test, y_pred))
 
# ----------------------------------------------------------------------------
# 6. Save the trained pipeline (preprocessing + SMOTE + model all bundled)
# ----------------------------------------------------------------------------
joblib.dump(pipeline, MODEL_OUT)
print(f"\nSaved trained model to {MODEL_OUT}")