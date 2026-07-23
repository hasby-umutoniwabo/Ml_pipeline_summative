"""Streamlit UI: prediction, visualizations, uptime, upload and retrain controls."""
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"  

st.set_page_config(page_title="Malaria Cell Classifier", layout="wide")
st.title("Malaria Cell Classification")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Visualizations", "Upload & Retrain", "Model Uptime"])

#  Predict tab 
with tab1:
    st.header("Predict a single cell image")
    uploaded_file = st.file_uploader("Upload a cell image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, width=200, caption="Uploaded image")
        if st.button("Run Prediction"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['label']} (confidence: {result['confidence']})")
                else:
                    st.error(f"API error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Make sure it's running.")

#  Visualizations tab 
with tab2:
    st.header("Dataset insights")
    st.markdown("""
    **Class balance:** 13,779 Parasitized and 13,779 Uninfected images, a fully balanced dataset.

    **Brightness:** Parasitized cells average 0.458 pixel brightness vs 0.485 for Uninfected,
    consistent with visible parasite staining creating darker contrast regions.

    **Image width:** Parasitized averages 133.4px vs 132.1px for Uninfected, nearly identical,
    confirming the model learns real cell features rather than a size shortcut.
    """)
    st.info("Full plots with distributions are in notebook/malaria_cnn.ipynb on GitHub.")

#  Upload & Retrain tab
with tab3:
    st.header("Upload bulk data for retraining")
    label_choice = st.selectbox("Label for these images", ["Parasitized", "Uninfected"])
    bulk_files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if bulk_files and st.button("Upload Files"):
        files = [("files", (f.name, f.getvalue())) for f in bulk_files]
        try:
            response = requests.post(f"{API_URL}/upload", params={"label": label_choice}, files=files)
            if response.status_code == 200:
                st.success(f"Uploaded {response.json()['saved']} files.")
            else:
                st.error(f"API error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure it's running.")

    st.divider()
    st.subheader("Trigger retraining")
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=5)
    if st.button("Start Retraining"):
        with st.spinner("Retraining in progress..."):
            try:
                response = requests.post(f"{API_URL}/retrain", params={"epochs": epochs})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Retrained for {result['epochs']} epochs. "
                               f"Final accuracy: {result['final_accuracy']}, final loss: {result['final_loss']}")
                else:
                    st.error(f"API error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Make sure it's running.")

# Uptime tab
with tab4:
    st.header("Model / API uptime")
    if st.button("Refresh Status"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                st.metric("Status", data["status"])
                st.metric("Uptime (seconds)", data["uptime_seconds"])
            else:
                st.error("API not responding correctly.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure it's running.")