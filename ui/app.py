"""Streamlit UI: prediction, visualizations, uptime, upload and retrain controls."""
import streamlit as st
import requests

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Malaria Cell Classifier", layout="wide", page_icon="🔬")

# dark lab theme: deep teal-black background, teal accent, coral for infected, sage for healthy
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

.stApp { background-color: #0F1A18; }
html, body, [class*="css"], p, span, label, div { font-family: 'Inter', sans-serif; color: #E4E9E7; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #E4E9E7 !important; }

.app-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #2DD4BF;
    margin-bottom: 1.5rem;
}
.app-header h1 { color: #2DD4BF !important; font-weight: 700; margin-bottom: 0.3rem; }
.app-header p { color: #9BB0AC; font-size: 0.95rem; }

.result-card {
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid;
    margin-top: 1rem;
    font-size: 1.05rem;
}
.result-parasitized { background-color: #2A1414; border-color: #E5484D; color: #FFB4B4; }
.result-uninfected { background-color: #12261C; border-color: #5FBF8C; color: #B7EBCB; }

/* explicit tab colors so nothing blends into the dark background */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    color: #9BB0AC !important;
    background-color: #182924;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #2DD4BF !important;
    background-color: #1F3831 !important;
    border-bottom: 2px solid #2DD4BF !important;
}

.stButton button {
    background-color: #2DD4BF;
    color: #0F1A18;
    border-radius: 6px;
    border: none;
    font-weight: 600;
}
.stButton button:hover { background-color: #21B3A0; color: #0F1A18; }

[data-testid="stMetric"] {
    background-color: #182924;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #2A423B;
}
[data-testid="stMetricValue"] { color: #2DD4BF !important; }

.stFileUploader section { background-color: #182924; border: 1px dashed #2A423B; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🔬 Malaria Cell Classifier</h1>
    <p>Upload a blood cell image for instant diagnosis, or contribute data to retrain the model.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Visualizations", "Upload & Retrain", "Model Uptime"])

# Predict tab 
with tab1:
    st.subheader("Predict a single cell image")
    uploaded_file = st.file_uploader("Upload a cell image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, width=180, caption="Uploaded image")
        with col2:
            if st.button("Run Prediction", use_container_width=True):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    response = requests.post(f"{API_URL}/predict", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        css_class = "result-parasitized" if result["label"] == "Parasitized" else "result-uninfected"
                        st.markdown(f"""
                        <div class="result-card {css_class}">
                            <strong>Diagnosis: {result['label']}</strong><br>
                            Confidence: {result['confidence'] * 100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"API error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Make sure it's running.")

# Visualizations tab
with tab2:
    st.subheader("Dataset insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total images", "27,558")
    col2.metric("Class balance", "50 / 50")
    col3.metric("Test AUC", "0.99")

    st.markdown("""
    **Class balance.** 13,779 Parasitized and 13,779 Uninfected images. A fully balanced dataset,
    so the model has no built-in bias toward either class.

    **Brightness.** Parasitized cells average 0.458 pixel brightness vs 0.485 for Uninfected.
    Darker, more contrasted regions line up with visible parasite staining.

    **Image width.** Parasitized averages 133.4px vs 132.1px for Uninfected, nearly identical.
    This rules out image size as an accidental shortcut. The model is reading real cell features.
    """)
    st.info("Full distribution plots are in notebook/malaria_cnn.ipynb on GitHub.")

# Upload & Retrain tab
with tab3:
    st.subheader("Upload bulk data for retraining")
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
                response = requests.post(f"{API_URL}/retrain", params={"epochs": epochs}, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Retrained for {result['epochs']} epochs. "
                               f"Final accuracy: {result['final_accuracy']}, final loss: {result['final_loss']}")
                else:
                    st.error(f"API error (status {response.status_code}): {response.text or 'No response body — likely a server timeout or crash.'}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Make sure it's running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Retraining may be too resource-intensive for this server.")

# Uptime tab 
with tab4:
    st.subheader("Model / API uptime")
    if st.button("Refresh Status"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                col1, col2 = st.columns(2)
                col1.metric("Status", data["status"].capitalize())
                col2.metric("Uptime (seconds)", data["uptime_seconds"])
            else:
                st.error("API not responding correctly.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure it's running.")