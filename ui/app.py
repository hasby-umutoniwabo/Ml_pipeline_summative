"""Streamlit UI: prediction, visualizations, uptime, upload and retrain controls."""
import streamlit as st
import requests

API_URL = "http://localhost:8000"  

st.set_page_config(page_title="Malaria Cell Classifier", layout="wide", page_icon="🔬")

# custom styling: lab-inspired palette, distinct typography, clean cards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.main { background-color: #FAFAF7; }

.app-header {
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 2px solid #0E4B4F;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    color: #0E4B4F;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.app-header p {
    color: #5A6B6D;
    font-size: 0.95rem;
}

.result-card {
    padding: 1.2rem 1.5rem;
    border-radius: 8px;
    border-left: 5px solid;
    margin-top: 1rem;
    font-size: 1.05rem;
}
.result-parasitized {
    background-color: #FBEAE8;
    border-color: #C1443B;
    color: #7A2A22;
}
.result-uninfected {
    background-color: #EAF2ED;
    border-color: #4C7A63;
    color: #2E4D3D;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}

.stButton button {
    background-color: #0E4B4F;
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 500;
}
.stButton button:hover {
    background-color: #0A3538;
    color: white;
}
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

#  Visualizations tab
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