# Malaria Cell Classifier — Machine Learning Cycle

A full MLOps pipeline for classifying malaria infected blood cells from microscope images. Covers data acquisition, preprocessing, model training, evaluation, retraining, an API, a UI, Docker deployment, and load testing.

## Video Demo

[Watch on YouTube](https://youtube.com/your-video-link-here)

## Live URLs

- UI: https://malaria-ui-j37y.onrender.com
- API: https://malaria-api-saiv.onrender.com
- API docs (Swagger): https://malaria-api-saiv.onrender.com/docs

Note: Render's free tier spins down after inactivity, so the first request may take 30 to 60 seconds to wake up.

## Project Description

Malaria diagnosis usually requires a trained technician manually inspecting blood smears under a microscope. This project trains a CNN to automate that first pass, classifying cell images as Parasitized or Uninfected. The system also supports uploading new labeled images and retraining the model on demand, so it can keep improving as more data becomes available.

Dataset: NIH Malaria Cell Images, fetched automatically in the notebook, no manual download needed.

## Model Performance

- Accuracy: 96.47%
- Precision: 98.15%
- Recall: 94.73%
- F1 Score: 96.41%
- ROC AUC: 0.9918

## Project Structure

    Ml_pipeline_summative/
    ├── README.md
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    ├── nginx.conf
    ├── notebook/
    │   └── malaria_cnn.ipynb
    ├── src/
    │   ├── preprocessing.py
    │   ├── model.py
    │   └── prediction.py
    ├── api/
    │   └── main.py
    ├── ui/
    │   └── app.py
    ├── data/
    │   └── uploads/
    ├── models/
    │   └── malaria_cnn.h5
    └── locust/
        ├── locustfile.py
        └── sample_cell.png

## Features

- **Predict**: upload one cell image, get an instant Parasitized or Uninfected diagnosis with confidence score
- **Visualizations**: dataset insights on class balance, brightness, and image size
- **Upload & Retrain**: upload multiple labeled images and trigger fine tuning on the existing model
- **Model Uptime**: live health check showing API status and uptime

## Setup Instructions

### 1. Clone the repo

    git clone https://github.com/hasby-umutoniwabo/Ml_pipeline_summative.git
    cd Ml_pipeline_summative

### 2. Create a virtual environment and install dependencies

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 3. Run the API

    uvicorn api.main:app --reload --port 8000

Visit `http://localhost:8000/health` to confirm it's running, or `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Run the UI

In a separate terminal:

    source venv/bin/activate
    streamlit run ui/app.py

Visit `http://localhost:8501`.

### 5. Run with Docker instead

    docker compose up --build

This starts the API on port 8000 and the UI on port 8501.

### 6. Run the notebook

Open `notebook/malaria_cnn.ipynb` in Google Colab or Jupyter. It downloads the dataset automatically, no manual download required. A GPU runtime is strongly recommended for training speed.

## Retraining Workflow

1. Go to the **Upload & Retrain** tab in the UI
2. Select the correct label (Parasitized or Uninfected) and upload one or more images
3. Click **Start Retraining** to fine tune the existing model on the newly uploaded data
4. The API reloads the updated model automatically after retraining completes

## Load Testing Results (Locust)

Tested with 20 concurrent users, spawn rate 5 per second, against the `/predict` and `/health` endpoints.

| Metric | 1 container | 3 containers |
|---|---|---|
| Total requests | 563 | 555 |
| Failures | 0 (0%) | 60 (11%) |
| Predict median (ms) | 57 | 270 |
| Predict 95th percentile (ms) | 270 | 410 |
| Aggregate requests/sec | 8.8 | 10 |

**Interpretation:** scaling to 3 containers increased raw throughput, but on a single local machine all 3 replicas competed for the same limited CPU and memory (each loads its own full copy of the model), causing predict request failures. In a cloud environment with dedicated resources per container, scaling would improve throughput without this contention.

To reproduce:

    docker compose up --build --scale api=1 -d
    locust -f locust/locustfile.py --host http://localhost:8000

    # then, after stopping:
    docker compose up --build --scale api=3 -d
    locust -f locust/locustfile.py --host http://localhost:8000

## Tech Stack

- TensorFlow / Keras for the CNN
- FastAPI for the prediction and retraining API
- Streamlit for the UI
- Docker and Docker Compose for containerization
- Nginx for load balancing across scaled containers
- Locust for load testing
- Render for cloud deployment
