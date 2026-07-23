"""FastAPI app exposing predict, upload, retrain and health endpoints."""
import os
import shutil
import sys
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# allow importing from src/ when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction import load_model, predict_image
from src.model import retrain_model

app = FastAPI(title="Malaria Classification API")

# allow the streamlit UI (different port/origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/malaria_cnn.h5"
UPLOAD_DIR = "data/uploads"
TEMP_PREDICT_DIR = "data/temp_predict"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREDICT_DIR, exist_ok=True)

# track when the server started, used for uptime display in the UI
START_TIME = time.time()
model = load_model(MODEL_PATH)


@app.get("/health")
def health_check():
    # reports uptime in seconds, used by the UI's "model uptime" panel
    uptime_seconds = round(time.time() - START_TIME, 1)
    return {"status": "healthy", "uptime_seconds": uptime_seconds}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save uploaded image temporarily, run prediction, then clean up
    temp_path = os.path.join(TEMP_PREDICT_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict_image(model, temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)

    return result


@app.post("/upload")
async def upload_bulk(label: str, files: list[UploadFile] = File(...)):
    """
    Bulk upload images for retraining. label must be 'Parasitized' or 'Uninfected'.
    Saved into data/uploads/<label>/ so retrain endpoint can find them.
    """
    if label not in ["Parasitized", "Uninfected"]:
        raise HTTPException(status_code=400, detail="label must be Parasitized or Uninfected")

    label_dir = os.path.join(UPLOAD_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    saved_files = []
    for file in files:
        dest_path = os.path.join(label_dir, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file.filename)

    return {"saved": len(saved_files), "files": saved_files}


@app.post("/retrain")
def trigger_retrain(epochs: int = 5):
    # fine-tunes the existing model on everything currently in data/uploads/
    global model
    try:
        model, history = retrain_model(MODEL_PATH, UPLOAD_DIR, epochs=epochs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    final_accuracy = float(history.history["accuracy"][-1])
    final_loss = float(history.history["loss"][-1])
    return {
        "status": "retrained",
        "epochs": epochs,
        "final_accuracy": round(final_accuracy, 4),
        "final_loss": round(final_loss, 4)
    }