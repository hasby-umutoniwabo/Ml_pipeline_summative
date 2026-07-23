"""Single-image prediction utilities for the API."""
import tensorflow as tf
from src.preprocessing import preprocess_single_image

CLASS_NAMES = ["Uninfected", "Parasitized"]

def load_model(model_path):
    # loads the saved .h5 model from disk
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path):
    """
    Runs prediction on a single image file.
    Returns the predicted label and confidence score.
    """
    image = preprocess_single_image(image_path)
    prob = float(model.predict(image, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "raw_score": round(prob, 4)
    }