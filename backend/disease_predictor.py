
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_models"

MODEL_PATH = MODEL_DIR / "blackgram_disease_model.keras"
CLASS_PATH = MODEL_DIR / "disease_classes.json"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60


# ==========================================
# 2. CLEAN DISPLAY NAMES
# ==========================================

DISPLAY_NAMES = {
    "Anthracnose 230": "Anthracnose",
    "Healthy 220": "Healthy",
    "Leaf Crinckle 150": "Leaf Crinkle",
    "Powdery Mildew 180": "Powdery Mildew",
    "Yellow Mosaic 220": "Yellow Mosaic"
}


# ==========================================
# 3. LOAD MODEL
# ==========================================

_model = None
_class_names = None


def load_disease_model():

    global _model, _class_names

    if _model is not None and _class_names is not None:
        return _model, _class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Disease model not found.")

    if not CLASS_PATH.exists():
        raise FileNotFoundError("Disease class file not found.")

    _model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as file:
        _class_names = json.load(file)

    return _model, _class_names


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    try:
        with Image.open(image_path) as image:

            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        return np.expand_dims(image_array, axis=0)

    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError(
            "Invalid image. Please upload a valid leaf photograph."
        )


# ==========================================
# 5. DISEASE PREDICTION
# ==========================================

def predict_disease(image_path):

    try:

        model, class_names = load_disease_model()

        image_array = preprocess_image(image_path)

        predictions = model.predict(
            image_array,
            verbose=0
        )

        probabilities = predictions[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_class = DISPLAY_NAMES.get(
            class_names[predicted_index],
            class_names[predicted_index]
        )

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # Get top three predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = []

        for index in top_indices:

            original_name = class_names[int(index)]

            display_name = DISPLAY_NAMES.get(
                original_name,
                original_name
            )

            top_predictions.append({
                "disease": display_name,
                "confidence": round(
                    float(probabilities[index] * 100), 2
                )
            })

        # Confidence-based warning
        if confidence < CONFIDENCE_THRESHOLD * 100:

            return {
                "success": True,
                "reliable_prediction": False,
                "message": (
                    "The model is uncertain about this image. "
                    "Please upload a clearer leaf photograph "
                    "or consult an agricultural expert."
                ),
                "predicted_disease": None,
                "confidence": round(confidence, 2),
                "top_predictions": top_predictions
            }

        return {
            "success": True,
            "reliable_prediction": True,
            "message": "Prediction generated successfully.",
            "predicted_disease": predicted_class,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions
        }

    except (FileNotFoundError, ValueError) as error:

        return {
            "success": False,
            "message": str(error)
        }

    except Exception as error:

        return {
            "success": False,
            "message": f"Prediction failed: {error}"
        }
