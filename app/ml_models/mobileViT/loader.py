from tensorflow.keras.models import load_model
import os

def load_mobileViT_model():
    model_path = os.path.join(os.path.dirname(__file__), "weights", "best_model.keras")
    model = load_model(model_path)
    classes = {
        "0": "Mild",
        "1": "Moderate",
        "2": "Severe",
        "3": "Very Severe",
    }
    return model, classes
