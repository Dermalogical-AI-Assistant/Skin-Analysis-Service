from ultralytics import YOLO
import os

def load_yolov9_model():
    model_path = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
    model = YOLO(model_path)
    return model, model.names
