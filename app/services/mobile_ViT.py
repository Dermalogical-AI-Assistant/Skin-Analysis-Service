from app.ml_models.mobileViT.loader import load_mobileViT_model
from app.core import config
from PIL import Image
import numpy as np
import io


# Load mobileViT model
model, classes  = load_mobileViT_model()

def detect_objects_mobileViT(image_bytes: bytes, conf_threshold: float = 0.5):
    # Đọc ảnh và chuyển sang grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    image = image.resize((256, 256))  # Resize về input shape của model

    # Chuyển sang numpy array và normalize
    image_array = np.array(image).astype("float32") / 255.0

    # Thêm channel axis và batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # (256, 256, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 256, 256, 1)

    # Dự đoán
    result = model.predict(image_array)

    predicted_index = np.argmax(result)
    predicted_label = classes.get(str(predicted_index))
    confidence = result[0][predicted_index]

    if confidence < conf_threshold:
       return None, conf_threshold, classes

    result_return = {
        "name": predicted_label,
        "confidence": float(confidence),
        "classes": int(predicted_index),
    }

    return result_return, conf_threshold, classes
