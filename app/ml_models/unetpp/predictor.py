import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from fastapi import UploadFile
import time

class AcneDetector:
    def __init__(self):
        self.IMG_SIZE = (256, 256)
        self.SMOOTH = 1.0
        self.model = self._load_model(os.path.join(os.path.dirname(__file__), "weights", "best_model.keras"))

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.SMOOTH)

    def iou_metric(self, y_true, y_pred):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection
        return (intersection + 1e-7) / (union + 1e-7)

    def combined_loss(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice_loss = 1 - self.dice_coef(y_true, y_pred)
        return 0.5 * bce + 0.5 * dice_loss

    def _load_model(self, model_path: str):
        print(f"Loading segmentation model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            custom_objects = {
                'dice_coef': self.dice_coef,
                'iou_metric': self.iou_metric,
                'combined_loss': self.combined_loss
            }

            model = load_model(model_path, custom_objects=custom_objects)
            print(f"✅ Segmentation model loaded successfully!")
            return model  # This was missing!

        except Exception as e:
            raise Exception(f"Error loading segmentation model: {e}")

    def preprocess_uploaded_image(self, uploaded_file: UploadFile):
        try:
            # Read file content
            file_content = uploaded_file.file.read()
            uploaded_file.file.seek(0)  # Reset file pointer

            # Convert to numpy array
            nparr = np.frombuffer(file_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Cannot read image from file: {uploaded_file.filename}")

            # Save original size
            original_size = (img.shape[1], img.shape[0])  # (width, height)

            # Convert BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for model
            img_resized = cv2.resize(img_rgb, self.IMG_SIZE)

            # Normalize to [0, 1]
            img_resized = img_resized.astype(np.float32) / 255.0

            # Add batch dimension
            img_resized = np.expand_dims(img_resized, axis=0)

            return img_resized, original_size, img_rgb

        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None, None

    def predict_mask(self, uploaded_file: UploadFile, threshold=0.5):
        """Generate mask from uploaded image"""
        print(f"🔍 Generating mask for: {uploaded_file.filename}")

        # Check if model is loaded
        if self.model is None:
            print("❌ Model not loaded properly")
            return None

        # Preprocess
        img, original_size, original_img = self.preprocess_uploaded_image(uploaded_file)
        if img is None:
            return None

        try:
            # Predict
            start_time = time.time()
            prediction = self.model.predict(img, verbose=0)[0, :, :, 0]
            prediction_time = time.time() - start_time

            # Create binary mask
            binary_mask = (prediction > threshold).astype(np.uint8)

            # Resize to original size
            binary_mask_resized = cv2.resize(
                binary_mask,
                original_size,
                interpolation=cv2.INTER_NEAREST
            )

            # Calculate statistics
            total_pixels = binary_mask_resized.shape[0] * binary_mask_resized.shape[1]
            acne_pixels = np.sum(binary_mask_resized)
            acne_percentage = (acne_pixels / total_pixels) * 100

            result = {
                'success': True,
                'binary_mask': binary_mask_resized,
                'original_image': original_img,
                'original_size': original_size,
                'acne_percentage': round(acne_percentage, 2),
                'prediction_time': round(prediction_time, 3),
                'filename': uploaded_file.filename
            }

            print(f"✅ Mask generation completed! Acne area: {acne_percentage:.2f}%")
            return result

        except Exception as e:
            print(f"❌ Error during mask prediction: {e}")
            return None