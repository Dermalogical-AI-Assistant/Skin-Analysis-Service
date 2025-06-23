import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class AcneSeverityPredictor:
    def __init__(self, device: str = None, img_size: int = 300):
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.label_index = {"Mild": 0, "Moderate": 1, "Severe": 2, "Very Severe": 3}
        self.index_label = {v: k for k, v in self.label_index.items()}

        # Setup transforms
        self._setup_transforms()

        # Load model

        self.model = self._load_model(os.path.join(os.path.dirname(__file__), "weights", "best.pth"))

        print(f"✅ AcneSeverityPredictor initialized successfully!")
        print(f"📱 Device: {self.device}")

    def _setup_transforms(self):
        # Transform for image
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transform for mask
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size), antialias=True),
            transforms.ToTensor()
        ])

    def _get_efficientnet_b3_model(self, out_classes: int = 4):
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = efficientnet_b3(weights=weights)

        # Modify first conv to accept 4 channels (RGB + mask)
        original_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,  # 3 RGB + 1 mask
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Copy weights from pretrained model for RGB channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            # Initialize weights for mask channel
            new_conv.weight[:, 3:4, :, :] = original_conv.weight[:, :1, :, :] * 0.1
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

        # Replace first conv
        model.features[0][0] = new_conv

        # Modify classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.SiLU(),
            nn.Linear(num_ftrs // 2, out_classes)
        )
        return model

    def _load_model(self, model_path: str):
        try:
            model = self._get_efficientnet_b3_model(out_classes=4)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

            print(f"✅ Severity classification model loaded from: {model_path}")
            return model

        except Exception as e:
            raise RuntimeError(f"❌ Error loading severity model: {e}")

    def predict_severity(self, image_array: np.ndarray, mask_array: np.ndarray):
        """
        Predict acne severity from image and mask arrays

        Args:
            image_array: RGB image as numpy array
            mask_array: Binary mask as numpy array
        """
        try:
            # Preprocess image and mask
            img_tensor = self.image_transform(image_array)
            mask_tensor = self.mask_transform(mask_array)

            # Combine image and mask
            combined = torch.cat([img_tensor, mask_tensor], dim=0)  # Shape: (4, H, W)
            combined = combined.unsqueeze(0)  # Add batch dimension: (1, 4, H, W)

            # Move to device
            combined = combined.to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(combined)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(1).item()
                confidence = probabilities[0][predicted_class].item()

            # Convert to label
            predicted_label = self.index_label[predicted_class]
            return predicted_class, predicted_label, confidence,
        except Exception as e:
            print(f"❌ Error during severity prediction: {e}")
