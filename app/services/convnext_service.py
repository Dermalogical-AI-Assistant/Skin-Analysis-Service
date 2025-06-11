import torch
import numpy as np
import mediapipe as mp
import cv2
from torchvision import transforms
from app.ml_models.convnext.loader import load_convnext_model

IMG_SIZE = 224 

model, classes = load_convnext_model()

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_skin_type(image_bytes: bytes):
    # Convert bytes to numpy array
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Initialize face detection
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=3,  # Use model_selection=3 for better accuracy
        min_detection_confidence=0.6
    )
    
    # Detect faces
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        return {"error": "No face detected"}, classes
    
    # Get first face (you might want to handle multiple faces differently)
    detection = results.detections[0]
    if detection:
        bbox = detection.location_data.relative_bounding_box
        ih, iw = img.shape[:2]
        
        # Calculate face bounding box
        x = max(0, int(bbox.xmin * iw))
        y = max(0, int(bbox.ymin * ih))
        w = min(iw - x, int(bbox.width * iw))
        h = min(ih - y, int(bbox.height * ih))
        
        # Extract face ROI
        face = img[y:y+h, x:x+w]
        img = face
    
    # Convert to tensor and apply transforms
    try:
        img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}, classes
    
    # Move to device and make prediction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)
        confidence = float(confidence.item())
        class_index = int(predicted_index.item())
        predicted_label = classes[str(class_index)]
    
    # Prepare result
    result_return = {
        "name": predicted_label,
        "confidence": confidence,
        "class_index": class_index,
    }
    
    return result_return, classes
