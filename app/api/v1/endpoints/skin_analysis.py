from fastapi import APIRouter, UploadFile, File, Query
from starlette.responses import JSONResponse

from app.services.yolo_service import detect_objects_yolov9
from app.services.mobile_ViT import detect_objects_mobileViT
from app.services.convnext_service import classify_skin_type 

router = APIRouter()

@router.post("/predict")
async def predict_yolo(
    image: UploadFile = File(...),
    conf: float = Query(default=0.5, ge=0, le=1, description="Confidence threshold between 0 and 1")
):
    # READ IMAGE
    image_bytes = await image.read()

    # Call model to predict
    acne_detection_results_json, acne_detection_conf_threshold, acne_detection_image_url, acne_detection__classes = detect_objects_yolov9(
        image_bytes,
        conf_threshold = conf
    )
    acne_severity_results, acne_severity_conf_threshold, acne_severity_classes = detect_objects_mobileViT(
        image_bytes,
        conf_threshold = conf
    )

    result_return, classes = classify_skin_type(image_bytes=image_bytes)

    # RETURN
    return JSONResponse(content={
        "acneDetection": {
            "meta":{
                "classes":acne_detection__classes,
                "conf_threshold":acne_detection_conf_threshold
            },
            "predicts":acne_detection_results_json,
        },
        "acneSeverity": {
            "meta": {
                "classes": acne_severity_classes,
                "conf_threshold": acne_severity_conf_threshold
            },
            "predicts": [acne_severity_results] if acne_severity_results else []
        },
        "skinType": {
            "meta": {
                "classes": classes
            },
            "predicts": result_return
        },
        "imageURL": acne_detection_image_url
    })



