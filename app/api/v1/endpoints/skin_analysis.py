# ===== app/api/v1/endpoints/skin_analysis.py =====
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException
from starlette.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional
import uuid
from app.schemas.convert.skin_analysis import acne_detection_to_response
from app.services.acne_severity import CompleteAcneAnalyzer
from app.services.yolo_service import detect_objects_yolov9
from app.services.convnext_service import classify_skin_type

from app.database.connection import get_db
from app.services.skin_analysis import SkinAnalysisService
from app.schemas.skin_analysis import SkinAnalysisResponse, AnalysisStatistics, SkinAnalysisHistoryItem
from app.core.auth import get_current_user_id, security_optional, decode_access_token, get_current_user_id_optional
from jose import JWTError
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

analyzer = CompleteAcneAnalyzer()


def get_skin_analysis_service(db: AsyncSession = Depends(get_db)) -> SkinAnalysisService:
    return SkinAnalysisService(db)


@router.post("/predict")
async def predict_yolo(
        image: UploadFile = File(...),
        conf: float = Query(default=0.01, ge=0, le=1),
        save_to_db: bool = Query(default=True),
        service: SkinAnalysisService = Depends(get_skin_analysis_service),
        current_user_id: Optional[str] = Depends(get_current_user_id_optional)
):
    try:
        print("current_user_id:", current_user_id)

        # READ IMAGE một lần
        image_bytes = await image.read()

        # Tạo wrapper cho các hàm sync
        async def run_acne_detection():
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(
                    executor,
                    detect_objects_yolov9,
                    image_bytes,
                    conf
                )

        async def run_skin_type_classification():
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(
                    executor,
                    classify_skin_type,
                    image_bytes
                )

        # Chạy song song tất cả models
        (acne_detection_results,
         acne_severity_results,
         skin_type_results) = await asyncio.gather(
            run_acne_detection(),
            analyzer.analyze_acne(image),  # Đã là async
            run_skin_type_classification()
        )

        # Unpack results
        (acne_detection_results_json,
         acne_detection_conf_threshold,
         acne_detection_image_url,
         acne_detection__classes) = acne_detection_results

        result_return, classes = skin_type_results

        # Prepare prediction results
        prediction_results = {
            "acneDetection": {
                "meta": {
                    "classes": acne_detection__classes,
                    "conf_threshold": acne_detection_conf_threshold
                },
                "predicts": acne_detection_results_json,
            },
            "acneSeverity": acne_severity_results,
            "skinType": {
                "meta": {
                    "classes": classes
                },
                "predicts": result_return
            },
            "image_url": acne_detection_image_url
        }

        # Save to database logic (unchanged)
        saved_analysis = None
        if save_to_db and current_user_id:
            try:
                saved_analysis = await service.create_analysis(prediction_results, current_user_id)
            except Exception as e:
                print(f"Warning: Failed to save analysis to database: {str(e)}")

        # Return response
        response_data = prediction_results.copy()
        if saved_analysis:
            response_data["id"] = str(saved_analysis["id"])
            response_data["created_at"] = saved_analysis["created_at"].isoformat() if saved_analysis[
                "created_at"] else None
            response_data["message"] = "Analysis saved successfully"
        elif save_to_db and not current_user_id:
            response_data["save_status"] = {
                "message": "Analysis not saved - user authentication required"
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history", response_model=Dict[str, Any])
async def get_user_analysis_history(
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=10, ge=1, le=100),
        service: SkinAnalysisService = Depends(get_skin_analysis_service),
        current_user_id: str = Depends(get_current_user_id)
):
    """Get current user's skin analysis history"""
    return await service.get_user_history(current_user_id, page=page, per_page=per_page)


@router.get("/{analysis_id}")
async def get_skin_analysis(
        analysis_id: str,
        service: SkinAnalysisService = Depends(get_skin_analysis_service),
        current_user_id: str = Depends(get_current_user_id)
):
    """Get skin analysis by ID (only user's own analysis)"""
    try:
        # Validate UUID format
        uuid.UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")

    analysis = await service.get_analysis(analysis_id, current_user_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    res_result = acne_detection_to_response(analysis)

    return res_result

# Admin endpoints (optional)
@router.get("/admin/all", response_model=Dict[str, Any])
async def get_all_analyses_admin(
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=10, ge=1, le=100),
        service: SkinAnalysisService = Depends(get_skin_analysis_service),
        # current_user_id: str = Depends(get_current_admin_user)  # Implement admin check
):
    """Get all skin analyses with pagination (Admin only)"""
    return await service.get_all_analyses(page=page, per_page=per_page)