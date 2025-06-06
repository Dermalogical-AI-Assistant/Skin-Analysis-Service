from fastapi import APIRouter
from app.api.v1.endpoints import skin_analysis

api_router = APIRouter()
api_router.include_router(skin_analysis.router, prefix="/skin-analysis", tags=["SKIN-ANALYSIS"])
