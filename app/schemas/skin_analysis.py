# ===== app/schemas/skin_analysis.py =====
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID


class AcneDetectionSchema(BaseModel):
    """Schema for individual acne detection"""
    id: UUID
    name: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    color_r: int
    color_g: int
    color_b: int
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat() if v else None
        }


# Base schemas for individual detections
class AcneDetectionItem(BaseModel):
    """Individual acne detection item"""
    name: str
    class_id: int = Field(..., alias="class")
    confidence: float
    box: Dict[str, float]  # {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
    color: List[int]  # [r, g, b]

    class Config:
        allow_population_by_field_name = True


class AcneSeveritySchema(BaseModel):
    """Schema for acne severity classification"""
    id: UUID
    severity_name: Optional[str] = None
    severity_class: Optional[int] = None
    confidence: Optional[float] = None
    conf_threshold: float = 0.5
    available_classes: Optional[Union[List[str], Dict[str, str]]] = []
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat() if v else None
        }

    @validator('available_classes', pre=True)
    def validate_available_classes(cls, v):
        """Convert dict to list if needed"""
        if isinstance(v, dict):
            # Convert dict values to list, maintaining order by keys
            return [v.get(str(i), f"Class_{i}") for i in sorted([int(k) for k in v.keys()])]
        return v or []


class SkinTypeSchema(BaseModel):
    """Schema for skin type classification"""
    id: UUID
    skin_type_name: str
    class_index: int
    confidence: float
    available_classes: Optional[Union[List[str], Dict[str, str]]] = []
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat() if v else None
        }

    @validator('available_classes', pre=True)
    def validate_available_classes(cls, v):
        """Convert dict to list if needed"""
        if isinstance(v, dict):
            # Convert dict values to list, maintaining order by keys
            return [v.get(str(i), f"Class_{i}") for i in sorted([int(k) for k in v.keys()])]
        return v or []


class SkinAnalysisResponse(BaseModel):
    """Complete skin analysis response"""
    id: UUID
    user_id: str
    image_url: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    acne_detections: List[AcneDetectionSchema] = []
    acne_severity: Optional[AcneSeveritySchema] = None
    skin_type: Optional[SkinTypeSchema] = None

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat() if v else None
        }

    @validator('acne_detections', pre=True)
    def validate_acne_detections(cls, v):
        return v or []


class SkinAnalysisCreate(BaseModel):
    """Schema for creating skin analysis"""
    analysis_data: Dict[str, Any]
    user_id: str


class SkinAnalysisHistoryItem(BaseModel):
    """Simplified schema for history listing"""
    id: str
    image_url: str
    created_at: datetime
    skin_type: Optional[str] = None
    severity_level: Optional[str] = None
    total_detections: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AnalysisStatistics(BaseModel):
    """Analysis statistics schema"""
    total_detections: int = 0
    detections_by_type: Dict[str, int] = {}
    average_confidence: float = 0.0
    skin_type: Optional[str] = None
    severity_level: Optional[str] = None


class PaginatedAnalysisResponse(BaseModel):
    """Paginated analysis response"""
    analyses: List[SkinAnalysisResponse]
    page: int
    per_page: int
    total: int
    total_pages: int


class PaginatedHistoryResponse(BaseModel):
    """Paginated history response"""
    analyses: List[SkinAnalysisHistoryItem]
    page: int
    per_page: int
    total: int
    total_pages: int