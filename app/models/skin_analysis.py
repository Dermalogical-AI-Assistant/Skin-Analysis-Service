# ===== app/models/skin_analysis.py =====
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.models.base_model import Base


class SkinAnalysisSession(Base):
    """Main analysis session"""
    __tablename__ = "skin_analysis_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String(36), nullable=False, index=True)  # Added user_id field
    image_url = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    acne_detections = relationship("AcneDetection", back_populates="session", cascade="all, delete-orphan")
    acne_severity = relationship("AcneSeverity", back_populates="session", uselist=False, cascade="all, delete-orphan")
    skin_type = relationship("SkinType", back_populates="session", uselist=False, cascade="all, delete-orphan")


class AcneDetection(Base):
    """Individual acne detection results"""
    __tablename__ = "acne_detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("skin_analysis_sessions.id"), nullable=False)

    # Detection details
    name = Column(String(50), nullable=False)  # whitehead, blackhead, etc.
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

    # Bounding box coordinates
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    # Color information (RGB)
    color_r = Column(Integer, nullable=False)
    color_g = Column(Integer, nullable=False)
    color_b = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("SkinAnalysisSession", back_populates="acne_detections")


class AcneSeverity(Base):
    """Acne severity classification"""
    __tablename__ = "acne_severity"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("skin_analysis_sessions.id"), nullable=False)

    # Classification (if available)
    severity_name = Column(String(20))  # Mild, Moderate, Severe, Very Severe
    severity_class = Column(Integer)
    confidence = Column(Float)

    # Metadata
    conf_threshold = Column(Float, default=0.5)
    available_classes = Column(JSON)  # Store class mapping

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("SkinAnalysisSession", back_populates="acne_severity")


class SkinType(Base):
    """Skin type classification"""
    __tablename__ = "skin_types"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("skin_analysis_sessions.id"), nullable=False)

    # Classification
    skin_type_name = Column(String(20), nullable=False)  # Oily, Dry, Normal, Combination
    class_index = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

    # Metadata
    available_classes = Column(JSON)  # Store class mapping

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("SkinAnalysisSession", back_populates="skin_type")