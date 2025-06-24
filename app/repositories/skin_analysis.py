# ===== app/repositories/skin_analysis.py =====
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import func
from typing import List, Optional, Dict, Any
import json
import uuid

from app.models.skin_analysis import SkinAnalysisSession, AcneDetection, AcneSeverity, SkinType
from app.schemas.skin_analysis import SkinAnalysisCreate


class SkinAnalysisRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_analysis(self, analysis_data: Dict[str, Any], user_id: str) -> SkinAnalysisSession:
        """Create complete skin analysis session with user_id"""

        # Create main session
        session = SkinAnalysisSession(
            user_id=user_id,
            image_url=analysis_data.get("image_url", "")
        )
        self.db.add(session)
        await self.db.flush()  # Get session.id

        # Process acne detections
        acne_data = analysis_data.get("acneDetection", {})
        for detection in acne_data.get("predicts", []):
            try:
                acne_detection = AcneDetection(
                    session_id=session.id,
                    name=detection.get("name", "unknown"),
                    class_id=detection.get("class", 0),
                    confidence=float(detection.get("confidence", 0.0)),
                    x1=float(detection.get("box", {}).get("x1", 0)),
                    y1=float(detection.get("box", {}).get("y1", 0)),
                    x2=float(detection.get("box", {}).get("x2", 0)),
                    y2=float(detection.get("box", {}).get("y2", 0)),
                    color_r=int(detection.get("color", [0, 0, 0])[0]),
                    color_g=int(detection.get("color", [0, 0, 0])[1]),
                    color_b=int(detection.get("color", [0, 0, 0])[2])
                )
                self.db.add(acne_detection)
            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Skipping invalid acne detection data: {e}")
                continue

        # Process acne severity
        severity_data = analysis_data.get("acneSeverity", {})
        try:
            severity = AcneSeverity(
                session_id=session.id,
                conf_threshold=float(severity_data.get("meta", {}).get("conf_threshold", 0.5)),
                available_classes=severity_data.get("meta", {}).get("classes", [])
            )

            # Add severity prediction if available
            severity_predicts = severity_data.get("predicts", [])
            if severity_predicts and len(severity_predicts) > 0:
                first_predict = severity_predicts[0]
                severity.severity_name = first_predict.get("name")
                severity.severity_class = first_predict.get("class")
                severity.confidence = float(first_predict.get("confidence", 0.0)) if first_predict.get("confidence") else None

            self.db.add(severity)
        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing acne severity data: {e}")

        # Process skin type
        skin_data = analysis_data.get("skinType", {})
        try:
            skin_predicts = skin_data.get("predicts", {})
            skin_type = SkinType(
                session_id=session.id,
                skin_type_name=skin_predicts.get("name", "unknown"),
                class_index=int(skin_predicts.get("class_index", 0)),
                confidence=float(skin_predicts.get("confidence", 0.0)),
                available_classes=skin_data.get("meta", {}).get("classes", [])
            )
            self.db.add(skin_type)
        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing skin type data: {e}")

        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_analysis_by_id(self, session_id: str, user_id: Optional[str] = None) -> Optional[SkinAnalysisSession]:
        """Get analysis with all related data, optionally filter by user_id"""
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            return None

        query = select(SkinAnalysisSession).options(
            selectinload(SkinAnalysisSession.acne_detections),
            selectinload(SkinAnalysisSession.acne_severity),
            selectinload(SkinAnalysisSession.skin_type)
        ).where(SkinAnalysisSession.id == session_uuid)

        if user_id is not None:
            query = query.where(SkinAnalysisSession.user_id == user_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_analyses(self, user_id: str, skip: int = 0, limit: int = 100) -> List[SkinAnalysisSession]:
        """Get all analyses for a specific user with pagination"""
        result = await self.db.execute(
            select(SkinAnalysisSession)
            .options(
                selectinload(SkinAnalysisSession.acne_detections),
                selectinload(SkinAnalysisSession.acne_severity),
                selectinload(SkinAnalysisSession.skin_type)
            )
            .where(SkinAnalysisSession.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(SkinAnalysisSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_all_analyses(self, skip: int = 0, limit: int = 100) -> List[SkinAnalysisSession]:
        """Get all analyses with pagination (admin only)"""
        result = await self.db.execute(
            select(SkinAnalysisSession)
            .options(
                selectinload(SkinAnalysisSession.acne_detections),
                selectinload(SkinAnalysisSession.acne_severity),
                selectinload(SkinAnalysisSession.skin_type)
            )
            .offset(skip)
            .limit(limit)
            .order_by(SkinAnalysisSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_user_analysis_count(self, user_id: str) -> int:
        """Get total count of analyses for a user"""
        result = await self.db.execute(
            select(func.count(SkinAnalysisSession.id))
            .where(SkinAnalysisSession.user_id == user_id)
        )
        return result.scalar() or 0

    async def get_total_analysis_count(self) -> int:
        """Get total count of all analyses"""
        result = await self.db.execute(
            select(func.count(SkinAnalysisSession.id))
        )
        return result.scalar() or 0

    async def get_statistics(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analysis statistics"""
        session = await self.get_analysis_by_id(session_id, user_id)
        if not session:
            return {}

        # Count detections by type
        detections_by_type = {}
        total_confidence = 0

        for detection in session.acne_detections:
            detections_by_type[detection.name] = detections_by_type.get(detection.name, 0) + 1
            total_confidence += detection.confidence

        avg_confidence = total_confidence / len(session.acne_detections) if session.acne_detections else 0

        return {
            "total_detections": len(session.acne_detections),
            "detections_by_type": detections_by_type,
            "average_confidence": round(avg_confidence, 3),
            "skin_type": session.skin_type.skin_type_name if session.skin_type else None,
            "severity_level": session.acne_severity.severity_name if session.acne_severity else None
        }