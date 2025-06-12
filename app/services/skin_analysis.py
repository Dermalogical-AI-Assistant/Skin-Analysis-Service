# ===== app/services/skin_analysis.py =====
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any

from app.repositories.skin_analysis import SkinAnalysisRepository
from app.schemas.skin_analysis import SkinAnalysisResponse, AnalysisStatistics, SkinAnalysisHistoryItem

class SkinAnalysisService:
    def __init__(self, db: AsyncSession):
        self.repo = SkinAnalysisRepository(db)

    async def create_analysis(self, analysis_data: Dict[str, Any], user_id: str):
        """Create new skin analysis with user_id"""
        session = await self.repo.create_analysis(analysis_data, user_id)
        return {
            "id": session.id,
            "created_at": session.created_at,
        }

    async def get_analysis(self, session_id: str, user_id: Optional[str] = None) -> Optional[SkinAnalysisResponse]:
        """Get analysis by ID, optionally filter by user_id"""
        session = await self.repo.get_analysis_by_id(session_id, user_id)
        return SkinAnalysisResponse.from_orm(session) if session else None

    async def get_user_history(self, user_id: str, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Get user's analysis history with pagination"""
        skip = (page - 1) * per_page
        sessions = await self.repo.get_user_analyses(user_id, skip=skip, limit=per_page)
        total_count = await self.repo.get_user_analysis_count(user_id)

        # Create simplified history items
        history_items = []
        for session in sessions:
            try:
                history_item_data = {
                    "id": str(session.id),
                    "image_url": session.image_url or "",
                    "created_at": session.created_at,
                    "skin_type": session.skin_type.skin_type_name if session.skin_type else None,
                    "severity_level": session.acne_severity.severity_name if session.acne_severity else None,
                    "total_detections": len(session.acne_detections) if session.acne_detections else 0
                }
                history_items.append(SkinAnalysisHistoryItem(**history_item_data))
            except Exception as e:
                print(f"Warning: Error processing history item for session {session.id}: {e}")
                continue

        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1

        return {
            "analyses": history_items,
            "page": page,
            "per_page": per_page,
            "total": total_count,
            "total_pages": total_pages
        }

    async def get_all_analyses(self, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Get paginated analyses (admin only)"""
        skip = (page - 1) * per_page
        sessions = await self.repo.get_all_analyses(skip=skip, limit=per_page)
        total_count = await self.repo.get_total_analysis_count()

        analyses = []
        for session in sessions:
            try:
                analyses.append(SkinAnalysisResponse.from_orm(session))
            except Exception as e:
                print(f"Warning: Error processing analysis for session {session.id}: {e}")
                continue

        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1

        return {
            "analyses": analyses,
            "page": page,
            "per_page": per_page,
            "total": total_count,
            "total_pages": total_pages
        }
