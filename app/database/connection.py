# ===== app/database/database/connection.py =====
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from app.models.base_model import Base

load_dotenv()

DATABASE_URL = f"{os.getenv('SKIN_ANALYSIS_DATABASE_URL')}"
# DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
print(DATABASE_URL)
# Async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """Tự động tạo tất cả tables từ models"""
    async with engine.begin() as conn:
        # Tạo tất cả tables được define trong Base.metadata
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Dependency để lấy database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()