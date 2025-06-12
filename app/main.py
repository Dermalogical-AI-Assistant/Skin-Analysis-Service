from fastapi import FastAPI
from app.api.v1.api_v1 import api_router
from app.database.connection import init_db
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.development")
load_dotenv(dotenv_path=".env")

app = FastAPI(title="Object Detection API")

@app.on_event("startup")
async def on_startup():
    await init_db()

app.include_router(api_router, prefix="/api/v1")
