from fastapi import FastAPI
from dotenv import load_dotenv
app = FastAPI()

# main.py
from fastapi import FastAPI
from app.api.v1.api_v1 import api_router
load_dotenv(dotenv_path=".env.development")
load_dotenv(dotenv_path=".env")

app = FastAPI(title="Object Detection API")

app.include_router(api_router, prefix="/api/v1")
