# ===== app/models/models.py =====
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# SQLAlchemy Models (Database)
Base = declarative_base()