# ===== app/core/auth.py =====
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional
import os
from datetime import datetime, timedelta

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "ngoctramxinhdep")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)  # Won't raise error if no token


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    """Extract user ID from JWT token - Required authentication"""
    try:
        payload = decode_access_token(credentials.credentials)
        user_id: string = payload.get("user").get("id")
        return user_id
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_id_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional)) -> \
Optional[int]:
    """Extract user ID from JWT token - Optional authentication, returns None if no valid token"""
    if not credentials:
        return None

    try:
        payload = decode_access_token(credentials.credentials)
        print("payload",payload)
        user_id: int = payload.get("user").get("id")
        return user_id
    except (HTTPException, JWTError):
        return None