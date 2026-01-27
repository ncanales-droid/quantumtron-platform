"""Security utilities for JWT authentication."""

from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer authentication
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta

    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT access token.

    Args:
        token: JWT token string

    Returns:
        Optional[dict]: Decoded token data if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict]:
    """
    Get current user if authenticated, otherwise return None.
    
    This is the 'optional' version that doesn't require authentication.
    For testing, returns a mock user if token is provided.

    Args:
        credentials: Optional HTTP authorization credentials

    Returns:
        Optional[Dict]: User data if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        # Decode the token
        payload = decode_access_token(credentials.credentials)
        if not payload:
            return None
        
        # Extract user information from token payload
        user_id = payload.get("sub")
        username = payload.get("username")
        email = payload.get("email")
        
        if not user_id:
            return None
            
        return {
            "id": user_id,
            "username": username or f"user_{user_id}",
            "email": email or f"user_{user_id}@example.com",
            "token_payload": payload
        }
    except Exception:
        # If any error occurs during validation, return None
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    Get current user (requires authentication).
    
    Raises HTTPException if not authenticated.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Dict: User data
        
    Raises:
        HTTPException: If authentication fails
    """
    user = await get_optional_user(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user