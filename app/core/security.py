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

# Lovable testing token - use this in Lovable UI
LOVABLE_TEST_TOKEN = "lovable-test-token-12345"


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
    Accepts special Lovable test token for development.

    Args:
        credentials: Optional HTTP authorization credentials

    Returns:
        Optional[Dict]: User data if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    
    # SPECIAL CASE: Accept Lovable test token for development
    if token == LOVABLE_TEST_TOKEN:
        return {
            "id": 999,
            "username": "lovable_user",
            "email": "lovable@example.com",
            "source": "lovable_test_token"
        }
    
    # SPECIAL CASE: Accept any token starting with "test-" for testing
    if token.startswith("test-"):
        return {
            "id": 1,
            "username": f"test_user_{hash(token) % 1000}",
            "email": f"test{hash(token) % 1000}@example.com",
            "source": "test_token"
        }
    
    try:
        # Decode the actual JWT token
        payload = decode_access_token(token)
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
            "token_payload": payload,
            "source": "jwt_token"
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
    Accepts Lovable test token for development.

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
            detail="Could not validate credentials. For Lovable testing, use token: 'lovable-test-token-12345'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def create_lovable_test_token() -> str:
    """
    Create a test token for Lovable development.
    
    Returns:
        str: Lovable test token
    """
    return LOVABLE_TEST_TOKEN


def is_lovable_test_token(token: str) -> bool:
    """
    Check if token is the Lovable test token.
    
    Args:
        token: Token to check
        
    Returns:
        bool: True if token is Lovable test token
    """
    return token == LOVABLE_TEST_TOKEN