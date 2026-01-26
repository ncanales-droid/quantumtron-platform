"""API dependencies for authentication and authorization."""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token
from app.core.exceptions import AuthenticationError

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        dict: User information from token

    Raises:
        AuthenticationError: If token is invalid or missing
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise AuthenticationError(detail="Invalid authentication token")

    # Extract user information from token
    # In a real implementation, you would validate the user exists in the database
    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise AuthenticationError(detail="Token missing user information")

    return {"user_id": user_id, "email": payload.get("email")}


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise return None.

    Args:
        credentials: Optional HTTP Bearer token credentials

    Returns:
        Optional[dict]: User information if authenticated, None otherwise
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except AuthenticationError:
        return None


async def get_database_session(
    db: AsyncSession = Depends(get_db),
) -> AsyncSession:
    """
    Get database session dependency.

    Args:
        db: Database session

    Returns:
        AsyncSession: Database session
    """
    return db
