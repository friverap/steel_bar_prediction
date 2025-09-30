"""
Security utilities for API authentication and rate limiting
"""

from fastapi import HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import time
from collections import defaultdict
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Simple in-memory rate limiting (in production, use Redis)
rate_limit_storage = defaultdict(list)

security = HTTPBearer()


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """
    Verify API key from X-API-Key header
    
    Args:
        x_api_key: API key from header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if x_api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return x_api_key


async def verify_admin_api_key(x_api_key: str = Header(...)) -> str:
    """
    Verify admin API key for pipeline operations
    
    Args:
        x_api_key: API key from header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid or not admin
    """
    # Verificar que sea un API key válido primero
    await verify_api_key(x_api_key)
    
    # Verificar que sea admin key (o el API key regular si no hay admin configurado)
    admin_key = settings.ADMIN_API_KEY or settings.API_KEY
    
    if x_api_key != admin_key:
        logger.warning(f"Unauthorized admin operation attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Admin access required for pipeline operations"
        )
    
    return x_api_key


async def check_rate_limit(request: Request, api_key: str) -> None:
    """
    Check rate limiting for API key
    
    Args:
        request: FastAPI request object
        api_key: Validated API key
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host
    key = f"{api_key}:{client_ip}"
    current_time = time.time()
    
    # Clean old requests (older than 1 hour)
    rate_limit_storage[key] = [
        timestamp for timestamp in rate_limit_storage[key]
        if current_time - timestamp < 3600  # 1 hour in seconds
    ]
    
    # Check if rate limit exceeded
    if len(rate_limit_storage[key]) >= settings.RATE_LIMIT_PER_HOUR:
        logger.warning(f"Rate limit exceeded for key: {api_key[:10]}... from IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_HOUR} requests per hour."
        )
    
    # Add current request timestamp
    rate_limit_storage[key].append(current_time)
    
    logger.info(f"Request {len(rate_limit_storage[key])}/{settings.RATE_LIMIT_PER_HOUR} for key: {api_key[:10]}...")


def get_api_key_info(api_key: str) -> dict:
    """
    Get information about API key usage
    
    Args:
        api_key: API key to check
        
    Returns:
        Dictionary with usage information
    """
    current_time = time.time()
    total_requests = 0
    
    # Count requests across all IPs for this API key
    for key, timestamps in rate_limit_storage.items():
        if key.startswith(api_key):
            # Clean old timestamps
            valid_timestamps = [
                ts for ts in timestamps
                if current_time - ts < 3600
            ]
            total_requests += len(valid_timestamps)
    
    return {
        "api_key": api_key[:10] + "...",
        "requests_last_hour": total_requests,
        "remaining_requests": max(0, settings.RATE_LIMIT_PER_HOUR - total_requests),
        "rate_limit": settings.RATE_LIMIT_PER_HOUR
    }
