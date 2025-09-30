"""
Configuration settings for DeAcero Steel Price Predictor API
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Security
    API_KEY: str = Field(..., description="API key for authentication")
    ADMIN_API_KEY: str = Field(default="", description="Admin API key for pipeline operations")
    RATE_LIMIT_PER_HOUR: int = Field(default=100, description="Rate limit per hour per API key")
    
    # Cache Configuration
    CACHE_EXPIRY_HOURS: int = Field(default=1, description="Cache expiry time in hours")
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL for caching")
    
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./deacero.db", description="Database URL")
    
    # External APIs (todas las del .env)
    BANXICO_API_TOKEN: str = Field(default="", description="BANXICO API token")
    FRED_API_KEY: str = Field(default="", description="FRED API key")
    ALPHA_VANTAGE_API_KEY: str = Field(default="", description="Alpha Vantage API key")
    TRADING_ECONOMICS_API_KEY: str = Field(default="", description="Trading Economics API key")
    QUANDL_API_KEY: str = Field(default="", description="Quandl API key")
    INEGI_API_TOKEN: str = Field(default="", description="INEGI API token")
    
    # Data Configuration (del .env)
    DATA_START_DATE: str = Field(default="2020-01-01", description="Start date for data collection")
    # NOTA: DATA_END_DATE se calcula dinámicamente en scripts usando datetime.now()
    
    # Model Configuration
    MODEL_PATH: str = Field(default="./data/models/steel_price_model.pkl", description="Path to trained model")
    MODEL_UPDATE_INTERVAL_HOURS: int = Field(default=24, description="Model update interval in hours")
    
    # Data Sources
    DATA_UPDATE_INTERVAL_HOURS: int = Field(default=6, description="Data update interval in hours")
    HISTORICAL_DATA_DAYS: int = Field(default=730, description="Days of historical data to maintain")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="./logs/deacero_api.log", description="Log file path")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Permitir campos extra del .env


# Global settings instance
settings = Settings()
