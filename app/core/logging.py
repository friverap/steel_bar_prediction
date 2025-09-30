"""
Logging configuration for DeAcero Steel Price Predictor API
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

from app.core.config import settings


def setup_logging() -> None:
    """
    Setup logging configuration for the application
    """
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Application logger
    app_logger = logging.getLogger("app")
    app_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    app_logger.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"app.{name}")


class APILogger:
    """
    Custom logger for API requests and responses
    """
    
    def __init__(self):
        self.logger = get_logger("api")
    
    def log_request(self, method: str, path: str, client_ip: str, api_key: str = None):
        """Log API request"""
        api_key_masked = api_key[:10] + "..." if api_key else "None"
        self.logger.info(
            f"REQUEST - {method} {path} - IP: {client_ip} - API Key: {api_key_masked}"
        )
    
    def log_response(self, method: str, path: str, status_code: int, response_time: float):
        """Log API response"""
        self.logger.info(
            f"RESPONSE - {method} {path} - Status: {status_code} - Time: {response_time:.3f}s"
        )
    
    def log_prediction(self, predicted_price: float, confidence: float, features_used: int):
        """Log prediction details"""
        self.logger.info(
            f"PREDICTION - Price: ${predicted_price:.2f} - Confidence: {confidence:.3f} - Features: {features_used}"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"ERROR - {context}: {str(error)}", exc_info=True)


# Global API logger instance
api_logger = APILogger()
