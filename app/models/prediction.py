"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum


class PredictionResponse(BaseModel):
    """
    Response model for steel rebar price prediction
    Matches the exact format required by DeAcero technical test
    """
    prediction_date: str = Field(..., description="Date of prediction in YYYY-MM-DD format")
    predicted_price_usd: float = Field(..., description="Predicted price in USD per metric ton")
    currency: str = Field(default="USD", description="Currency of the prediction")
    unit: str = Field(default="metric ton", description="Unit of measurement")
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    timestamp: str = Field(..., description="Timestamp of prediction in ISO format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_date": "2025-01-15",
                "predicted_price_usd": 750.45,
                "currency": "USD",
                "unit": "metric ton",
                "model_confidence": 0.85,
                "timestamp": "2025-01-14T10:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_status: Optional[str] = Field(None, description="Model status")
    last_data_update: Optional[str] = Field(None, description="Last data update timestamp")


class RootResponse(BaseModel):
    """Root endpoint response model"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    documentation_url: str = Field(..., description="API documentation URL")
    data_sources: List[str] = Field(..., description="List of data sources used")
    last_model_update: str = Field(..., description="Last model update timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    r2_score: float = Field(..., description="R-squared score")
    training_samples: int = Field(..., description="Number of training samples")
    features_count: int = Field(..., description="Number of features used")
    last_updated: str = Field(..., description="Last model update timestamp")


class DataSourceStatus(BaseModel):
    """Data source status information"""
    source_name: str = Field(..., description="Name of data source")
    last_update: str = Field(..., description="Last successful update")
    status: str = Field(..., description="Current status (active/error/warning)")
    records_count: Optional[int] = Field(None, description="Number of records")
    error_message: Optional[str] = Field(None, description="Error message if any")


class PredictionInput(BaseModel):
    """
    Optional input model for custom prediction parameters
    """
    target_date: Optional[date] = Field(None, description="Target date for prediction")
    include_confidence_interval: bool = Field(default=False, description="Include confidence interval")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_date": "2025-01-15",
                "include_confidence_interval": True,
                "model_version": "v1.0"
            }
        }


class ExtendedPredictionResponse(PredictionResponse):
    """
    Extended prediction response with additional information
    """
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")
    key_factors: Optional[List[Dict[str, Any]]] = Field(None, description="Key factors influencing the prediction")
    market_trend: Optional[str] = Field(None, description="Overall market trend")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_date": "2025-01-15",
                "predicted_price_usd": 750.45,
                "currency": "USD",
                "unit": "metric ton",
                "model_confidence": 0.85,
                "timestamp": "2025-01-14T10:00:00Z",
                "confidence_interval_lower": 720.30,
                "confidence_interval_upper": 780.60,
                "key_factors": [
                    {"factor": "iron_ore_price", "impact": 0.35, "value": 120.50},
                    {"factor": "usd_mxn_rate", "impact": 0.28, "value": 18.45}
                ],
                "market_trend": "upward"
            }
        }


# ========== MODELOS PARA EXPLICABILIDAD V2 ==========

class FeatureFactor(BaseModel):
    """Modelo para un factor individual con su importancia"""
    feature: str = Field(..., description="Nombre del factor")
    average_importance: float = Field(..., description="Importancia promedio entre modelos")
    category: str = Field(..., description="Categoría del factor")
    models_count: int = Field(..., description="Número de modelos que usan este factor")
    models: List[str] = Field(..., description="Lista de modelos que usan este factor")
    description: str = Field(..., description="Descripción del factor")


class FeatureImportanceResponse(BaseModel):
    """Respuesta para análisis de feature importance"""
    models_analyzed: List[str] = Field(..., description="Modelos analizados")
    total_factors_analyzed: int = Field(..., description="Total de factores analizados")
    top_factors: List[FeatureFactor] = Field(..., description="Factores más importantes")
    factors_by_category: Dict[str, List[FeatureFactor]] = Field(..., description="Factores agrupados por categoría")
    analysis_timestamp: str = Field(..., description="Timestamp del análisis")
    requested_model: Optional[str] = Field(None, description="Modelo específico solicitado")
    top_n_requested: int = Field(..., description="Número de factores solicitados")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models_analyzed": ["XGBoost_V2_regime", "MIDAS_V2_hibrida"],
                "total_factors_analyzed": 45,
                "top_factors": [
                    {
                        "feature": "precio_varilla_lme_lag_1",
                        "average_importance": 0.342,
                        "category": "Autorregresivo",
                        "models_count": 2,
                        "models": ["XGBoost_V2_regime", "MIDAS_V2_hibrida"],
                        "description": "Precio de varilla del día anterior"
                    }
                ],
                "factors_by_category": {},
                "analysis_timestamp": "2025-01-14T10:00:00Z",
                "requested_model": None,
                "top_n_requested": 20
            }
        }


class CausalStep(BaseModel):
    """Paso individual en una cadena causal"""
    step: int = Field(..., description="Número del paso")
    mechanism: str = Field(..., description="Mecanismo causal")
    description: str = Field(..., description="Descripción del paso")


class CausalFactor(BaseModel):
    """Factor causal con explicación detallada"""
    factor_name: str = Field(..., description="Nombre del factor")
    importance_score: float = Field(..., description="Puntuación de importancia")
    category: str = Field(..., description="Categoría económica")
    causal_mechanism: str = Field(..., description="Mecanismo causal principal")
    economic_rationale: str = Field(..., description="Justificación económica")
    impact_direction: str = Field(..., description="Dirección del impacto")
    time_horizon: str = Field(..., description="Horizonte temporal del impacto")
    description: str = Field(..., description="Descripción detallada")


class CausalFactorsResponse(BaseModel):
    """Respuesta para análisis de factores causales"""
    total_factors_analyzed: int = Field(..., description="Total de factores analizados")
    factors_returned: int = Field(..., description="Factores retornados")
    causal_factors: List[CausalFactor] = Field(..., description="Factores causales con explicaciones")
    available_categories: List[str] = Field(..., description="Categorías disponibles")
    filter_applied: Dict[str, Any] = Field(..., description="Filtros aplicados")
    economic_context: Dict[str, str] = Field(..., description="Contexto económico general")
    analysis_timestamp: str = Field(..., description="Timestamp del análisis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_factors_analyzed": 45,
                "factors_returned": 15,
                "causal_factors": [
                    {
                        "factor_name": "iron",
                        "importance_score": 0.234,
                        "category": "Materias Primas",
                        "causal_mechanism": "Costo de materia prima",
                        "economic_rationale": "El mineral de hierro representa ~70% del costo de producción del acero",
                        "impact_direction": "Positiva (mayor costo → mayor precio)",
                        "time_horizon": "Corto plazo (1-5 días)",
                        "description": "Precio del mineral de hierro"
                    }
                ],
                "available_categories": ["Autorregresivo", "Materias Primas", "Mercados Financieros"],
                "filter_applied": {"category": None, "min_importance": 0.01},
                "economic_context": {},
                "analysis_timestamp": "2025-01-14T10:00:00Z"
            }
        }


class ModelComparisonResponse(BaseModel):
    """Respuesta para comparación de modelos"""
    models_compared: List[str] = Field(..., description="Modelos comparados")
    comparison: Dict[str, Any] = Field(..., description="Comparación detallada")
    recommendation: Dict[str, str] = Field(..., description="Recomendación de uso")
    comparison_timestamp: str = Field(..., description="Timestamp de la comparación")
