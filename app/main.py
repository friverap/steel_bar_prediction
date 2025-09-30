"""
DeAcero Steel Price Predictor API
FastAPI application for predicting steel rebar prices
"""

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging
from pathlib import Path
import pickle
import os

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.endpoints import predict, explainability
from app.core.security import verify_api_key

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="DeAcero Steel Rebar Price Predictor",
    description="API REST que predice el precio de cierre del día siguiente para la varilla corrugada",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predict.router,
    prefix="/predict",
    tags=["predictions"],
    dependencies=[Depends(verify_api_key)]
)

app.include_router(
    explainability.router,
    prefix="/explainability",
    tags=["explainability"],
    dependencies=[Depends(verify_api_key)]
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    
    # Variables utilizadas para predicción (según documentación técnica)
    predictor_variables = [
        # Metales Base (4)
        "cobre_lme",                    # Cobre - London Metal Exchange
        "zinc_lme",                     # Zinc - London Metal Exchange  
        "steel",                        # Índice acero LME
        "aluminio_lme",                 # Aluminio - London Metal Exchange
        # Materias Primas (2)
        "coking",                       # Carbón metalúrgico
        "iron",                         # Mineral de hierro
        # Macro/Financial (3)
        "dxy",                          # Índice Dólar US
        "treasury",                     # Treasury Yield 10Y
        "tasa_interes_banxico",         # Tasa de referencia Banxico
        # Risk/Market (2)
        "VIX",                          # Índice de Volatilidad
        "infrastructure",               # Índice de Infraestructura
        # Autorregresivas (2)
        "precio_varilla_lme_lag_1",    # Precio t-1
        "precio_varilla_lme_lag_20"    # Precio t-20
    ]
    
    # Obtener fecha de última actualización de modelos
    last_update = "No disponible"
    try:
        models_dir = Path(__file__).parent.parent / "models"
        
        # Buscar modelos en múltiples ubicaciones
        model_files = []
        for location in ["production", "test"]:
            location_path = models_dir / location
            if location_path.exists():
                model_files.extend(list(location_path.glob("*.pkl")))
        
        if model_files:
            # Obtener el modelo más reciente
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            last_update_timestamp = datetime.fromtimestamp(latest_model.stat().st_mtime)
            last_update = last_update_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"No se pudo obtener fecha de actualización de modelos: {str(e)}")
    
    return {
        "service": "Steel Rebar Price Predictor",
        "version": "2.0.0",
        "description": "API REST para predicción del precio de cierre del día siguiente (t+1) de varilla corrugada",
        "documentation_url": "/docs",
        "target_variable": "precio_varilla_lme (Steel Rebar - USD/metric ton)",
        "predictor_variables": predictor_variables,
        "total_variables": len(predictor_variables),
        "models_available": ["XGBoost_V2_regime", "MIDAS_V2_hibrida"],
        "last_model_update": last_update,
        "prediction_horizon": "t+1 (próximo día hábil)",
        "endpoints": {
            "prediction": "/predict/steel-rebar-price",
            "explainability": "/explainability/feature-importance",
            "causal_factors": "/explainability/causal-factors",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
