"""
Prediction endpoints for steel rebar price forecasting
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from datetime import datetime, timedelta
import logging
from typing import Optional

from app.models.prediction import PredictionResponse, PredictionInput, ExtendedPredictionResponse
from app.services.predictor import SteelPricePredictor
from app.services.cache_manager import CacheManager
from app.core.security import check_rate_limit, verify_api_key, verify_admin_api_key
from app.core.logging import api_logger
from src.ml_pipeline.prediction_cache import get_instant_api_response, PredictionCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
predictor = SteelPricePredictor()
cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get cache manager instance, initialize if needed"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
        cache_manager._start_cleanup_task()
    return cache_manager


@router.get("/steel-rebar-price", response_model=PredictionResponse)
async def predict_steel_rebar_price(
    request: Request,
    force_refresh: bool = False,
    api_key: str = Depends(verify_api_key)
) -> PredictionResponse:
    """
    Predecir precio de varilla corrugada para el día siguiente
    
    Endpoint principal que utiliza el pipeline completo V2 con modelos XGBoost_V2_regime 
    y MIDAS_V2_hibrida para generar predicciones precisas del precio de cierre.
    
    Args:
        force_refresh: Forzar actualización completa del pipeline de datos
    
    Returns:
        PredictionResponse: Predicción con precio, confianza y metadatos V2
    """
    try:
        # Log request
        api_logger.log_request("GET", "/predict/steel-rebar-price", request.client.host, api_key)
        
        # Check rate limiting (configurado en settings.RATE_LIMIT_PER_HOUR = 100)
        await check_rate_limit(request, api_key)
        
        # RESPUESTA INSTANTÁNEA (<2 segundos) desde cache pre-calculado
        if not force_refresh:
            logger.info("⚡ Obteniendo predicción instantánea desde cache...")
            
            try:
                # Usar sistema de cache optimizado para respuesta <2s
                instant_prediction = await get_instant_api_response()
                
                if instant_prediction:
                    # Obtener timestamp de última actualización del modelo
                    model_timestamp = _get_model_last_update_timestamp(instant_prediction.get("best_model"))
                    
                    # Crear respuesta desde cache con timestamp del modelo
                    response = PredictionResponse(
                        prediction_date=instant_prediction["prediction_date"],
                        predicted_price_usd=instant_prediction["predicted_price_usd"],
                        currency=instant_prediction["currency"],
                        unit=instant_prediction["unit"],
                        model_confidence=instant_prediction["model_confidence"],
                        timestamp=model_timestamp  # Timestamp de última actualización del modelo
                    )
                    
                    # Log respuesta instantánea
                    api_logger.log_prediction(
                        response.predicted_price_usd,
                        response.model_confidence,
                        0  # No features usadas en cache
                    )
                    
                    logger.info(f"⚡ Respuesta instantánea ({instant_prediction.get('response_time_seconds', 0):.3f}s):")
                    logger.info(f"   💰 Precio: ${response.predicted_price_usd:.2f}")
                    logger.info(f"   🤖 Modelo: {instant_prediction.get('best_model', 'Unknown')}")
                    logger.info(f"   📦 Desde cache: {instant_prediction.get('from_cache', True)}")
                    
                    return response
                    
            except Exception as cache_error:
                logger.warning(f"⚠️ Error en cache instantáneo: {str(cache_error)}")
        
        # FALLBACK: Generar predicción completa (más lento)
        logger.info("🔄 Cache no disponible - ejecutando pipeline completo...")
        
        # Usar predictor V2 con pipeline integrado
        prediction_result = await predictor.predict_next_day_price(force_pipeline_refresh=force_refresh)
        
        # Crear respuesta en formato requerido
        response = PredictionResponse(
            prediction_date=prediction_result.get("prediction_date", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")),
            predicted_price_usd=prediction_result["price"],
            currency="USD",
            unit="metric ton",
            model_confidence=prediction_result["confidence"],
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Log predicción completa
        api_logger.log_prediction(
            response.predicted_price_usd,
            response.model_confidence,
            prediction_result.get("features_used", 0)
        )
        
        logger.info(f"✅ Predicción completa generada:")
        logger.info(f"   💰 Precio: ${response.predicted_price_usd:.2f}")
        logger.info(f"   📊 Confianza: {response.model_confidence:.1%}")
        logger.info(f"   🤖 Modelo: {prediction_result.get('best_model', 'Unknown')}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generando predicción V2: {str(e)}", exc_info=True)
        api_logger.log_error(e, "steel_rebar_price_prediction_v2")
        
        raise HTTPException(
            status_code=500,
            detail="Error generando predicción de precio. Por favor intente nuevamente."
        )


@router.post("/steel-rebar-price/custom", response_model=ExtendedPredictionResponse)
async def predict_steel_rebar_price_custom(
    prediction_input: PredictionInput,
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> ExtendedPredictionResponse:
    """
    Generate custom steel rebar price prediction with additional parameters
    
    This endpoint allows for more customized predictions with specific dates,
    confidence intervals, and detailed factor analysis.
    
    Args:
        prediction_input: Custom prediction parameters
        
    Returns:
        ExtendedPredictionResponse: Extended prediction with additional details
    """
    try:
        # Log request
        api_logger.log_request("POST", "/predict/steel-rebar-price/custom", request.client.host, api_key)
        
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        # Generate prediction with custom parameters
        prediction_result = await predictor.predict_custom(
            target_date=prediction_input.target_date,
            include_confidence_interval=prediction_input.include_confidence_interval,
            model_version=prediction_input.model_version
        )
        
        # Create extended response
        target_date = prediction_input.target_date or (datetime.now().date() + timedelta(days=1))
        
        response_data = {
            "prediction_date": target_date.strftime("%Y-%m-%d"),
            "predicted_price_usd": prediction_result["price"],
            "currency": "USD",
            "unit": "metric ton",
            "model_confidence": prediction_result["confidence"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Add optional fields if requested
        if prediction_input.include_confidence_interval and "confidence_interval" in prediction_result:
            response_data["confidence_interval_lower"] = prediction_result["confidence_interval"]["lower"]
            response_data["confidence_interval_upper"] = prediction_result["confidence_interval"]["upper"]
        
        if "key_factors" in prediction_result:
            response_data["key_factors"] = prediction_result["key_factors"]
            
        if "market_trend" in prediction_result:
            response_data["market_trend"] = prediction_result["market_trend"]
        
        response = ExtendedPredictionResponse(**response_data)
        
        # Log successful prediction
        api_logger.log_prediction(
            response.predicted_price_usd,
            response.model_confidence,
            prediction_result.get("features_used", 0)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating custom prediction: {str(e)}", exc_info=True)
        api_logger.log_error(e, "custom_steel_rebar_price_prediction")
        
        raise HTTPException(
            status_code=500,
            detail="Error generating custom price prediction. Please try again later."
        )


@router.get("/model/status")
async def get_model_status(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get current model status and performance metrics
    
    Returns:
        Model status, metrics, and last update information
    """
    try:
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        status = await predictor.get_model_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model status"
        )


@router.post("/pipeline/refresh")
async def refresh_pipeline(
    request: Request,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Forzar actualización completa del pipeline de datos (ADMIN ONLY)
    
    ⚠️ Este endpoint requiere ADMIN_API_KEY (no accesible para usuarios finales)
    
    En producción GCP, este endpoint es llamado automáticamente por:
    - Cloud Scheduler (diario a las 18:00 MX)
    - Cloud Function de reentrenamiento
    
    Flujo completo:
    1. Ingesta de datos actualizados
    2. Consolidación y limpieza
    3. Feature engineering
    4. Generación de nuevas predicciones
    5. Actualización de cache
    
    Returns:
        Resultado de la actualización del pipeline
    """
    try:
        # Log request
        api_logger.log_request("POST", "/predict/pipeline/refresh", request.client.host, api_key)
        
        # Check rate limiting (operaciones costosas - usar rate limit estándar)
        await check_rate_limit(request, api_key)
        
        logger.info("🔄 Iniciando refresh forzado del pipeline...")
        
        # Ejecutar refresh completo
        refresh_result = await predictor.force_pipeline_refresh()
        
        if refresh_result['status'] == 'success':
            logger.info("✅ Pipeline actualizado exitosamente")
            
            return {
                "status": "success",
                "message": "Pipeline actualizado exitosamente",
                "refresh_timestamp": refresh_result['refresh_timestamp'],
                "pipeline_summary": refresh_result.get('pipeline_results', {}).get('summary', {}),
                "next_prediction_ready": True
            }
        else:
            logger.error(f"❌ Error en refresh del pipeline: {refresh_result.get('error')}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Error actualizando pipeline: {refresh_result.get('error', 'Unknown')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en refresh del pipeline: {str(e)}", exc_info=True)
        api_logger.log_error(e, "pipeline_refresh")
        
        raise HTTPException(
            status_code=500,
            detail="Error ejecutando refresh del pipeline"
        )


@router.get("/pipeline/status")
async def get_pipeline_status(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Obtener estado detallado del pipeline de producción
    
    Returns:
        Estado completo del pipeline, modelos, y archivos de datos
    """
    try:
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        # Obtener estado completo del servicio V2
        status = await predictor.get_model_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error obteniendo estado del pipeline"
        )


def _get_model_last_update_timestamp(model_name: str) -> str:
    """
    Obtener timestamp de última actualización del modelo
    
    Args:
        model_name: Nombre del modelo (XGBoost_V2_regime, MIDAS_V2_hibrida)
        
    Returns:
        Timestamp ISO format de última modificación del archivo del modelo
    """
    from pathlib import Path
    from datetime import datetime
    
    try:
        # Buscar modelo en múltiples ubicaciones
        models_base = Path(__file__).parent.parent.parent / "models"
        
        model_files = []
        for location in ["production", "test"]:
            location_path = models_base / location
            if location_path.exists():
                # Buscar por nombre específico
                for pattern in [f"{model_name}_latest.pkl", f"{model_name}.pkl"]:
                    model_file = location_path / pattern
                    if model_file.exists():
                        model_files.append(model_file)
        
        if model_files:
            # Obtener el más reciente
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            last_update = datetime.fromtimestamp(latest_model.stat().st_mtime)
            return last_update.isoformat() + "Z"
        else:
            # Si no se encuentra el modelo específico, usar timestamp actual
            logger.warning(f"Modelo {model_name} no encontrado, usando timestamp actual")
            return datetime.utcnow().isoformat() + "Z"
            
    except Exception as e:
        logger.error(f"Error obteniendo timestamp del modelo: {str(e)}")
        # Fallback a timestamp actual
        return datetime.utcnow().isoformat() + "Z"
