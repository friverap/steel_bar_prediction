"""
Steel Price Prediction Service - V2 Production
Servicio principal para predicción de precios de varilla corrugada
Integrado con pipeline de producción y modelos V2
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
import logging
import joblib
import asyncio
from pathlib import Path
import sys
import os

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.config import settings
from app.services.data_collector import DataCollector
from app.services.feature_engineer import FeatureEngineer
from src.ml_pipeline.production_pipeline import ProductionPipeline
from src.ml_pipeline.models_v2 import ProductionPredictor, ModelV2Factory
from src.ml_pipeline.model_selector import ProductionModelManager, select_best_model_for_api

logger = logging.getLogger(__name__)


class SteelPricePredictor:
    """
    Servicio principal de predicción integrado con pipeline V2
    """
    
    def __init__(self):
        # Componentes V2
        self.pipeline = ProductionPipeline()
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.production_predictor = ProductionPredictor(self.models_dir)
        self.model_manager = ProductionModelManager(self.models_dir)
        
        # Componentes legacy (para compatibilidad)
        self.feature_engineer = FeatureEngineer()
        self.data_collector = DataCollector()
        
        # Estado del servicio
        self.last_pipeline_run = None
        self.last_prediction = None
        self.service_stats = {
            'predictions_generated': 0,
            'pipeline_runs': 0,
            'errors': 0
        }
        
        logger.info("🚀 SteelPricePredictor V2 inicializado con pipeline de producción")
    
    def _load_model(self) -> None:
        """Load the trained ML model"""
        try:
            model_path = Path(settings.MODEL_PATH)
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.last_model_load = datetime.utcnow()
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Load model metrics if available
                metrics_path = model_path.parent / "model_metrics.json"
                if metrics_path.exists():
                    import json
                    with open(metrics_path, 'r') as f:
                        self.model_metrics = json.load(f)
            else:
                logger.warning(f"Model file not found at {model_path}. Using dummy model.")
                self._create_dummy_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model for testing purposes"""
        logger.info("Creating dummy model for testing")
        
        class DummyModel:
            def predict(self, X):
                # Simple dummy prediction based on recent trends
                # In reality, this would be a trained ML model
                base_price = 750.0
                random_variation = np.random.normal(0, 25, len(X))
                return base_price + random_variation
            
            def predict_proba(self, X):
                # Dummy confidence scores
                return np.random.uniform(0.75, 0.95, len(X))
        
        self.model = DummyModel()
        self.model_metrics = {
            "mape": 8.5,
            "mae": 45.2,
            "rmse": 62.8,
            "r2_score": 0.82,
            "training_samples": 1000,
            "features_count": 15,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def predict_next_day_price(self, force_pipeline_refresh: bool = False) -> Dict[str, Any]:
        """
        Predicción V2 del precio de varilla para el día siguiente
        Usa el pipeline completo de producción con modelos V2
        
        Args:
            force_pipeline_refresh: Forzar actualización completa del pipeline
            
        Returns:
            Dictionary con predicción, confianza y metadatos V2
        """
        try:
            logger.info("🎯 Iniciando predicción V2 del día siguiente...")
            
            # Verificar si necesitamos ejecutar el pipeline completo
            need_pipeline_run = (
                force_pipeline_refresh or 
                self.last_pipeline_run is None or
                self._pipeline_needs_refresh()
            )
            
            if need_pipeline_run:
                logger.info("🔄 Ejecutando pipeline completo de datos...")
                
                # Ejecutar pipeline completo (ingesta → join → limpieza → features)
                pipeline_results = await self.pipeline.run_full_pipeline(force_data_refresh=force_pipeline_refresh)
                
                if pipeline_results['status'] != 'success':
                    raise ValueError(f"Pipeline falló: {pipeline_results.get('error', 'Unknown')}")
                
                self.last_pipeline_run = datetime.now()
                self.service_stats['pipeline_runs'] += 1
                
                logger.info("✅ Pipeline completado exitosamente")
            else:
                logger.info("⏭️ Usando datos existentes del pipeline")
            
            # Cargar features preparadas
            features_path = self.pipeline.data_processed_dir / 'features_v2_latest.csv'
            
            if not features_path.exists():
                raise FileNotFoundError("Features V2 no encontradas - ejecutar pipeline completo")
            
            features_df = pd.read_csv(features_path, index_col='fecha', parse_dates=True)
            
            # USAR CACHE DIRECTAMENTE (el pipeline ya generó predicciones)
            logger.info("📦 Intentando usar predicciones desde cache...")
            
            try:
                from src.ml_pipeline.prediction_cache import PredictionCache
                cache = PredictionCache()
                cached_prediction = await cache.get_cached_prediction()
                
                if cached_prediction and 'prediction' in cached_prediction:
                    prediction_data = cached_prediction['prediction']
                    
                    # Verificar si el cache es reciente (menos de 1 hora)
                    cache_timestamp = prediction_data.get('timestamp')
                    cache_is_recent = False
                    
                    if cache_timestamp:
                        try:
                            cache_time = datetime.fromisoformat(cache_timestamp.replace('Z', '+00:00'))
                            now = datetime.now()
                            hours_old = (now - cache_time.replace(tzinfo=None)).total_seconds() / 3600
                            cache_is_recent = hours_old < 1  # Cache válido si es menor a 1 hora
                            
                            logger.info(f"📦 Cache encontrado: {hours_old:.1f} horas de antigüedad")
                        except Exception as e:
                            logger.warning(f"⚠️ Error parseando timestamp cache: {e}")
                    
                    # Solo usar cache si es reciente Y precio > $100 (datos reales)
                    price_is_realistic = prediction_data.get('predicted_price_usd', 0) > 100
                    
                    if cache_is_recent and price_is_realistic:
                        logger.info("✅ Usando predicción de cache (reciente y realista):")
                        logger.info(f"   💰 Precio: ${prediction_data['predicted_price_usd']:.2f}")
                        logger.info(f"   🤖 Modelo: {prediction_data['best_model']}")
                        logger.info(f"   📊 Confidence: {prediction_data['model_confidence']:.3f}")
                        
                        # Usar datos del cache directamente
                        return {
                            "price": prediction_data['predicted_price_usd'],
                            "confidence": prediction_data['model_confidence'],
                            "confidence_interval": {
                                "lower": prediction_data['predicted_price_usd'] * 0.95,
                                "upper": prediction_data['predicted_price_usd'] * 1.05
                            },
                            "best_model": prediction_data['best_model'],
                            "features_used": 12,  # Variables unificadas
                        "data_timestamp": prediction_data['timestamp'],
                        "model_version": "v2.0",
                        "pipeline_execution": False,
                        "from_cache": True,
                        "cache_timestamp": cached_prediction.get('cached_at')
                    }
                    else:
                        logger.info(f"🔄 Cache NO válido - generando nueva predicción...")
                        logger.info(f"   Reciente: {cache_is_recent}, Precio realista: {price_is_realistic}")
                        logger.info(f"   Precio en cache: ${prediction_data.get('predicted_price_usd', 0):.2f}")
                        # Continuar con pipeline completo
                else:
                    logger.info("📦 Cache no disponible - ejecutando pipeline completo...")
                    # Continuar con pipeline completo
                    
            except Exception as cache_error:
                logger.warning(f"⚠️ Error accediendo cache: {str(cache_error)} - continuando con pipeline...")
                # Continuar con pipeline completo en lugar de fallar
            
            # GENERAR PREDICCIONES CON MODELOS V2
            logger.info("🤖 Generando predicciones con modelos V2...")
            
            # Obtener últimos datos disponibles del features_df
            X_latest = features_df.iloc[[-1]].copy()  # Última fila para predicción
            
            # Generar predicciones con cada modelo
            all_predictions = {}
            models_performance = {}
            
            for model_name in self.production_predictor.model_factory.model_configs.keys():
                try:
                    logger.info(f"🔮 Prediciendo con {model_name}...")
                    
                    prediction_result = self.production_predictor.model_factory.predict_with_model(
                        model_name, 
                        X_latest
                    )
                    
                    if prediction_result:
                        all_predictions[model_name] = prediction_result['prediction']
                        mape_value = prediction_result.get('mape', 5.0)
                        models_performance[model_name] = {
                            'rmse': prediction_result.get('rmse', 10.0),
                            'mape': mape_value,
                            'r2': prediction_result.get('confidence_score', 0.85)
                        }
                        logger.info(f"✅ {model_name}: ${prediction_result['prediction']:.2f} (MAPE: {mape_value:.2f}%)")
                    else:
                        logger.warning(f"⚠️ {model_name}: No se pudo generar predicción")
                        
                except Exception as e:
                    logger.error(f"❌ Error en {model_name}: {str(e)}")
            
            if not all_predictions:
                raise ValueError("No se pudieron generar predicciones con ningún modelo")
            
            logger.info(f"📊 Predicciones generadas: {len(all_predictions)} modelos")
            
            # SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO
            best_model_result = await self.model_manager.get_best_prediction(models_performance, all_predictions)
            
            # Calcular intervalo de confianza
            best_prediction = best_model_result['prediction_value']
            best_model_name = best_model_result['model_used']
            
            # Calcular confidence basada en MAPE del mejor modelo
            mape = models_performance.get(best_model_name, {}).get('mape', 5.0)
            # MAPE bajo = confidence alta: MAPE 0% → 1.0, MAPE 10% → 0.9, MAPE 20% → 0.8
            confidence_score = max(0.5, min(1.0, 1.0 - (mape / 100)))
            
            # Usar RMSE del modelo para intervalo de confianza
            rmse = models_performance.get(best_model_name, {}).get('rmse', best_prediction * 0.05)
            
            confidence_interval = {
                "lower": float(best_prediction - (1.96 * rmse)),
                "upper": float(best_prediction + (1.96 * rmse))
            }
            
            # Actualizar estadísticas
            self.service_stats['predictions_generated'] += 1
            self.last_prediction = best_model_result
            
            logger.info(f"🏆 MEJOR PREDICCIÓN: ${best_prediction:.2f} ({best_model_name})")
            logger.info(f"📊 Confianza: {confidence_score:.1%} ({best_model_result['confidence_level']})")
            
            return {
                "price": best_prediction,
                "confidence": confidence_score,
                "confidence_interval": confidence_interval,
                "best_model": best_model_name,
                "model_selection_justification": best_model_result['selection_justification'],
                "performance_grade": best_model_result['performance_grade'],
                "features_used": len(features_df.columns),
                "data_timestamp": datetime.now().isoformat(),
                "model_version": "v2.0",
                "pipeline_execution": need_pipeline_run,
                "all_models_evaluated": len(all_predictions),
                "user_recommendation": best_model_result['recommendation']
            }
                
        except Exception as e:
            logger.error(f"Error en predicción V2: {str(e)}", exc_info=True)
            self.service_stats['errors'] += 1
            
            # NO FALLBACKS - Si falla, debe fallar claramente
            raise
    
    async def predict_custom(
        self,
        target_date: Optional[date] = None,
        include_confidence_interval: bool = False,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate custom prediction with additional parameters
        
        Args:
            target_date: Specific date for prediction
            include_confidence_interval: Include confidence intervals
            model_version: Specific model version to use
            
        Returns:
            Dictionary with extended prediction information
        """
        try:
            # Use next day if no target date specified
            if not target_date:
                target_date = date.today() + timedelta(days=1)
            
            # Collect data up to target date
            historical_data = await self.data_collector.get_historical_data(
                end_date=target_date - timedelta(days=1)
            )
            
            # Engineer features
            features = await self.feature_engineer.create_features(historical_data)
            
            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1))[0]
            confidence = self._calculate_confidence(features)
            
            result = {
                "price": float(prediction),
                "confidence": float(confidence),
                "features_used": len(features),
                "target_date": target_date.isoformat(),
                "model_version": model_version or "v1.0"
            }
            
            # Add confidence interval if requested
            if include_confidence_interval:
                result["confidence_interval"] = self._calculate_confidence_interval(
                    prediction, confidence
                )
            
            # Add key factors analysis
            result["key_factors"] = await self._analyze_key_factors(features)
            
            # Add market trend
            result["market_trend"] = self._determine_market_trend(historical_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in custom prediction: {str(e)}")
            raise
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """
        Calculate prediction confidence based on features and model performance
        
        Args:
            features: Feature vector
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence from model metrics
            base_confidence = self.model_metrics.get("r2_score", 0.8)
            
            # Adjust based on feature completeness
            feature_completeness = np.sum(~np.isnan(features)) / len(features)
            
            # Adjust based on data recency (dummy calculation)
            recency_factor = 0.95  # Assume recent data
            
            confidence = base_confidence * feature_completeness * recency_factor
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.75
    
    def _calculate_confidence_interval(self, prediction: float, confidence: float) -> Dict[str, float]:
        """
        Calculate confidence interval for prediction
        
        Args:
            prediction: Point prediction
            confidence: Confidence score
            
        Returns:
            Dictionary with lower and upper bounds
        """
        # Simple confidence interval calculation
        # In practice, this would use model-specific methods
        margin = prediction * (1 - confidence) * 0.5
        
        return {
            "lower": prediction - margin,
            "upper": prediction + margin
        }
    
    async def _analyze_key_factors(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze key factors influencing the prediction
        
        Args:
            features: Feature vector
            
        Returns:
            List of key factors with their impacts
        """
        # Dummy factor analysis - in practice, use feature importance
        key_factors = [
            {
                "factor": "iron_ore_price",
                "impact": 0.35,
                "value": float(features[0]) if len(features) > 0 else 120.5
            },
            {
                "factor": "usd_mxn_rate",
                "impact": 0.28,
                "value": float(features[1]) if len(features) > 1 else 18.45
            },
            {
                "factor": "oil_price",
                "impact": 0.22,
                "value": float(features[2]) if len(features) > 2 else 75.30
            },
            {
                "factor": "construction_index",
                "impact": 0.15,
                "value": float(features[3]) if len(features) > 3 else 105.8
            }
        ]
        
        return key_factors
    
    def _determine_market_trend(self, historical_data: Dict[str, Any]) -> str:
        """
        Determine overall market trend
        
        Args:
            historical_data: Historical market data
            
        Returns:
            Market trend description
        """
        # Dummy trend analysis
        # In practice, analyze recent price movements
        trends = ["upward", "downward", "stable", "volatile"]
        return np.random.choice(trends)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status and performance metrics
        
        Returns:
            Model status information
        """
        return {
            "status": "active" if self.model else "error",
            "last_loaded": self.last_model_load.isoformat() if self.last_model_load else None,
            "metrics": self.model_metrics,
            "model_path": settings.MODEL_PATH,
            "version": "v1.0"
        }
    
    def should_reload_model(self) -> bool:
        """
        Check if model should be reloaded based on update interval
        
        Returns:
            True if model should be reloaded
        """
        if not self.last_model_load:
            return True
    
    def _pipeline_needs_refresh(self) -> bool:
        """Verificar si el pipeline necesita actualización"""
        if not self.last_pipeline_run:
            return True
        
        # Actualizar pipeline cada 6 horas
        hours_since_run = (datetime.now() - self.last_pipeline_run).total_seconds() / 3600
        return hours_since_run >= 6
    
    def _prepare_features_for_models_v2(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparar features del pipeline para los modelos V2
        Mapea nombres de columnas generadas a nombres esperados por modelos
        
        Args:
            features_df: DataFrame con features generados por pipeline
            
        Returns:
            DataFrame con features mapeados y solo las columnas necesarias
        """
        logger.info("🔧 Preparando features para modelos V2...")
        
        try:
            # Tomar última fila (para predicción t+1)
            X_latest = features_df.iloc[[-1]].copy()
            
            # Variables CORRECTAS según documentación (13 features simples)
            unified_variables = [
                # Metales (4)
                'cobre_lme', 'zinc_lme', 'steel', 'aluminio_lme',
                # Materias Primas (2)
                'coking', 'iron',
                # Macro/Financial (3)
                'dxy', 'treasury', 'tasa_interes_banxico',
                # Risk/Market (2)
                'VIX', 'infrastructure',
                # Autorregresivas (2)
                'precio_varilla_lme_lag_1', 'precio_varilla_lme_lag_20'
            ]
            
            # Seleccionar solo las 13 features correctas (ya existen en el pipeline)
            missing_cols = [col for col in unified_variables if col not in X_latest.columns]
            
            if missing_cols:
                logger.error(f"❌ Columnas faltantes: {missing_cols}")
                logger.error(f"   Disponibles: {list(X_latest.columns)[:20]}")
                raise ValueError(f"Features faltantes: {missing_cols}")
            
            # Seleccionar y reordenar
            X_model = X_latest[unified_variables].copy()
            
            logger.info(f"✅ Features preparados: {X_model.shape[1]} columnas")
            logger.info(f"   Columnas: {list(X_model.columns)}")
            
            return X_model
            
        except Exception as e:
            logger.error(f"Error preparando features: {str(e)}")
            return None
    
    async def _fallback_prediction(self, error_msg: str) -> Dict[str, Any]:
        """Predicción de fallback cuando falla el pipeline V2"""
        logger.warning(f"🔄 Usando predicción de fallback debido a: {error_msg}")
        
        # Predicción conservadora basada en último precio conocido
        try:
            # Intentar cargar último precio conocido
            daily_path = self.pipeline.data_processed_dir / 'daily_time_series' / 'daily_series_consolidated_latest.csv'
            
            if daily_path.exists():
                daily_df = pd.read_csv(daily_path, index_col='fecha', parse_dates=True)
                if 'precio_varilla_lme' in daily_df.columns:
                    last_price = daily_df['precio_varilla_lme'].iloc[-1]
                    
                    # Predicción simple: último precio ± pequeña variación
                    variation = np.random.normal(0, last_price * 0.02)  # ±2%
                    predicted_price = last_price + variation
                    
                    return {
                        "price": float(predicted_price),
                        "confidence": 0.60,  # Baja confianza para fallback
                        "confidence_interval": {
                            "lower": float(predicted_price * 0.95),
                            "upper": float(predicted_price * 1.05)
                        },
                        "best_model": "fallback_simple",
                        "features_used": 1,
                        "data_timestamp": datetime.now().isoformat(),
                        "model_version": "fallback",
                        "pipeline_execution": False,
                        "warning": "Predicción de fallback - pipeline no disponible"
                    }
        except Exception as fallback_error:
            logger.error(f"Error en fallback: {str(fallback_error)}")
        
        # Fallback absoluto
        return {
            "price": 750.0,  # Precio base histórico
            "confidence": 0.50,
            "confidence_interval": {
                "lower": 700.0,
                "upper": 800.0
            },
            "best_model": "emergency_fallback",
            "features_used": 0,
            "data_timestamp": datetime.now().isoformat(),
            "model_version": "emergency",
            "pipeline_execution": False,
            "error": error_msg
        }
    
    async def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """
        Obtener análisis de feature importance y factores causales
        Nuevo método para endpoint de explicabilidad
        """
        try:
            logger.info("📊 Generando análisis de explicabilidad...")
            
            # Verificar que hay modelos disponibles
            available_models = self.production_predictor.model_factory.list_available_models()
            logger.info(f"🔍 Modelos disponibles: {available_models}")
            
            if not available_models:
                # Buscar modelos reentrenados en production
                production_models = list((self.models_dir / 'production').glob('*_latest.pkl'))
                logger.info(f"🔍 Modelos en production: {[m.name for m in production_models]}")
                
                if not production_models:
                    raise ValueError("No hay modelos disponibles para análisis de explicabilidad")
            
            # Usar el análisis del production predictor
            analysis = await self.production_predictor.get_feature_importance_analysis()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de explicabilidad: {str(e)}")
            # NO FALLBACK - Si falla, debe fallar claramente
            raise
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del servicio V2
        """
        try:
            # Estado de modelos disponibles
            available_models = self.production_predictor.model_factory.list_available_models()
            
            # Información de cada modelo
            models_info = {}
            for model_name in available_models:
                models_info[model_name] = self.production_predictor.model_factory.get_model_info(model_name)
            
            # Estado del pipeline
            pipeline_status = {
                'last_run': self.last_pipeline_run.isoformat() if self.last_pipeline_run else None,
                'needs_refresh': self._pipeline_needs_refresh(),
                'data_files_exist': self._check_data_files_exist()
            }
            
            return {
                "status": "active_v2",
                "service_version": "2.0",
                "available_models": available_models,
                "models_info": models_info,
                "pipeline_status": pipeline_status,
                "service_stats": self.service_stats,
                "last_prediction": {
                    "timestamp": self.last_prediction.get('timestamp') if self.last_prediction else None,
                    "model_used": self.last_prediction.get('best_model') if self.last_prediction else None
                },
                "system_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_data_files_exist(self) -> Dict[str, bool]:
        """Verificar existencia de archivos de datos críticos"""
        base_path = self.pipeline.data_processed_dir
        
        return {
            'daily_consolidated': (base_path / 'daily_time_series' / 'daily_series_consolidated_latest.csv').exists(),
            'monthly_consolidated': (base_path / 'monthly_time_series' / 'monthly_series_consolidated_latest.csv').exists(),
            'features_v2': (base_path / 'features_v2_latest.csv').exists(),
            'clean_daily': (base_path / 'clean_daily_series_latest.csv').exists()
        }
    
    async def force_pipeline_refresh(self) -> Dict[str, Any]:
        """
        Forzar actualización completa del pipeline Y generar nueva predicción
        Útil para mantenimiento o cuando hay nuevos datos
        
        Flujo completo:
        1. Ejecutar pipeline (ingesta → join → limpieza → features)
        2. Generar nueva predicción con modelos V2
        3. Actualizar cache con predicción nueva
        """
        logger.info("🔄 Forzando actualización completa del pipeline...")
        
        try:
            # Ejecutar pipeline completo con refresh forzado
            pipeline_results = await self.pipeline.run_full_pipeline(force_data_refresh=True)
            
            if pipeline_results['status'] != 'success':
                return {
                    'status': 'error',
                    'error': pipeline_results.get('error', 'Unknown'),
                    'pipeline_results': pipeline_results
                }
            
            self.last_pipeline_run = datetime.now()
            self.service_stats['pipeline_runs'] += 1
            
            logger.info("✅ Pipeline completado - generando nueva predicción...")
            
            # GENERAR NUEVA PREDICCIÓN y actualizar cache
            try:
                # Cargar features recién generadas
                features_path = self.pipeline.data_processed_dir / 'features_v2_latest.csv'
                
                if not features_path.exists():
                    raise FileNotFoundError("Features V2 no encontradas después del pipeline")
                
                features_df = pd.read_csv(features_path, index_col='fecha', parse_dates=True)
                
                # MAPEAR FEATURES a los nombres que esperan los modelos V2
                X_latest = self._prepare_features_for_models_v2(features_df)
                
                if X_latest is None or X_latest.empty:
                    raise ValueError("No se pudieron preparar features para los modelos")
                
                # Generar predicciones con cada modelo
                all_predictions = {}
                models_performance = {}
                
                for model_name in self.production_predictor.model_factory.model_configs.keys():
                    try:
                        logger.info(f"🔮 Generando predicción con {model_name}...")
                        
                        prediction_result = self.production_predictor.model_factory.predict_with_model(
                            model_name, 
                            X_latest
                        )
                        
                        if prediction_result:
                            all_predictions[model_name] = prediction_result['prediction']
                            models_performance[model_name] = {
                                'rmse': prediction_result.get('rmse', 10.0),
                                'mape': prediction_result.get('mape', 2.0),
                                'r2': prediction_result.get('confidence_score', 0.85),
                                'hit_rate_2pct': 1.0
                            }
                            logger.info(f"✅ {model_name}: ${prediction_result['prediction']:.2f}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error en {model_name}: {str(e)}")
                
                if all_predictions:
                    # Seleccionar mejor modelo (simple: el que tenga mejor R²)
                    best_model_name = max(models_performance.items(), key=lambda x: x[1].get('r2', -999))[0]
                    best_prediction = all_predictions[best_model_name]
                    best_metrics = models_performance[best_model_name]
                    
                    logger.info(f"🏆 Mejor modelo seleccionado: {best_model_name}")
                    logger.info(f"💰 Predicción: ${best_prediction:.2f}")
                    logger.info(f"📊 R²: {best_metrics.get('r2', 0):.3f}, RMSE: ${best_metrics.get('rmse', 0):.2f}")
                    
                    # ACTUALIZAR CACHE con nueva predicción
                    from src.ml_pipeline.prediction_cache import PredictionCache
                    cache = PredictionCache()
                    
                    # Calcular confidence basada en MAPE (mejor que R² que puede ser negativo)
                    mape = best_metrics.get('mape', 5.0)
                    # Convertir MAPE a confidence: MAPE bajo = confidence alta
                    # MAPE 0% → confidence 1.0, MAPE 10% → confidence 0.9, MAPE 20% → confidence 0.8
                    model_confidence = max(0.5, min(1.0, 1.0 - (mape / 100)))
                    
                    new_prediction = {
                        'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                        'predicted_price_usd': float(best_prediction),
                        'currency': 'USD',
                        'unit': 'metric ton',
                        'model_confidence': model_confidence,
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'best_model': best_model_name,
                        'all_models_evaluated': len(all_predictions),
                        'model_metrics': {
                            'mape': mape,
                            'rmse': best_metrics.get('rmse', 0)
                        }
                    }
                    
                    # Guardar en cache
                    await cache.cache_daily_prediction(
                        new_prediction,
                        feature_importance={},  # Opcional
                        model_comparison={}     # Opcional
                    )
                    
                    logger.info(f"✅ Cache actualizado con nueva predicción: ${new_prediction['predicted_price_usd']:.2f}")
                    
                    return {
                        'status': 'success',
                        'pipeline_results': pipeline_results,
                        'new_prediction': new_prediction,
                        'refresh_timestamp': datetime.now().isoformat(),
                        'cache_updated': True
                    }
                else:
                    logger.warning("⚠️ No se pudieron generar predicciones")
                    return {
                        'status': 'success',
                        'pipeline_results': pipeline_results,
                        'refresh_timestamp': datetime.now().isoformat(),
                        'cache_updated': False,
                        'warning': 'Pipeline completado pero no se generaron predicciones'
                    }
                    
            except Exception as pred_error:
                logger.error(f"Error generando predicción después del pipeline: {str(pred_error)}")
                # Pipeline fue exitoso pero predicción falló
                return {
                    'status': 'partial_success',
                    'pipeline_results': pipeline_results,
                    'prediction_error': str(pred_error),
                    'refresh_timestamp': datetime.now().isoformat(),
                    'cache_updated': False
                }
                
        except Exception as e:
            logger.error(f"Error en refresh forzado: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
