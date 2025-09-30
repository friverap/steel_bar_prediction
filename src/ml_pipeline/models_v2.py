#!/usr/bin/env python3
"""
Modelos V2 para Producción - DeAcero Steel Price Predictor
Implementación exacta de los modelos XGBoost_V2_regime y MIDAS_V2_hibrida
optimizados para predicción de precios de varilla corrugada

Estos modelos replican exactamente la implementación exitosa del notebook 03_AB_TESTING.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Imports condicionales para modelos específicos
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logging.warning("XGBoost no disponible")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logging.warning("LightGBM no disponible")

logger = logging.getLogger(__name__)


class ModelV2Factory:
    """
    Factory para crear y cargar modelos V2 optimizados para producción
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        # CONFIGURACIÓN CORRECTA según documentación (features simples)
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
        
        self.model_configs = {
            'XGBoost_V2_regime': {
                'variables': unified_variables,
                'type': 'xgboost',
                'description': 'XGBoost con variables unificadas SIN data leakage',
                'features_expected': unified_variables
            },
            'MIDAS_V2_hibrida': {
                'variables': unified_variables,
                'type': 'midas', 
                'description': 'MIDAS con variables unificadas SIN data leakage',
                'features_expected': unified_variables
            }
        }
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Cargar modelo entrenado desde archivo
        
        Args:
            model_name: Nombre del modelo (XGBoost_V2_regime, MIDAS_V2_hibrida)
            
        Returns:
            Diccionario con modelo y metadatos
        """
        logger.info(f"📂 Cargando modelo: {model_name}")
        
        if model_name in self.loaded_models:
            logger.info(f"✅ Modelo {model_name} ya cargado en memoria")
            return self.loaded_models[model_name]
        
        # Buscar archivo de modelo en múltiples ubicaciones (priorizar más recientes)
        locations_to_check = [
            self.models_dir / 'production' / f"{model_name}_latest.pkl",     # Más reciente
            self.models_dir / 'production' / f"{model_name}.pkl",            # Production
            self.models_dir / f"{model_name}_latest.pkl",                    # Latest en root
            self.models_dir / f"{model_name}.pkl",                           # Root
            self.models_dir / 'test' / f"{model_name}.pkl"                   # Test (fallback)
        ]
        
        model_file = None
        for location in locations_to_check:
            if location.exists():
                model_file = location
                logger.info(f"📂 Modelo encontrado en: {location}")
                break
        
        if not model_file:
            logger.error(f"❌ Modelo {model_name} no encontrado en ninguna ubicación")
            logger.error(f"🔍 Ubicaciones verificadas: {[str(loc) for loc in locations_to_check]}")
            return None
        
        try:
            # Cargar modelo
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validar estructura del modelo
            required_keys = ['model', 'scalers', 'test_metrics']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                logger.warning(f"⚠️ Modelo {model_name} falta claves: {missing_keys}")
            
            # Agregar configuración
            model_data['config'] = self.model_configs.get(model_name, {})
            model_data['loaded_at'] = datetime.now().isoformat()
            model_data['file_path'] = str(model_file)
            
            # Cachear en memoria
            self.loaded_models[model_name] = model_data
            
            logger.info(f"✅ Modelo {model_name} cargado exitosamente")
            logger.info(f"   📊 MAPE: {model_data.get('test_metrics', {}).get('mape', 'N/A'):.2f}%")
            logger.info(f"   📊 R²: {model_data.get('test_metrics', {}).get('r2', 'N/A'):.3f}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {model_name}: {str(e)}")
            return None
    
    def predict_with_model(self, model_name: str, X: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generar predicción con modelo específico
        
        Args:
            model_name: Nombre del modelo
            X: Features preparadas
            
        Returns:
            Diccionario con predicción y metadatos
        """
        logger.info(f"🔮 Generando predicción con {model_name}")
        
        # Cargar modelo si no está en memoria
        model_data = self.load_model(model_name)
        if model_data is None:
            return None
        
        try:
            # Preparar datos según el tipo de modelo
            if model_name.startswith('XGBoost'):
                return self._predict_xgboost_v2(model_data, X, model_name)
            elif model_name.startswith('MIDAS'):
                return self._predict_midas_v2(model_data, X, model_name)
            else:
                logger.error(f"Tipo de modelo no reconocido: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error en predicción con {model_name}: {str(e)}")
            return None
    
    def _predict_xgboost_v2(self, model_data: Dict[str, Any], X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Predicción específica para XGBoost V2"""
        
        # Obtener configuración del modelo
        config = model_data['config']
        required_vars = config.get('variables', [])
        
        # Verificar si X ya tiene exactamente las features necesarias
        has_all_required = all(var in X.columns for var in required_vars)
        
        if has_all_required and set(X.columns) == set(required_vars):
            # Features ya están preparadas correctamente - usarlas directamente
            logger.info(f"✅ Features ya preparadas correctamente para {model_name}")
            X_model = X[required_vars].iloc[-1:].copy()
        else:
            # Preparar features usando función helper
            logger.info(f"🔧 Preparando features para {model_name}")
            X_model = self._prepare_xgboost_features(X, required_vars)
        
        # Verificar que tenemos las features necesarias
        if X_model.empty:
            raise ValueError(f"No se pudieron preparar features para {model_name}")
        
        # Usar modelo y scalers
        model = model_data['model']
        scalers = model_data['scalers']
        
        # Escalar features
        X_scaled = scalers['X'].transform(X_model)
        
        # Generar predicción
        y_pred_scaled = model.predict(X_scaled)
        
        # Desescalar predicción
        y_pred = scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Calcular métricas de confianza
        test_metrics = model_data.get('test_metrics', {})
        confidence_score = test_metrics.get('r2', 0.8)
        rmse = test_metrics.get('rmse', 0.05)
        
        # Intervalo de confianza (95%)
        confidence_interval = {
            'lower': float(y_pred - (1.96 * rmse)),
            'upper': float(y_pred + (1.96 * rmse))
        }
        
        # Feature importance si está disponible
        feature_importance = []
        if 'feature_importance' in model_data:
            importance_df = model_data['feature_importance']
            feature_importance = importance_df.head(10).to_dict('records')
        
        return {
            'model_name': model_name,
            'model_type': 'XGBoost_V2',
            'prediction': float(y_pred),
            'confidence_score': float(confidence_score),
            'confidence_interval': confidence_interval,
            'rmse': float(rmse),
            'mape': test_metrics.get('mape', 0.0),
            'features_used': list(X_model.columns),
            'feature_importance': feature_importance,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _predict_midas_v2(self, model_data: Dict[str, Any], X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Predicción específica para MIDAS V2"""
        
        # Para MIDAS, usar features pre-procesadas si están disponibles
        if 'midas_features' in model_data:
            # Usar las features MIDAS específicas del modelo entrenado
            midas_features = model_data['midas_features']
            if 'test' in midas_features:
                # Tomar la estructura de features del test set
                test_features = midas_features['test']
                if isinstance(test_features, pd.DataFrame):
                    # Usar la misma estructura pero con datos actuales
                    X_model = self._align_features_to_midas(X, test_features.columns)
                else:
                    # Crear features MIDAS dinámicamente
                    X_model = self._create_midas_features_dynamic(X)
            else:
                X_model = self._create_midas_features_dynamic(X)
        else:
            X_model = self._create_midas_features_dynamic(X)
        
        if X_model.empty:
            raise ValueError(f"No se pudieron preparar features MIDAS para {model_name}")
        
        # Usar modelo y scalers
        model = model_data['model']
        scalers = model_data['scalers']
        
        # Escalar
        X_scaled = scalers['X'].transform(X_model)
        
        # Predicción
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Métricas
        test_metrics = model_data.get('test_metrics', {})
        confidence_score = test_metrics.get('r2', 0.8)
        rmse = test_metrics.get('rmse', 0.05)
        
        confidence_interval = {
            'lower': float(y_pred - (1.96 * rmse)),
            'upper': float(y_pred + (1.96 * rmse))
        }
        
        return {
            'model_name': model_name,
            'model_type': 'MIDAS_V2',
            'prediction': float(y_pred),
            'confidence_score': float(confidence_score),
            'confidence_interval': confidence_interval,
            'rmse': float(rmse),
            'mape': test_metrics.get('mape', 0.0),
            'features_used': list(X_model.columns),
            'midas_features_count': X_model.shape[1],
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def _prepare_xgboost_features(self, X: pd.DataFrame, required_vars: List[str]) -> pd.DataFrame:
        """Preparar features específicas para XGBoost V2"""
        
        # Buscar variables disponibles
        available_features = []
        
        for var in required_vars:
            if var in X.columns:
                available_features.append(var)
            else:
                # Buscar coincidencias parciales
                matching = [col for col in X.columns if var.lower() in col.lower()]
                if matching:
                    available_features.append(matching[0])
                    logger.info(f"   Usando {matching[0]} para {var}")
                else:
                    logger.warning(f"   Variable {var} no encontrada")
        
        if len(available_features) < len(required_vars) * 0.8:
            logger.warning(f"Solo {len(available_features)}/{len(required_vars)} variables disponibles")
        
        # Crear DataFrame con features (SOLO las requeridas, sin extras)
        X_model = X[available_features].copy()
        
        # Tomar última observación
        X_latest = X_model.iloc[-1:].copy()
        
        # Llenar NaN con estrategia conservadora
        for col in X_latest.columns:
            if X_latest[col].isnull().any():
                # Usar el último valor válido o la mediana
                if not X_model[col].isnull().all():
                    fill_value = X_model[col].fillna(method='ffill').iloc[-1]
                    if pd.isnull(fill_value):
                        fill_value = X_model[col].median()
                    X_latest[col] = X_latest[col].fillna(fill_value)
                else:
                    X_latest[col] = X_latest[col].fillna(0)
        
        return X_latest
    
    def _align_features_to_midas(self, X: pd.DataFrame, midas_columns: List[str]) -> pd.DataFrame:
        """Alinear features actuales con la estructura MIDAS entrenada"""
        
        # Crear DataFrame con las columnas MIDAS esperadas
        X_aligned = pd.DataFrame(index=X.index[-1:], columns=midas_columns)
        
        # Llenar con ceros inicialmente
        X_aligned = X_aligned.fillna(0)
        
        # Intentar mapear features disponibles
        for col in midas_columns:
            if col in X.columns:
                X_aligned[col] = X[col].iloc[-1]
            else:
                # Buscar coincidencias parciales
                partial_matches = [x_col for x_col in X.columns if any(word in x_col.lower() for word in col.lower().split('_'))]
                if partial_matches:
                    # Usar la primera coincidencia
                    X_aligned[col] = X[partial_matches[0]].iloc[-1]
                    logger.debug(f"   MIDAS: {col} ← {partial_matches[0]}")
        
        return X_aligned
    
    def _create_midas_features_dynamic(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crear features MIDAS dinámicamente"""
        
        target_var = 'precio_varilla_lme'
        if target_var not in X.columns:
            logger.error(f"Variable objetivo {target_var} no encontrada para MIDAS")
            return pd.DataFrame()
        
        # Crear features MIDAS básicas
        midas_features = []
        
        # 1. Lags de precio
        for lag in [1, 5, 10, 20]:
            if len(X) > lag:
                feature = X[target_var].shift(lag).iloc[-1]
                midas_features.append(feature)
        
        # 2. Medias móviles
        for window in [5, 10, 20]:
            if len(X) >= window:
                feature = X[target_var].rolling(window).mean().iloc[-1]
                midas_features.append(feature)
        
        # 3. Volatilidad
        for window in [5, 10, 20]:
            if len(X) >= window:
                feature = X[target_var].pct_change().rolling(window).std().iloc[-1]
                midas_features.append(feature)
        
        # 4. Variables exógenas
        exog_vars = ['iron', 'coking', 'VIX', 'sp500', 'commodities']
        for var in exog_vars:
            matching = [col for col in X.columns if var.lower() in col.lower()]
            if matching:
                feature = X[matching[0]].iloc[-1]
                midas_features.append(feature)
        
        # Crear DataFrame
        feature_names = [f'midas_feature_{i}' for i in range(len(midas_features))]
        X_midas = pd.DataFrame([midas_features], columns=feature_names, index=X.index[-1:])
        
        # Llenar NaN
        X_midas = X_midas.fillna(0)
        
        return X_midas
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if model_name in self.loaded_models:
            model_data = self.loaded_models[model_name]
            return {
                'name': model_name,
                'type': model_data.get('config', {}).get('type', 'unknown'),
                'description': model_data.get('config', {}).get('description', ''),
                'loaded_at': model_data.get('loaded_at'),
                'test_metrics': model_data.get('test_metrics', {}),
                'variables': model_data.get('config', {}).get('variables', []),
                'file_path': model_data.get('file_path')
            }
        else:
            return self.model_configs.get(model_name, {})
    
    def get_feature_importance(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Obtener feature importance del modelo
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Lista de features con su importancia
        """
        # Asegurar que el modelo esté cargado
        model_data = self.loaded_models.get(model_name)
        if not model_data:
            # Intentar cargar el modelo
            model_data = self.load_model(model_name)
            if not model_data:
                logger.warning(f"No se pudo cargar {model_name} para feature importance")
                return []
        
        logger.info(f"🔍 Extrayendo feature importance de {model_name}...")
        
        # Buscar feature importance en diferentes ubicaciones
        if 'feature_importance' in model_data:
            # Feature importance guardada como DataFrame
            importance_df = model_data['feature_importance']
            if isinstance(importance_df, pd.DataFrame):
                logger.info(f"✅ Feature importance encontrada en DataFrame: {len(importance_df)} features")
                return importance_df.to_dict('records')
        
        # Para modelos XGBoost/LightGBM con feature_importances_
        model = model_data.get('model')
        if hasattr(model, 'feature_importances_'):
            logger.info(f"🔍 Extrayendo de model.feature_importances_...")
            
            # Obtener nombres de features
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            elif 'feature_names' in model_data:
                feature_names = model_data['feature_names']
            else:
                # Crear nombres genéricos
                feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
                logger.warning(f"⚠️ Usando nombres de features genéricos para {model_name}")
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"✅ Feature importance extraída: {len(importance_df)} features")
            return importance_df.to_dict('records')
        
        logger.warning(f"⚠️ Feature importance no disponible para {model_name}")
        return []
    
    def list_available_models(self) -> List[str]:
        """Listar modelos disponibles"""
        available = []
        
        for model_name in self.model_configs.keys():
            # Buscar en múltiples ubicaciones
            locations_to_check = [
                self.models_dir / f"{model_name}.pkl",                    # models/
                self.models_dir / f"{model_name}_latest.pkl",             # models/production/
                self.models_dir / 'production' / f"{model_name}.pkl",     # models/production/
                self.models_dir / 'production' / f"{model_name}_latest.pkl", # models/production/latest
                self.models_dir / 'test' / f"{model_name}.pkl"            # models/test/
            ]
            
            for model_file in locations_to_check:
                if model_file.exists():
                    available.append(model_name)
                    break
        
        return available


class ProductionPredictor:
    """
    Predictor de producción que usa los mejores modelos V2
    """
    
    def __init__(self, models_dir: Path):
        self.model_factory = ModelV2Factory(models_dir)
        self.prediction_history = []
        
    async def predict_next_day_price(self, features_df: pd.DataFrame, model_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        Predecir precio del día siguiente
        
        Args:
            features_df: DataFrame con features preparadas
            model_preference: Modelo preferido (opcional)
            
        Returns:
            Predicción con intervalos de confianza y explicabilidad
        """
        logger.info("🎯 Generando predicción para el día siguiente...")
        
        # Determinar modelos a usar
        available_models = self.model_factory.list_available_models()
        
        if model_preference and model_preference in available_models:
            models_to_use = [model_preference]
        else:
            # Usar modelos por defecto en orden de preferencia
            preferred_order = ['MIDAS_V2_hibrida', 'XGBoost_V2_regime']
            models_to_use = [m for m in preferred_order if m in available_models]
        
        if not models_to_use:
            raise ValueError("No hay modelos V2 disponibles para predicción")
        
        logger.info(f"🤖 Usando modelos: {models_to_use}")
        
        # Generar predicciones con cada modelo
        predictions = {}
        
        for model_name in models_to_use:
            try:
                pred_result = self.model_factory.predict_with_model(model_name, features_df)
                if pred_result:
                    predictions[model_name] = pred_result
                    logger.info(f"✅ {model_name}: ${pred_result['prediction']:.2f}")
            except Exception as e:
                logger.error(f"Error con {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No se pudieron generar predicciones")
        
        # Seleccionar mejor predicción
        best_model_name = self._select_best_model(predictions)
        best_prediction = predictions[best_model_name]
        
        # Crear resultado consolidado
        result = {
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'predicted_price_usd_per_ton': best_prediction['prediction'],
            'currency': 'USD',
            'unit': 'metric ton',
            'model_confidence': best_prediction['confidence_score'],
            'confidence_interval_lower': best_prediction['confidence_interval']['lower'],
            'confidence_interval_upper': best_prediction['confidence_interval']['upper'],
            'best_model': best_model_name,
            'all_predictions': predictions,
            'timestamp': datetime.now().isoformat() + 'Z'
        }
        
        # Guardar en historial
        self.prediction_history.append(result)
        
        # Mantener solo últimas 100 predicciones
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        logger.info(f"🎯 Predicción final: ${result['predicted_price_usd_per_ton']:.2f} (modelo: {best_model_name})")
        
        return result
    
    def _select_best_model(self, predictions: Dict[str, Any]) -> str:
        """Seleccionar el mejor modelo basado en métricas históricas"""
        
        best_model = None
        best_score = -1
        
        # Criterio 1: R² más alto
        for model_name, pred_data in predictions.items():
            r2 = pred_data.get('confidence_score', 0)
            if r2 > best_score:
                best_score = r2
                best_model = model_name
        
        # Criterio 2: MAPE más bajo (como tiebreaker)
        if len(predictions) > 1:
            models_with_same_r2 = [m for m, p in predictions.items() 
                                 if abs(p.get('confidence_score', 0) - best_score) < 0.01]
            
            if len(models_with_same_r2) > 1:
                best_mape = float('inf')
                for model_name in models_with_same_r2:
                    mape = predictions[model_name].get('mape', float('inf'))
                    if mape < best_mape:
                        best_mape = mape
                        best_model = model_name
        
        return best_model
    
    async def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """
        Obtener análisis consolidado de feature importance
        """
        logger.info("📊 Analizando importancia de features...")
        
        available_models = self.model_factory.list_available_models()
        
        if not available_models:
            return {'error': 'No hay modelos disponibles'}
        
        # Recopilar feature importance de todos los modelos
        all_importance = {}
        
        for model_name in available_models:
            try:
                # Primero cargar el modelo
                model_data = self.model_factory.load_model(model_name)
                if model_data:
                    # Luego extraer feature importance
                    importance = self.model_factory.get_feature_importance(model_name)
                    if importance:
                        all_importance[model_name] = importance
                        logger.info(f"✅ {model_name}: {len(importance)} features extraídas")
                    else:
                        logger.warning(f"⚠️ {model_name}: No feature importance disponible")
                else:
                    logger.warning(f"⚠️ {model_name}: No se pudo cargar modelo")
            except Exception as e:
                logger.error(f"❌ Error con {model_name}: {str(e)}")
                continue
        
        # Verificar que tenemos datos para analizar
        if not all_importance:
            logger.warning("⚠️ No se pudo extraer feature importance de ningún modelo")
            return {
                'models_analyzed': [],
                'total_factors_analyzed': 0,
                'top_factors': [],
                'factors_by_category': {},
                'analysis_timestamp': datetime.now().isoformat(),
                'error': 'No se pudo extraer feature importance de los modelos disponibles'
            }
        
        # Consolidar y analizar
        consolidated_importance = self._consolidate_feature_importance(all_importance)
        
        # Categorizar factores
        categorized_factors = self._categorize_causal_factors(consolidated_importance)
        
        return {
            'models_analyzed': list(all_importance.keys()),
            'total_factors_analyzed': len(consolidated_importance),
            'top_factors': consolidated_importance[:15],
            'factors_by_category': categorized_factors,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _consolidate_feature_importance(self, all_importance: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Consolidar feature importance de múltiples modelos"""
        
        feature_scores = {}
        
        # Agregar importancia de todos los modelos
        for model_name, importance_list in all_importance.items():
            for feature_info in importance_list:
                feature = feature_info['feature']
                importance = feature_info['importance']
                
                if feature not in feature_scores:
                    feature_scores[feature] = []
                
                feature_scores[feature].append({
                    'model': model_name,
                    'importance': importance
                })
        
        # Calcular importancia promedio
        consolidated = []
        for feature, scores in feature_scores.items():
            avg_importance = np.mean([s['importance'] for s in scores])
            
            consolidated.append({
                'feature': feature,
                'average_importance': float(avg_importance),
                'models_count': len(scores),
                'models': [s['model'] for s in scores],
                'category': self._categorize_feature(feature),
                'description': self._get_feature_description(feature)
            })
        
        # Ordenar por importancia
        consolidated.sort(key=lambda x: x['average_importance'], reverse=True)
        
        return consolidated
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorizar feature según su naturaleza"""
        feature_lower = feature_name.lower()
        
        if any(word in feature_lower for word in ['precio', 'price', 'varilla', 'steel', 'lag', 'current']):
            return 'Autorregresivo'
        elif any(word in feature_lower for word in ['iron', 'coking', 'mineral', 'carbon']):
            return 'Materias Primas'
        elif any(word in feature_lower for word in ['vix', 'volatility', 'sp500']):
            return 'Mercados Financieros'
        elif any(word in feature_lower for word in ['tipo_cambio', 'usd', 'mxn']):
            return 'Tipo de Cambio'
        elif any(word in feature_lower for word in ['tasa', 'interes', 'tiie']):
            return 'Tasas de Interés'
        elif any(word in feature_lower for word in ['ma', 'moving', 'average', 'volatility']):
            return 'Indicadores Técnicos'
        else:
            return 'Otros'
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Obtener descripción detallada del factor"""
        descriptions = {
            'precio_varilla_lme_lag_1': 'Precio de varilla del día anterior (componente autorregresivo crítico)',
            'current_price': 'Precio actual de referencia (feature crítica V2)',
            'iron': 'Precio del mineral de hierro (materia prima principal)',
            'coking': 'Precio del carbón de coque (materia prima para acero)',
            'VIX': 'Índice de volatilidad implícita (sentimiento de mercado)',
            'sp500': 'Índice S&P 500 (condiciones macroeconómicas)',
            'tipo_cambio_usdmxn': 'Tipo de cambio USD/MXN (impacto en costos)',
            'tasa_interes_banxico': 'Tasa de interés de Banxico (política monetaria)',
            'volatility_20': 'Volatilidad rolling de 20 días (riesgo de mercado)',
            'price_ma20': 'Media móvil de 20 días (tendencia de mediano plazo)',
            'commodities': 'Índice general de commodities (contexto sectorial)',
            'steel': 'Precios de acero relacionados'
        }
        
        return descriptions.get(feature_name, f"Factor: {feature_name}")
    
    def _categorize_causal_factors(self, consolidated_importance: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorizar factores causales por tipo"""
        
        categories = {}
        
        for factor in consolidated_importance:
            category = factor['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(factor)
        
        return categories
    
    async def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener historial de predicciones"""
        return self.prediction_history[-limit:] if self.prediction_history else []


# Funciones de utilidad para el pipeline

def evaluate_model_v2(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluar modelo con métricas V2 estándar
    """
    # Asegurar arrays numpy
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Filtrar NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'error': 'No hay datos válidos para evaluación'}
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    directional_accuracy = 0.0
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    
    # Hit rate ±2%
    threshold = 0.02
    within_threshold = np.abs((y_pred - y_true) / y_true) <= threshold
    hit_rate = np.mean(within_threshold) * 100
    
    return {
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy),
        'hit_rate_2pct': float(hit_rate),
        'samples_evaluated': len(y_true)
    }


def safe_to_datetime(fecha_series, normalize=True):
    """
    Función helper para manejo seguro de fechas (replica de join_daily_series.py)
    """
    try:
        result = pd.to_datetime(fecha_series)
    except (ValueError, TypeError):
        try:
            result = pd.to_datetime(fecha_series, utc=True)
            result = result.dt.tz_localize(None)
        except:
            result = fecha_series.apply(lambda x: 
                pd.to_datetime(x).tz_localize(None) if hasattr(pd.to_datetime(x), 'tz_localize') 
                else pd.to_datetime(x))
    
    # Eliminar timezone si existe
    if hasattr(result.dtype, 'tz') and result.dt.tz is not None:
        result = result.dt.tz_localize(None)
    
    # Normalizar a fecha (sin hora) para evitar duplicados
    if normalize:
        result = result.dt.normalize()
    
    return result
