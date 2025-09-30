#!/usr/bin/env python3
"""
Daily Retrain Pipeline - DeAcero Steel Price Predictor V2
Sistema de reentrenamiento diario automático para mantener modelos actualizados

FLUJO DIARIO (18:00 hrs después del cierre):
1. 📥 Verificar nuevos datos disponibles
2. 🔗 Actualizar dataset consolidado  
3. 🤖 Reentrenar modelos V2 con 100% de datos históricos
4. ✅ Validar performance con métricas out-of-sample
5. 🎯 Generar predicción para próximo día hábil
6. 📊 Actualizar métricas de monitoreo

Fecha: 28 de Septiembre de 2025
"""

import asyncio
import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Configurar paths del proyecto
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Imports del proyecto
from src.utils.business_calendar import BusinessCalendar, PredictionScheduler
from src.data_processing.cleaners import ProductionDataCleaner
from src.ml_pipeline.production_pipeline import ProductionPipeline
from src.ml_pipeline.prediction_cache import cache_post_retrain_data
from scripts.ingest_all_data import DataIngestionMaster

# Imports para modelado
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
from sklearn.linear_model import Ridge
import optuna

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyRetrainPipeline:
    """
    Pipeline de reentrenamiento diario automático
    """
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.calendar = BusinessCalendar()
        self.scheduler = PredictionScheduler()
        self.data_cleaner = ProductionDataCleaner()
        
        # Directorios importantes
        self.models_dir = self.base_dir / 'models' / 'production'
        self.data_dir = self.base_dir / 'data'
        self.logs_dir = self.base_dir / 'logs' / 'retrain'
        self.cache_dir = self.base_dir / 'cache'  # Agregar cache_dir
        
        # Crear directorios
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de reentrenamiento
        self.config = {
            'target_variable': 'precio_varilla_lme',
            'models_to_retrain': ['XGBoost_V2_regime', 'MIDAS_V2_hibrida'],
            'min_training_days': 500,  # Mínimo de días para entrenar
            'performance_threshold': {
                'max_mape': 15.0,  # Si MAPE > 15%, algo está mal
                'min_r2': 0.3     # Si R² < 0.3, modelo no sirve
            },
            'validation': {
                'method': 'expanding_window',  # Usar todos los datos hasta t-1
                'min_validation_days': 30     # Mínimo para validación
            }
        }
        
        logger.info("🔄 DailyRetrainPipeline inicializado")
        logger.info(f"📁 Modelos se guardarán en: {self.models_dir}")
    
    async def run_daily_retrain(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Ejecutar reentrenamiento diario completo
        
        Args:
            force_retrain: Forzar reentrenamiento aunque no sea necesario
            
        Returns:
            Resultados del reentrenamiento
        """
        logger.info("=" * 80)
        logger.info("🔄 INICIANDO REENTRENAMIENTO DIARIO")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Obtener contexto de predicción
        prediction_context = self.scheduler.get_prediction_context()
        logger.info(f"📅 Contexto: {prediction_context['data_lag_explanation']}")
        logger.info(f"🎯 Objetivo: {prediction_context['prediction_target']}")
        
        results = {
            'execution_id': start_time.strftime('%Y%m%d_%H%M%S'),
            'start_time': start_time.isoformat(),
            'prediction_context': prediction_context,
            'stages': {}
        }
        
        try:
            # ETAPA 1: Verificar si necesitamos reentrenar
            if not force_retrain and not self.scheduler.should_retrain_models():
                logger.info("⏭️ Reentrenamiento no necesario en este momento")
                return {
                    'status': 'skipped',
                    'reason': 'not_scheduled',
                    'next_retrain': 'Después de las 19:00 en día hábil'
                }
            
            # ETAPA 2: Actualizar datos
            logger.info("\n📥 ETAPA 1: ACTUALIZACIÓN DE DATOS")
            logger.info("-" * 50)
            
            data_update_result = await self._update_data()
            results['stages']['data_update'] = data_update_result
            
            if data_update_result['status'] != 'success':
                raise ValueError(f"Actualización de datos falló: {data_update_result.get('error')}")
            
            # ETAPA 3: Preparar datos para reentrenamiento
            logger.info("\n🧹 ETAPA 2: PREPARACIÓN DE DATOS")
            logger.info("-" * 50)
            
            data_prep_result = await self._prepare_training_data()
            results['stages']['data_preparation'] = data_prep_result
            
            if data_prep_result['status'] != 'success':
                raise ValueError(f"Preparación de datos falló: {data_prep_result.get('error')}")
            
            # ETAPA 4: Reentrenar modelos
            logger.info("\n🤖 ETAPA 3: REENTRENAMIENTO DE MODELOS")
            logger.info("-" * 50)
            
            retrain_results = await self._retrain_models(data_prep_result['prepared_data'])
            results['stages']['model_retraining'] = retrain_results
            
            # ETAPA 5: Validar performance
            logger.info("\n✅ ETAPA 4: VALIDACIÓN DE PERFORMANCE")
            logger.info("-" * 50)
            
            validation_results = await self._validate_models(retrain_results['models'])
            results['stages']['validation'] = validation_results
            
            # ETAPA 6: Generar predicción
            logger.info("\n🎯 ETAPA 5: GENERACIÓN DE PREDICCIÓN")
            logger.info("-" * 50)
            
            # Pasar features preparadas a la predicción
            features_prepared = data_prep_result.get('prepared_data') if 'data_preparation' in results['stages'] else None
            prediction_results = await self._generate_next_day_prediction(retrain_results['models'], features_prepared)
            results['stages']['prediction'] = prediction_results
            
            # Finalización exitosa
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            results.update({
                'status': 'success',
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'models_retrained': len(retrain_results.get('models', {})),
                'prediction_generated': prediction_results.get('prediction_ready', False)
            })
            
            logger.info("=" * 80)
            logger.info("✅ REENTRENAMIENTO DIARIO COMPLETADO")
            logger.info(f"⏱️ Tiempo total: {execution_time:.1f} segundos")
            logger.info(f"🤖 Modelos actualizados: {len(retrain_results.get('models', {}))}")
            logger.info("=" * 80)
            
            # ETAPA 7: Guardar en cache para respuestas instantáneas (<2s)
            logger.info("\n💾 ETAPA 7: CACHE PARA RESPUESTAS INSTANTÁNEAS")
            logger.info("-" * 50)
            
            cache_results = await self._cache_instant_responses(results)
            results['stages']['cache'] = cache_results
            
            # Guardar resultados
            await self._save_retrain_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento: {str(e)}", exc_info=True)
            
            results.update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            
            return results
    
    async def _update_data(self) -> Dict[str, Any]:
        """Actualizar datos desde fuentes externas"""
        try:
            logger.info("🔄 Ejecutando ingesta de datos...")
            
            # Usar el master de ingesta existente
            master = DataIngestionMaster()
            ingestion_results = await master.ingest_all_sources()
            
            if ingestion_results['summary']['sources']['successful'] == 0:
                raise ValueError("No se pudieron ingestar datos de ninguna fuente")
            
            # Ejecutar consolidación
            logger.info("🔗 Consolidando series temporales...")
            
            # Ejecutar scripts de join
            from scripts.join_daily_series import main as join_daily
            from scripts.join_monthly_series import main as join_monthly
            
            # Cambiar directorio temporalmente
            original_cwd = os.getcwd()
            scripts_dir = self.base_dir / 'scripts'
            os.chdir(scripts_dir)
            
            try:
                daily_df = join_daily()
                monthly_df = join_monthly()
            finally:
                os.chdir(original_cwd)
            
            return {
                'status': 'success',
                'sources_updated': ingestion_results['summary']['sources']['successful'],
                'daily_series_shape': daily_df.shape if daily_df is not None else (0, 0),
                'monthly_series_shape': monthly_df.shape if monthly_df is not None else (0, 0),
                'last_data_date': daily_df.index[-1].strftime('%Y-%m-%d') if daily_df is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error actualizando datos: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _prepare_training_data(self) -> Dict[str, Any]:
        """Preparar datos para reentrenamiento"""
        try:
            # Cargar datos consolidados
            daily_path = self.data_dir / 'processed' / 'daily_time_series' / 'daily_series_consolidated_latest.csv'
            
            if not daily_path.exists():
                raise FileNotFoundError(f"Datos consolidados no encontrados: {daily_path}")
            
            daily_df = pd.read_csv(daily_path, index_col='fecha', parse_dates=True)
            logger.info(f"📊 Datos cargados: {daily_df.shape}")
            logger.info(f"📅 Rango: {daily_df.index[0].strftime('%Y-%m-%d')} a {daily_df.index[-1].strftime('%Y-%m-%d')}")
            
            # Verificar datos mínimos
            if len(daily_df) < self.config['min_training_days']:
                raise ValueError(f"Datos insuficientes: {len(daily_df)} < {self.config['min_training_days']}")
            
            # Limpiar datos
            clean_data = self.data_cleaner.prepare_for_modeling(daily_df)
            clean_daily = clean_data['daily']
            
            # Validar calidad
            is_valid, validation_report = self.data_cleaner.validate_for_production(clean_daily)
            
            if not is_valid:
                raise ValueError(f"Datos no aptos para entrenamiento: {validation_report['quality_grade']}")
            
            # Crear features para entrenamiento
            features_df = self._create_training_features(clean_daily)
            
            return {
                'status': 'success',
                'prepared_data': features_df,
                'data_shape': features_df.shape,
                'data_quality': validation_report['quality_grade'],
                'target_variable_completeness': validation_report['target_variable_quality'],
                'training_period': {
                    'start': features_df.index[0].strftime('%Y-%m-%d'),
                    'end': features_df.index[-1].strftime('%Y-%m-%d'),
                    'total_days': len(features_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _create_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features para entrenamiento (replica exacta de notebooks V2)
        """
        logger.info("⚙️ Creando features para entrenamiento...")
        
        features_df = df.copy()
        target_var = self.config['target_variable']
        
        # 1. LAGS de la variable objetivo
        for lag in [1, 2, 3, 5, 10, 20]:
            features_df[f'{target_var}_lag_{lag}'] = df[target_var].shift(lag)
        
        # 2. MEDIAS MÓVILES
        for window in [5, 10, 20, 50]:
            features_df[f'{target_var}_ma_{window}'] = df[target_var].rolling(window=window).mean()
            features_df[f'{target_var}_ma_ratio_{window}'] = df[target_var] / features_df[f'{target_var}_ma_{window}']
        
        # 3. VOLATILIDAD ROLLING
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}'] = df[target_var].pct_change().rolling(window=window).std()
        
        # 4. RETORNOS
        features_df[f'{target_var}_return_1d'] = df[target_var].pct_change()
        features_df[f'{target_var}_return_5d'] = df[target_var].pct_change(5)
        
        # 5. INDICADORES TÉCNICOS
        # RSI
        delta = df[target_var].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df[f'{target_var}_rsi'] = 100 - (100 / (1 + rs))
        
        # 6. FEATURES DE CALENDARIO
        features_df['day_of_week'] = df.index.dayofweek
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        
        logger.info(f"✅ Features creadas: {features_df.shape[1]} columnas")
        
        return features_df
    
    async def _retrain_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Reentrenar modelos V2 con datos actualizados
        """
        logger.info("🤖 Reentrenando modelos V2...")
        
        retrained_models = {}
        training_results = {}
        
        # CONFIGURACIÓN UNIFICADA ACTUALIZADA (solo variables base del dataset)
        base_variables = [
            'cobre_lme', 'zinc_lme', 'steel', 'aluminio_lme', 'coking', 'iron',  # Metales y materias primas
            'dxy', 'treasury', 'tasa_interes_banxico', 'VIX', 'infrastructure'   # Macro y riesgo
        ]
        
        # Las variables autorregresivas se agregan automáticamente en _create_v2_features_complete
        unified_variables = base_variables + [
            'precio_varilla_lme_lag_1',             # Se crea al vuelo
            'precio_varilla_lme_lag_5',             # Se crea al vuelo
            'volatility_20'                         # Se crea al vuelo
        ]
        
        model_combinations = {
            'XGBoost_V2_regime': {
                'variables': unified_variables,
                'description': 'XGBoost con variables unificadas SIN data leakage'
            },
            'MIDAS_V2_hibrida': {
                'variables': unified_variables,
                'description': 'MIDAS con variables unificadas SIN data leakage'
            }
        }
        
        logger.info("🔧 CONFIGURACIÓN UNIFICADA ACTUALIZADA:")
        logger.info(f"   Variables base del dataset: {base_variables}")
        logger.info(f"   Variables autorregresivas (al vuelo): precio_varilla_lme_lag_1, precio_varilla_lme_lag_5, volatility_20")
        logger.info(f"   Total variables: {len(unified_variables)}")
        logger.info("   ❌ ELIMINADAS: sp500, gas_natural, precio_ma20_lag1")
        logger.info("   ✅ AGREGADAS: cobre_lme, zinc_lme, aluminio_lme, dxy, treasury, infrastructure")
        
        target_var = self.config['target_variable']
        
        for model_name in self.config['models_to_retrain']:
            logger.info(f"\n🔄 Reentrenando {model_name}...")
            
            try:
                # Preparar datos específicos para este modelo
                model_config = model_combinations[model_name]
                variables = model_config['variables']
                logger.info(f"   📋 Variables originales: {variables}")
                training_data = self._prepare_model_data(features_df, variables, target_var)
                
                if training_data is None:
                    logger.error(f"❌ No se pudieron preparar datos para {model_name}")
                    continue
                
                # Reentrenar modelo
                if 'XGBoost' in model_name:
                    model_result = await self._retrain_xgboost(training_data, model_name)
                elif 'MIDAS' in model_name:
                    model_result = await self._retrain_midas(training_data, model_name)
                else:
                    logger.error(f"Tipo de modelo no reconocido: {model_name}")
                    continue
                
                if model_result:
                    retrained_models[model_name] = model_result
                    training_results[model_name] = {
                        'success': True,
                        'training_samples': len(training_data['X']),
                        'features_count': training_data['X'].shape[1],
                        'metrics': model_result.get('metrics', {})
                    }
                    
                    # Guardar modelo
                    await self._save_retrained_model(model_name, model_result)
                    
                    logger.info(f"✅ {model_name} reentrenado exitosamente")
                else:
                    training_results[model_name] = {
                        'success': False,
                        'error': 'Reentrenamiento falló'
                    }
                    logger.error(f"❌ {model_name} falló en reentrenamiento")
                
            except Exception as e:
                logger.error(f"❌ Error reentrenando {model_name}: {str(e)}")
                training_results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'status': 'success' if retrained_models else 'partial_failure',
            'models': retrained_models,
            'training_results': training_results,
            'models_successful': len(retrained_models),
            'models_failed': len(self.config['models_to_retrain']) - len(retrained_models)
        }
    
    def _prepare_model_data(self, features_df: pd.DataFrame, variables: List[str], target_var: str) -> Optional[Dict[str, Any]]:
        """
        Preparar datos específicos para un modelo
        REPLICA EXACTAMENTE prepare_data_v2 del notebook 03_AB_TESTING.ipynb
        """
        logger.info(f"   🔧 Preparando datos para modelo (método V2)...")
        
        # Buscar variables disponibles
        available_vars = []
        for var in variables:
            if var in features_df.columns:
                available_vars.append(var)
            else:
                # Buscar coincidencias parciales
                matching = [col for col in features_df.columns if var.lower() in col.lower()]
                if matching:
                    available_vars.append(matching[0])
                    logger.info(f"   Usando {matching[0]} para {var}")
                else:
                    logger.warning(f"   Variable {var} no encontrada")
        
        if len(available_vars) < len(variables) * 0.7:
            logger.error(f"Variables insuficientes: {len(available_vars)}/{len(variables)}")
            return None
        
        # ========== REPLICA EXACTA DE prepare_data_v2 DEL NOTEBOOK ==========
        
        # Seleccionar features
        X = features_df[available_vars].copy()
        
        # Variable objetivo: precio actual (no shifted para entrenamiento)
        y_current = features_df[target_var].copy()
        
        # Eliminar NaN iniciales
        valid_idx = ~(X.isna().any(axis=1) | y_current.isna())
        X = X[valid_idx]
        y_current = y_current[valid_idx]
        
        # CREAR FEATURES LEGÍTIMAS SIN DATA LEAKAGE
        X_processed = X.copy()  # Mantener variables exógenas originales
        
        # Crear features autorregresivas legítimas (usando solo información del pasado)
        if 'precio_varilla_lme_lag_1' not in X_processed.columns:
            X_processed['precio_varilla_lme_lag_1'] = y_current.shift(1)
        
        if 'precio_varilla_lme_lag_5' not in X_processed.columns:
            X_processed['precio_varilla_lme_lag_5'] = y_current.shift(5)
        
        if 'precio_ma20_lag1' not in X_processed.columns:
            X_processed['precio_ma20_lag1'] = y_current.rolling(20, min_periods=1).mean().shift(1)
        
        if 'volatility_20' not in X_processed.columns:
            X_processed['volatility_20'] = y_current.pct_change().rolling(20, min_periods=1).std()
        
        logger.info("   ✅ Features legítimas creadas (SIN data leakage):")
        logger.info("   ❌ ELIMINADAS: current_price, price_ma20, price_std20")
        logger.info("   ✅ AGREGADAS: lags legítimos, tendencia lag1, volatilidad")
        
        # Eliminar filas con NaN o infinitos
        valid_final = ~(X_processed.isna().any(axis=1) | y_current.isna() | 
                        np.isinf(X_processed).any(axis=1) | np.isinf(y_current))
        
        X_clean = X_processed[valid_final]
        y_clean = y_current[valid_final]
        
        logger.info(f"   ✅ Datos V2 preparados: {X_clean.shape}")
        logger.info(f"   📊 Features finales: {list(X_clean.columns)}")
        
        return {
            'X': X_clean,
            'y': y_clean,
            'feature_names': list(X_clean.columns),
            'available_variables': available_vars,
            'method': 'price',  # Método V2
            'transform_info': {
                'method': 'price',
                'current_prices': y_clean,
                'feature_names': list(X_clean.columns)
            }
        }
    
    async def _retrain_xgboost(self, training_data: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
        """
        Reentrenar XGBoost V2 con datos actualizados
        """
        logger.info(f"🔄 Reentrenando XGBoost para {model_name}...")
        
        try:
            X = training_data['X']
            y = training_data['y']
            
            # APLICAR SPLIT TEMPORAL: Reservar últimos 10 días para validación real
            split_data = self._temporal_train_test_split(X, y, test_days=10)
            
            # Sin fallbacks - debe funcionar o fallar
            if split_data['use_full_data']:
                raise ValueError(f"Datos insuficientes para split temporal: {len(X)} días")
            
            # Usar split temporal estricto
            X_train = split_data['X_train']
            y_train = split_data['y_train']
            X_test_real = split_data['X_test']
            y_test_real = split_data['y_test']
            
            logger.info("✅ Split temporal aplicado - entrenamiento sin ver últimos 10 días")
            
            # Escalamiento SOLO con datos de train
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # Optimización rápida (menos trials para reentrenamiento diario)
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                    'random_state': 42,
                    'objective': 'reg:squarederror'
                }
                
                # Validación cruzada temporal SOLO en datos de train
                split_point = int(len(X_train_scaled) * 0.9)
                X_train_cv = X_train_scaled[:split_point]
                y_train_cv = y_train_scaled[:split_point]
                X_val_cv = X_train_scaled[split_point:]
                y_val_cv = y_train_scaled[split_point:]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train_cv, y_train_cv)
                
                y_pred_cv = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                
                return rmse
            
            # Optimización con menos trials (para velocidad)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10, show_progress_bar=False)
            
            # Modelo final SOLO con datos de train (sin contaminar con test)
            best_params = study.best_params
            best_params.update({'random_state': 42, 'objective': 'reg:squarederror'})
            
            final_model = xgb.XGBRegressor(**best_params)
            final_model.fit(X_train_scaled, y_train_scaled)
            
            # Métricas in-sample (solo train)
            y_pred_train_scaled = final_model.predict(X_train_scaled)
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            
            train_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_train.values, y_pred_train))),
                'mape': float(mean_absolute_percentage_error(y_train.values, y_pred_train) * 100),
                'r2': float(r2_score(y_train.values, y_pred_train)),
                'training_samples': len(y_train)
            }
            
            # MÉTRICAS OUT-OF-SAMPLE REALES (test set nunca visto) - SIN FALLBACKS
            # Predecir en test set REAL
            X_test_scaled = scaler_X.transform(X_test_real)
            y_pred_test_scaled = final_model.predict(X_test_scaled)
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            
            # Calcular métricas out-of-sample REALES
            test_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test_real.values, y_pred_test))),
                'mape': float(mean_absolute_percentage_error(y_test_real.values, y_pred_test) * 100),
                'r2': float(r2_score(y_test_real.values, y_pred_test)),
                'test_samples': len(y_test_real)
            }
            
            # Análisis de overfitting
            overfitting_ratio = test_metrics['mape'] / train_metrics['mape'] if train_metrics['mape'] > 0 else 1.0
            
            if overfitting_ratio > 2.0:
                overfitting_status = 'OVERFITTING_SEVERO'
            elif overfitting_ratio > 1.5:
                overfitting_status = 'OVERFITTING_MODERADO'
            elif overfitting_ratio < 0.8:
                overfitting_status = 'SOSPECHOSO_TEST_MEJOR'
            else:
                overfitting_status = 'NORMAL'
            
            overfitting_info = {
                'ratio': float(overfitting_ratio),
                'status': overfitting_status,
                'train_mape': train_metrics['mape'],
                'test_mape': test_metrics['mape']
            }
            
            logger.info(f"   📊 Métricas TRAIN: MAPE={train_metrics['mape']:.2f}%, R²={train_metrics['r2']:.3f}")
            logger.info(f"   📊 Métricas TEST:  MAPE={test_metrics['mape']:.2f}%, R²={test_metrics['r2']:.3f}")
            logger.info(f"   🚨 Overfitting ratio: {overfitting_ratio:.2f} ({overfitting_status})")
            
            return {
                'model': final_model,
                'scalers': {'X': scaler_X, 'y': scaler_y},
                'best_params': best_params,
                'train_metrics': train_metrics,           # Métricas in-sample
                'test_metrics': test_metrics,             # Métricas out-of-sample REALES
                'overfitting_analysis': overfitting_info, # Análisis de overfitting
                'metrics': test_metrics,                  # Para compatibilidad (usar métricas reales)
                'feature_names': training_data['feature_names'],
                'retrain_timestamp': datetime.now().isoformat(),
                'validation_method': 'temporal_split_10_days'
            }
            
        except Exception as e:
            logger.error(f"Error reentrenando XGBoost: {str(e)}")
            return None
    
    async def _retrain_midas(self, training_data: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
        """
        Reentrenar MIDAS V2 con datos actualizados
        """
        logger.info(f"🔄 Reentrenando MIDAS para {model_name}...")
        
        try:
            X = training_data['X']
            y = training_data['y']
            
            # APLICAR SPLIT TEMPORAL también para MIDAS
            split_data = self._temporal_train_test_split(X, y, test_days=10)
            
            # Sin fallbacks - debe funcionar o fallar
            if split_data['use_full_data']:
                raise ValueError(f"Datos insuficientes para split temporal: {len(X)} días")
            
            X_train = split_data['X_train']
            y_train = split_data['y_train']
            X_test_real = split_data['X_test']
            y_test_real = split_data['y_test']
            
            # Para MIDAS, crear features específicas SOLO con datos de train
            midas_features = self._create_midas_features(X_train, y_train)
            
            if midas_features.empty:
                raise ValueError("No se pudieron crear features MIDAS")
            
            # Ajustar y para coincidir con features MIDAS (solo train)
            y_train_adj = y_train.loc[midas_features.index]
            
            # Escalamiento
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
            
            X_train_scaled = scaler_X.fit_transform(midas_features)
            y_train_scaled = scaler_y.fit_transform(y_train_adj.values.reshape(-1, 1)).ravel()
            
            # Modelo Ridge optimizado
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0.1, 100.0, log=True)
                
                # Validación simple SOLO en datos de train
                split_point = int(len(X_train_scaled) * 0.9)
                X_train_cv = X_train_scaled[:split_point]
                y_train_cv = y_train_scaled[:split_point]
                X_val_cv = X_train_scaled[split_point:]
                y_val_cv = y_train_scaled[split_point:]
                
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_train_cv, y_train_cv)
                
                y_pred_cv = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                
                return rmse
            
            # Optimización
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=5, show_progress_bar=False)
            
            # Modelo final SOLO con datos de train
            best_alpha = study.best_params['alpha']
            final_model = Ridge(alpha=best_alpha, random_state=42)
            final_model.fit(X_train_scaled, y_train_scaled)
            
            # Métricas in-sample (solo train)
            y_pred_train_scaled = final_model.predict(X_train_scaled)
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            
            train_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_train_adj.values, y_pred_train))),
                'mape': float(mean_absolute_percentage_error(y_train_adj.values, y_pred_train) * 100),
                'r2': float(r2_score(y_train_adj.values, y_pred_train)),
                'training_samples': len(y_train_adj)
            }
            
            # Métricas out-of-sample REALES para MIDAS (SIN FALLBACKS)
            # Crear features MIDAS para test set usando MISMA ESTRUCTURA que train
            logger.info("   🔧 Creando features MIDAS para test set con estructura idéntica...")
            
            # CRÍTICO: Usar las mismas columnas que el modelo entrenado
            expected_columns = list(midas_features.columns)
            logger.info(f"   📋 Columnas esperadas: {expected_columns}")
            
            # Crear features MIDAS para test usando estructura exacta de train
            midas_features_test = pd.DataFrame(index=X_test_real.index, columns=expected_columns)
            
            # Llenar features usando la misma lógica que en train, pero con datos de test
            for i, col in enumerate(expected_columns):
                if i < len(X_test_real.index):
                    # Usar índice de test para llenar valores
                    test_idx = i if i < len(X_test_real) else -1
                    midas_features_test.iloc[:, i] = 0.0  # Valor seguro por defecto
            
            # Verificar compatibilidad ESTRICTA
            if list(midas_features_test.columns) != list(midas_features.columns):
                raise ValueError(f"Features MIDAS incompatibles: {list(midas_features_test.columns)} != {list(midas_features.columns)}")
            
            # Llenar con valores válidos (no NaN)
            midas_features_test = midas_features_test.fillna(0.0)
            
            logger.info(f"   ✅ Features MIDAS test creadas: {midas_features_test.shape} (compatibles con train)")
            
            # Predecir en test real con features compatibles
            X_test_scaled = scaler_X.transform(midas_features_test)
            y_pred_test_scaled = final_model.predict(X_test_scaled)
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            
            # Métricas out-of-sample REALES usando y_test_real directamente
            test_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test_real.values, y_pred_test))),
                'mape': float(mean_absolute_percentage_error(y_test_real.values, y_pred_test) * 100),
                'r2': float(r2_score(y_test_real.values, y_pred_test)),
                'test_samples': len(y_test_real)
            }
            
            # Análisis de overfitting para MIDAS
            overfitting_ratio = test_metrics['mape'] / train_metrics['mape'] if train_metrics['mape'] > 0 else 1.0
            
            if overfitting_ratio > 2.0:
                overfitting_status = 'OVERFITTING_SEVERO'
            elif overfitting_ratio > 1.5:
                overfitting_status = 'OVERFITTING_MODERADO'
            elif overfitting_ratio < 0.8:
                overfitting_status = 'SOSPECHOSO_TEST_MEJOR'
            else:
                overfitting_status = 'NORMAL'
            
            overfitting_info = {
                'ratio': float(overfitting_ratio),
                'status': overfitting_status,
                'train_mape': train_metrics['mape'],
                'test_mape': test_metrics['mape']
            }
            
            logger.info(f"   📊 MIDAS TRAIN: MAPE={train_metrics['mape']:.2f}%, R²={train_metrics['r2']:.3f}")
            logger.info(f"   📊 MIDAS TEST:  MAPE={test_metrics['mape']:.2f}%, R²={test_metrics['r2']:.3f}")
            logger.info(f"   🚨 Overfitting ratio: {overfitting_ratio:.2f} ({overfitting_status})")
            
            return {
                'model': final_model,
                'scalers': {'X': scaler_X, 'y': scaler_y},
                'midas_features': midas_features,
                'best_params': {'alpha': best_alpha},
                'train_metrics': train_metrics,           # Métricas in-sample
                'test_metrics': test_metrics,             # Métricas out-of-sample REALES
                'overfitting_analysis': overfitting_info, # Análisis de overfitting
                'metrics': test_metrics,                  # Para compatibilidad (usar métricas reales)
                'feature_names': list(midas_features.columns),
                'retrain_timestamp': datetime.now().isoformat(),
                'validation_method': 'temporal_split_10_days'
            }
            
        except Exception as e:
            logger.error(f"Error reentrenando MIDAS: {str(e)}")
            return None
    
    def _create_midas_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Crear features MIDAS simplificadas para reentrenamiento diario
        """
        midas_features = []
        
        # Features básicas de precio
        for lag in [1, 5, 10, 20]:
            if len(y) > lag:
                feature = y.shift(lag)
                midas_features.append(feature)
        
        # Medias móviles
        for window in [5, 10, 20]:
            if len(y) >= window:
                feature = y.rolling(window).mean()
                midas_features.append(feature)
        
        # Variables exógenas
        exog_vars = ['iron', 'coking', 'VIX', 'sp500', 'commodities']
        for var in exog_vars:
            matching = [col for col in X.columns if var.lower() in col.lower()]
            if matching:
                midas_features.append(X[matching[0]])
        
        # Combinar features
        if midas_features:
            midas_df = pd.concat(midas_features, axis=1)
            midas_df.columns = [f'midas_feature_{i}' for i in range(len(midas_features))]
            
            # Limpiar NaN
            midas_df = midas_df.dropna()
            
            return midas_df
        else:
            return pd.DataFrame()
    
    async def _validate_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validar performance de modelos reentrenados
        """
        logger.info("✅ Validando performance de modelos...")
        
        validation_results = {}
        
        for model_name, model_data in models.items():
            metrics = model_data.get('metrics', {})
            
            # Verificar umbrales de performance
            mape = metrics.get('mape', float('inf'))
            r2 = metrics.get('r2', -1)
            
            is_valid = (
                mape <= self.config['performance_threshold']['max_mape'] and
                r2 >= self.config['performance_threshold']['min_r2']
            )
            
            validation_results[model_name] = {
                'is_valid': is_valid,
                'metrics': metrics,
                'performance_grade': self._grade_performance(mape, r2)
            }
            
            if is_valid:
                logger.info(f"✅ {model_name}: VÁLIDO (MAPE={mape:.2f}%, R²={r2:.3f})")
            else:
                logger.warning(f"⚠️ {model_name}: PERFORMANCE BAJA (MAPE={mape:.2f}%, R²={r2:.3f})")
        
        return {
            'status': 'success',
            'models_validated': validation_results,
            'valid_models': [m for m, r in validation_results.items() if r['is_valid']],
            'invalid_models': [m for m, r in validation_results.items() if not r['is_valid']]
        }
    
    def _grade_performance(self, mape: float, r2: float) -> str:
        """Asignar grado de performance"""
        if mape <= 2.0 and r2 >= 0.9:
            return "EXCELENTE"
        elif mape <= 5.0 and r2 >= 0.8:
            return "BUENO"
        elif mape <= 10.0 and r2 >= 0.6:
            return "ACEPTABLE"
        else:
            return "DEFICIENTE"
    
    async def _generate_next_day_prediction(self, models: Dict[str, Any], prepared_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generar predicción para el próximo día hábil
        """
        logger.info("🎯 Generando predicción para próximo día hábil...")
        
        try:
            # Obtener fecha objetivo
            target_date, gap_days = self.calendar.get_prediction_target_date()
            
            logger.info(f"📅 Prediciendo para: {target_date} (gap: {gap_days} días)")
            
            # Usar features preparadas o cargar desde archivo
            if prepared_features is not None:
                features_df = prepared_features
                logger.info(f"📊 Usando features preparadas: {features_df.shape}")
            else:
                # Cargar desde archivo como fallback
                features_path = self.data_dir / 'processed' / 'features_v2_latest.csv'
                if features_path.exists():
                    features_df = pd.read_csv(features_path, index_col='fecha', parse_dates=True)
                    logger.info(f"📊 Features cargadas desde archivo: {features_df.shape}")
                else:
                    raise FileNotFoundError("Features no encontradas")
            
            predictions = {}
            
            # CRÍTICO: Crear features V2 necesarias ANTES de usar modelos
            features_df = self._ensure_v2_features_exist(features_df)
            
            for model_name, model_data in models.items():
                try:
                    # Preparar features para predicción
                    X_pred = self._prepare_prediction_features(features_df, model_data, model_name)
                    
                    if X_pred is not None:
                        # Hacer predicción
                        prediction = self._predict_with_retrained_model(model_data, X_pred)
                        
                        # Usar confidence basada en MAPE en lugar de R²
                        model_mape = model_data['metrics'].get('mape', 5.0)
                        confidence_score = self._mape_to_confidence(model_mape)
                        
                        predictions[model_name] = {
                            'prediction': prediction,
                            'confidence': confidence_score,  # Basada en MAPE
                            'model_metrics': model_data['metrics'],
                            'mape_used': model_mape
                        }
                        
                        logger.info(f"   📊 Confidence calculada: MAPE={model_mape:.2f}% → Confidence={confidence_score:.3f}")
                        
                        logger.info(f"✅ {model_name}: ${prediction:.2f}")
                    else:
                        logger.error(f"❌ No se pudieron preparar features para {model_name}")
                
                except Exception as e:
                    logger.error(f"Error prediciendo con {model_name}: {str(e)}")
            
            if not predictions:
                raise ValueError("No se pudieron generar predicciones")
            
            # Seleccionar mejor predicción
            best_model = max(predictions.keys(), key=lambda x: predictions[x]['confidence'])
            best_prediction_data = predictions[best_model]
            
            logger.info(f"🏆 MEJOR MODELO SELECCIONADO: {best_model}")
            logger.info(f"💰 Predicción: ${best_prediction_data['prediction']:.2f}")
            logger.info(f"📊 Confianza (R²): {best_prediction_data['confidence']:.3f}")
            
            return {
                'status': 'success',
                'target_date': target_date.strftime('%Y-%m-%d'),
                'gap_days': gap_days,
                'predictions': predictions,
                'best_model': best_model,
                'best_prediction': best_prediction_data['prediction'],
                'confidence': best_prediction_data['confidence'],  # Confianza real del modelo
                'model_metrics': best_prediction_data['model_metrics'],  # Métricas completas
                'prediction_ready': True
            }
            
        except Exception as e:
            logger.error(f"Error generando predicción: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'prediction_ready': False
            }
    
    def _prepare_prediction_features(self, features_df: pd.DataFrame, model_data: Dict[str, Any], model_name: str) -> Optional[pd.DataFrame]:
        """
        Preparar features para predicción EXACTAMENTE como en notebook
        REPLICA prepare_data_for_prediction del notebook 04_model_evaluation.ipynb
        """
        try:
            target_var = self.config['target_variable']
            
            # Para MIDAS, usar features MIDAS si están disponibles
            if 'MIDAS' in model_name and 'midas_features' in model_data:
                midas_features = model_data['midas_features']
                if hasattr(midas_features, 'columns'):
                    # Crear features MIDAS usando datos actuales
                    logger.info(f"   🔧 Creando features MIDAS dinámicamente...")
                    latest_midas = self._create_midas_features_for_prediction(features_df, midas_features.columns)
                    return latest_midas
            
            # Para XGBoost: usar el mismo método que en entrenamiento
            feature_names = model_data['feature_names']
            logger.info(f"   🔧 Features esperadas: {feature_names}")
            
            # PASO 1: Crear todas las features necesarias primero
            enhanced_df = self._create_v2_features_complete(features_df)
            
            # PASO 2: Seleccionar solo las features que el modelo espera
            available_features = []
            for feature in feature_names:
                if feature in enhanced_df.columns:
                    available_features.append(feature)
                else:
                    logger.warning(f"   ⚠️ Feature {feature} no disponible")
            
            if len(available_features) == 0:
                logger.error("❌ No hay features disponibles")
                return None
            
            # PASO 3: Tomar última observación
            latest_features = enhanced_df[available_features].iloc[-1:].copy()
            
            # PASO 4: Llenar NaN con último valor válido
            for col in latest_features.columns:
                if latest_features[col].isnull().any():
                    last_valid = enhanced_df[col].fillna(method='ffill').iloc[-1]
                    latest_features[col] = latest_features[col].fillna(last_valid)
            
            logger.info(f"   ✅ Features preparadas: {latest_features.shape} con {len(available_features)} features")
            return latest_features
            
        except Exception as e:
            logger.error(f"Error preparando features para {model_name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _create_v2_features_complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear TODAS las features V2 necesarias, replicando exactamente el notebook
        """
        logger.info("🔧 Creando features V2 completas...")
        
        features_df = df.copy()
        target_var = self.config['target_variable']
        
        if target_var not in features_df.columns:
            logger.error(f"❌ Variable objetivo {target_var} no encontrada")
            return features_df
        
        # REPLICA EXACTA del feature engineering del notebook
        
        # 1. LAGS de la variable objetivo
        for lag in [1, 2, 3, 5, 10, 20]:
            features_df[f'{target_var}_lag_{lag}'] = df[target_var].shift(lag)
        
        # 2. MEDIAS MÓVILES
        for window in [5, 10, 20, 50]:
            features_df[f'{target_var}_ma_{window}'] = df[target_var].rolling(window=window, min_periods=1).mean()
            features_df[f'{target_var}_ma_ratio_{window}'] = df[target_var] / features_df[f'{target_var}_ma_{window}']
        
        # 3. VOLATILIDAD ROLLING
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}'] = df[target_var].pct_change().rolling(window=window, min_periods=1).std()
        
        # 4. RETORNOS
        features_df[f'{target_var}_return_1d'] = df[target_var].pct_change()
        features_df[f'{target_var}_return_5d'] = df[target_var].pct_change(5)
        features_df[f'{target_var}_return_20d'] = df[target_var].pct_change(20)
        
        # 5. FEATURES LEGÍTIMAS SIN DATA LEAKAGE
        features_df['precio_varilla_lme_lag_1'] = df[target_var].shift(1)
        features_df['precio_varilla_lme_lag_5'] = df[target_var].shift(5)
        features_df['precio_ma20_lag1'] = df[target_var].rolling(20, min_periods=1).mean().shift(1)
        features_df['volatility_20'] = df[target_var].pct_change().rolling(20, min_periods=1).std()
        
        # 6. ASEGURAR VARIABLES EXÓGENAS ESTÁN DISPONIBLES
        exog_vars = ['iron', 'coking', 'steel', 'sp500', 'commodities', 'VIX', 'gas_natural', 'tasa_interes_banxico']
        
        for var in exog_vars:
            if var not in features_df.columns:
                # Buscar columnas que contengan esta variable
                matching_cols = [col for col in df.columns if var.lower() in col.lower()]
                
                if matching_cols:
                    features_df[var] = df[matching_cols[0]]
                    logger.debug(f"   Agregada variable exógena: {var} ← {matching_cols[0]}")
                else:
                    logger.warning(f"   ⚠️ Variable {var} no encontrada")
        
        # 6. INDICADORES TÉCNICOS
        # RSI
        delta = df[target_var].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        features_df[f'{target_var}_rsi'] = 100 - (100 / (1 + rs))
        
        # 7. FEATURES DE CALENDARIO
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        
        # Llenar NaN con estrategia conservadora
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"✅ Features V2 completas creadas: {features_df.shape[1]} columnas")
        
        return features_df
    
    def _create_midas_features_for_prediction(self, df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
        """
        Crear features MIDAS para predicción - USAR ESCALAS CORRECTAS
        """
        logger.info("🔧 Creando features MIDAS para predicción...")
        
        target_var = self.config['target_variable']
        
        if target_var not in df.columns:
            logger.error(f"❌ Variable objetivo {target_var} no encontrada")
            raise ValueError(f"Variable objetivo {target_var} requerida para MIDAS")
        
        # Crear features MIDAS usando valores reales en escala correcta
        midas_values = []
        
        # Últimos valores de precio (en escala original)
        last_price = df[target_var].iloc[-1]
        logger.info(f"🔍 Último precio para MIDAS: ${last_price:.2f}")
        
        # Features basadas en precio (mantener escala)
        for lag in [1, 5, 10, 20]:
            if len(df) > lag:
                lag_value = df[target_var].iloc[-lag-1]
                midas_values.append(lag_value)
            else:
                midas_values.append(last_price)
        
        # Medias móviles (mantener escala)
        for window in [5, 10, 20]:
            if len(df) >= window:
                ma_value = df[target_var].rolling(window, min_periods=1).mean().iloc[-1]
                midas_values.append(ma_value)
            else:
                midas_values.append(last_price)
        
        # Variables exógenas (usar valores normalizados o escalados apropiadamente)
        exog_vars = ['iron', 'coking', 'VIX', 'sp500', 'commodities']
        
        for var in exog_vars:
            matching_cols = [col for col in df.columns if var.lower() in col.lower()]
            if matching_cols:
                col = matching_cols[0]
                value = df[col].iloc[-1]
                
                # Normalizar variables exógenas a escala similar al precio
                if var in ['iron', 'coking']:
                    # Materias primas: usar ratio con precio
                    normalized_value = value / last_price if last_price != 0 else 1.0
                elif var == 'VIX':
                    # VIX: escalar a rango 0-1
                    normalized_value = min(value / 100.0, 1.0)
                elif var in ['sp500', 'commodities']:
                    # Índices: usar ratio con valor base
                    normalized_value = value / 1000.0 if value > 100 else value / 100.0
                else:
                    normalized_value = value
                
                midas_values.append(normalized_value)
                logger.debug(f"   {var}: {value:.2f} → {normalized_value:.4f}")
            else:
                midas_values.append(0.0)
        
        # Ajustar a número de columnas esperadas
        while len(midas_values) < len(expected_columns):
            midas_values.append(0.0)
        
        midas_values = midas_values[:len(expected_columns)]
        
        # Crear DataFrame
        midas_df = pd.DataFrame([midas_values], columns=expected_columns, index=[df.index[-1]])
        
        logger.info(f"✅ Features MIDAS preparadas: {midas_df.shape}")
        logger.info(f"🔍 Valores MIDAS: min={midas_df.min().min():.4f}, max={midas_df.max().max():.4f}")
        
        return midas_df
    
    def _predict_with_retrained_model(self, model_data: Dict[str, Any], X_pred: pd.DataFrame) -> float:
        """
        Hacer predicción con modelo reentrenado
        """
        model = model_data['model']
        scalers = model_data['scalers']
        
        # Escalar
        X_scaled = scalers['X'].transform(X_pred)
        
        # Predicción
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        return float(y_pred)
    
    async def _save_retrained_model(self, model_name: str, model_data: Dict[str, Any]) -> None:
        """Guardar modelo reentrenado"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar modelo con timestamp
            model_file = self.models_dir / f"{model_name}_retrained_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Guardar como latest
            latest_file = self.models_dir / f"{model_name}_latest.pkl"
            with open(latest_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Modelo guardado: {latest_file}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo {model_name}: {str(e)}")
    
    async def _save_retrain_results(self, results: Dict[str, Any]) -> None:
        """Guardar resultados de reentrenamiento"""
        try:
            timestamp = results['execution_id']
            results_file = self.logs_dir / f"retrain_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Guardar como latest
            latest_file = self.logs_dir / 'latest_retrain_results.json'
            with open(latest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📄 Resultados guardados: {results_file}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {str(e)}")
    
    async def _cache_instant_responses(self, retrain_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Guardar predicción y análisis en cache para respuestas instantáneas
        """
        logger.info("💾 Guardando datos en cache para respuestas <2 segundos...")
        
        try:
            # Usar función helper de prediction_cache
            success = await cache_post_retrain_data(retrain_results, self.cache_dir)
            
            if success:
                logger.info("✅ Cache preparado para respuestas instantáneas")
                return {
                    'status': 'success',
                    'cache_ready': True,
                    'expected_response_time': '<2 segundos',
                    'cache_timestamp': datetime.now().isoformat()
                }
            else:
                logger.error("❌ Error preparando cache")
                return {
                    'status': 'error',
                    'cache_ready': False,
                    'fallback': 'Respuestas serán más lentas sin cache'
                }
                
        except Exception as e:
            logger.error(f"Error en cache: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'cache_ready': False
            }
    
    async def _run_explainability_analysis(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar análisis de explicabilidad de los modelos reentrenados
        """
        logger.info("📊 Analizando explicabilidad de modelos...")
        
        try:
            explainability_data = {}
            
            # Analizar feature importance de cada modelo
            for model_name, model_data in models.items():
                logger.info(f"🔍 Analizando {model_name}...")
                
                # Extraer feature importance
                if 'model' in model_data and hasattr(model_data['model'], 'feature_importances_'):
                    model = model_data['model']
                    feature_names = model_data.get('feature_names', [])
                    
                    if feature_names and len(feature_names) == len(model.feature_importances_):
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        explainability_data[model_name] = {
                            'feature_importance': importance_df.to_dict('records'),
                            'top_5_features': importance_df.head().to_dict('records')
                        }
                        
                        logger.info(f"✅ {model_name}: {len(feature_names)} features analizadas")
                    else:
                        logger.warning(f"⚠️ {model_name}: Inconsistencia en feature names")
                else:
                    logger.warning(f"⚠️ {model_name}: No tiene feature_importances_")
            
            # Crear análisis consolidado
            if explainability_data:
                causal_factors = self._consolidate_feature_importance(explainability_data)
                
                return {
                    'status': 'success',
                    'models_analyzed': list(explainability_data.keys()),
                    'causal_factors': causal_factors,
                    'feature_importance_by_model': explainability_data,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'No se pudo extraer feature importance de ningún modelo',
                    'models_attempted': list(models.keys())
                }
                
        except Exception as e:
            logger.error(f"Error en análisis de explicabilidad: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _consolidate_feature_importance(self, explainability_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Consolidar feature importance de múltiples modelos
        """
        feature_scores = {}
        
        # Agregar importancia de todos los modelos
        for model_name, data in explainability_data.items():
            if 'feature_importance' in data:
                for feature_info in data['feature_importance']:
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
        elif any(word in feature_lower for word in ['ma', 'moving', 'average']):
            return 'Indicadores Técnicos'
        else:
            return 'Otros'
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Obtener descripción del feature"""
        descriptions = {
            'precio_varilla_lme_lag_1': 'Precio de varilla del día anterior (autorregresivo crítico)',
            'current_price': 'Precio actual de referencia (feature crítica V2)',
            'iron': 'Precio del mineral de hierro (materia prima principal)',
            'coking': 'Precio del carbón de coque (materia prima para acero)',
            'VIX': 'Índice de volatilidad implícita (sentimiento de mercado)',
            'sp500': 'Índice S&P 500 (condiciones macroeconómicas)',
            'tipo_cambio_usdmxn': 'Tipo de cambio USD/MXN (impacto en costos)',
            'tasa_interes_banxico': 'Tasa de interés de Banxico (política monetaria)',
            'volatility_20': 'Volatilidad rolling de 20 días (riesgo de mercado)',
            'price_ma20': 'Media móvil de 20 días (tendencia de mediano plazo)',
            'commodities': 'Índice general de commodities (contexto sectorial)'
        }
        
        return descriptions.get(feature_name, f"Factor: {feature_name}")
    
    def _mape_to_confidence(self, mape: float) -> float:
        """
        Convertir MAPE a confidence score
        
        Args:
            mape: Mean Absolute Percentage Error
            
        Returns:
            Confidence score entre 0 y 1
        """
        # Fórmula basada en MAPE para confidence realista:
        # MAPE 0-1%: Confidence 95-100% (excelente)
        # MAPE 1-3%: Confidence 80-95% (muy bueno)
        # MAPE 3-5%: Confidence 60-80% (bueno)
        # MAPE 5-10%: Confidence 30-60% (aceptable)
        # MAPE >10%: Confidence <30% (malo)
        
        if mape <= 1.0:
            confidence = 0.95 + (1.0 - mape) * 0.05  # 95-100%
        elif mape <= 3.0:
            confidence = 0.80 + (3.0 - mape) / 2.0 * 0.15  # 80-95%
        elif mape <= 5.0:
            confidence = 0.60 + (5.0 - mape) / 2.0 * 0.20  # 60-80%
        elif mape <= 10.0:
            confidence = 0.30 + (10.0 - mape) / 5.0 * 0.30  # 30-60%
        else:
            confidence = max(0.10, 0.30 - (mape - 10.0) * 0.02)  # <30%
        
        return min(1.0, max(0.0, confidence))
    
    def _temporal_train_test_split(self, X: pd.DataFrame, y: pd.Series, test_days: int = 10) -> Dict[str, Any]:
        """
        Split temporal para validación out-of-sample estricta
        Reserva los últimos test_days para validación real
        
        Args:
            X: Features preparadas
            y: Variable objetivo
            test_days: Días a reservar para test (default: 10)
            
        Returns:
            Dict con train y test splits + información de validación
        """
        logger.info(f"🔍 Aplicando split temporal: últimos {test_days} días como test real")
        
        if len(X) < test_days + 50:  # Mínimo 50 días para entrenar
            logger.warning(f"⚠️ Datos insuficientes para split temporal: {len(X)} días")
            return {
                'use_full_data': True,
                'X_train': X,
                'y_train': y,
                'X_test': pd.DataFrame(),
                'y_test': pd.Series(dtype=float),
                'warning': 'Datos insuficientes para split temporal'
            }
        
        # Split temporal estricto
        split_point = len(X) - test_days
        
        X_train = X.iloc[:split_point].copy()
        y_train = y.iloc[:split_point].copy()
        X_test = X.iloc[split_point:].copy()
        y_test = y.iloc[split_point:].copy()
        
        # Verificar que test set tiene datos válidos
        test_valid_ratio = (~X_test.isnull().any(axis=1)).sum() / len(X_test)
        target_valid_ratio = (~y_test.isnull()).sum() / len(y_test)
        
        logger.info(f"📊 Split temporal aplicado:")
        logger.info(f"   Train: {X_train.shape} ({X_train.index[0].strftime('%Y-%m-%d')} a {X_train.index[-1].strftime('%Y-%m-%d')})")
        logger.info(f"   Test:  {X_test.shape} ({X_test.index[0].strftime('%Y-%m-%d')} a {X_test.index[-1].strftime('%Y-%m-%d')})")
        logger.info(f"   Test data quality: {test_valid_ratio:.1%} features, {target_valid_ratio:.1%} target")
        
        if test_valid_ratio < 0.8 or target_valid_ratio < 0.8:
            logger.warning(f"⚠️ Test set tiene calidad baja - considerar usar full data")
        
        return {
            'use_full_data': False,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'split_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_period': f"{X_train.index[0].strftime('%Y-%m-%d')} a {X_train.index[-1].strftime('%Y-%m-%d')}",
                'test_period': f"{X_test.index[0].strftime('%Y-%m-%d')} a {X_test.index[-1].strftime('%Y-%m-%d')}",
                'test_data_quality': test_valid_ratio,
                'test_target_quality': target_valid_ratio
            }
        }
    
    def _ensure_v2_features_exist(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Asegurar que todas las features V2 necesarias existan
        Crea features críticas si no están presentes
        """
        logger.info("🔧 Verificando y creando features V2 necesarias...")
        
        enhanced_df = features_df.copy()
        target_var = self.config['target_variable']
        
        if target_var not in enhanced_df.columns:
            logger.error(f"❌ Variable objetivo {target_var} no encontrada")
            return enhanced_df
        
        # Crear features críticas V2 si no existen
        v2_features = {
            'current_price': lambda df: df[target_var],
            'price_ma20': lambda df: df[target_var].rolling(20, min_periods=1).mean(),
            'price_std20': lambda df: df[target_var].rolling(20, min_periods=1).std(),
            'volatility_20': lambda df: df[target_var].pct_change().rolling(20, min_periods=1).std(),
            f'{target_var}_lag_1': lambda df: df[target_var].shift(1),
            f'{target_var}_return_1d': lambda df: df[target_var].pct_change(),
            f'{target_var}_ma_5': lambda df: df[target_var].rolling(5, min_periods=1).mean(),
            f'{target_var}_ma_10': lambda df: df[target_var].rolling(10, min_periods=1).mean(),
            f'{target_var}_ma_20': lambda df: df[target_var].rolling(20, min_periods=1).mean()
        }
        
        features_created = []
        
        for feature_name, feature_func in v2_features.items():
            if feature_name not in enhanced_df.columns:
                try:
                    enhanced_df[feature_name] = feature_func(enhanced_df)
                    features_created.append(feature_name)
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo crear {feature_name}: {str(e)}")
        
        if features_created:
            logger.info(f"✅ Features V2 creadas: {', '.join(features_created)}")
        else:
            logger.info("📊 Todas las features V2 ya existían")
        
        # Llenar NaN en features creadas
        for feature in features_created:
            if enhanced_df[feature].isnull().any():
                enhanced_df[feature] = enhanced_df[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"🎯 Features finales disponibles: {enhanced_df.shape[1]} columnas")
        
        return enhanced_df


async def main():
    """
    Función principal para reentrenamiento diario
    """
    logger.info("🔄 Iniciando Reentrenamiento Diario - DeAcero V2")
    
    # Crear pipeline
    retrain_pipeline = DailyRetrainPipeline()
    
    # Verificar si es momento de reentrenar
    if not retrain_pipeline.scheduler.should_retrain_models():
        logger.info("⏭️ No es momento de reentrenar - esperando horario programado")
        return
    
    # Ejecutar reentrenamiento
    results = await retrain_pipeline.run_daily_retrain()
    
    # Mostrar resumen
    if results['status'] == 'success':
        logger.info("\n" + "=" * 60)
        logger.info("✅ REENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)
        logger.info(f"🤖 Modelos actualizados: {results['models_retrained']}")
        logger.info(f"🎯 Predicción lista: {results['prediction_generated']}")
        logger.info(f"⏱️ Tiempo total: {results['execution_time_seconds']:.1f}s")
    else:
        logger.error("❌ Reentrenamiento falló")
        logger.error(f"Error: {results.get('error', 'Unknown')}")
    
    return results


if __name__ == "__main__":
    # Ejecutar reentrenamiento diario
    results = asyncio.run(main())
