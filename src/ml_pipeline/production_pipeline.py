#!/usr/bin/env python3
"""
Production Pipeline - DeAcero Steel Price Predictor
Pipeline completo de producción para predicción de precios de varilla corrugada

Este script orquesta todo el proceso:
1. Ingesta de datos desde múltiples fuentes
2. Join y consolidación de series temporales
3. Limpieza y validación de datos
4. Feature engineering para modelos V2
5. Carga y ejecución de modelos entrenados
6. Generación de predicciones con intervalos de confianza

Fecha: 27 de Septiembre de 2025
"""

import asyncio
import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Configurar paths del proyecto
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Imports del proyecto
from src.data_processing.cleaners import ProductionDataCleaner
from scripts.ingest_all_data import DataIngestionMaster
from scripts.join_daily_series import main as join_daily_main
from scripts.join_monthly_series import main as join_monthly_main

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionPipeline:
    """
    Pipeline completo de producción para predicción de precios de varilla corrugada
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.base_dir = BASE_DIR
        self.data_cleaner = ProductionDataCleaner()
        
        # Directorios importantes
        self.models_dir = self.base_dir / 'models' / 'production'
        self.data_processed_dir = self.base_dir / 'data' / 'processed'
        self.data_raw_dir = self.base_dir / 'data' / 'raw'
        
        # Crear directorios si no existen
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Variables de estado
        self.pipeline_state = {}
        self.execution_stats = {}
        
        logger.info(f"🚀 Pipeline de producción inicializado")
        logger.info(f"📁 Directorio base: {self.base_dir}")
        logger.info(f"🤖 Modelos disponibles: {self.config['models_to_use']}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del pipeline"""
        return {
            'models_to_use': ['XGBoost_V2_regime', 'MIDAS_V2_hibrida'],
            'target_variable': 'precio_varilla_lme',
            'forecast_horizon': 1,  # Predicción para t+1
            'data_validation': {
                'min_data_quality': 0.85,
                'require_target_quality': 0.95,
                'max_missing_days': 5
            },
            'feature_engineering': {
                'create_lags': True,
                'create_moving_averages': True,
                'create_volatility': True,
                'create_technical_indicators': True
            },
            'output': {
                'save_predictions': True,
                'save_feature_importance': True,
                'save_model_metrics': True
            }
        }
    
    async def run_full_pipeline(self, force_data_refresh: bool = False) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo de producción
        
        Args:
            force_data_refresh: Forzar actualización de datos aunque existan
            
        Returns:
            Resultados del pipeline con predicciones y métricas
        """
        logger.info("=" * 80)
        logger.info("🚀 INICIANDO PIPELINE DE PRODUCCIÓN COMPLETO")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        pipeline_results = {
            'execution_id': start_time.strftime('%Y%m%d_%H%M%S'),
            'start_time': start_time.isoformat(),
            'stages': {}
        }
        
        try:
            # ETAPA 1: Ingesta de datos
            logger.info("\n📥 ETAPA 1: INGESTA DE DATOS")
            logger.info("-" * 50)
            
            if force_data_refresh or self._needs_data_refresh():
                ingestion_results = await self._run_data_ingestion()
                pipeline_results['stages']['data_ingestion'] = ingestion_results
            else:
                logger.info("⏭️ Datos recientes disponibles, saltando ingesta")
                pipeline_results['stages']['data_ingestion'] = {'status': 'skipped', 'reason': 'recent_data_available'}
            
            # ETAPA 2: Consolidación de series temporales
            logger.info("\n🔗 ETAPA 2: CONSOLIDACIÓN DE SERIES")
            logger.info("-" * 50)
            
            consolidation_results = await self._run_data_consolidation()
            pipeline_results['stages']['data_consolidation'] = consolidation_results
            
            # ETAPA 3: Limpieza y validación
            logger.info("\n🧹 ETAPA 3: LIMPIEZA Y VALIDACIÓN")
            logger.info("-" * 50)
            
            cleaning_results = await self._run_data_cleaning()
            pipeline_results['stages']['data_cleaning'] = cleaning_results
            
            # ETAPA 4: Feature Engineering
            logger.info("\n⚙️ ETAPA 4: FEATURE ENGINEERING")
            logger.info("-" * 50)
            
            features_results = await self._run_feature_engineering()
            pipeline_results['stages']['feature_engineering'] = features_results
            
            # ETAPA 5: Carga de modelos y predicción
            logger.info("\n🤖 ETAPA 5: PREDICCIÓN CON MODELOS V2")
            logger.info("-" * 50)
            
            prediction_results = await self._run_prediction()
            pipeline_results['stages']['prediction'] = prediction_results
            
            # ETAPA 6: Análisis de explicabilidad
            logger.info("\n📊 ETAPA 6: ANÁLISIS DE EXPLICABILIDAD")
            logger.info("-" * 50)
            
            explainability_results = await self._run_explainability_analysis()
            pipeline_results['stages']['explainability'] = explainability_results
            
            # Finalización
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            pipeline_results.update({
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'status': 'success',
                'summary': self._generate_pipeline_summary(pipeline_results)
            })
            
            logger.info("=" * 80)
            logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info(f"⏱️ Tiempo total: {execution_time:.1f} segundos")
            logger.info("=" * 80)
            
            # Guardar resultados
            await self._save_pipeline_results(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Error en pipeline: {str(e)}", exc_info=True)
            
            pipeline_results.update({
                'end_time': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            })
            
            return pipeline_results
    
    def _needs_data_refresh(self) -> bool:
        """Verificar si se necesita actualizar los datos"""
        # Verificar si existen archivos consolidados recientes (últimas 24 horas)
        daily_file = self.data_processed_dir / 'daily_time_series' / 'daily_series_consolidated_latest.csv'
        monthly_file = self.data_processed_dir / 'monthly_time_series' / 'monthly_series_consolidated_latest.csv'
        
        if not daily_file.exists():
            logger.info("📥 Archivos diarios no encontrados - requiere ingesta")
            return True
        
        # Verificar antigüedad
        file_age = datetime.now() - datetime.fromtimestamp(daily_file.stat().st_mtime)
        if file_age > timedelta(hours=24):
            logger.info(f"📥 Datos antiguos ({file_age}) - requiere actualización")
            return True
        
        logger.info(f"📥 Datos recientes disponibles (edad: {file_age})")
        return False
    
    async def _run_data_ingestion(self) -> Dict[str, Any]:
        """Ejecutar ingesta completa de datos"""
        logger.info("🔄 Ejecutando ingesta de datos...")
        
        try:
            # Usar el master de ingesta existente
            master = DataIngestionMaster()
            results = await master.ingest_all_sources()
            
            return {
                'status': 'success',
                'sources_successful': results['summary']['sources']['successful'],
                'sources_failed': results['summary']['sources']['failed'],
                'total_series': results['summary']['data']['total_series'],
                'total_points': results['summary']['data']['total_points']
            }
            
        except Exception as e:
            logger.error(f"Error en ingesta: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _run_data_consolidation(self) -> Dict[str, Any]:
        """Ejecutar consolidación de series diarias y mensuales"""
        logger.info("🔄 Consolidando series temporales...")
        
        try:
            # Ejecutar join de series diarias
            logger.info("📊 Consolidando series diarias...")
            
            # Cambiar al directorio de scripts temporalmente
            original_cwd = os.getcwd()
            scripts_dir = self.base_dir / 'scripts'
            os.chdir(scripts_dir)
            
            try:
                # Ejecutar join diario
                daily_df = join_daily_main()
                daily_success = daily_df is not None
                daily_shape = daily_df.shape if daily_df is not None else (0, 0)
                
                # Ejecutar join mensual
                logger.info("📊 Consolidando series mensuales...")
                monthly_df = join_monthly_main()
                monthly_success = monthly_df is not None
                monthly_shape = monthly_df.shape if monthly_df is not None else (0, 0)
                
            finally:
                # Restaurar directorio original
                os.chdir(original_cwd)
            
            return {
                'status': 'success',
                'daily_consolidation': {
                    'success': daily_success,
                    'shape': daily_shape
                },
                'monthly_consolidation': {
                    'success': monthly_success,
                    'shape': monthly_shape
                }
            }
            
        except Exception as e:
            logger.error(f"Error en consolidación: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _run_data_cleaning(self) -> Dict[str, Any]:
        """Ejecutar limpieza y validación de datos"""
        logger.info("🔄 Limpiando y validando datos...")
        
        try:
            # Cargar datos consolidados
            daily_path = self.data_processed_dir / 'daily_time_series' / 'daily_series_consolidated_latest.csv'
            monthly_path = self.data_processed_dir / 'monthly_time_series' / 'monthly_series_consolidated_latest.csv'
            
            if not daily_path.exists():
                raise FileNotFoundError(f"Archivo diario no encontrado: {daily_path}")
            
            # Cargar datos
            daily_df = pd.read_csv(daily_path, index_col='fecha', parse_dates=True)
            logger.info(f"📊 Datos diarios cargados: {daily_df.shape}")
            
            monthly_df = None
            if monthly_path.exists():
                monthly_df = pd.read_csv(monthly_path, index_col='fecha', parse_dates=True)
                logger.info(f"📊 Datos mensuales cargados: {monthly_df.shape}")
            
            # Limpiar datos
            cleaned_data = self.data_cleaner.prepare_for_modeling(daily_df, monthly_df)
            
            # Validar para producción
            is_valid, validation_report = self.data_cleaner.validate_for_production(cleaned_data['daily'])
            
            if not is_valid:
                raise ValueError(f"Datos no aptos para producción: {validation_report['quality_grade']}")
            
            # Guardar datos limpios
            clean_daily_path = self.data_processed_dir / 'clean_daily_series_latest.csv'
            cleaned_data['daily'].to_csv(clean_daily_path)
            
            if 'monthly' in cleaned_data:
                clean_monthly_path = self.data_processed_dir / 'clean_monthly_series_latest.csv'
                cleaned_data['monthly'].to_csv(clean_monthly_path)
            
            return {
                'status': 'success',
                'validation_report': validation_report,
                'daily_shape': cleaned_data['daily'].shape,
                'monthly_shape': cleaned_data.get('monthly', pd.DataFrame()).shape,
                'data_quality_grade': validation_report['quality_grade'],
                'files_saved': [str(clean_daily_path)]
            }
            
        except Exception as e:
            logger.error(f"Error en limpieza: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _run_feature_engineering(self) -> Dict[str, Any]:
        """Ejecutar feature engineering para modelos V2"""
        logger.info("🔄 Aplicando feature engineering...")
        
        try:
            # Cargar datos limpios
            clean_daily_path = self.data_processed_dir / 'clean_daily_series_latest.csv'
            daily_df = pd.read_csv(clean_daily_path, index_col='fecha', parse_dates=True)
            
            # Aplicar interpolación final para asegurar completitud
            for col in daily_df.columns:
                daily_df[col] = daily_df[col].interpolate(method='linear', limit_direction='both')
            
            # Feature engineering específico para modelos V2
            features_df = self._create_v2_features(daily_df)
            
            # Guardar features
            features_path = self.data_processed_dir / 'features_v2_latest.csv'
            features_df.to_csv(features_path)
            
            # Preparar datos para cada modelo
            model_data = self._prepare_model_specific_data(features_df)
            
            return {
                'status': 'success',
                'features_shape': features_df.shape,
                'features_created': len(features_df.columns),
                'target_variable': self.config['target_variable'],
                'models_prepared': list(model_data.keys()),
                'features_file': str(features_path)
            }
            
        except Exception as e:
            logger.error(f"Error en feature engineering: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _create_v2_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features específicas para modelos V2
        Replica exactamente el feature engineering de los notebooks
        """
        logger.info("⚙️ Creando features V2...")
        
        features_df = df.copy()
        target_var = self.config['target_variable']
        
        if target_var not in features_df.columns:
            raise ValueError(f"Variable objetivo '{target_var}' no encontrada")
        
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
        features_df[f'{target_var}_return_20d'] = df[target_var].pct_change(20)
        
        # 5. INDICADORES TÉCNICOS
        # RSI (Relative Strength Index)
        delta = df[target_var].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df[f'{target_var}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        bb_ma = df[target_var].rolling(bb_window).mean()
        bb_std_dev = df[target_var].rolling(bb_window).std()
        features_df[f'{target_var}_bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        features_df[f'{target_var}_bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        features_df[f'{target_var}_bb_position'] = (df[target_var] - features_df[f'{target_var}_bb_lower']) / (features_df[f'{target_var}_bb_upper'] - features_df[f'{target_var}_bb_lower'])
        
        # MACD
        ema_12 = df[target_var].ewm(span=12).mean()
        ema_26 = df[target_var].ewm(span=26).mean()
        features_df[f'{target_var}_macd'] = ema_12 - ema_26
        features_df[f'{target_var}_macd_signal'] = features_df[f'{target_var}_macd'].ewm(span=9).mean()
        
        # 6. FEATURES DE CALENDARIO
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        features_df['is_month_end'] = df.index.is_month_end.astype(int)
        features_df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # 7. INTERACCIONES CON OTRAS VARIABLES
        # Buscar variables clave para interacciones
        key_vars = ['cobre_lme', 'tipo_cambio_usdmxn', 'sp500', 'vix_volatilidad']
        
        for var in key_vars:
            matching_cols = [col for col in df.columns if var in col.lower()]
            if matching_cols:
                other_var = matching_cols[0]
                features_df[f'steel_{var}_ratio'] = df[target_var] / df[other_var]
                features_df[f'steel_{var}_spread'] = df[target_var] - df[other_var]
                features_df[f'steel_{var}_correlation_5d'] = df[target_var].rolling(5).corr(df[other_var])
        
        logger.info(f"✅ Features V2 creadas: {features_df.shape[1]} columnas")
        
        return features_df
    
    def _prepare_model_specific_data(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Preparar datos específicos para cada modelo V2
        """
        logger.info("🎯 Preparando datos específicos por modelo...")
        
        # Definir combinaciones de variables para cada modelo
        model_combinations = {
            'XGBoost_V2_regime': {
                'variables': ['iron', 'coking', 'steel', 'VIX', 'sp500', 'tasa_interes_banxico'],
                'additional_features': ['current_price', 'price_ma20', 'price_std20']
            },
            'MIDAS_V2_hibrida': {
                'variables': ['precio_varilla_lme_lag_1', 'volatility_20', 'iron', 'coking', 'commodities', 'VIX'],
                'additional_features': ['current_price', 'price_ma20', 'price_std20']
            }
        }
        
        model_data = {}
        target_var = self.config['target_variable']
        
        for model_name, config in model_combinations.items():
            logger.info(f"📋 Preparando datos para {model_name}...")
            
            # Identificar variables disponibles
            available_vars = []
            for var in config['variables']:
                if var in features_df.columns:
                    available_vars.append(var)
                else:
                    # Buscar variables similares
                    matching = [col for col in features_df.columns if var.lower() in col.lower()]
                    if matching:
                        available_vars.append(matching[0])
                        logger.info(f"   Usando {matching[0]} para {var}")
                    else:
                        logger.warning(f"   Variable {var} no encontrada")
            
            if len(available_vars) < len(config['variables']) * 0.8:
                logger.warning(f"Solo {len(available_vars)}/{len(config['variables'])} variables disponibles para {model_name}")
            
            # Preparar X e y
            X = features_df[available_vars].copy()
            y = features_df[target_var].copy()
            
            # Agregar features adicionales (críticas para V2)
            y_current = features_df[target_var]
            X['current_price'] = y_current
            X['price_ma20'] = y_current.rolling(20).mean()
            X['price_std20'] = y_current.rolling(20).std()
            
            # Eliminar NaN
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            model_data[model_name] = {
                'X': X_clean,
                'y': y_clean,
                'feature_names': list(X_clean.columns),
                'available_variables': available_vars,
                'data_range': {
                    'start': X_clean.index[0].strftime('%Y-%m-%d'),
                    'end': X_clean.index[-1].strftime('%Y-%m-%d'),
                    'observations': len(X_clean)
                }
            }
            
            logger.info(f"   ✅ {model_name}: {X_clean.shape} datos preparados")
        
        return model_data
    
    async def _run_prediction(self) -> Dict[str, Any]:
        """Ejecutar predicción con modelos V2"""
        logger.info("🔄 Ejecutando predicción con modelos...")
        
        try:
            # Cargar features preparadas
            features_path = self.data_processed_dir / 'features_v2_latest.csv'
            features_df = pd.read_csv(features_path, index_col='fecha', parse_dates=True)
            
            predictions = {}
            
            for model_name in self.config['models_to_use']:
                logger.info(f"🤖 Cargando modelo {model_name}...")
                
                # Buscar archivo de modelo
                model_files = list(self.models_dir.glob(f"{model_name}*.pkl"))
                if not model_files:
                    # Buscar en directorio de test
                    test_models_dir = self.base_dir / 'models' / 'test'
                    model_files = list(test_models_dir.glob(f"{model_name}*.pkl"))
                
                if not model_files:
                    logger.warning(f"Modelo {model_name} no encontrado")
                    continue
                
                model_path = model_files[0]
                
                try:
                    # Cargar modelo
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Generar predicción
                    prediction_result = await self._generate_prediction_v2(model_data, features_df, model_name)
                    predictions[model_name] = prediction_result
                    
                    logger.info(f"✅ {model_name}: Predicción generada")
                    
                except Exception as e:
                    logger.error(f"Error con modelo {model_name}: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("No se pudieron cargar modelos para predicción")
            
            # Seleccionar mejor predicción
            best_model = self._select_best_prediction(predictions)
            
            return {
                'status': 'success',
                'predictions': predictions,
                'best_model': best_model,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _generate_prediction_v2(self, model_data: Dict[str, Any], features_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Generar predicción usando modelo V2 específico
        """
        try:
            # Preparar datos según el tipo de modelo
            if 'XGBoost' in model_name:
                return await self._predict_xgboost_v2(model_data, features_df, model_name)
            elif 'MIDAS' in model_name:
                return await self._predict_midas_v2(model_data, features_df, model_name)
            else:
                raise ValueError(f"Tipo de modelo no reconocido: {model_name}")
                
        except Exception as e:
            logger.error(f"Error generando predicción para {model_name}: {str(e)}")
            raise
    
    async def _predict_xgboost_v2(self, model_data: Dict[str, Any], features_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Predicción específica para XGBoost V2"""
        # Determinar combinación de variables
        if 'regime' in model_name:
            variables = ['iron', 'coking', 'steel', 'VIX', 'sp500', 'tasa_interes_banxico']
        else:  # hibrida
            variables = ['precio_varilla_lme_lag_1', 'volatility_20', 'iron', 'coking', 'commodities', 'VIX']
        
        # Preparar features
        available_vars = []
        for var in variables:
            matching = [col for col in features_df.columns if var.lower() in col.lower()]
            if matching:
                available_vars.append(matching[0])
        
        # Crear X con features adicionales críticas
        X = features_df[available_vars].copy()
        target_var = self.config['target_variable']
        
        X['current_price'] = features_df[target_var]
        X['price_ma20'] = features_df[target_var].rolling(20).mean()
        X['price_std20'] = features_df[target_var].rolling(20).std()
        
        # Tomar última observación para predicción
        X_latest = X.iloc[-1:].copy()
        
        # Eliminar NaN
        X_latest = X_latest.fillna(X_latest.median())
        
        # Usar modelo y scalers
        model = model_data['model']
        scalers = model_data['scalers']
        
        # Escalar
        X_scaled = scalers['X'].transform(X_latest)
        
        # Predicción
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Calcular intervalo de confianza basado en métricas históricas
        test_metrics = model_data.get('test_metrics', {})
        rmse = test_metrics.get('rmse', 0.05)
        
        confidence_interval = {
            'lower': y_pred - (1.96 * rmse),  # 95% CI
            'upper': y_pred + (1.96 * rmse)
        }
        
        return {
            'model_name': model_name,
            'prediction': float(y_pred),
            'confidence_interval': confidence_interval,
            'confidence_score': test_metrics.get('r2', 0.8),
            'features_used': list(X_latest.columns),
            'historical_metrics': test_metrics
        }
    
    async def _predict_midas_v2(self, model_data: Dict[str, Any], features_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Predicción específica para MIDAS V2"""
        # Usar features MIDAS pre-procesadas si están disponibles
        if 'midas_features' in model_data and 'test' in model_data['midas_features']:
            # Usar features MIDAS guardadas (tomar última observación)
            midas_features = model_data['midas_features']['test']
            if isinstance(midas_features, pd.DataFrame):
                X_latest = midas_features.iloc[-1:].copy()
            else:
                X_latest = pd.DataFrame(midas_features[-1:])
        else:
            # Crear features MIDAS dinámicamente
            variables = ['precio_varilla_lme_lag_1', 'volatility_20', 'iron', 'coking', 'commodities', 'VIX']
            available_vars = []
            
            for var in variables:
                matching = [col for col in features_df.columns if var.lower() in col.lower()]
                if matching:
                    available_vars.append(matching[0])
            
            X_latest = features_df[available_vars].iloc[-1:].copy()
        
        # Eliminar NaN
        X_latest = X_latest.fillna(X_latest.median())
        
        # Usar modelo y scalers
        model = model_data['model']
        scalers = model_data['scalers']
        
        # Escalar
        X_scaled = scalers['X'].transform(X_latest)
        
        # Predicción
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Intervalo de confianza
        test_metrics = model_data.get('test_metrics', {})
        rmse = test_metrics.get('rmse', 0.05)
        
        confidence_interval = {
            'lower': y_pred - (1.96 * rmse),
            'upper': y_pred + (1.96 * rmse)
        }
        
        return {
            'model_name': model_name,
            'prediction': float(y_pred),
            'confidence_interval': confidence_interval,
            'confidence_score': test_metrics.get('r2', 0.8),
            'features_used': list(X_latest.columns),
            'historical_metrics': test_metrics
        }
    
    def _select_best_prediction(self, predictions: Dict[str, Any]) -> str:
        """Seleccionar la mejor predicción basada en métricas históricas"""
        best_model = None
        best_score = -1
        
        for model_name, pred_data in predictions.items():
            # Usar R² como criterio principal
            r2 = pred_data.get('confidence_score', 0)
            if r2 > best_score:
                best_score = r2
                best_model = model_name
        
        logger.info(f"🏆 Mejor modelo seleccionado: {best_model} (R²: {best_score:.3f})")
        return best_model
    
    async def _run_explainability_analysis(self) -> Dict[str, Any]:
        """Ejecutar análisis de explicabilidad y factores causales"""
        logger.info("🔄 Analizando explicabilidad...")
        
        try:
            # Cargar modelos y extraer feature importance
            explainability_data = {}
            
            for model_name in self.config['models_to_use']:
                model_files = list((self.base_dir / 'models' / 'test').glob(f"{model_name}*.pkl"))
                if model_files:
                    with open(model_files[0], 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Extraer feature importance
                    if 'feature_importance' in model_data:
                        importance_df = model_data['feature_importance']
                        explainability_data[model_name] = {
                            'feature_importance': importance_df.to_dict('records'),
                            'top_5_features': importance_df.head().to_dict('records')
                        }
                    elif hasattr(model_data.get('model'), 'feature_importances_'):
                        # Para modelos XGBoost/LightGBM
                        model = model_data['model']
                        if hasattr(model, 'feature_names_in_'):
                            importance_df = pd.DataFrame({
                                'feature': model.feature_names_in_,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            explainability_data[model_name] = {
                                'feature_importance': importance_df.to_dict('records'),
                                'top_5_features': importance_df.head().to_dict('records')
                            }
            
            # Crear análisis consolidado de factores causales
            causal_factors = self._analyze_causal_factors(explainability_data)
            
            return {
                'status': 'success',
                'models_analyzed': list(explainability_data.keys()),
                'causal_factors': causal_factors,
                'feature_importance_by_model': explainability_data
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de explicabilidad: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_causal_factors(self, explainability_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analizar factores causales consolidados de todos los modelos
        """
        # Consolidar importancia de features de todos los modelos
        feature_importance_consolidated = {}
        
        for model_name, data in explainability_data.items():
            if 'feature_importance' in data:
                for feature_info in data['feature_importance']:
                    feature = feature_info['feature']
                    importance = feature_info['importance']
                    
                    if feature not in feature_importance_consolidated:
                        feature_importance_consolidated[feature] = []
                    
                    feature_importance_consolidated[feature].append({
                        'model': model_name,
                        'importance': importance
                    })
        
        # Calcular importancia promedio y crear ranking
        causal_factors = []
        
        for feature, model_importances in feature_importance_consolidated.items():
            avg_importance = np.mean([mi['importance'] for mi in model_importances])
            
            # Categorizar el factor
            category = self._categorize_factor(feature)
            
            causal_factors.append({
                'factor_name': feature,
                'average_importance': float(avg_importance),
                'category': category,
                'models_using': [mi['model'] for mi in model_importances],
                'description': self._get_factor_description(feature)
            })
        
        # Ordenar por importancia
        causal_factors.sort(key=lambda x: x['average_importance'], reverse=True)
        
        return causal_factors[:20]  # Top 20 factores
    
    def _categorize_factor(self, feature_name: str) -> str:
        """Categorizar factor según su naturaleza económica"""
        feature_lower = feature_name.lower()
        
        if any(word in feature_lower for word in ['precio', 'price', 'varilla', 'steel', 'lag']):
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
            return 'Otros Indicadores'
    
    def _get_factor_description(self, feature_name: str) -> str:
        """Obtener descripción del factor"""
        descriptions = {
            'precio_varilla_lme_lag_1': 'Precio de varilla del día anterior (autorregresivo)',
            'iron': 'Precio del mineral de hierro',
            'coking': 'Precio del carbón de coque',
            'VIX': 'Índice de volatilidad del mercado',
            'sp500': 'Índice S&P 500',
            'tipo_cambio_usdmxn': 'Tipo de cambio USD/MXN',
            'tasa_interes_banxico': 'Tasa de interés de Banxico',
            'volatility_20': 'Volatilidad rolling de 20 días',
            'current_price': 'Precio actual (feature crítica V2)',
            'price_ma20': 'Media móvil de 20 días',
            'commodities': 'Índice de commodities'
        }
        
        return descriptions.get(feature_name, f"Indicador: {feature_name}")
    
    async def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Guardar resultados del pipeline"""
        try:
            # Crear directorio de resultados
            results_dir = self.data_processed_dir / 'pipeline_results'
            results_dir.mkdir(exist_ok=True)
            
            # Guardar resultados completos
            results_file = results_dir / f"pipeline_results_{results['execution_id']}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Guardar última ejecución
            latest_file = results_dir / 'latest_pipeline_results.json'
            with open(latest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Resultados guardados en: {results_file}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {str(e)}")
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generar resumen ejecutivo del pipeline"""
        summary = {
            'execution_time': results.get('execution_time_seconds', 0),
            'stages_completed': len([s for s in results['stages'].values() if s.get('status') == 'success']),
            'total_stages': len(results['stages']),
            'data_quality': 'Unknown',
            'models_executed': 0,
            'predictions_generated': 0
        }
        
        # Extraer información clave
        if 'data_cleaning' in results['stages']:
            cleaning = results['stages']['data_cleaning']
            if cleaning.get('status') == 'success':
                summary['data_quality'] = cleaning.get('data_quality_grade', 'Unknown')
        
        if 'prediction' in results['stages']:
            prediction = results['stages']['prediction']
            if prediction.get('status') == 'success':
                summary['models_executed'] = len(prediction.get('predictions', {}))
                summary['predictions_generated'] = len(prediction.get('predictions', {}))
                summary['best_model'] = prediction.get('best_model', 'Unknown')
        
        return summary


async def main():
    """
    Función principal para ejecutar el pipeline
    """
    logger.info("🚀 Iniciando Pipeline de Producción - DeAcero Steel Price Predictor")
    
    # Crear pipeline
    pipeline = ProductionPipeline()
    
    # Ejecutar pipeline completo
    results = await pipeline.run_full_pipeline(force_data_refresh=False)
    
    # Mostrar resumen
    if results['status'] == 'success':
        summary = results['summary']
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMEN DE EJECUCIÓN")
        logger.info("=" * 60)
        logger.info(f"✅ Etapas completadas: {summary['stages_completed']}/{summary['total_stages']}")
        logger.info(f"📊 Calidad de datos: {summary['data_quality']}")
        logger.info(f"🤖 Modelos ejecutados: {summary['models_executed']}")
        logger.info(f"🎯 Predicciones generadas: {summary['predictions_generated']}")
        logger.info(f"🏆 Mejor modelo: {summary.get('best_model', 'N/A')}")
        logger.info(f"⏱️ Tiempo total: {summary['execution_time']:.1f}s")
        
        # Mostrar predicción final si está disponible
        if 'prediction' in results['stages'] and results['stages']['prediction']['status'] == 'success':
            pred_data = results['stages']['prediction']
            best_model = pred_data['best_model']
            best_prediction = pred_data['predictions'][best_model]
            
            logger.info("\n" + "=" * 60)
            logger.info("🎯 PREDICCIÓN FINAL")
            logger.info("=" * 60)
            logger.info(f"🤖 Modelo: {best_model}")
            logger.info(f"💰 Precio predicho: ${best_prediction['prediction']:.2f} USD/ton")
            logger.info(f"📊 Confianza: {best_prediction['confidence_score']:.1%}")
            logger.info(f"📈 Intervalo: ${best_prediction['confidence_interval']['lower']:.2f} - ${best_prediction['confidence_interval']['upper']:.2f}")
    else:
        logger.error("❌ Pipeline falló")
        logger.error(f"Error: {results.get('error', 'Unknown')}")
    
    return results


if __name__ == "__main__":
    # Ejecutar pipeline
    results = asyncio.run(main())
