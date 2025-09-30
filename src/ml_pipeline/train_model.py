"""
Model Training Pipeline
Pipeline para entrenamiento de modelos de predicción de precios de acero
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import joblib
import json
from pathlib import Path

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)


class SteelPriceModelTrainer:
    """
    Entrenador de modelos para predicción de precios de acero
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.best_model = None
        self.best_model_name = None
        
        # Configuración de modelos
        self.model_configs = {
            'xgboost': {
                'model_class': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'random_state': [random_state]
                },
                'requires_scaling': False
            },
            'lightgbm': {
                'model_class': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'random_state': [random_state]
                },
                'requires_scaling': False
            },
            'random_forest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'random_state': [random_state]
                },
                'requires_scaling': False
            },
            'gradient_boosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9],
                    'random_state': [random_state]
                },
                'requires_scaling': False
            },
            'ridge': {
                'model_class': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'random_state': [random_state]
                },
                'requires_scaling': True
            },
            'lasso': {
                'model_class': Lasso,
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'random_state': [random_state]
                },
                'requires_scaling': True
            },
            'elastic_net': {
                'model_class': ElasticNet,
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9],
                    'random_state': [random_state]
                },
                'requires_scaling': True
            }
        }
    
    def prepare_time_series_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Preparar datos para entrenamiento de series temporales
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proporción para test set
            val_size: Proporción para validation set
            
        Returns:
            Tuple con train, val, test splits
        """
        logger.info(f"Preparando datos temporales: {X.shape[0]} observaciones")
        
        # Para series temporales, usar splits temporales (no aleatorios)
        n_samples = len(X)
        
        # Calcular índices de corte
        test_start_idx = int(n_samples * (1 - test_size))
        val_start_idx = int(n_samples * (1 - test_size - val_size))
        
        # Splits temporales
        X_train = X.iloc[:val_start_idx]
        X_val = X.iloc[val_start_idx:test_start_idx]
        X_test = X.iloc[test_start_idx:]
        
        y_train = y.iloc[:val_start_idx]
        y_val = y.iloc[val_start_idx:test_start_idx]
        y_test = y.iloc[test_start_idx:]
        
        logger.info(f"Splits creados:")
        logger.info(f"   Train: {len(X_train)} observaciones ({len(X_train)/n_samples:.1%})")
        logger.info(f"   Validation: {len(X_val)} observaciones ({len(X_val)/n_samples:.1%})")
        logger.info(f"   Test: {len(X_test)} observaciones ({len(X_test)/n_samples:.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        use_grid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Entrenar un modelo específico
        
        Args:
            model_name: Nombre del modelo a entrenar
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            use_grid_search: Usar búsqueda de hiperparámetros
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Modelo '{model_name}' no encontrado en configuración")
        
        logger.info(f"Entrenando modelo: {model_name}")
        
        config = self.model_configs[model_name]
        start_time = datetime.now()
        
        # Preparar datos
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy() if X_val is not None else None
        
        # Aplicar scaling si es necesario
        scaler = None
        if config['requires_scaling']:
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
        
        # Entrenar modelo
        if use_grid_search and len(config['params']) > 1:
            # Usar GridSearchCV con validación temporal
            tscv = TimeSeriesSplit(n_splits=3)
            
            model = GridSearchCV(
                config['model_class'](),
                param_grid=config['params'],
                cv=tscv,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1,
                verbose=1
            )
            
            model.fit(X_train_scaled, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_
            
        else:
            # Entrenar con parámetros por defecto
            default_params = {k: v[0] if isinstance(v, list) else v 
                            for k, v in config['params'].items()}
            
            best_model = config['model_class'](**default_params)
            best_model.fit(X_train_scaled, y_train)
            best_params = default_params
        
        # Evaluación
        train_pred = best_model.predict(X_train_scaled)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = best_model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, val_pred)
        
        # Guardar modelo y scaler
        self.models[model_name] = best_model
        if scaler:
            self.scalers[model_name] = scaler
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Resultados del entrenamiento
        training_result = {
            'model_name': model_name,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time_seconds': training_time,
            'features_count': X_train.shape[1],
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if X_val is not None else 0,
            'requires_scaling': config['requires_scaling'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history[model_name] = training_result
        
        logger.info(f"Modelo {model_name} entrenado exitosamente")
        logger.info(f"   Train MAPE: {train_metrics['mape']:.2f}%")
        if val_metrics:
            logger.info(f"   Val MAPE: {val_metrics['mape']:.2f}%")
        
        return training_result
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Entrenar todos los modelos configurados
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            models_to_train: Lista de modelos a entrenar (None para todos)
            
        Returns:
            Diccionario con resultados de todos los entrenamientos
        """
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        logger.info(f"Entrenando {len(models_to_train)} modelos...")
        
        all_results = {}
        
        for model_name in models_to_train:
            try:
                result = self.train_model(model_name, X_train, y_train, X_val, y_val)
                all_results[model_name] = result
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo basado en validation MAPE
        if all_results:
            best_model_name = min(
                all_results.keys(),
                key=lambda x: all_results[x]['val_metrics'].get('mape', float('inf'))
                if all_results[x]['val_metrics'] else all_results[x]['train_metrics']['mape']
            )
            
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"🏆 Mejor modelo: {best_model_name}")
            best_mape = all_results[best_model_name]['val_metrics'].get('mape') or all_results[best_model_name]['train_metrics']['mape']
            logger.info(f"   MAPE: {best_mape:.2f}%")
        
        return all_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de evaluación"""
        
        # Asegurar que no hay valores NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No hay valores válidos para calcular métricas")
            return {'mape': 100.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        metrics = {
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'r2': r2_score(y_true_clean, y_pred_clean),
            'mape': mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
        }
        
        return metrics
    
    def save_best_model(self, model_path: str = '../data/models/steel_price_model.pkl') -> bool:
        """
        Guardar el mejor modelo entrenado
        
        Args:
            model_path: Ruta donde guardar el modelo
            
        Returns:
            True si se guardó exitosamente
        """
        if not self.best_model:
            logger.error("No hay modelo entrenado para guardar")
            return False
        
        try:
            # Crear directorio si no existe
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar modelo
            joblib.dump(self.best_model, model_path)
            
            # Guardar scaler si existe
            if self.best_model_name in self.scalers:
                scaler_path = model_path.replace('.pkl', '_scaler.pkl')
                joblib.dump(self.scalers[self.best_model_name], scaler_path)
            
            # Guardar métricas del modelo
            metrics_path = model_path.replace('.pkl', '_metrics.json')
            model_info = {
                'model_name': self.best_model_name,
                'training_timestamp': datetime.now().isoformat(),
                'metrics': self.training_history[self.best_model_name],
                'feature_importance': self._get_feature_importance(),
                'model_parameters': self.training_history[self.best_model_name]['best_params']
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            logger.info(f"Modelo guardado exitosamente: {model_path}")
            logger.info(f"Métricas guardadas: {metrics_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            return False
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Obtener importancia de features del mejor modelo"""
        
        if not self.best_model:
            return {}
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                # Tree-based models
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                # Linear models
                importances = np.abs(self.best_model.coef_)
            else:
                logger.warning("Modelo no soporta feature importance")
                return {}
            
            # Asumir que tenemos nombres de features disponibles
            # En producción, esto vendría del pipeline de features
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Normalizar importancias
            importances_normalized = importances / importances.sum()
            
            return dict(zip(feature_names, importances_normalized))
            
        except Exception as e:
            logger.error(f"Error obteniendo feature importance: {str(e)}")
            return {}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtener resumen del entrenamiento"""
        
        if not self.training_history:
            return {"status": "no_models_trained"}
        
        # Comparar todos los modelos entrenados
        models_comparison = []
        
        for model_name, history in self.training_history.items():
            val_metrics = history.get('val_metrics', {})
            train_metrics = history.get('train_metrics', {})
            
            models_comparison.append({
                'model_name': model_name,
                'train_mape': train_metrics.get('mape', np.nan),
                'val_mape': val_metrics.get('mape', np.nan),
                'train_r2': train_metrics.get('r2', np.nan),
                'val_r2': val_metrics.get('r2', np.nan),
                'training_time': history.get('training_time_seconds', 0),
                'features_count': history.get('features_count', 0)
            })
        
        comparison_df = pd.DataFrame(models_comparison)
        
        summary = {
            'total_models_trained': len(self.training_history),
            'best_model': self.best_model_name,
            'models_comparison': comparison_df.to_dict('records'),
            'training_timestamp': datetime.now().isoformat(),
            'best_model_metrics': self.training_history.get(self.best_model_name, {}) if self.best_model_name else {}
        }
        
        return summary


def create_lag_features(df: pd.DataFrame, target_col: str, lags: List[int] = [1, 2, 3, 5, 7]) -> pd.DataFrame:
    """
    Crear features con lags temporales
    
    Args:
        df: DataFrame con datos temporales
        target_col: Columna objetivo
        lags: Lista de lags a crear
        
    Returns:
        DataFrame con features de lag
    """
    lag_df = df.copy()
    
    for lag in lags:
        lag_df[f"{target_col}_lag_{lag}"] = lag_df[target_col].shift(lag)
    
    return lag_df


def create_rolling_features(
    df: pd.DataFrame, 
    target_col: str, 
    windows: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Crear features de ventanas móviles
    
    Args:
        df: DataFrame con datos temporales
        target_col: Columna objetivo
        windows: Lista de ventanas a crear
        
    Returns:
        DataFrame con features de rolling
    """
    rolling_df = df.copy()
    
    for window in windows:
        rolling_df[f"{target_col}_ma_{window}"] = rolling_df[target_col].rolling(window=window).mean()
        rolling_df[f"{target_col}_std_{window}"] = rolling_df[target_col].rolling(window=window).std()
        rolling_df[f"{target_col}_min_{window}"] = rolling_df[target_col].rolling(window=window).min()
        rolling_df[f"{target_col}_max_{window}"] = rolling_df[target_col].rolling(window=window).max()
    
    return rolling_df


# Función principal de entrenamiento
def train_steel_price_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_to_train: Optional[List[str]] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Función principal para entrenar modelos de predicción de precios de acero
    
    Args:
        X: Features DataFrame
        y: Target Series
        models_to_train: Lista de modelos a entrenar
        save_model: Guardar mejor modelo
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    logger.info("Iniciando entrenamiento de modelos de predicción de precios de acero")
    
    # Inicializar entrenador
    trainer = SteelPriceModelTrainer()
    
    # Preparar datos
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_time_series_data(X, y)
    
    # Entrenar modelos
    training_results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, models_to_train
    )
    
    # Guardar mejor modelo si se solicita
    if save_model and trainer.best_model:
        trainer.save_best_model()
    
    # Obtener resumen
    summary = trainer.get_training_summary()
    
    return {
        'training_results': training_results,
        'summary': summary,
        'data_splits': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        'trainer': trainer  # Para uso posterior
    }
