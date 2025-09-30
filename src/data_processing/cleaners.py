"""
Data Cleaning Utilities - DeAcero Steel Price Predictor
Funciones avanzadas para limpieza y preprocesamiento de series temporales consolidadas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clase avanzada para limpieza y preprocesamiento de series temporales consolidadas
    Optimizada para el pipeline de producción de DeAcero Steel Price Predictor
    """
    
    def __init__(self):
        self.cleaning_stats = {}
        self.interpolation_stats = {}
        self.target_variable = 'precio_varilla_lme'
    
    def clean_time_series(self, df: pd.DataFrame, date_col: str = 'fecha', value_col: str = 'valor') -> pd.DataFrame:
        """
        Limpiar serie temporal
        
        Args:
            df: DataFrame con datos de serie temporal
            date_col: Nombre de columna de fechas
            value_col: Nombre de columna de valores
            
        Returns:
            DataFrame limpio
        """
        logger.info(f"Limpiando serie temporal con {len(df)} registros")
        
        original_count = len(df)
        
        # Copiar DataFrame
        clean_df = df.copy()
        
        # 1. Convertir fechas
        clean_df[date_col] = pd.to_datetime(clean_df[date_col])
        
        # 2. Ordenar por fecha
        clean_df = clean_df.sort_values(date_col)
        
        # 3. Eliminar duplicados por fecha
        duplicates_count = clean_df.duplicated(subset=[date_col]).sum()
        if duplicates_count > 0:
            logger.warning(f"Eliminando {duplicates_count} fechas duplicadas")
            clean_df = clean_df.drop_duplicates(subset=[date_col], keep='last')
        
        # 4. Manejar valores faltantes
        missing_count = clean_df[value_col].isna().sum()
        if missing_count > 0:
            logger.info(f"Manejando {missing_count} valores faltantes")
            clean_df = self._handle_missing_values(clean_df, value_col)
        
        # 5. Detectar y manejar outliers
        outliers_count = self._detect_outliers(clean_df[value_col])
        if outliers_count > 0:
            logger.info(f"Detectados {outliers_count} outliers")
            clean_df = self._handle_outliers(clean_df, value_col)
        
        # 6. Validar consistencia temporal
        clean_df = self._validate_temporal_consistency(clean_df, date_col, value_col)
        
        # Estadísticas de limpieza
        final_count = len(clean_df)
        self.cleaning_stats = {
            'original_records': original_count,
            'final_records': final_count,
            'records_removed': original_count - final_count,
            'duplicates_removed': duplicates_count,
            'missing_values_handled': missing_count,
            'outliers_detected': outliers_count,
            'data_quality_score': self._calculate_quality_score(clean_df, value_col)
        }
        
        logger.info(f"Limpieza completada: {original_count} → {final_count} registros")
        
        return clean_df
    
    def _handle_missing_values(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Manejar valores faltantes"""
        
        # Estrategias según porcentaje de missing values
        missing_pct = df[value_col].isna().sum() / len(df)
        
        if missing_pct > 0.3:
            logger.warning(f"Alto porcentaje de valores faltantes: {missing_pct:.1%}")
            # Para series con muchos faltantes, usar interpolación simple
            df[value_col] = df[value_col].interpolate(method='linear')
        elif missing_pct > 0.1:
            # Interpolación temporal
            df[value_col] = df[value_col].interpolate(method='time')
        else:
            # Forward fill para pocos faltantes
            df[value_col] = df[value_col].fillna(method='ffill')
        
        # Backward fill para cualquier faltante restante
        df[value_col] = df[value_col].fillna(method='bfill')
        
        return df
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> int:
        """Detectar outliers usando IQR o Z-score"""
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        else:  # z-score
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = z_scores > 3
        
        return outliers.sum()
    
    def _handle_outliers(self, df: pd.DataFrame, value_col: str, method: str = 'cap') -> pd.DataFrame:
        """Manejar outliers"""
        
        series = df[value_col]
        
        if method == 'cap':
            # Winsorization - cap outliers at percentiles
            lower_cap = series.quantile(0.01)
            upper_cap = series.quantile(0.99)
            df[value_col] = series.clip(lower=lower_cap, upper=upper_cap)
        elif method == 'remove':
            # Remover outliers extremos
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Más conservador
            upper_bound = Q3 + 3 * IQR
            df = df[(series >= lower_bound) & (series <= upper_bound)]
        
        return df
    
    def _validate_temporal_consistency(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Validar consistencia temporal"""
        
        # Verificar orden temporal
        if not df[date_col].is_monotonic_increasing:
            logger.warning("Datos no están en orden temporal, reordenando")
            df = df.sort_values(date_col)
        
        # Detectar gaps temporales grandes
        df['date_diff'] = df[date_col].diff()
        median_diff = df['date_diff'].median()
        
        large_gaps = df['date_diff'] > median_diff * 5
        if large_gaps.any():
            gap_count = large_gaps.sum()
            logger.warning(f"Detectados {gap_count} gaps temporales grandes")
        
        # Remover columna temporal auxiliar
        df = df.drop('date_diff', axis=1)
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame, value_col: str) -> float:
        """Calcular score de calidad de datos (0-100)"""
        
        scores = []
        
        # Completitud
        completeness = (1 - df[value_col].isna().sum() / len(df)) * 100
        scores.append(completeness)
        
        # Consistencia (sin valores negativos para precios)
        if df[value_col].min() >= 0:
            scores.append(100)
        else:
            negative_pct = (df[value_col] < 0).sum() / len(df)
            scores.append(max(0, 100 - negative_pct * 100))
        
        # Variabilidad (no demasiado constante)
        cv = df[value_col].std() / df[value_col].mean() if df[value_col].mean() > 0 else 0
        variability_score = min(100, cv * 100) if cv > 0.01 else 50
        scores.append(variability_score)
        
        return np.mean(scores)
    
    def clean_multiple_series(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Limpiar múltiples series temporales"""
        
        cleaned_data = {}
        total_stats = {
            'series_count': len(data_dict),
            'total_original_records': 0,
            'total_final_records': 0,
            'avg_quality_score': 0
        }
        
        for series_name, df in data_dict.items():
            logger.info(f"Limpiando serie: {series_name}")
            
            cleaned_df = self.clean_time_series(df)
            cleaned_data[series_name] = cleaned_df
            
            # Acumular estadísticas
            total_stats['total_original_records'] += self.cleaning_stats['original_records']
            total_stats['total_final_records'] += self.cleaning_stats['final_records']
            total_stats['avg_quality_score'] += self.cleaning_stats['data_quality_score']
        
        # Promediar quality score
        total_stats['avg_quality_score'] /= len(data_dict)
        
        logger.info(f"Limpieza completada para {len(data_dict)} series")
        logger.info(f"Quality score promedio: {total_stats['avg_quality_score']:.1f}")
        
        return cleaned_data, total_stats


class FeaturesCleaner:
    """
    Clase especializada para limpieza de features para ML
    """
    
    def __init__(self):
        self.feature_stats = {}
    
    def clean_features_matrix(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Limpiar matriz de features para ML
        
        Args:
            X: DataFrame de features
            y: Serie target
            
        Returns:
            Tuple con X e y limpios
        """
        logger.info(f"Limpiando matriz de features: {X.shape}")
        
        original_shape = X.shape
        
        # 1. Eliminar features con demasiados valores faltantes
        missing_threshold = 0.5
        missing_pct = X.isnull().sum() / len(X)
        features_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if features_to_drop:
            logger.warning(f"Eliminando {len(features_to_drop)} features con >50% missing values")
            X = X.drop(columns=features_to_drop)
        
        # 2. Eliminar features constantes
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            logger.warning(f"Eliminando {len(constant_features)} features constantes")
            X = X.drop(columns=constant_features)
        
        # 3. Eliminar features altamente correlacionadas
        correlation_threshold = 0.95
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        highly_correlated = [column for column in upper_triangle.columns 
                           if any(upper_triangle[column] > correlation_threshold)]
        
        if highly_correlated:
            logger.info(f"Eliminando {len(highly_correlated)} features altamente correlacionadas")
            X = X.drop(columns=highly_correlated)
        
        # 4. Imputar valores faltantes restantes
        X = X.fillna(X.median())
        
        # 5. Alinear X e y (eliminar filas donde y es NaN)
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        final_shape = X.shape
        
        self.feature_stats = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'features_dropped_missing': len(features_to_drop),
            'features_dropped_constant': len(constant_features),
            'features_dropped_correlated': len(highly_correlated),
            'final_missing_values': X.isnull().sum().sum(),
            'target_missing_values': y.isnull().sum()
        }
        
        logger.info(f"Features limpieza completada: {original_shape} → {final_shape}")
        
        return X, y
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Obtener reporte de limpieza"""
        return self.feature_stats


def create_unified_dataset(
    banxico_data: Dict,
    inegi_data: Dict,
    lme_data: Dict,
    target_metal: str = 'steel_rebar'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Crear dataset unificado para modelado
    
    Args:
        banxico_data: Datos de BANXICO
        inegi_data: Datos de INEGI
        lme_data: Datos de LME
        target_metal: Metal objetivo para predicción
        
    Returns:
        Tuple con features DataFrame y target Series
    """
    logger.info("Creando dataset unificado...")
    
    # Obtener target (precio de varilla)
    if target_metal not in lme_data['metals_data']:
        raise ValueError(f"Metal objetivo '{target_metal}' no encontrado en datos LME")
    
    target_data = lme_data['metals_data'][target_metal]['data']
    target_series = target_data.set_index('fecha')['valor']
    
    # Crear DataFrame de features
    features_list = []
    feature_names = []
    
    # Features de BANXICO
    for key, series_data in banxico_data['series_data'].items():
        df = series_data['data'].set_index('fecha')['valor']
        df.name = f"banxico_{key}"
        features_list.append(df)
        feature_names.append(f"banxico_{key}")
    
    # Features de INEGI
    for key, indicator_data in inegi_data['indicators_data'].items():
        df = indicator_data['data'].set_index('fecha')['valor']
        df.name = f"inegi_{key}"
        features_list.append(df)
        feature_names.append(f"inegi_{key}")
    
    # Features de otros metales
    for key, metal_data in lme_data['metals_data'].items():
        if key != target_metal:  # No incluir el target como feature
            df = metal_data['data'].set_index('fecha')['valor']
            df.name = f"lme_{key}"
            features_list.append(df)
            feature_names.append(f"lme_{key}")
    
    # Combinar todas las features
    features_df = pd.concat(features_list, axis=1, join='outer')
    
    # Alinear con target
    aligned_data = pd.concat([features_df, target_series], axis=1, join='inner')
    
    if aligned_data.empty:
        raise ValueError("No hay fechas comunes entre features y target")
    
    # Separar features y target
    X = aligned_data.iloc[:, :-1]
    y = aligned_data.iloc[:, -1]
    
    logger.info(f"Dataset unificado creado: {X.shape[0]} observaciones, {X.shape[1]} features")
    
    return X, y


def validate_data_quality(df: pd.DataFrame, min_quality_score: float = 70.0) -> Dict[str, Any]:
    """
    Validar calidad general de datos
    
    Args:
        df: DataFrame a validar
        min_quality_score: Score mínimo de calidad
        
    Returns:
        Diccionario con métricas de calidad
    """
    quality_metrics = {}
    
    # Completitud
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    # Consistencia de tipos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    type_consistency = len(numeric_cols) / len(df.columns) * 100
    
    # Variabilidad
    variability_scores = []
    for col in numeric_cols:
        if df[col].std() > 0:
            cv = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else 0
            variability_scores.append(min(100, cv * 100))
    
    avg_variability = np.mean(variability_scores) if variability_scores else 0
    
    # Score general
    overall_score = (completeness + type_consistency + avg_variability) / 3
    
    quality_metrics = {
        'completeness_pct': completeness,
        'type_consistency_pct': type_consistency,
        'avg_variability_score': avg_variability,
        'overall_quality_score': overall_score,
        'meets_minimum_quality': overall_score >= min_quality_score,
        'total_records': len(df),
        'total_features': len(df.columns),
        'missing_cells': missing_cells,
        'numeric_features': len(numeric_cols)
    }
    
    return quality_metrics


# ========== EXTENSIONES PARA PIPELINE DE PRODUCCIÓN ==========

class ConsolidatedDataCleaner(DataCleaner):
    """
    Limpiador especializado para series temporales consolidadas
    Optimizado para el pipeline de producción de DeAcero
    """
    
    def __init__(self, target_variable: str = 'precio_varilla_lme'):
        super().__init__()
        self.target_variable = target_variable
        self.interpolation_stats = {}
    
    def clean_consolidated_daily_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza específica para series diarias consolidadas
        Optimizada para el pipeline de producción
        
        Args:
            df: DataFrame consolidado con series diarias (index=fecha, columns=variables)
            
        Returns:
            DataFrame limpio y listo para feature engineering
        """
        logger.info(f"🧹 Iniciando limpieza de series diarias consolidadas: {df.shape}")
        
        # Copiar y verificar estructura
        clean_df = df.copy()
        
        # 1. Verificar que el índice sea datetime
        if not isinstance(clean_df.index, pd.DatetimeIndex):
            logger.error("El índice debe ser DatetimeIndex")
            raise ValueError("DataFrame debe tener índice de fechas")
        
        # 2. Ordenar por fecha
        clean_df = clean_df.sort_index()
        
        # 3. Eliminar duplicados en el índice
        if clean_df.index.duplicated().any():
            duplicates = clean_df.index.duplicated().sum()
            logger.warning(f"Eliminando {duplicates} fechas duplicadas")
            clean_df = clean_df[~clean_df.index.duplicated(keep='last')]
        
        # 4. Verificar variable objetivo
        if self.target_variable not in clean_df.columns:
            logger.error(f"Variable objetivo '{self.target_variable}' no encontrada")
            raise ValueError(f"Variable objetivo '{self.target_variable}' requerida")
        
        # 5. Limpieza por columna con estrategias específicas
        for col in clean_df.columns:
            clean_df[col] = self._clean_individual_series(clean_df[col], col)
        
        # 6. Interpolación inteligente para llenar NaN restantes
        clean_df = self._smart_interpolation(clean_df)
        
        # 7. Validación final de calidad
        quality_report = self._validate_data_quality(clean_df)
        
        logger.info(f"✅ Limpieza completada: {df.shape} → {clean_df.shape}")
        logger.info(f"📊 Calidad final: {quality_report['overall_quality']:.1%}")
        
        return clean_df
    
    def clean_consolidated_monthly_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza específica para series mensuales consolidadas
        
        Args:
            df: DataFrame consolidado con series mensuales
            
        Returns:
            DataFrame limpio
        """
        logger.info(f"🧹 Iniciando limpieza de series mensuales: {df.shape}")
        
        clean_df = df.copy()
        
        # Verificar estructura
        if not isinstance(clean_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame debe tener índice de fechas")
        
        clean_df = clean_df.sort_index()
        
        # Limpieza específica para datos mensuales
        for col in clean_df.columns:
            # Estrategia más conservadora para datos mensuales
            clean_df[col] = self._clean_monthly_series(clean_df[col], col)
        
        # Interpolación mensual (más conservadora)
        clean_df = self._interpolate_monthly_data(clean_df)
        
        logger.info(f"✅ Series mensuales limpiadas: {clean_df.shape}")
        
        return clean_df
    
    def _clean_individual_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Limpieza de serie individual con estrategias específicas por tipo
        """
        original_nulls = series.isnull().sum()
        
        # Identificar tipo de variable para estrategia específica
        if any(keyword in col_name.lower() for keyword in ['precio', 'price', 'varilla', 'lme']):
            # Precios: más conservador
            series = self._clean_price_series(series, col_name)
        elif any(keyword in col_name.lower() for keyword in ['volatility', 'vix']):
            # Volatilidad: manejar picos
            series = self._clean_volatility_series(series, col_name)
        elif any(keyword in col_name.lower() for keyword in ['tipo_cambio', 'usd', 'mxn']):
            # Tipo de cambio: suavizar
            series = self._clean_exchange_rate_series(series, col_name)
        else:
            # Estrategia general
            series = self._clean_general_series(series, col_name)
        
        final_nulls = series.isnull().sum()
        
        if col_name not in self.cleaning_stats:
            self.cleaning_stats[col_name] = {}
        
        self.cleaning_stats[col_name].update({
            'original_nulls': original_nulls,
            'final_nulls': final_nulls,
            'nulls_filled': original_nulls - final_nulls
        })
        
        return series
    
    def _clean_price_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """Limpieza específica para series de precios"""
        # 1. Eliminar valores negativos o cero (precios inválidos)
        invalid_mask = (series <= 0) | (series > series.quantile(0.99) * 3)
        if invalid_mask.any():
            logger.warning(f"{col_name}: Reemplazando {invalid_mask.sum()} valores inválidos")
            series.loc[invalid_mask] = np.nan
        
        # 2. Detectar y suavizar saltos extremos (>20% en un día)
        pct_change = series.pct_change().abs()
        extreme_changes = pct_change > 0.20
        if extreme_changes.any():
            logger.warning(f"{col_name}: Suavizando {extreme_changes.sum()} cambios extremos")
            # Reemplazar con interpolación local
            series.loc[extreme_changes] = np.nan
        
        # 3. Interpolación temporal para precios
        series = series.interpolate(method='time', limit=5)  # Máximo 5 días consecutivos
        
        return series
    
    def _clean_volatility_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """Limpieza específica para series de volatilidad"""
        # Volatilidad no puede ser negativa
        series = series.clip(lower=0)
        
        # Manejar picos extremos de volatilidad
        q99 = series.quantile(0.99)
        extreme_vol = series > q99 * 2
        if extreme_vol.any():
            logger.warning(f"{col_name}: Limitando {extreme_vol.sum()} picos de volatilidad")
            series.loc[extreme_vol] = q99
        
        # Interpolación lineal para volatilidad
        series = series.interpolate(method='linear', limit=3)
        
        return series
    
    def _clean_exchange_rate_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """Limpieza específica para tipos de cambio"""
        # Suavizar tipos de cambio con media móvil
        if series.isnull().sum() > 0:
            # Usar forward fill primero, luego interpolación
            series = series.fillna(method='ffill', limit=2)
            series = series.interpolate(method='linear', limit=3)
        
        return series
    
    def _clean_general_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """Limpieza general para otras series"""
        # Estrategia conservadora
        series = series.interpolate(method='linear', limit=5)
        return series
    
    def _clean_monthly_series(self, series: pd.Series, col_name: str) -> pd.Series:
        """Limpieza específica para series mensuales"""
        # Más conservador para datos mensuales
        # Solo interpolación lineal con límite bajo
        series = series.interpolate(method='linear', limit=2)
        return series
    
    def _smart_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolación inteligente final para llenar NaN restantes
        Prioriza la variable objetivo y usa diferentes estrategias
        """
        logger.info("🔧 Aplicando interpolación inteligente final...")
        
        interpolated_df = df.copy()
        
        # 1. Prioridad a la variable objetivo
        if self.target_variable in interpolated_df.columns:
            target_nulls_before = interpolated_df[self.target_variable].isnull().sum()
            if target_nulls_before > 0:
                logger.warning(f"Variable objetivo tiene {target_nulls_before} NaN - aplicando interpolación crítica")
                
                # Interpolación agresiva para variable objetivo
                interpolated_df[self.target_variable] = interpolated_df[self.target_variable].interpolate(
                    method='linear', limit_direction='both'
                )
                
                # Si aún hay NaN, usar forward/backward fill
                interpolated_df[self.target_variable] = interpolated_df[self.target_variable].fillna(method='ffill')
                interpolated_df[self.target_variable] = interpolated_df[self.target_variable].fillna(method='bfill')
                
                target_nulls_after = interpolated_df[self.target_variable].isnull().sum()
                logger.info(f"Variable objetivo: {target_nulls_before} → {target_nulls_after} NaN")
        
        # 2. Para otras variables críticas
        critical_vars = ['iron', 'coking', 'sp500', 'tipo_cambio_usdmxn', 'vix']
        
        for var in critical_vars:
            matching_cols = [col for col in interpolated_df.columns if var in col.lower()]
            for col in matching_cols:
                nulls_before = interpolated_df[col].isnull().sum()
                if nulls_before > 0:
                    # Interpolación temporal
                    interpolated_df[col] = interpolated_df[col].interpolate(method='time', limit=10)
                    # Forward fill como respaldo
                    interpolated_df[col] = interpolated_df[col].fillna(method='ffill', limit=5)
                    
                    nulls_after = interpolated_df[col].isnull().sum()
                    if nulls_after < nulls_before:
                        logger.info(f"{col}: {nulls_before} → {nulls_after} NaN")
        
        # 3. Para el resto de variables
        for col in interpolated_df.columns:
            if col not in [self.target_variable] + [c for v in critical_vars for c in interpolated_df.columns if v in c.lower()]:
                nulls_before = interpolated_df[col].isnull().sum()
                if nulls_before > 0:
                    # Estrategia más simple
                    interpolated_df[col] = interpolated_df[col].interpolate(method='linear', limit=5)
                    interpolated_df[col] = interpolated_df[col].fillna(interpolated_df[col].median())
        
        # 4. Verificación final
        remaining_nulls = interpolated_df.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning(f"⚠️ Quedan {remaining_nulls} NaN después de interpolación")
            
            # Última estrategia: llenar con mediana por columna
            for col in interpolated_df.columns:
                if interpolated_df[col].isnull().any():
                    median_val = interpolated_df[col].median()
                    interpolated_df[col] = interpolated_df[col].fillna(median_val)
        
        # Estadísticas de interpolación
        self.interpolation_stats = {
            'total_nulls_filled': df.isnull().sum().sum() - interpolated_df.isnull().sum().sum(),
            'final_nulls': interpolated_df.isnull().sum().sum(),
            'interpolation_success': interpolated_df.isnull().sum().sum() == 0
        }
        
        return interpolated_df
    
    def _interpolate_monthly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolación específica para datos mensuales"""
        logger.info("🔧 Interpolando datos mensuales...")
        
        interpolated_df = df.copy()
        
        for col in interpolated_df.columns:
            nulls_before = interpolated_df[col].isnull().sum()
            if nulls_before > 0:
                # Para mensuales: solo interpolación lineal conservadora
                interpolated_df[col] = interpolated_df[col].interpolate(method='linear', limit=3)
                # Forward fill como último recurso
                interpolated_df[col] = interpolated_df[col].fillna(method='ffill', limit=2)
                
                nulls_after = interpolated_df[col].isnull().sum()
                logger.info(f"{col}: {nulls_before} → {nulls_after} NaN (mensual)")
        
        return interpolated_df
    
    def prepare_for_modeling(self, daily_df: pd.DataFrame, monthly_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Preparación final de datos para modelado
        Aplica todas las transformaciones necesarias para los modelos V2
        
        Args:
            daily_df: Series diarias consolidadas y limpias
            monthly_df: Series mensuales consolidadas (opcional)
            
        Returns:
            Dictionary con datos preparados para cada tipo de modelo
        """
        logger.info("🎯 Preparando datos para modelado V2...")
        
        # 1. Limpiar series diarias
        clean_daily = self.clean_consolidated_daily_series(daily_df)
        
        # 2. Limpiar series mensuales si existen
        clean_monthly = None
        if monthly_df is not None and not monthly_df.empty:
            clean_monthly = self.clean_consolidated_monthly_series(monthly_df)
        
        # 3. Verificar alineación temporal con variable objetivo
        if self.target_variable in clean_daily.columns:
            target_start = clean_daily[self.target_variable].first_valid_index()
            target_end = clean_daily[self.target_variable].last_valid_index()
            
            logger.info(f"📅 Alineando datos al rango de variable objetivo: {target_start.date()} a {target_end.date()}")
            
            # Truncar todas las series al rango de la variable objetivo
            clean_daily = clean_daily.loc[target_start:target_end]
            
            if clean_monthly is not None:
                # Para mensuales, usar solo el rango que se solapa
                monthly_start = max(target_start, clean_monthly.index.min())
                monthly_end = min(target_end, clean_monthly.index.max())
                clean_monthly = clean_monthly.loc[monthly_start:monthly_end]
        
        # 4. Verificación final de calidad
        final_stats = {
            'daily_shape': clean_daily.shape,
            'daily_nulls': clean_daily.isnull().sum().sum(),
            'daily_completeness': (1 - clean_daily.isnull().sum().sum() / (clean_daily.shape[0] * clean_daily.shape[1])) * 100,
            'target_variable_nulls': clean_daily[self.target_variable].isnull().sum() if self.target_variable in clean_daily.columns else 'N/A'
        }
        
        if clean_monthly is not None:
            final_stats.update({
                'monthly_shape': clean_monthly.shape,
                'monthly_nulls': clean_monthly.isnull().sum().sum(),
                'monthly_completeness': (1 - clean_monthly.isnull().sum().sum() / (clean_monthly.shape[0] * clean_monthly.shape[1])) * 100
            })
        
        logger.info(f"📊 Estadísticas finales: {final_stats}")
        
        # Guardar estadísticas
        self.cleaning_stats['final_preparation'] = final_stats
        
        result = {'daily': clean_daily}
        if clean_monthly is not None:
            result['monthly'] = clean_monthly
        
        return result
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validación completa de calidad de datos
        """
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        completeness = (total_cells - null_cells) / total_cells
        
        # Verificar variable objetivo específicamente
        target_quality = 1.0
        if self.target_variable in df.columns:
            target_nulls = df[self.target_variable].isnull().sum()
            target_quality = (len(df) - target_nulls) / len(df)
        
        quality_report = {
            'overall_quality': completeness,
            'target_variable_quality': target_quality,
            'total_observations': df.shape[0],
            'total_variables': df.shape[1],
            'null_cells': null_cells,
            'data_range': {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d')
            },
            'business_days_only': self._check_business_days(df),
            'quality_grade': self._assign_quality_grade(completeness, target_quality)
        }
        
        return quality_report
    
    def _check_business_days(self, df: pd.DataFrame) -> bool:
        """Verificar si el dataset contiene solo días hábiles"""
        weekend_count = sum(df.index.dayofweek >= 5)  # Sábado=5, Domingo=6
        return weekend_count == 0
    
    def _assign_quality_grade(self, overall: float, target: float) -> str:
        """Asignar grado de calidad"""
        if target >= 0.95 and overall >= 0.90:
            return "EXCELENTE"
        elif target >= 0.90 and overall >= 0.85:
            return "BUENA"
        elif target >= 0.80 and overall >= 0.75:
            return "ACEPTABLE"
        else:
            return "DEFICIENTE"


class ProductionDataCleaner(ConsolidatedDataCleaner):
    """
    Versión especializada para producción con validaciones estrictas
    """
    
    def __init__(self, target_variable: str = 'precio_varilla_lme'):
        super().__init__(target_variable)
        self.production_config = {
            'max_interpolation_days': 5,
            'outlier_threshold': 3.0,
            'min_data_quality': 0.85,
            'require_target_quality': 0.95
        }
    
    def validate_for_production(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validación estricta para producción
        
        Returns:
            (is_valid, validation_report)
        """
        logger.info("🔍 Validando datos para producción...")
        
        validation_report = self._validate_data_quality(df)
        
        # Criterios estrictos para producción
        is_valid = (
            validation_report['target_variable_quality'] >= self.production_config['require_target_quality'] and
            validation_report['overall_quality'] >= self.production_config['min_data_quality'] and
            validation_report['null_cells'] == 0  # No NaN en producción
        )
        
        validation_report['production_ready'] = is_valid
        validation_report['validation_timestamp'] = datetime.now().isoformat()
        
        if not is_valid:
            logger.error("❌ Datos NO aptos para producción")
            logger.error(f"Calidad objetivo: {validation_report['target_variable_quality']:.3f} (req: {self.production_config['require_target_quality']})")
            logger.error(f"Calidad general: {validation_report['overall_quality']:.3f} (req: {self.production_config['min_data_quality']})")
        else:
            logger.info("✅ Datos APTOS para producción")
        
        return is_valid, validation_report
