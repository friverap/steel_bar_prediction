#!/usr/bin/env python3
"""
Script para hacer JOIN de series temporales mensuales actualizadas
Proyecto: DeAcero Steel Price Predictor
Fecha: 26 de Septiembre de 2025

Este script:
1. Carga todas las series mensuales actualizadas desde data/raw/
2. Filtra solo las series con datos posteriores a 2025
3. Realiza un JOIN ordenado con nombres descriptivos
4. Guarda el resultado en monthly_time_series/
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'monthly_time_series')

# Fecha de corte para considerar series actualizadas
FECHA_CORTE = pd.Timestamp('2025-01-01')

# Diccionario de mapeo de nombres de archivos a nombres descriptivos
VARIABLE_NAMES = {
    # FRED
    'federal_funds_rate': 'tasa_fed_usa',
    'steel_production': 'produccion_acero_usa',
    'ppi_metals': 'indice_precios_productor_metales_usa',
    'iron_steel_scrap': 'precio_chatarra_acero_usa',
    'GastoConstruccion': 'gasto_construccion_usa',
    'ProduccionIndustrial': 'produccion_industrial_usa',
    
    # INEGI
    'ProduccionConstruccion': 'produccion_construccion_mexico',
    'produccion_metalurgica': 'produccion_metalurgica_mexico',
    
    # Banxico
    'inflation_monthly': 'inflacion_mensual_mexico'
}

def safe_to_datetime(fecha_series):
    """
    Convierte de manera segura una serie de fechas a datetime, manejando timezones.
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
    
    if hasattr(result.dtype, 'tz') and result.dt.tz is not None:
        result = result.dt.tz_localize(None)
    
    # Normalizar a fecha (sin hora)
    result = result.dt.normalize()
    
    return result

def load_fred_series():
    """
    Carga las series mensuales de FRED que están actualizadas.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES FRED")
    print("="*60)
    
    fred_series = {}
    fred_files = glob.glob(os.path.join(DATA_RAW_DIR, 'FRED_*.csv'))
    
    for file_path in fred_files:
        filename = os.path.basename(file_path)
        
        # Extraer el nombre de la serie
        for key in ['federal_funds_rate', 'steel_production', 'ppi_metals', 
                   'iron_steel_scrap', 'GastoConstruccion', 'ProduccionIndustrial']:
            if key in filename:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Identificar columnas de fecha y valor
                    date_col = None
                    value_col = None
                    
                    for col in df.columns:
                        if col.lower() in ['date', 'fecha']:
                            date_col = col
                        elif col.lower() in ['value', 'valor'] or df[col].dtype in ['float64', 'int64']:
                            if value_col is None and col != date_col:
                                value_col = col
                    
                    if date_col and value_col:
                        df_clean = df[[date_col, value_col]].copy()
                        df_clean.columns = ['fecha', 'valor']
                        df_clean['fecha'] = safe_to_datetime(df_clean['fecha'])
                        
                        # Verificar si está actualizada
                        if df_clean['fecha'].max() >= FECHA_CORTE:
                            # Renombrar columna con nombre descriptivo
                            variable_name = VARIABLE_NAMES.get(key, key)
                            df_clean.columns = ['fecha', variable_name]
                            
                            # Configurar índice y eliminar duplicados
                            df_clean.set_index('fecha', inplace=True)
                            if df_clean.index.duplicated().any():
                                df_clean = df_clean.groupby(level=0).mean()
                            
                            fred_series[variable_name] = df_clean
                            print(f"✅ {variable_name}: {len(df_clean)} registros ({df_clean.index.min().date()} a {df_clean.index.max().date()})")
                        else:
                            print(f"⚠️ {key}: No actualizada (última fecha: {df_clean['fecha'].max().date()})")
                
                except Exception as e:
                    print(f"❌ Error procesando {filename}: {str(e)}")
                
                break
    
    return fred_series

def load_inegi_series():
    """
    Carga las series mensuales de INEGI que están actualizadas.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES INEGI")
    print("="*60)
    
    inegi_series = {}
    inegi_files = glob.glob(os.path.join(DATA_RAW_DIR, 'INEGI_*.csv'))
    
    for file_path in inegi_files:
        filename = os.path.basename(file_path)
        
        # Buscar series específicas
        for key in ['ProduccionConstruccion', 'produccion_metalurgica']:
            if key in filename:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Identificar columnas
                    date_col = None
                    value_col = None
                    
                    for col in df.columns:
                        if col.lower() in ['date', 'fecha']:
                            date_col = col
                        elif col.lower() in ['value', 'valor'] or df[col].dtype in ['float64', 'int64']:
                            if value_col is None and col != date_col:
                                value_col = col
                    
                    if date_col and value_col:
                        df_clean = df[[date_col, value_col]].copy()
                        df_clean.columns = ['fecha', 'valor']
                        df_clean['fecha'] = safe_to_datetime(df_clean['fecha'])
                        
                        # Verificar si está actualizada (INEGI hasta julio 2025)
                        if df_clean['fecha'].max() >= pd.Timestamp('2025-07-01'):
                            variable_name = VARIABLE_NAMES.get(key, key)
                            df_clean.columns = ['fecha', variable_name]
                            
                            df_clean.set_index('fecha', inplace=True)
                            if df_clean.index.duplicated().any():
                                df_clean = df_clean.groupby(level=0).mean()
                            
                            inegi_series[variable_name] = df_clean
                            print(f"✅ {variable_name}: {len(df_clean)} registros ({df_clean.index.min().date()} a {df_clean.index.max().date()})")
                        else:
                            print(f"⚠️ {key}: No actualizada (última fecha: {df_clean['fecha'].max().date()})")
                
                except Exception as e:
                    print(f"❌ Error procesando {filename}: {str(e)}")
                
                break
    
    return inegi_series

def load_banxico_series():
    """
    Carga las series mensuales de Banxico que están actualizadas.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES BANXICO")
    print("="*60)
    
    banxico_series = {}
    banxico_files = glob.glob(os.path.join(DATA_RAW_DIR, 'banxico_*.csv'))
    
    for file_path in banxico_files:
        filename = os.path.basename(file_path)
        
        if 'inflation_monthly' in filename:
            try:
                df = pd.read_csv(file_path)
                
                # Identificar columnas
                date_col = None
                value_col = None
                
                for col in df.columns:
                    if col.lower() in ['date', 'fecha']:
                        date_col = col
                    elif col.lower() in ['value', 'valor', 'inflation_rate'] or df[col].dtype in ['float64', 'int64']:
                        if value_col is None and col != date_col:
                            value_col = col
                
                if date_col and value_col:
                    df_clean = df[[date_col, value_col]].copy()
                    df_clean.columns = ['fecha', 'valor']
                    df_clean['fecha'] = safe_to_datetime(df_clean['fecha'])
                    
                    # Verificar si está actualizada
                    if df_clean['fecha'].max() >= FECHA_CORTE:
                        variable_name = VARIABLE_NAMES.get('inflation_monthly', 'inflation_monthly')
                        df_clean.columns = ['fecha', variable_name]
                        
                        df_clean.set_index('fecha', inplace=True)
                        if df_clean.index.duplicated().any():
                            df_clean = df_clean.groupby(level=0).mean()
                        
                        banxico_series[variable_name] = df_clean
                        print(f"✅ {variable_name}: {len(df_clean)} registros ({df_clean.index.min().date()} a {df_clean.index.max().date()})")
                    else:
                        print(f"⚠️ inflation_monthly: No actualizada (última fecha: {df_clean['fecha'].max().date()})")
            
            except Exception as e:
                print(f"❌ Error procesando {filename}: {str(e)}")
    
    return banxico_series

def join_all_series(all_series_dict):
    """
    Realiza el JOIN de todas las series mensuales.
    """
    print("\n" + "="*60)
    print("🔗 REALIZANDO JOIN DE SERIES MENSUALES")
    print("="*60)
    
    if not all_series_dict:
        print("❌ No hay series para hacer join")
        return None
    
    # Comenzar con la primera serie
    first_key = list(all_series_dict.keys())[0]
    consolidated_df = all_series_dict[first_key].copy()
    print(f"\nIniciando con: {first_key}")
    print(f"Shape inicial: {consolidated_df.shape}")
    
    # Hacer join con las demás series
    for i, (name, df) in enumerate(list(all_series_dict.items())[1:], 1):
        print(f"\n{i}. JOIN con {name}:")
        print(f"   • DataFrame actual: {consolidated_df.shape}")
        print(f"   • Serie a unir: {df.shape}")
        
        # Hacer el join
        nan_before = consolidated_df.isna().sum().sum()
        consolidated_df = consolidated_df.join(df, how='outer')
        nan_after = consolidated_df.isna().sum().sum()
        
        print(f"   • Resultado después del join: {consolidated_df.shape}")
        print(f"   • NaN introducidos: {nan_after - nan_before}")
    
    # Ordenar por fecha
    consolidated_df = consolidated_df.sort_index()
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS FINALES DEL DATASET")
    print("="*60)
    print(f"• Dimensiones: {consolidated_df.shape}")
    print(f"• Rango temporal: {consolidated_df.index.min().date()} a {consolidated_df.index.max().date()}")
    print(f"• Total de observaciones: {consolidated_df.shape[0] * consolidated_df.shape[1]}")
    print(f"• Valores no nulos: {consolidated_df.count().sum()}")
    print(f"• Valores nulos: {consolidated_df.isna().sum().sum()}")
    print(f"• Porcentaje de completitud: {(consolidated_df.count().sum() / (consolidated_df.shape[0] * consolidated_df.shape[1]) * 100):.2f}%")
    
    # Análisis por variable
    print("\n📈 COMPLETITUD POR VARIABLE:")
    print("-"*40)
    for col in consolidated_df.columns:
        completeness = (consolidated_df[col].count() / len(consolidated_df)) * 100
        print(f"• {col:40s}: {completeness:6.2f}% completo")
    
    return consolidated_df

def save_results(df, all_series_dict):
    """
    Guarda el DataFrame consolidado y metadata.
    """
    print("\n" + "="*60)
    print("💾 GUARDANDO RESULTADOS")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar CSV principal
    csv_path = os.path.join(OUTPUT_DIR, f'monthly_series_consolidated_{timestamp}.csv')
    df.to_csv(csv_path)
    print(f"✅ CSV guardado: {csv_path}")
    
    # Guardar versión sin timestamp para uso fácil
    csv_latest = os.path.join(OUTPUT_DIR, 'monthly_series_consolidated_latest.csv')
    df.to_csv(csv_latest)
    print(f"✅ CSV latest: {csv_latest}")
    
    # Guardar en formato Parquet (más eficiente) - opcional si está disponible
    parquet_path = None
    try:
        parquet_path = os.path.join(OUTPUT_DIR, f'monthly_series_consolidated_{timestamp}.parquet')
        df.to_parquet(parquet_path)
        print(f"✅ Parquet guardado: {parquet_path}")
        
        # Guardar versión latest en Parquet
        parquet_latest = os.path.join(OUTPUT_DIR, 'monthly_series_consolidated_latest.parquet')
        df.to_parquet(parquet_latest)
        print(f"✅ Parquet latest: {parquet_latest}")
    except ImportError:
        print("⚠️ Parquet no disponible (instalar pyarrow o fastparquet para habilitar)")
    
    # Crear metadata
    metadata = {
        'timestamp_generacion': timestamp,
        'fecha_corte_actualizacion': str(FECHA_CORTE.date()),
        'total_series': len(df.columns),
        'total_observaciones': len(df),
        'rango_temporal': {
            'inicio': str(df.index.min().date()),
            'fin': str(df.index.max().date())
        },
        'series_incluidas': list(df.columns),
        'fuentes': {
            'FRED': [col for col in df.columns if 'usa' in col],
            'INEGI': [col for col in df.columns if 'construccion_mexico' in col or 'metalurgica_mexico' in col],
            'Banxico': [col for col in df.columns if 'inflacion' in col]
        },
        'estadisticas': {
            'porcentaje_completitud': float((df.count().sum() / (df.shape[0] * df.shape[1]) * 100)),
            'valores_no_nulos': int(df.count().sum()),
            'valores_nulos': int(df.isna().sum().sum())
        },
        'completitud_por_variable': {col: float((df[col].count() / len(df)) * 100) for col in df.columns}
    }
    
    # Guardar metadata
    metadata_path = os.path.join(OUTPUT_DIR, f'metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata guardada: {metadata_path}")
    
    # Guardar versión latest de metadata
    metadata_latest = os.path.join(OUTPUT_DIR, 'metadata_latest.json')
    with open(metadata_latest, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata latest: {metadata_latest}")
    
    return csv_path, parquet_path, metadata_path

def main():
    """
    Función principal que orquesta todo el proceso.
    """
    print("\n" + "="*80)
    print("🚀 INICIANDO PROCESO DE JOIN DE SERIES MENSUALES ACTUALIZADAS")
    print("="*80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio de entrada: {DATA_RAW_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Fecha de corte para actualización: {FECHA_CORTE.date()}")
    
    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cargar series de cada fuente
    all_series = {}
    
    # FRED
    fred_series = load_fred_series()
    all_series.update(fred_series)
    
    # INEGI
    inegi_series = load_inegi_series()
    all_series.update(inegi_series)
    
    # Banxico
    banxico_series = load_banxico_series()
    all_series.update(banxico_series)
    
    # Resumen de carga
    print("\n" + "="*60)
    print("📋 RESUMEN DE CARGA")
    print("="*60)
    print(f"Total de series cargadas: {len(all_series)}")
    print(f"• FRED: {len(fred_series)} series")
    print(f"• INEGI: {len(inegi_series)} series")
    print(f"• Banxico: {len(banxico_series)} series")
    
    if len(all_series) == 0:
        print("\n❌ No se encontraron series mensuales actualizadas. Abortando proceso.")
        return
    
    # Realizar JOIN
    consolidated_df = join_all_series(all_series)
    
    if consolidated_df is not None:
        # Guardar resultados
        csv_path, parquet_path, metadata_path = save_results(consolidated_df, all_series)
        
        print("\n" + "="*80)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"Archivos generados en: {OUTPUT_DIR}")
        print("\nPara usar los datos en Python:")
        print(f"  df = pd.read_csv('{csv_path}')")
        print(f"  df = pd.read_parquet('{parquet_path}')  # Más eficiente")
        
        return consolidated_df
    else:
        print("\n❌ Error en el proceso de JOIN")
        return None

if __name__ == "__main__":
    df_result = main()
    
    if df_result is not None:
        print("\n" + "="*80)
        print("📊 VISTA PREVIA DEL DATASET FINAL")
        print("="*80)
        print("\nPrimeras 5 filas:")
        print(df_result.head())
        print("\nÚltimas 5 filas:")
        print(df_result.tail())
        print("\nDescripción estadística:")
        print(df_result.describe())
