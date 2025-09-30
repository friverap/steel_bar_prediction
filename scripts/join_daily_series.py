#!/usr/bin/env python3
"""
Script para hacer JOIN de series temporales diarias actualizadas
Proyecto: DeAcero Steel Price Predictor
Fecha: 26 de Septiembre de 2025

Este script:
1. Carga todas las series diarias actualizadas desde data/raw/
2. Ofrece múltiples estrategias para manejar fines de semana
3. Realiza un JOIN eficiente con nombres descriptivos
4. Analiza gaps y completitud
5. Guarda el resultado en daily_time_series/
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'daily_time_series')

# Fecha de corte para considerar series actualizadas
FECHA_CORTE = pd.Timestamp('2025-01-01')

# Estrategia de manejo de fines de semana
WEEKEND_STRATEGY = 'business_days'  # Opciones: 'business_days', 'forward_fill', 'interpolate', 'keep_gaps'

# Diccionario de mapeo de nombres de archivos a nombres descriptivos
VARIABLE_NAMES = {
    # LME
    'steel_rebar': 'precio_varilla_lme',
    'Aluminio': 'aluminio_lme',
    'Cobre': 'cobre_lme',
    'Zinc': 'zinc_lme',
    'iron_ore': 'mineral_hierro_lme',
    'coking_coal': 'carbon_coque_lme',
    'steel_etf': 'etf_acero_lme',
    
    # Yahoo Finance
    'Petroleo_WTI': 'petroleo_wti',
    'Petroleo_Brent': 'petroleo_brent',
    'GasNatural': 'gas_natural',
    'SP500': 'sp500',
    'VIX_Volatilidad': 'vix_volatilidad',
    'dxy_index': 'dxy_index_yahoo',
    'treasury_10y': 'bonos_10y',
    'commodities_etf': 'commodities_etf',
    'materials_etf': 'materials_etf',
    'china_etf': 'china_etf',
    'emerging_markets': 'emerging_markets_etf',
    'infrastructure_etf': 'infrastructure_etf',
    
    # AHMSA
    'ahmsa': 'precio_acero_ahmsa',
    'nucor': 'nucor_acciones',
    'arcelormittal': 'arcelormittal_acciones',
    'ternium_mexico': 'ternium_mexico_acciones',
    'steel_etf': 'steel_etf_ahmsa',
    'materials_etf': 'materials_etf_ahmsa',
    'emerging_markets_etf': 'emerging_markets_etf_ahmsa',
    
    # Raw Materials
    'MineralHierro_VALE': 'vale_mineral_hierro',
    'MineralHierro_RIO': 'rio_mineral_hierro',
    'MineralHierro_BHP': 'bhp_mineral_hierro',
    'CarbonCoque_TECK': 'teck_carbon_coque',
    'CarbonCoque_AAL': 'aal_carbon_coque',
    'ETF_Acero_SLX': 'slx_etf_acero',
    'ETF_Mineria_XME': 'xme_etf_mineria',
    'ETF_Materiales_XLB': 'xlb_etf_materiales',
    
    # Banxico
    'usd_mxn': 'tipo_cambio_usdmxn',
    'tiie_28': 'tiie_28_dias',
    'tiie_91': 'tiie_91_dias',
    'udis': 'udis_valor',
    'interest_rate': 'tasa_interes_banxico',
    
    # FRED
    'dxy_index': 'dxy_index_fred',
    'natural_gas': 'gas_natural_fred'
}

def safe_to_datetime(fecha_series, normalize=True):
    """
    Convierte de manera segura una serie de fechas a datetime, manejando timezones.
    IMPORTANTE: Normaliza a fecha sin hora para evitar duplicados.
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
    
    # CRÍTICO: Normalizar a fecha (sin hora) para evitar duplicados como 00:00:00 vs 05:00:00
    if normalize:
        result = result.dt.normalize()  # Pone todo en 00:00:00
        # Alternativamente, podríamos usar: result = result.dt.date para solo fecha
    
    return result

def analyze_gaps(df):
    """
    Analiza los gaps (fines de semana y días faltantes) en una serie temporal.
    """
    if df.empty:
        return {}
    
    # Calcular diferencias entre fechas consecutivas
    date_diffs = df.index.to_series().diff().dropna()
    
    # Contar tipos de gaps
    one_day = date_diffs[date_diffs == pd.Timedelta(days=1)]
    weekend_gaps = date_diffs[date_diffs == pd.Timedelta(days=3)]  # Viernes a Lunes
    other_gaps = date_diffs[(date_diffs != pd.Timedelta(days=1)) & 
                            (date_diffs != pd.Timedelta(days=3))]
    
    return {
        'total_dias': len(df),
        'dias_habiles': len(one_day),
        'gaps_fin_semana': len(weekend_gaps),
        'otros_gaps': len(other_gaps),
        'max_gap_dias': date_diffs.max().days if not date_diffs.empty else 0,
        'pct_dias_habiles': (len(one_day) / len(date_diffs) * 100) if len(date_diffs) > 0 else 0
    }

def apply_weekend_strategy(df, strategy='business_days'):
    """
    Aplica la estrategia seleccionada para manejar fines de semana.
    """
    if strategy == 'business_days':
        # Mantener solo días hábiles (no hacer nada)
        return df
    
    elif strategy == 'forward_fill':
        # Resamplear a calendario diario y forward fill
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_reindexed = df.reindex(date_range)
        df_filled = df_reindexed.ffill()
        return df_filled
    
    elif strategy == 'interpolate':
        # Resamplear e interpolar linealmente
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_reindexed = df.reindex(date_range)
        df_interpolated = df_reindexed.interpolate(method='linear')
        return df_interpolated
    
    elif strategy == 'keep_gaps':
        # Mantener gaps como NaN
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_reindexed = df.reindex(date_range)
        return df_reindexed
    
    else:
        return df

def load_lme_series():
    """
    Carga las series diarias de LME.
    Usa columna 'Close' para precio de cierre.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES LME")
    print("="*60)
    
    lme_series = {}
    steel_rebar_loaded = False  # Bandera para evitar sobrescritura
    
    # CARGAR STEEL REBAR REAL DE INVESTING.COM
    steel_real_file = os.path.join(DATA_RAW_DIR, 'Investing_steel_rebar_real.csv')
    if os.path.exists(steel_real_file):
        try:
            df = pd.read_csv(steel_real_file)
            df['fecha'] = safe_to_datetime(df['fecha'], normalize=True)
            
            print(f"   📅 Fecha máxima en archivo: {df['fecha'].max()}")
            print(f"   📅 Fecha de corte: {FECHA_CORTE}")
            if df['fecha'].max() >= FECHA_CORTE:
                df_clean = df[['fecha', 'valor']].copy()
                df_clean.columns = ['fecha', 'precio_varilla_lme']
                df_clean.set_index('fecha', inplace=True)
                df_clean = df_clean.dropna()
                
                lme_series['precio_varilla_lme'] = df_clean
                print(f"✅ STEEL REBAR REAL (Investing.com): {len(df_clean)} registros")
                print(f"   Último precio: ${df_clean.iloc[-1, 0]:.2f} USD/tonelada")
                print(f"   Rango: ${df_clean['precio_varilla_lme'].min():.2f} - ${df_clean['precio_varilla_lme'].max():.2f}")
                
                # MARCAR QUE YA TENEMOS DATOS REALES - NO SOBRESCRIBIR
                steel_rebar_loaded = True
            
        except Exception as e:
            print(f"⚠️ Error cargando steel real: {e}")
    
    lme_files = glob.glob(os.path.join(DATA_RAW_DIR, 'LME_*.csv'))
    
    for file_path in lme_files:
        filename = os.path.basename(file_path)
        
        try:
            df = pd.read_csv(file_path)
            
            # Buscar columna de precio de cierre
            value_col = None
            if 'Close' in df.columns:
                value_col = 'Close'
            elif 'precio_cierre' in df.columns:
                value_col = 'precio_cierre'
            elif 'valor' in df.columns:
                value_col = 'valor'
            
            if 'fecha' in df.columns and value_col:
                df_clean = df[['fecha', value_col]].copy()
                df_clean.columns = ['fecha', 'valor']
                df_clean['fecha'] = safe_to_datetime(df_clean['fecha'], normalize=True)
                
                # IMPORTANTE: Manejar duplicados después de normalizar fechas
                # (ej: 00:00:00 y 05:00:00 del mismo día)
                df_clean = df_clean.groupby('fecha')['valor'].mean().reset_index()
                
                # Verificar si está actualizada
                if df_clean['fecha'].max() >= FECHA_CORTE:
                    # Procesar TODAS las columnas del archivo
                    for col in df.columns:
                        if col != 'fecha' and col != value_col:  # Evitar fecha y la columna ya procesada
                            continue
                            
                    # Extraer nombre de la serie - manejar steel_rebar correctamente
                    if 'steel_rebar' in filename:
                        series_name = 'steel_rebar'
                    else:
                        series_name = filename.replace('LME_', '').replace('.csv', '').split('_')[0]
                    variable_name = VARIABLE_NAMES.get(series_name, series_name)
                    
                    # EVITAR SOBRESCRIBIR SOLO PRECIO_VARILLA_LME
                    if variable_name == 'precio_varilla_lme' and steel_rebar_loaded:
                        print(f"   ⚠️ Saltando {variable_name} de {filename} - usando datos reales de Investing.com")
                        continue  # Solo salta esta variable, no todo el archivo
                    
                    df_clean.columns = ['fecha', variable_name]
                    df_clean.set_index('fecha', inplace=True)
                    df_clean = df_clean.dropna()
                    
                    # Análisis de gaps ANTES de aplicar estrategia
                    gap_analysis = analyze_gaps(df_clean)
                    
                    # Aplicar estrategia de fines de semana
                    df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                    
                    lme_series[variable_name] = df_clean
                    print(f"✅ {variable_name}: {len(df_clean)} registros ({df_clean.index.min().date()} a {df_clean.index.max().date()})")
                    print(f"   Gaps: {gap_analysis['gaps_fin_semana']} fines de semana, {gap_analysis['otros_gaps']} otros")
        
        except Exception as e:
            print(f"❌ Error procesando {filename}: {str(e)}")
    
    return lme_series

def load_yahoo_series():
    """
    Carga las series diarias de Yahoo Finance.
    Usa columna 'Close' para precio de cierre.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES YAHOO FINANCE")
    print("="*60)
    
    yahoo_series = {}
    yahoo_files = glob.glob(os.path.join(DATA_RAW_DIR, 'YahooFinance_*.csv'))
    
    for file_path in yahoo_files:
        filename = os.path.basename(file_path)
        
        try:
            df = pd.read_csv(file_path)
            
            # Buscar columna de precio de cierre
            value_col = None
            if 'Close' in df.columns:
                value_col = 'Close'
            elif 'precio_cierre' in df.columns:
                value_col = 'precio_cierre'
            elif 'valor' in df.columns:
                value_col = 'valor'
            
            if 'fecha' in df.columns and value_col:
                df_clean = df[['fecha', value_col]].copy()
                df_clean.columns = ['fecha', 'valor']
                
                # CRÍTICO: Convertir valor a numérico para evitar errores con mean()
                df_clean['valor'] = pd.to_numeric(df_clean['valor'], errors='coerce')
                
                df_clean['fecha'] = safe_to_datetime(df_clean['fecha'], normalize=True)
                
                # Manejar duplicados después de normalizar
                df_clean = df_clean.groupby('fecha')['valor'].mean().reset_index()
                
                if df_clean['fecha'].max() >= FECHA_CORTE:
                    series_name = filename.replace('YahooFinance_', '').replace('.csv', '').split('_')[0]
                    variable_name = VARIABLE_NAMES.get(series_name, series_name)
                    
                    df_clean.columns = ['fecha', variable_name]
                    df_clean.set_index('fecha', inplace=True)
                    df_clean = df_clean.dropna()
                    
                    gap_analysis = analyze_gaps(df_clean)
                    df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                    
                    yahoo_series[variable_name] = df_clean
                    print(f"✅ {variable_name}: {len(df_clean)} registros")
                    print(f"   Gaps: {gap_analysis['gaps_fin_semana']} fines de semana")
        
        except Exception as e:
            print(f"❌ Error procesando {filename}: {str(e)}")
    
    return yahoo_series

def load_banxico_series():
    """
    Carga las series diarias de Banxico.
    NOTA: UDIS tiene datos en fines de semana, se filtran para mantener solo días hábiles.
    """
    print("\n" + "="*60)
    print("📊 CARGANDO SERIES BANXICO")
    print("="*60)
    
    banxico_series = {}
    banxico_files = glob.glob(os.path.join(DATA_RAW_DIR, 'banxico_*.csv'))
    
    for file_path in banxico_files:
        filename = os.path.basename(file_path)
        
        # Solo cargar series diarias
        daily_series = ['usd_mxn', 'tiie_28', 'udis', 'interest_rate']
        
        if any(s in filename for s in daily_series):
            try:
                df = pd.read_csv(file_path)
                
                # Identificar columnas
                date_col = 'fecha' if 'fecha' in df.columns else 'date'
                value_cols = [col for col in df.columns if col not in [date_col, 'metadata']]
                
                if date_col in df.columns and value_cols:
                    value_col = value_cols[0]
                    df_clean = df[[date_col, value_col]].copy()
                    df_clean.columns = ['fecha', 'valor']
                    df_clean['fecha'] = safe_to_datetime(df_clean['fecha'])
                    
                    if df_clean['fecha'].max() >= FECHA_CORTE:
                        # Extraer nombre de la serie
                        for key in daily_series:
                            if key in filename:
                                variable_name = VARIABLE_NAMES.get(key, key)
                                break
                        
                        df_clean.columns = ['fecha', variable_name]
                        df_clean.set_index('fecha', inplace=True)
                        
                        if df_clean.index.duplicated().any():
                            df_clean = df_clean.groupby(level=0).mean()
                        
                        df_clean = df_clean.dropna()
                        
                        # ESPECIAL PARA UDIS: Filtrar fines de semana ANTES de aplicar estrategia
                        if 'udis' in filename.lower():
                            print(f"   ⚠️ UDIS detectado - filtrando fines de semana...")
                            before_filter = len(df_clean)
                            # Mantener solo días hábiles (lunes=0 a viernes=4)
                            df_clean = df_clean[df_clean.index.dayofweek < 5]
                            after_filter = len(df_clean)
                            print(f"   Registros antes: {before_filter}, después: {after_filter}")
                            print(f"   Fines de semana eliminados: {before_filter - after_filter}")
                        
                        gap_analysis = analyze_gaps(df_clean)
                        df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                        
                        banxico_series[variable_name] = df_clean
                        print(f"✅ {variable_name}: {len(df_clean)} registros")
            
            except Exception as e:
                print(f"❌ Error procesando {filename}: {str(e)}")
    
    return banxico_series

def load_other_series():
    """
    Carga otras series diarias (AHMSA, Raw Materials, FRED diarias).
    """
    print("\n" + "="*60)
    print("📊 CARGANDO OTRAS SERIES DIARIAS")
    print("="*60)
    
    other_series = {}
    
    # AHMSA
    ahmsa_files = glob.glob(os.path.join(DATA_RAW_DIR, 'ahmsa_*.csv'))
    for file_path in ahmsa_files:
        try:
            df = pd.read_csv(file_path)
            
            # Buscar columna de precio de cierre
            value_col = None
            if 'Close' in df.columns:
                value_col = 'Close'
            elif 'precio_cierre' in df.columns:
                value_col = 'precio_cierre'
            elif 'valor' in df.columns:
                value_col = 'valor'
            
            if 'fecha' in df.columns and value_col:
                filename = os.path.basename(file_path)
                
                # CASO ESPECIAL: Manejar archivos con doble prefijo "ahmsa_ahmsa_"
                if filename.startswith('ahmsa_ahmsa_'):
                    # Si tiene doble ahmsa, extraer correctamente el segundo ahmsa
                    series_name = 'ahmsa'
                else:
                    # Caso normal: extraer el nombre después del primer ahmsa_
                    series_name = filename.replace('ahmsa_', '').replace('.csv', '').split('_')[0]
                
                df_clean = df[['fecha', value_col]].copy()
                df_clean.columns = ['fecha', 'valor']
                df_clean['fecha'] = safe_to_datetime(df_clean['fecha'], normalize=True)
                
                # Manejar duplicados después de normalizar
                df_clean = df_clean.groupby('fecha')['valor'].mean().reset_index()
                
                if df_clean['fecha'].max() >= FECHA_CORTE:
                    variable_name = VARIABLE_NAMES.get(series_name, series_name)
                    df_clean.columns = ['fecha', variable_name]
                    df_clean.set_index('fecha', inplace=True)
                    df_clean = df_clean.dropna()
                    df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                    
                    other_series[variable_name] = df_clean
                    print(f"✅ AHMSA - {variable_name}: {len(df_clean)} registros")
        except Exception as e:
            pass
    
    # Raw Materials
    raw_files = glob.glob(os.path.join(DATA_RAW_DIR, 'RawMaterials_*.csv'))
    for file_path in raw_files:
        try:
            df = pd.read_csv(file_path)
            
            # Raw Materials usa 'precio_cierre'
            value_col = None
            if 'precio_cierre' in df.columns:
                value_col = 'precio_cierre'
            elif 'Close' in df.columns:
                value_col = 'Close'
            elif 'valor' in df.columns:
                value_col = 'valor'
            
            if 'fecha' in df.columns and value_col:
                filename = os.path.basename(file_path)
                series_name = filename.replace('RawMaterials_', '').replace('.csv', '').split('_20')[0]
                
                df_clean = df[['fecha', value_col]].copy()
                df_clean.columns = ['fecha', 'valor']
                df_clean['fecha'] = safe_to_datetime(df_clean['fecha'], normalize=True)
                
                # Manejar duplicados después de normalizar
                df_clean = df_clean.groupby('fecha')['valor'].mean().reset_index()
                
                if df_clean['fecha'].max() >= FECHA_CORTE:
                    variable_name = VARIABLE_NAMES.get(series_name, series_name)
                    df_clean.columns = ['fecha', variable_name]
                    df_clean.set_index('fecha', inplace=True)
                    df_clean = df_clean.dropna()
                    df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                    
                    other_series[variable_name] = df_clean
                    print(f"✅ RawMaterials - {variable_name}: {len(df_clean)} registros")
        except Exception as e:
            pass
    
    # FRED diarias
    fred_files = glob.glob(os.path.join(DATA_RAW_DIR, 'FRED_*.csv'))
    daily_fred = ['dxy_index', 'natural_gas']
    
    for file_path in fred_files:
        filename = os.path.basename(file_path)
        if any(s in filename for s in daily_fred):
            try:
                df = pd.read_csv(file_path)
                
                # Identificar columnas
                date_col = None
                value_col = None
                
                for col in df.columns:
                    if col.lower() in ['date', 'fecha']:
                        date_col = col
                    elif df[col].dtype in ['float64', 'int64'] and col != date_col:
                        value_col = col
                        break
                
                if date_col and value_col:
                    df_clean = df[[date_col, value_col]].copy()
                    df_clean.columns = ['fecha', 'valor']
                    df_clean['fecha'] = safe_to_datetime(df_clean['fecha'])
                    
                    if df_clean['fecha'].max() >= FECHA_CORTE:
                        for key in daily_fred:
                            if key in filename:
                                variable_name = VARIABLE_NAMES.get(key, key) + '_fred'
                                break
                        
                        df_clean.columns = ['fecha', variable_name]
                        df_clean.set_index('fecha', inplace=True)
                        
                        if df_clean.index.duplicated().any():
                            df_clean = df_clean.groupby(level=0).mean()
                        
                        df_clean = df_clean.dropna()
                        df_clean = apply_weekend_strategy(df_clean, WEEKEND_STRATEGY)
                        
                        other_series[variable_name] = df_clean
                        print(f"✅ FRED - {variable_name}: {len(df_clean)} registros")
            except Exception as e:
                pass
    
    return other_series

def join_all_series(all_series_dict):
    """
    Realiza el JOIN de todas las series diarias.
    IMPORTANTE: Trunca el dataset a la fecha máxima de la variable objetivo (precio_varilla_lme).
    """
    print("\n" + "="*60)
    print("🔗 REALIZANDO JOIN DE SERIES DIARIAS")
    print("="*60)
    print(f"Estrategia de fines de semana: {WEEKEND_STRATEGY}")
    
    if not all_series_dict:
        print("❌ No hay series para hacer join")
        return None
    
    # CRÍTICO: Identificar la variable objetivo y sus fechas límite
    target_variable = 'precio_varilla_lme'
    target_min_date = None
    target_max_date = None
    
    if target_variable in all_series_dict:
        target_min_date = all_series_dict[target_variable].index.min()
        target_max_date = all_series_dict[target_variable].index.max()
        print(f"\n⭐ VARIABLE OBJETIVO DETECTADA: {target_variable}")
        print(f"   Fecha mínima: {target_min_date.date()}")
        print(f"   Fecha máxima: {target_max_date.date()}")
        print(f"   TODAS las series serán truncadas a este rango")
    else:
        print(f"\n⚠️ ADVERTENCIA: Variable objetivo '{target_variable}' no encontrada")
        print("   Usando todas las fechas disponibles")
    
    # Comenzar con la primera serie
    first_key = list(all_series_dict.keys())[0]
    consolidated_df = all_series_dict[first_key].copy()
    print(f"\nIniciando con: {first_key}")
    print(f"Shape inicial: {consolidated_df.shape}")
    
    # Hacer join con las demás series
    for i, (name, df) in enumerate(list(all_series_dict.items())[1:], 1):
        if i % 10 == 1:
            print(f"\nProcesando series {i} a {min(i+9, len(all_series_dict)-1)}...")
        
        nan_before = consolidated_df.isna().sum().sum()
        consolidated_df = consolidated_df.join(df, how='outer')
        nan_after = consolidated_df.isna().sum().sum()
    
    # Ordenar por fecha
    consolidated_df = consolidated_df.sort_index()
    
    # TRUNCAR AL RANGO DE FECHAS DE LA VARIABLE OBJETIVO
    if target_min_date is not None and target_max_date is not None:
        print(f"\n✂️ TRUNCANDO DATASET AL RANGO DE LA VARIABLE OBJETIVO")
        print(f"   Fecha mínima: {target_min_date.date()}")
        print(f"   Fecha máxima: {target_max_date.date()}")
        
        before_truncate = len(consolidated_df)
        
        # Aplicar filtro de fecha mínima y máxima
        consolidated_df = consolidated_df[
            (consolidated_df.index >= target_min_date) & 
            (consolidated_df.index <= target_max_date)
        ]
        
        after_truncate = len(consolidated_df)
        
        print(f"   Registros antes del truncado: {before_truncate}")
        print(f"   Registros después del truncado: {after_truncate}")
        print(f"   Registros eliminados: {before_truncate - after_truncate}")
        
        # Verificar que tenemos la variable objetivo en todo el rango
        if target_variable in consolidated_df.columns:
            non_null_target = consolidated_df[target_variable].count()
            print(f"   Registros no nulos de {target_variable}: {non_null_target}")
            print(f"   Completitud de variable objetivo: {(non_null_target/len(consolidated_df)*100):.1f}%")
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS FINALES DEL DATASET")
    print("="*60)
    print(f"• Dimensiones: {consolidated_df.shape}")
    print(f"• Rango temporal: {consolidated_df.index.min().date()} a {consolidated_df.index.max().date()}")
    
    # Análisis de gaps global
    gap_analysis = analyze_gaps(consolidated_df)
    print(f"\n📈 ANÁLISIS DE GAPS:")
    print(f"• Total de días: {gap_analysis['total_dias']}")
    print(f"• Gaps de fin de semana: {gap_analysis['gaps_fin_semana']}")
    print(f"• Otros gaps: {gap_analysis['otros_gaps']}")
    print(f"• Gap máximo: {gap_analysis['max_gap_dias']} días")
    
    # Completitud
    total_cells = consolidated_df.shape[0] * consolidated_df.shape[1]
    non_null = consolidated_df.count().sum()
    print(f"\n📊 COMPLETITUD:")
    print(f"• Total de observaciones: {total_cells:,}")
    print(f"• Valores no nulos: {non_null:,}")
    print(f"• Valores nulos: {total_cells - non_null:,}")
    print(f"• Porcentaje de completitud: {(non_null / total_cells * 100):.2f}%")
    
    return consolidated_df

def save_results(df, all_series_dict):
    """
    Guarda el DataFrame consolidado y metadata.
    """
    print("\n" + "="*60)
    print("💾 GUARDANDO RESULTADOS")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar CSV principal
    csv_path = os.path.join(OUTPUT_DIR, f'daily_series_consolidated_{timestamp}.csv')
    df.to_csv(csv_path)
    print(f"✅ CSV guardado: {csv_path}")
    
    # Guardar versión latest
    csv_latest = os.path.join(OUTPUT_DIR, 'daily_series_consolidated_latest.csv')
    df.to_csv(csv_latest)
    print(f"✅ CSV latest: {csv_latest}")
    
    # Crear metadata
    gap_analysis = analyze_gaps(df)
    
    metadata = {
        'timestamp_generacion': timestamp,
        'fecha_corte_actualizacion': str(FECHA_CORTE.date()),
        'estrategia_fines_semana': WEEKEND_STRATEGY,
        'total_series': len(df.columns),
        'total_observaciones': len(df),
        'rango_temporal': {
            'inicio': str(df.index.min().date()),
            'fin': str(df.index.max().date())
        },
        'analisis_gaps': gap_analysis,
        'series_incluidas': list(df.columns),
        'fuentes': {
            'LME': [col for col in df.columns if 'lme' in col],
            'YahooFinance': [col for col in df.columns if any(x in col for x in ['wti', 'brent', 'sp500', 'vix', 'etf'])],
            'Banxico': [col for col in df.columns if any(x in col for x in ['tipo_cambio', 'tiie', 'udis'])],
            'AHMSA': [col for col in df.columns if 'ahmsa' in col],
            'RawMaterials': [col for col in df.columns if any(x in col for x in ['vale', 'rio', 'bhp', 'teck', 'aal', 'slx', 'xme', 'xlb'])],
            'FRED': [col for col in df.columns if 'fred' in col]
        },
        'estadisticas': {
            'porcentaje_completitud': float(df.count().sum() / (df.shape[0] * df.shape[1]) * 100),
            'valores_no_nulos': int(df.count().sum()),
            'valores_nulos': int(df.isna().sum().sum())
        },
        'completitud_por_variable': {
            col: float((df[col].count() / len(df)) * 100) 
            for col in df.columns[:20]  # Solo primeras 20 para no hacer muy largo
        }
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
    
    return csv_path, metadata_path

def main():
    """
    Función principal que orquesta todo el proceso.
    """
    print("\n" + "="*80)
    print("🚀 INICIANDO PROCESO DE JOIN DE SERIES DIARIAS ACTUALIZADAS")
    print("="*80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio de entrada: {DATA_RAW_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Fecha de corte para actualización: {FECHA_CORTE.date()}")
    print(f"Estrategia de fines de semana: {WEEKEND_STRATEGY}")
    
    # Cargar series de cada fuente
    all_series = {}
    
    # LME (PRIORIDAD: contiene la variable objetivo)
    print("\n🎯 CARGANDO FUENTE PRIORITARIA (contiene variable objetivo)")
    lme_series = load_lme_series()
    all_series.update(lme_series)
    
    # Verificar que tenemos la variable objetivo
    if 'precio_varilla_lme' not in all_series:
        print("\n⚠️ ADVERTENCIA: Variable objetivo 'precio_varilla_lme' no encontrada en LME")
        print("   Verificando si existe con otro nombre...")
        for key in all_series.keys():
            if 'varilla' in key.lower() or 'rebar' in key.lower():
                print(f"   Posible variable objetivo encontrada: {key}")
    
    # Yahoo Finance
    yahoo_series = load_yahoo_series()
    all_series.update(yahoo_series)
    
    # Banxico
    banxico_series = load_banxico_series()
    all_series.update(banxico_series)
    
    # Otras series
    other_series = load_other_series()
    all_series.update(other_series)
    
    # Resumen de carga
    print("\n" + "="*60)
    print("📋 RESUMEN DE CARGA")
    print("="*60)
    print(f"Total de series cargadas: {len(all_series)}")
    print(f"• LME: {len(lme_series)} series")
    print(f"• Yahoo Finance: {len(yahoo_series)} series")
    print(f"• Banxico: {len(banxico_series)} series")
    print(f"• Otras: {len(other_series)} series")
    
    if len(all_series) == 0:
        print("\n❌ No se encontraron series diarias actualizadas. Abortando proceso.")
        return
    
    # Realizar JOIN
    consolidated_df = join_all_series(all_series)
    
    if consolidated_df is not None:
        # Guardar resultados
        csv_path, metadata_path = save_results(consolidated_df, all_series)
        
        print("\n" + "="*80)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"Archivos generados en: {OUTPUT_DIR}")
        
        # Recomendaciones según estrategia
        print("\n📝 RECOMENDACIONES:")
        if WEEKEND_STRATEGY == 'business_days':
            print("• Usando solo días hábiles (recomendado para modelos financieros)")
            print("• Los modelos LSTM/ARIMA manejarán bien esta estructura")
            print("• Cuidado al calcular retornos entre viernes y lunes")
        elif WEEKEND_STRATEGY == 'forward_fill':
            print("• Forward fill aplicado - datos continuos pero con autocorrelación artificial")
            print("• Útil para integración con datos de otras frecuencias")
            print("• Considerar el sesgo introducido en las métricas")
        
        return consolidated_df
    else:
        print("\n❌ Error en el proceso de JOIN")
        return None

if __name__ == "__main__":
    # Configurar estrategia de fines de semana
    print("\n" + "="*80)
    print("CONFIGURACIÓN DE ESTRATEGIA DE FINES DE SEMANA")
    print("="*80)
    print("Opciones disponibles:")
    print("1. business_days - Mantener solo días hábiles (RECOMENDADO)")
    print("2. forward_fill - Propagar último valor conocido")
    print("3. interpolate - Interpolación lineal")
    print("4. keep_gaps - Mantener gaps como NaN")
    
    # Por defecto usar business_days
    WEEKEND_STRATEGY = 'business_days'
    print(f"\nUsando estrategia: {WEEKEND_STRATEGY}")
    
    df_result = main()
    
    if df_result is not None:
        print("\n" + "="*80)
        print("📊 VISTA PREVIA DEL DATASET FINAL")
        print("="*80)
        print(f"\nPrimeras 5 filas ({WEEKEND_STRATEGY}):")
        print(df_result.head())
        print("\nInfo del DataFrame:")
        print(f"• Shape: {df_result.shape}")
        print(f"• Memoria: {df_result.memory_usage().sum() / 1024**2:.2f} MB")
