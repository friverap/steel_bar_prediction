#!/usr/bin/env python3
"""
Test de ingesta específico para INEGI usando INEGIpy
Basado en: https://github.com/andreslomeliv/DatosMex/tree/master/INEGIpy
"""

import asyncio
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.data_ingestion.inegi_collector import INEGICollector

class TestINEGIIngestion:
    """Test class para ingesta de INEGI usando INEGIpy"""
    
    @pytest.mark.asyncio
    async def test_inegi_data_coverage(self):
        """Test cobertura y granularidad de datos INEGI desde 2020"""
        
        print("\n" + "=" * 60)
        print("📈 TEST INEGI - INDICADORES ECONÓMICOS MEXICANOS")
        print("=" * 60)
        print("📚 Usando INEGIpy de https://github.com/andreslomeliv/DatosMex")
        print("=" * 60)
        
        # Verificar API token
        api_token = os.getenv('INEGI_API_TOKEN') or os.getenv('INEGI_API_KEY')
        
        if not api_token:
            print("❌ ERROR: INEGI_API_TOKEN no configurado en .env")
            print("   Obtener token en: https://www.inegi.org.mx/app/api/denue/v1/tokenVerify.aspx")
            assert False, "API token requerido"
        
        print(f"✅ API Token: {api_token[:8]}...{api_token[-8:]}")
        
        # Configurar período de análisis
        start_date = "2020-01-01"
        end_date = "2024-12-31"
        
        print(f"📅 Período de análisis: {start_date} a {end_date}")
        
        # Inicializar colector
        async with INEGICollector(api_token) as collector:
            print("✅ Colector INEGI inicializado")
            
            # Obtener información de configuración
            indicators_info = collector.get_indicators_info()
            
            print(f"\n📊 CONFIGURACIÓN DE INDICADORES:")
            print(f"   Total indicadores: {indicators_info['total_indicators']}")
            print(f"   INEGIpy disponible: {indicators_info['inegipy_available']}")
            print(f"   API token configurado: {indicators_info['api_token_configured']}")
            
            print(f"\n📊 POR IMPORTANCIA:")
            print(f"   🔴 Críticos: {len(indicators_info['critical_importance'])} - {indicators_info['critical_importance']}")
            print(f"   🟠 Altos: {len(indicators_info['high_importance'])} - {indicators_info['high_importance']}")
            print(f"   🟡 Medios: {len(indicators_info['medium_importance'])} - {indicators_info['medium_importance']}")
            
            print(f"\n📊 POR FRECUENCIA:")
            print(f"   📅 Mensuales: {len(indicators_info['frequencies']['monthly'])}")
            print(f"   📅 Trimestrales: {len(indicators_info['frequencies']['quarterly'])}")
            
            print(f"\n📊 POR CATEGORÍA:")
            for category, indicators in indicators_info['categories'].items():
                print(f"   {category}: {len(indicators)} indicadores")
            
            # Analizar cada indicador
            print("\n" + "=" * 60)
            print("📊 ANÁLISIS INDIVIDUAL DE INDICADORES")
            print("=" * 60)
            
            indicators_analysis = {}
            
            for indicator_key, config in collector.indicators_config.items():
                print(f"\n🔍 {indicator_key.upper()}")
                print(f"   📌 ID: {config['id']}")
                print(f"   📝 Nombre: {config['name']}")
                print(f"   📅 Frecuencia: {config['frequency']}")
                print(f"   ⭐ Importancia: {config['importance']}")
                print(f"   📁 Categoría: {config['category']}")
                
                try:
                    # Obtener datos del indicador
                    result = await collector.get_indicator_data(
                        indicator_key,
                        start_date,
                        end_date,
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal
                        date_analysis = self._analyze_temporal_granularity(df, config)
                        
                        indicators_analysis[indicator_key] = {
                            'config': config,
                            'result': result,
                            'temporal_analysis': date_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango: {df['fecha'].min():%Y-%m-%d} a {df['fecha'].max():%Y-%m-%d}")
                        print(f"   🔗 Fuente: {result['source']}")
                        print(f"   📊 Granularidad detectada: {date_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio entre datos: {date_analysis['avg_days_between']:.1f}")
                        
                        # Estadísticas del valor
                        print(f"   📈 Valor actual: {result['latest_value']:.2f}")
                        print(f"   📊 Media: {df['valor'].mean():.2f}")
                        print(f"   📊 Desv. Est.: {df['valor'].std():.2f}")
                        
                        # Información específica por categoría
                        if config['category'] == 'inflation':
                            if len(df) > 12:
                                inflacion_anual = ((df['valor'].iloc[-1] / df['valor'].iloc[-13]) - 1) * 100
                                print(f"   💹 Inflación anualizada: {inflacion_anual:.2f}%")
                        elif config['category'] == 'production':
                            print(f"   🏭 PRODUCCIÓN - Indicador clave para demanda de acero")
                        elif config['category'] == 'steel_sector':
                            print(f"   🏗️ SECTOR METALÚRGICO - Indicador directo del mercado")
                        elif config['category'] == 'producer_prices':
                            print(f"   💰 PRECIOS PRODUCTOR - Indicador de costos")
                        
                        # Calidad de datos para predicción diaria
                        if date_analysis['suitable_for_daily']:
                            print(f"   🎯 APTO para predicción diaria")
                        else:
                            print(f"   ⚠️ Requiere INTERPOLACIÓN para predicción diaria")
                    else:
                        indicators_analysis[indicator_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos disponibles'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        print(f"   🔗 Fuente: {result.get('source', 'N/A')}")
                        
                except Exception as e:
                    indicators_analysis[indicator_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)[:100]}")
            
            # Resumen general
            print("\n" + "=" * 60)
            print("📊 RESUMEN DE INGESTA INEGI")
            print("=" * 60)
            
            successful = len([i for i in indicators_analysis.values() if i['status'] == 'success'])
            with_data = len([i for i in indicators_analysis.values() 
                            if i['status'] == 'success' and i['result']['count'] > 0])
            monthly = len([i for i in indicators_analysis.values()
                         if i['status'] == 'success' and 
                         i['temporal_analysis']['detected_frequency'] == 'monthly'])
            daily_suitable = len([i for i in indicators_analysis.values()
                                if i['status'] == 'success' and
                                i['temporal_analysis']['suitable_for_daily']])
            
            print(f"✅ Indicadores exitosos: {successful}/{len(collector.indicators_config)}")
            print(f"📊 Con datos válidos: {with_data}")
            print(f"📅 Frecuencia mensual: {monthly}")
            print(f"🎯 Aptos para predicción diaria: {daily_suitable}")
            
            # Por categoría
            print(f"\n📊 RESUMEN POR CATEGORÍA:")
            category_summary = {}
            for indicator_key, analysis in indicators_analysis.items():
                if analysis['status'] == 'success':
                    category = analysis['config']['category']
                    if category not in category_summary:
                        category_summary[category] = []
                    category_summary[category].append(indicator_key)
            
            for category, indicators in category_summary.items():
                print(f"   {category}: {len(indicators)} indicadores exitosos")
            
            # Verificaciones de calidad
            print(f"\n✅ VERIFICACIONES:")
            assert successful >= 5, f"Se esperaban al menos 5 indicadores exitosos, obtenidos: {successful}"
            assert with_data >= 3, f"Se esperaban al menos 3 con datos, obtenidos: {with_data}"
            
            return indicators_analysis
    
    def _analyze_temporal_granularity(self, df: pd.DataFrame, config: dict) -> dict:
        """
        Analizar la granularidad temporal de los datos
        
        Args:
            df: DataFrame con columnas 'fecha' y 'valor'
            config: Configuración del indicador
            
        Returns:
            Diccionario con análisis temporal
        """
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'suitable_for_daily': False,
                'avg_days_between': 0,
                'total_points': 0,
                'date_range_days': 0,
                'coverage_from_2020': False
            }
        
        # Asegurar que fecha es datetime
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha')
        
        # Calcular diferencias entre fechas consecutivas
        date_diffs = df['fecha'].diff().dropna()
        avg_days = date_diffs.dt.days.mean()
        
        # Determinar frecuencia
        if avg_days <= 1.5:
            detected_freq = 'daily'
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 8:
            detected_freq = 'weekly'
            is_daily = False
            suitable_for_daily = True  # Se puede interpolar
        elif avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False  # Requiere interpolación significativa
        elif avg_days <= 95:
            detected_freq = 'quarterly'
            is_daily = False
            suitable_for_daily = False
        else:
            detected_freq = 'irregular'
            is_daily = False
            suitable_for_daily = False
        
        # Calcular métricas
        date_range = (df['fecha'].max() - df['fecha'].min()).days
        coverage_from_2020 = df['fecha'].min() <= pd.Timestamp('2020-01-01')
        
        return {
            'detected_frequency': detected_freq,
            'is_daily': is_daily,
            'suitable_for_daily': suitable_for_daily,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df['fecha'].min(),
            'end_date': df['fecha'].max(),
            'coverage_from_2020': coverage_from_2020,
            'data_density': (len(df) / date_range * 100) if date_range > 0 else 0
        }

# Función para ejecutar el test manualmente
async def run_inegi_test():
    """Ejecutar test de INEGI manualmente"""
    test_instance = TestINEGIIngestion()
    result = await test_instance.test_inegi_data_coverage()
    
    print("\n" + "=" * 60)
    print("📋 ANÁLISIS DETALLADO DE GRANULARIDAD")
    print("=" * 60)
    
    daily_indicators = []
    weekly_indicators = []
    monthly_indicators = []
    quarterly_indicators = []
    
    for indicator_key, analysis in result.items():
        if analysis['status'] == 'success':
            freq = analysis['temporal_analysis']['detected_frequency']
            
            if freq == 'daily':
                daily_indicators.append(indicator_key)
            elif freq == 'weekly':
                weekly_indicators.append(indicator_key)
            elif freq == 'monthly':
                monthly_indicators.append(indicator_key)
            elif freq == 'quarterly':
                quarterly_indicators.append(indicator_key)
    
    print(f"📅 Indicadores diarios: {len(daily_indicators)}")
    if daily_indicators:
        for ind in daily_indicators:
            print(f"   - {ind}")
    
    print(f"\n📅 Indicadores semanales: {len(weekly_indicators)}")
    if weekly_indicators:
        for ind in weekly_indicators:
            print(f"   - {ind}")
    
    print(f"\n📅 Indicadores mensuales: {len(monthly_indicators)}")
    if monthly_indicators:
        for ind in monthly_indicators:
            print(f"   - {ind}")
    
    print(f"\n📅 Indicadores trimestrales: {len(quarterly_indicators)}")
    if quarterly_indicators:
        for ind in quarterly_indicators:
            print(f"   - {ind}")
    
    print("\n" + "=" * 60)
    print("✅ Test INEGI con INEGIpy completado exitosamente")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    result = asyncio.run(run_inegi_test())