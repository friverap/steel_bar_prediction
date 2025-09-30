#!/usr/bin/env python3
"""
Test de ingesta específico para FRED
Analiza granularidad temporal de series económicas críticas para predicción de acero
"""

import asyncio
import sys
import os
import pytest

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.data_ingestion.fred_collector import FREDCollector
from datetime import datetime, timedelta
import pandas as pd


class TestFREDIngestion:
    """Test class para ingesta de FRED"""
    
    @pytest.mark.asyncio
    async def test_fred_data_coverage(self):
        """Test cobertura y granularidad de datos FRED desde 2020-01-01"""
        
        print("📊 TEST FRED - COBERTURA Y GRANULARIDAD")
        print("=" * 45)
        
        api_key = os.getenv('FRED_API_KEY')
        assert api_key is not None, "FRED API key debe estar configurado"
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with FREDCollector(api_key) as collector:
            print(f"✅ Colector FRED inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            
            # Obtener información de series
            series_info = collector.get_series_info()
            print(f"📊 Total series configuradas: {series_info['total_series']}")
            print(f"⭐ Series críticas: {len(series_info['critical_importance'])}")
            print(f"📈 Series diarias configuradas: {len(series_info['daily_series'])}")
            print(f"📅 Series mensuales configuradas: {len(series_info['monthly_series'])}")
            
            # Analizar cada serie individualmente
            series_analysis = {}
            
            for series_key, config in collector.series_config.items():
                print(f"\\n📈 Analizando serie FRED: {series_key}")
                print(f"   ID: {config['id']}")
                print(f"   Nombre: {config['name']}")
                print(f"   Frecuencia configurada: {config['frequency']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_series_data(
                        series_key, 
                        start_date, 
                        end_date, 
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_fred_temporal_granularity(df, series_key, config['id'])
                        
                        series_analysis[series_key] = {
                            'config': config,
                            'result': result,
                            'temporal_analysis': date_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango real: {df.iloc[:, 0].min()} a {df.iloc[:, 0].max()}")
                        print(f"   🔗 Fuente: {result['source']}")
                        print(f"   📊 Granularidad detectada: {date_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio entre datos: {date_analysis['avg_days_between']:.1f}")
                        print(f"   🎯 Apto para predicción diaria: {'SÍ' if date_analysis['suitable_for_daily'] else 'NO'}")
                        
                        # Verificar calidad de datos
                        if date_analysis['missing_data_pct'] > 0:
                            print(f"   ⚠️ Datos faltantes: {date_analysis['missing_data_pct']:.1f}%")
                        
                        if date_analysis['coverage_from_2020']:
                            print(f"   ✅ Cobertura desde 2020: SÍ")
                        else:
                            print(f"   ⚠️ Cobertura desde 2020: NO (inicia {date_analysis['start_date']})")
                        
                    else:
                        series_analysis[series_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        
                except Exception as e:
                    series_analysis[series_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Resumen de FRED
            successful_series = len([s for s in series_analysis.values() if s['status'] == 'success'])
            daily_series = len([s for s in series_analysis.values() 
                              if s['status'] == 'success' and s['temporal_analysis']['suitable_for_daily']])
            api_real_series = len([s for s in series_analysis.values() 
                                 if s['status'] == 'success' and s['result']['source'] == 'fred_api'])
            coverage_2020_series = len([s for s in series_analysis.values() 
                                      if s['status'] == 'success' and s['temporal_analysis']['coverage_from_2020']])
            
            print(f"\\n🎯 RESUMEN FRED:")
            print(f"   📊 Series exitosas: {successful_series}/{len(collector.series_config)}")
            print(f"   🎯 Series aptas para predicción diaria: {daily_series}")
            print(f"   🔗 APIs reales: {api_real_series}")
            print(f"   📅 Cobertura desde 2020: {coverage_2020_series}")
            
            # Assertions
            assert successful_series >= 6, f"FRED should have at least 6 working series"
            assert daily_series >= 2, f"FRED should have at least 2 daily series"
            assert api_real_series >= 6, f"FRED should have at least 6 real API series"
            
            return series_analysis
    
    def _analyze_fred_temporal_granularity(self, df: pd.DataFrame, series_name: str, series_id: str) -> dict:
        """Analizar granularidad temporal específica de FRED"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'suitable_for_daily': False,
                'avg_days_between': 0,
                'total_points': 0,
                'date_range_days': 0,
                'missing_data_pct': 100
            }
        
        # FRED devuelve datos en formato específico
        # Primera columna es fecha, segunda columna es el valor (series_id)
        date_col = df.columns[0]
        value_col = series_id
        
        # Convertir fechas y ordenar
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calcular diferencias entre fechas
        date_diffs = df[date_col].diff().dropna()
        avg_days = date_diffs.dt.days.mean()
        
        # Determinar frecuencia específica de FRED
        if avg_days <= 1.5:
            detected_freq = 'daily'
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 3.5:
            detected_freq = 'business_daily'  # Días laborables
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 8:
            detected_freq = 'weekly'
            is_daily = False
            suitable_for_daily = True  # Convertible a diaria
        elif avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False  # Requiere interpolación
        elif avg_days <= 95:
            detected_freq = 'quarterly'
            is_daily = False
            suitable_for_daily = False
        else:
            detected_freq = 'irregular'
            is_daily = False
            suitable_for_daily = False
        
        # Calcular estadísticas adicionales
        date_range = (df[date_col].max() - df[date_col].min()).days
        
        # Verificar datos faltantes
        if value_col in df.columns:
            missing_values = df[value_col].isna().sum()
            missing_pct = (missing_values / len(df)) * 100
        else:
            missing_pct = 0
        
        # Verificar cobertura desde 2020
        coverage_from_2020 = df[date_col].min() <= pd.Timestamp('2020-01-01')
        
        return {
            'detected_frequency': detected_freq,
            'is_daily': is_daily,
            'suitable_for_daily': suitable_for_daily,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df[date_col].min(),
            'end_date': df[date_col].max(),
            'coverage_from_2020': coverage_from_2020,
            'missing_data_pct': missing_pct,
            'data_density': (len(df) / date_range) * 100 if date_range > 0 else 0
        }


# Función para ejecutar manualmente
async def run_fred_test():
    """Ejecutar test de FRED manualmente"""
    
    test_instance = TestFREDIngestion()
    result = await test_instance.test_fred_data_coverage()
    
    print("\\n📋 ANÁLISIS DETALLADO FRED:")
    print("=" * 30)
    
    daily_suitable = []
    weekly_suitable = []
    monthly_only = []
    
    for series_key, analysis in result.items():
        if analysis['status'] == 'success':
            temp_analysis = analysis['temporal_analysis']
            
            if temp_analysis['suitable_for_daily']:
                if temp_analysis['is_daily']:
                    daily_suitable.append(series_key)
                else:
                    weekly_suitable.append(series_key)
            else:
                monthly_only.append(series_key)
    
    print(f"\\n🎯 CLASIFICACIÓN POR GRANULARIDAD:")
    print(f"   📅 Series diarias: {len(daily_suitable)} - {daily_suitable}")
    print(f"   📅 Series semanales (convertibles): {len(weekly_suitable)} - {weekly_suitable}")
    print(f"   📅 Series mensuales (interpolación): {len(monthly_only)} - {monthly_only}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_fred_test())
    print("\\n✅ Test FRED completado")
