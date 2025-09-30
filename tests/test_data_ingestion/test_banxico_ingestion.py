#!/usr/bin/env python3
"""
Test de ingesta específico para BANXICO
Analiza granularidad temporal y cobertura de datos desde 2020-01-01
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

from src.data_ingestion.banxico_collector import BANXICOCollector
from datetime import datetime, timedelta
import pandas as pd


class TestBANXICOIngestion:
    """Test class para ingesta de BANXICO"""
    
    @pytest.mark.asyncio
    async def test_banxico_data_coverage(self):
        """Test cobertura de datos BANXICO desde 2020-01-01"""
        
        print("🏦 TEST BANXICO - COBERTURA DE DATOS")
        print("=" * 40)
        
        api_token = os.getenv('BANXICO_API_TOKEN')
        assert api_token is not None, "BANXICO API token debe estar configurado"
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with BANXICOCollector(api_token) as collector:
            print(f"✅ Colector BANXICO inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            
            # Obtener información de series
            series_info = collector.get_series_info()
            print(f"📊 Total series configuradas: {series_info['total_series']}")
            
            # Analizar cada serie individualmente
            series_analysis = {}
            
            for series_key, config in collector.series_config.items():
                print(f"\\n📈 Analizando serie: {series_key}")
                print(f"   Nombre: {config['name']}")
                print(f"   Frecuencia configurada: {config['frequency']}")
                print(f"   Importancia: {config['importance']}")
                
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
                        date_analysis = self._analyze_temporal_granularity(df, series_key)
                        
                        series_analysis[series_key] = {
                            'config': config,
                            'result': result,
                            'temporal_analysis': date_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango: {df['fecha'].min()} a {df['fecha'].max()}")
                        print(f"   🔗 Fuente: {result['source']}")
                        print(f"   📊 Granularidad detectada: {date_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio entre datos: {date_analysis['avg_days_between']:.1f}")
                        
                        if date_analysis['is_daily']:
                            print(f"   🎯 ¡SERIE DIARIA! - Ideal para predicción")
                        elif date_analysis['detected_frequency'] == 'weekly':
                            print(f"   📅 Serie semanal - Convertible a diaria")
                        elif date_analysis['detected_frequency'] == 'monthly':
                            print(f"   📅 Serie mensual - Requiere interpolación")
                        
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
            
            # Resumen de BANXICO
            successful_series = len([s for s in series_analysis.values() if s['status'] == 'success'])
            daily_series = len([s for s in series_analysis.values() 
                              if s['status'] == 'success' and s['temporal_analysis']['is_daily']])
            api_real_series = len([s for s in series_analysis.values() 
                                 if s['status'] == 'success' and s['result']['source'] == 'banxico_api'])
            
            print(f"\\n🎯 RESUMEN BANXICO:")
            print(f"   📊 Series exitosas: {successful_series}/{len(collector.series_config)}")
            print(f"   📅 Series diarias: {daily_series}")
            print(f"   🔗 APIs reales: {api_real_series}")
            
            # Assertions
            assert successful_series >= 5, f"BANXICO should have at least 5 working series"
            assert api_real_series >= 2, f"BANXICO should have at least 2 real API series"
            
            return series_analysis
    
    def _analyze_temporal_granularity(self, df: pd.DataFrame, series_name: str) -> dict:
        """Analizar granularidad temporal de una serie"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'avg_days_between': 0,
                'total_points': 0,
                'date_range_days': 0
            }
        
        # Convertir fechas y ordenar
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha')
        
        # Calcular diferencias entre fechas
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
            suitable_for_daily = True
        elif avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False
        elif avg_days <= 95:
            detected_freq = 'quarterly'
            is_daily = False
            suitable_for_daily = False
        else:
            detected_freq = 'irregular'
            is_daily = False
            suitable_for_daily = False
        
        # Calcular estadísticas
        date_range = (df['fecha'].max() - df['fecha'].min()).days
        
        return {
            'detected_frequency': detected_freq,
            'is_daily': is_daily,
            'suitable_for_daily': suitable_for_daily,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df['fecha'].min(),
            'end_date': df['fecha'].max(),
            'coverage_from_2020': df['fecha'].min() <= pd.Timestamp('2020-01-01')
        }


# Función para ejecutar manualmente
async def run_banxico_test():
    """Ejecutar test de BANXICO manualmente"""
    
    test_instance = TestBANXICOIngestion()
    result = await test_instance.test_banxico_data_coverage()
    
    print("\\n📋 ANÁLISIS DETALLADO BANXICO:")
    print("=" * 35)
    
    for series_key, analysis in result.items():
        print(f"\\n📊 {series_key.upper()}:")
        
        if analysis['status'] == 'success':
            temp_analysis = analysis['temporal_analysis']
            print(f"   📅 Frecuencia: {temp_analysis['detected_frequency']}")
            print(f"   🎯 Es diaria: {'SÍ' if temp_analysis['is_daily'] else 'NO'}")
            print(f"   📊 Puntos: {temp_analysis['total_points']}")
            print(f"   📅 Cobertura desde 2020: {'SÍ' if temp_analysis['coverage_from_2020'] else 'NO'}")
            print(f"   ⏱️ Días promedio entre datos: {temp_analysis['avg_days_between']:.1f}")
        else:
            print(f"   ❌ Estado: {analysis['status']}")
            if 'error' in analysis:
                print(f"   Error: {analysis['error']}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_banxico_test())
    print("\\n✅ Test BANXICO completado")
