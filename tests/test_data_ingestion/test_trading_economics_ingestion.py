#!/usr/bin/env python3
"""
Test de ingesta específico para Trading Economics
Analiza granularidad temporal de indicadores mexicanos
Usando la librería oficial: https://github.com/tradingeconomics/tradingeconomics-python
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

from src.data_ingestion.trading_economics_collector import TradingEconomicsCollector
from datetime import datetime, timedelta
import pandas as pd

try:
    import tradingeconomics as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


class TestTradingEconomicsIngestion:
    """Test class para ingesta de Trading Economics"""
    
    @pytest.mark.asyncio
    async def test_trading_economics_data_coverage(self):
        """Test cobertura y granularidad de datos Trading Economics desde 2020-01-01"""
        
        print("📈 TEST TRADING ECONOMICS - INDICADORES MÉXICO")
        print("=" * 50)
        
        if not TE_AVAILABLE:
            print("⚠️ Librería tradingeconomics no disponible")
            print("   Instalar con: pip install tradingeconomics")
            return {}
        
        api_key = os.getenv('TRADING_ECONOMICS_API_KEY')
        assert api_key is not None, "Trading Economics API key debe estar configurado"
        print(f"🔑 API Key: {api_key[:15]}...")
        print(f"📚 Usando librería oficial tradingeconomics")
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with TradingEconomicsCollector(api_key) as collector:
            print(f"✅ Colector Trading Economics inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            print(f"🇲🇽 Enfoque: Indicadores de México (plan gratuito)")
            
            # Analizar cada indicador individualmente
            indicators_analysis = {}
            
            for indicator_key, config in collector.indicators_config.items():
                print(f"\\n📈 Analizando indicador: {indicator_key}")
                print(f"   País: {config['country']}")
                print(f"   Indicador: {config['indicator']}")
                print(f"   Nombre: {config['name']}")
                print(f"   Frecuencia esperada: {config['frequency']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_indicator_data(
                        indicator_key, 
                        start_date, 
                        end_date, 
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_indicator_temporal_granularity(df, indicator_key, config)
                        
                        indicators_analysis[indicator_key] = {
                            'config': config,
                            'result': result,
                            'temporal_analysis': date_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango: {df['fecha'].min()} a {df['fecha'].max()}")
                        print(f"   🔗 Fuente: {result['source']}")
                        print(f"   📊 Granularidad: {date_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio: {date_analysis['avg_days_between']:.1f}")
                        print(f"   🎯 Para predicción diaria: {'EXCELENTE' if date_analysis['suitable_for_daily'] else 'INTERPOLACIÓN'}")
                        
                        # Información específica para indicadores críticos
                        if config['importance'] == 'critical':
                            print(f"   ⭐ INDICADOR CRÍTICO para modelo")
                        
                    else:
                        indicators_analysis[indicator_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        
                except Exception as e:
                    indicators_analysis[indicator_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Resumen de Trading Economics
            successful_indicators = len([i for i in indicators_analysis.values() if i['status'] == 'success'])
            daily_indicators = len([i for i in indicators_analysis.values() 
                                  if i['status'] == 'success' and i['temporal_analysis']['suitable_for_daily']])
            api_real_indicators = len([i for i in indicators_analysis.values() 
                                     if i['status'] == 'success' and i['result']['source'] == 'trading_economics_api'])
            coverage_2020_indicators = len([i for i in indicators_analysis.values() 
                                          if i['status'] == 'success' and i['temporal_analysis'].get('coverage_from_2020', False)])
            
            print(f"\\n🎯 RESUMEN TRADING ECONOMICS:")
            print(f"   📊 Indicadores exitosos: {successful_indicators}/{len(collector.indicators_config)}")
            print(f"   🎯 Indicadores aptos para predicción diaria: {daily_indicators}")
            print(f"   🔗 APIs reales: {api_real_indicators}")
            print(f"   📅 Cobertura desde 2020: {coverage_2020_indicators}")
            
            return indicators_analysis
    
    def _analyze_indicator_temporal_granularity(self, df: pd.DataFrame, indicator_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de indicadores económicos"""
        
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
        
        # Convertir fechas y ordenar
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha')
        
        # Calcular diferencias entre fechas
        date_diffs = df['fecha'].diff().dropna()
        avg_days = date_diffs.dt.days.mean()
        
        # Determinar frecuencia para indicadores económicos
        if avg_days <= 1.5:
            detected_freq = 'daily'
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
        
        # Calcular estadísticas
        date_range = (df['fecha'].max() - df['fecha'].min()).days
        
        # Verificar datos faltantes
        missing_values = df['valor'].isna().sum()
        missing_pct = (missing_values / len(df)) * 100
        
        # Verificar cobertura desde 2020
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
            'missing_data_pct': missing_pct,
            'data_density': (len(df) / date_range) * 100 if date_range > 0 else 0
        }


# Función para ejecutar manualmente
async def run_trading_economics_test():
    """Ejecutar test de Trading Economics manualmente"""
    
    test_instance = TestTradingEconomicsIngestion()
    result = await test_instance.test_trading_economics_data_coverage()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_trading_economics_test())
    print("\\n✅ Test Trading Economics completado")
