#!/usr/bin/env python3
"""
Test de ingesta específico para Yahoo Finance
Analiza granularidad temporal de datos financieros adicionales
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

from src.data_ingestion.yahoo_finance import YahooFinanceCollector
from datetime import datetime, timedelta
import pandas as pd


class TestYahooFinanceIngestion:
    """Test class para ingesta de Yahoo Finance"""
    
    @pytest.mark.asyncio
    async def test_yahoo_finance_data_coverage(self):
        """Test cobertura y granularidad de datos Yahoo Finance desde 2020-01-01"""
        
        print("💰 TEST YAHOO FINANCE - DATOS FINANCIEROS")
        print("=" * 45)
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        collector = YahooFinanceCollector()
        print(f"✅ Colector Yahoo Finance inicializado")
        print(f"📅 Período: {start_date} a {end_date}")
        print(f"🆓 API gratuita - Sin API key requerida")
        
        # Analizar cada instrumento individualmente
        instruments_analysis = {}
        
        # Obtener configuración completa
        all_instruments = {**collector.financial_instruments, **collector.crypto_instruments}
        
        for instrument_key, config in all_instruments.items():
            print(f"\\n💰 Analizando instrumento: {instrument_key}")
            print(f"   Símbolo: {config['symbol']}")
            print(f"   Nombre: {config['name']}")
            print(f"   Categoría: {config['category']}")
            print(f"   Importancia: {config['importance']}")
            print(f"   Unidad: {config['unit']}")
            
            try:
                # Obtener datos sin guardar
                result = await collector.get_instrument_data(
                    instrument_key, 
                    start_date, 
                    end_date, 
                    period="5y",
                    include_crypto=True,
                    save_raw=False
                )
                
                if result and result['data'] is not None and not result['data'].empty:
                    df = result['data']
                    
                    # Análisis temporal detallado
                    date_analysis = self._analyze_yahoo_temporal_granularity(df, instrument_key, config)
                    
                    instruments_analysis[instrument_key] = {
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
                    
                    # Información específica por categoría
                    if config['category'] == 'energy':
                        print(f"   ⚡ ENERGÍA - Relevante para costos de producción")
                    elif config['category'] == 'currency':
                        print(f"   💱 DIVISA - Relevante para competitividad")
                    elif config['category'] == 'commodities':
                        print(f"   🥇 COMMODITY - Relevante para materias primas")
                    elif config['category'] == 'materials':
                        print(f"   🏗️ MATERIALES - Relevante para sector")
                    
                    # Mostrar métricas de performance si están disponibles
                    if 'performance_metrics' in result:
                        metrics = result['performance_metrics']
                        print(f"   📈 Retorno anualizado: {metrics['annualized_return']:.2f}")
                        print(f"   📊 Volatilidad: {metrics['volatility']:.2f}")
                    
                else:
                    instruments_analysis[instrument_key] = {
                        'config': config,
                        'result': result,
                        'status': 'no_data',
                        'error': 'Sin datos obtenidos'
                    }
                    print(f"   ❌ Sin datos obtenidos")
                    
            except Exception as e:
                instruments_analysis[instrument_key] = {
                    'config': config,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"   ❌ Error: {str(e)}")
        
        # Resumen de Yahoo Finance
        successful_instruments = len([i for i in instruments_analysis.values() if i['status'] == 'success'])
        daily_instruments = len([i for i in instruments_analysis.values() 
                               if i['status'] == 'success' and i['temporal_analysis']['suitable_for_daily']])
        api_real_instruments = len([i for i in instruments_analysis.values() 
                                  if i['status'] == 'success' and i['result']['source'] == 'yahoo_finance'])
        coverage_2020_instruments = len([i for i in instruments_analysis.values() 
                                       if i['status'] == 'success' and i['temporal_analysis']['coverage_from_2020']])
        
        print(f"\\n🎯 RESUMEN YAHOO FINANCE:")
        print(f"   📊 Instrumentos exitosos: {successful_instruments}/{len(all_instruments)}")
        print(f"   🎯 Instrumentos aptos para predicción diaria: {daily_instruments}")
        print(f"   🔗 APIs reales: {api_real_instruments}")
        print(f"   📅 Cobertura desde 2020: {coverage_2020_instruments}")
        
        # Assertions
        assert successful_instruments >= 8, f"Yahoo Finance should have at least 8 working instruments"
        assert daily_instruments >= 6, f"Yahoo Finance should have at least 6 daily instruments"
        assert api_real_instruments >= 8, f"Yahoo Finance should have at least 8 real API instruments"
        
        return instruments_analysis
    
    def _analyze_yahoo_temporal_granularity(self, df: pd.DataFrame, instrument_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de instrumentos Yahoo Finance"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'suitable_for_daily': False,
                'trading_days_only': False,
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
        
        # Verificar si son solo días laborables
        trading_days_only = self._check_trading_days_pattern(df['fecha'])
        
        # Yahoo Finance típicamente proporciona datos diarios para instrumentos financieros
        if avg_days <= 1.2:
            detected_freq = 'daily'
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 1.8 and trading_days_only:
            detected_freq = 'business_daily'  # Solo días laborables
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
            'trading_days_only': trading_days_only,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df['fecha'].min(),
            'end_date': df['fecha'].max(),
            'coverage_from_2020': coverage_from_2020,
            'missing_data_pct': missing_pct,
            'data_density': (len(df) / date_range) * 100 if date_range > 0 else 0
        }
    
    def _check_trading_days_pattern(self, dates: pd.Series) -> bool:
        """Verificar si las fechas siguen patrón de días laborables"""
        
        if len(dates) < 10:
            return False
        
        # Contar días por día de la semana
        weekdays = dates.dt.dayofweek
        weekend_count = len(weekdays[(weekdays == 5) | (weekdays == 6)])  # Sábado y domingo
        
        # Si menos del 5% son fines de semana, probablemente son solo días laborables
        weekend_pct = (weekend_count / len(dates)) * 100
        
        return weekend_pct < 5


# Función para ejecutar manualmente
async def run_yahoo_finance_test():
    """Ejecutar test de Yahoo Finance manualmente"""
    
    test_instance = TestYahooFinanceIngestion()
    result = await test_instance.test_yahoo_finance_data_coverage()
    
    print("\\n📋 ANÁLISIS DETALLADO YAHOO FINANCE:")
    print("=" * 40)
    
    daily_instruments = []
    weekly_instruments = []
    monthly_instruments = []
    
    for instrument_key, analysis in result.items():
        if analysis['status'] == 'success':
            temp_analysis = analysis['temporal_analysis']
            
            if temp_analysis['suitable_for_daily']:
                if temp_analysis['is_daily']:
                    daily_instruments.append(instrument_key)
                else:
                    weekly_instruments.append(instrument_key)
            else:
                monthly_instruments.append(instrument_key)
    
    print(f"\\n🎯 CLASIFICACIÓN POR GRANULARIDAD:")
    print(f"   📅 Instrumentos diarios: {len(daily_instruments)} - {daily_instruments}")
    print(f"   📅 Instrumentos semanales: {len(weekly_instruments)} - {weekly_instruments}")
    print(f"   📅 Instrumentos mensuales: {len(monthly_instruments)} - {monthly_instruments}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_yahoo_finance_test())
    print("\\n✅ Test Yahoo Finance completado")
