#!/usr/bin/env python3
"""
Test de ingesta específico para AHMSA y empresas siderúrgicas
Analiza granularidad temporal de acciones y ETFs del sector
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

from src.data_ingestion.ahmsa_collector import AHMSACollector
from datetime import datetime, timedelta
import pandas as pd


class TestAHMSAIngestion:
    """Test class para ingesta de AHMSA y empresas siderúrgicas"""
    
    @pytest.mark.asyncio
    async def test_ahmsa_data_coverage(self):
        """Test cobertura y granularidad de datos AHMSA desde 2020-01-01"""
        
        print("🏭 TEST AHMSA - EMPRESAS SIDERÚRGICAS")
        print("=" * 40)
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with AHMSACollector() as collector:
            print(f"✅ Colector AHMSA inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            print(f"🆓 Yahoo Finance - Sin API key requerida")
            print(f"🎯 Enfoque: Empresas siderúrgicas y ETFs del sector")
            
            # Analizar cada empresa/ETF individualmente
            companies_analysis = {}
            
            for company_key, config in collector.companies_config.items():
                print(f"\\n🏭 Analizando empresa/ETF: {company_key}")
                print(f"   Símbolo: {config.get('symbol', 'N/A')}")
                print(f"   Nombre: {config['name']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                print(f"   País: {config['country']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_company_data(
                        company_key, 
                        start_date, 
                        end_date, 
                        period="5y",
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_company_temporal_granularity(df, company_key, config)
                        
                        companies_analysis[company_key] = {
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
                        
                        # Información específica por tipo
                        if config['category'] == 'steel_company':
                            print(f"   🏭 EMPRESA SIDERÚRGICA - Indicador directo del sector")
                        elif config['category'] == 'industry_index':
                            print(f"   📊 ETF SECTORIAL - Indicador agregado de la industria")
                        
                        # Mostrar métricas financieras si están disponibles
                        if 'financial_metrics' in result:
                            metrics = result['financial_metrics']
                            print(f"   💰 Precio actual: ${metrics['current_price']:.2f}")
                            print(f"   📈 Volatilidad 20d: {metrics['volatility_20d']:.2%}")
                        
                    else:
                        companies_analysis[company_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos o empresa no cotiza'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        
                except Exception as e:
                    companies_analysis[company_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Resumen de AHMSA
            successful_companies = len([c for c in companies_analysis.values() if c['status'] == 'success'])
            daily_companies = len([c for c in companies_analysis.values() 
                                 if c['status'] == 'success' and c['temporal_analysis']['suitable_for_daily']])
            api_real_companies = len([c for c in companies_analysis.values() 
                                    if c['status'] == 'success' and c['result']['source'] == 'yahoo_finance'])
            steel_companies = len([c for c in companies_analysis.values() 
                                 if c['status'] == 'success' and c['config']['category'] == 'steel_company'])
            
            print(f"\\n🎯 RESUMEN AHMSA:")
            print(f"   📊 Empresas/ETFs exitosos: {successful_companies}/{len(collector.companies_config)}")
            print(f"   🎯 Aptos para predicción diaria: {daily_companies}")
            print(f"   🔗 APIs reales (Yahoo Finance): {api_real_companies}")
            print(f"   🏭 Empresas siderúrgicas: {steel_companies}")
            
            return companies_analysis
    
    def _analyze_company_temporal_granularity(self, df: pd.DataFrame, company_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de empresas/ETFs"""
        
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
        
        # Verificar patrón de días laborables
        trading_days_only = self._check_trading_days_pattern(df['fecha'])
        
        # Acciones típicamente tienen datos diarios (días laborables)
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
        weekend_count = len(weekdays[(weekdays == 5) | (weekdays == 6)])
        
        # Si menos del 5% son fines de semana, son días laborables
        weekend_pct = (weekend_count / len(dates)) * 100
        
        return weekend_pct < 5


# Función para ejecutar manualmente
async def run_ahmsa_test():
    """Ejecutar test de AHMSA manualmente"""
    
    test_instance = TestAHMSAIngestion()
    result = await test_instance.test_ahmsa_data_coverage()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_ahmsa_test())
    print("\\n✅ Test AHMSA completado")
