#!/usr/bin/env python3
"""
Test de ingesta para Nasdaq Data Link (Quandl)
Analiza datos fundamentales de empresas de acero y series temporales
"""

import asyncio
import sys
import os
import pytest
from datetime import datetime, timedelta
import pandas as pd

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.data_ingestion.quandl_collector import QuandlCollector


class TestQuandlIngestion:
    """Test class para ingesta de Nasdaq Data Link"""
    
    @pytest.mark.asyncio
    async def test_steel_companies_fundamentals(self):
        """Test de datos fundamentales de empresas de acero"""
        
        print("🏗️ TEST NASDAQ DATA LINK - FUNDAMENTALES DE ACERO")
        print("=" * 50)
        
        api_key = os.getenv('QUANDL_API_KEY')
        assert api_key is not None, "Nasdaq Data Link API key debe estar configurada"
        print(f"🔑 API Key: {api_key[:10]}...")
        
        async with QuandlCollector(api_key) as collector:
            print(f"✅ Colector Nasdaq Data Link inicializado")
            print(f"🏗️ Empresas de acero configuradas: {len(collector.steel_companies)}")
            
            companies_analysis = {}
            
            # Analizar cada empresa de acero
            for company_key, company_info in collector.steel_companies.items():
                print(f"\n🏗️ Analizando: {company_key}")
                print(f"   Ticker: {company_info['ticker']}")
                print(f"   Nombre: {company_info['name']}")
                print(f"   Importancia: {company_info['importance']}")
                
                try:
                    result = await collector.get_company_fundamentals(
                        company_key,
                        save_raw=False
                    )
                    
                    if result and 'data' in result and not result['data'].empty:
                        df = result['data']
                        metrics = result.get('metrics', {})
                        
                        companies_analysis[company_key] = {
                            'info': company_info,
                            'result': result,
                            'metrics': metrics,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} períodos")
                        print(f"   📅 Último período: {result.get('latest_date', 'N/A')}")
                        
                        # Mostrar métricas clave
                        if metrics:
                            if 'revenue' in metrics and metrics['revenue']:
                                print(f"   💰 Revenue: ${metrics['revenue']:,.0f}")
                            if 'ebitda' in metrics and metrics['ebitda']:
                                print(f"   📊 EBITDA: ${metrics['ebitda']:,.0f}")
                            if 'eps_basic' in metrics and metrics['eps_basic']:
                                print(f"   📈 EPS: ${metrics['eps_basic']:.2f}")
                            if 'current_ratio' in metrics and metrics['current_ratio']:
                                print(f"   💧 Current Ratio: {metrics['current_ratio']:.2f}")
                            if 'debt_to_equity' in metrics and metrics['debt_to_equity']:
                                print(f"   💸 Debt/Equity: {metrics['debt_to_equity']:.2f}")
                            if 'gross_margin' in metrics and metrics['gross_margin']:
                                print(f"   📊 Gross Margin: {metrics['gross_margin']:.1f}%")
                    else:
                        companies_analysis[company_key] = {
                            'info': company_info,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos'
                        }
                        print(f"   ❌ Sin datos disponibles")
                        
                except Exception as e:
                    companies_analysis[company_key] = {
                        'info': company_info,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)[:100]}")
            
            # Resumen de empresas
            successful = len([c for c in companies_analysis.values() if c['status'] == 'success'])
            critical = len([c for c in companies_analysis.values() 
                          if c['status'] == 'success' and c['info']['importance'] == 'critical'])
            
            print(f"\n🎯 RESUMEN FUNDAMENTALES:")
            print(f"   📊 Empresas exitosas: {successful}/{len(collector.steel_companies)}")
            print(f"   ⭐ Empresas críticas con datos: {critical}")
            
            return companies_analysis
    
    @pytest.mark.asyncio
    async def test_timeseries_data_coverage(self):
        """Test de series temporales para commodities y economía"""
        
        print("\n📈 TEST NASDAQ DATA LINK - SERIES TEMPORALES")
        print("=" * 50)
        
        api_key = os.getenv('QUANDL_API_KEY')
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with QuandlCollector(api_key) as collector:
            print(f"📅 Período: {start_date} a {end_date}")
            print(f"📊 Series configuradas: {len(collector.timeseries_datasets)}")
            
            timeseries_analysis = {}
            
            for dataset_key, dataset_info in collector.timeseries_datasets.items():
                print(f"\n📈 Analizando: {dataset_key}")
                print(f"   Código: {dataset_info['code']}")
                print(f"   Nombre: {dataset_info['name']}")
                print(f"   Frecuencia: {dataset_info['frequency']}")
                print(f"   Categoría: {dataset_info['category']}")
                
                try:
                    result = await collector.get_timeseries_data(
                        dataset_key,
                        start_date,
                        end_date,
                        save_raw=False
                    )
                    
                    if result and 'data' in result and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal
                        temporal_analysis = self._analyze_temporal_granularity(df, dataset_info)
                        
                        timeseries_analysis[dataset_key] = {
                            'info': dataset_info,
                            'result': result,
                            'temporal_analysis': temporal_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango: {df['fecha'].min()} a {df['fecha'].max()}")
                        print(f"   📊 Granularidad detectada: {temporal_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio entre datos: {temporal_analysis['avg_days_between']:.1f}")
                        print(f"   🎯 Apto para predicción diaria: {'SÍ' if temporal_analysis['suitable_for_daily'] else 'NO'}")
                        
                        if result.get('latest_value'):
                            print(f"   💰 Último valor: {result['latest_value']:.2f}")
                    else:
                        timeseries_analysis[dataset_key] = {
                            'info': dataset_info,
                            'status': 'no_data',
                            'error': 'Sin datos o sin acceso'
                        }
                        print(f"   ❌ Sin datos (posible restricción de suscripción)")
                        
                except Exception as e:
                    timeseries_analysis[dataset_key] = {
                        'info': dataset_info,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)[:100]}")
            
            # Resumen de series temporales
            successful = len([t for t in timeseries_analysis.values() if t['status'] == 'success'])
            daily = len([t for t in timeseries_analysis.values() 
                        if t['status'] == 'success' and t['temporal_analysis']['suitable_for_daily']])
            
            print(f"\n🎯 RESUMEN SERIES TEMPORALES:")
            print(f"   📊 Series exitosas: {successful}/{len(collector.timeseries_datasets)}")
            print(f"   📅 Series diarias aptas: {daily}")
            
            # Clasificar por categoría
            by_category = {}
            for key, analysis in timeseries_analysis.items():
                if analysis['status'] == 'success':
                    category = analysis['info']['category']
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(key)
            
            print(f"\n📊 POR CATEGORÍA:")
            for category, items in by_category.items():
                print(f"   • {category}: {len(items)} series")
                for item in items:
                    print(f"     - {item}")
            
            return timeseries_analysis
    
    def _analyze_temporal_granularity(self, df: pd.DataFrame, dataset_info: dict) -> dict:
        """Analizar granularidad temporal de series"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'suitable_for_daily': False,
                'avg_days_between': 0,
                'total_points': 0
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
            suitable = True
        elif avg_days <= 3:
            detected_freq = 'business_daily'
            suitable = True
        elif avg_days <= 8:
            detected_freq = 'weekly'
            suitable = True
        elif avg_days <= 35:
            detected_freq = 'monthly'
            suitable = False
        else:
            detected_freq = 'irregular'
            suitable = False
        
        # Verificar cobertura desde 2020
        coverage_from_2020 = df['fecha'].min() <= pd.Timestamp('2020-01-01')
        
        return {
            'detected_frequency': detected_freq,
            'suitable_for_daily': suitable,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': (df['fecha'].max() - df['fecha'].min()).days,
            'coverage_from_2020': coverage_from_2020
        }
    
    @pytest.mark.asyncio
    async def test_full_ingestion(self):
        """Test completo de ingesta de Nasdaq Data Link"""
        
        print("\n🎯 TEST COMPLETO NASDAQ DATA LINK")
        print("=" * 50)
        
        api_key = os.getenv('QUANDL_API_KEY')
        
        async with QuandlCollector(api_key) as collector:
            # Test empresas
            companies = await self.test_steel_companies_fundamentals()
            
            # Test series temporales
            timeseries = await self.test_timeseries_data_coverage()
            
            # Resumen global
            print("\n" + "=" * 50)
            print("📋 RESUMEN GLOBAL NASDAQ DATA LINK")
            print("=" * 50)
            
            companies_success = len([c for c in companies.values() if c['status'] == 'success'])
            timeseries_success = len([t for t in timeseries.values() if t['status'] == 'success'])
            
            print(f"\n🏗️ EMPRESAS DE ACERO:")
            print(f"   ✅ Con datos: {companies_success}/{len(companies)}")
            critical_companies = [k for k, v in companies.items() 
                                 if v['status'] == 'success' and v['info']['importance'] == 'critical']
            if critical_companies:
                print(f"   ⭐ Críticas disponibles: {', '.join(critical_companies)}")
            
            print(f"\n📈 SERIES TEMPORALES:")
            print(f"   ✅ Con datos: {timeseries_success}/{len(timeseries)}")
            daily_series = [k for k, v in timeseries.items() 
                          if v['status'] == 'success' and v.get('temporal_analysis', {}).get('suitable_for_daily')]
            if daily_series:
                print(f"   📅 Diarias disponibles: {', '.join(daily_series)}")
            
            print(f"\n💡 CONCLUSIÓN:")
            if companies_success > 0 and timeseries_success > 0:
                print("   ✅ Nasdaq Data Link FUNCIONAL para predicción")
                print("   • Datos fundamentales de empresas disponibles")
                print("   • Series temporales económicas disponibles")
            elif companies_success > 0:
                print("   ⚠️ Parcialmente funcional")
                print("   • Solo datos fundamentales disponibles")
            else:
                print("   ❌ Limitado por suscripción")
                print("   • Considerar plan premium para más datos")
            
            return {
                'companies': companies,
                'timeseries': timeseries,
                'summary': {
                    'companies_success': companies_success,
                    'timeseries_success': timeseries_success,
                    'total_success': companies_success + timeseries_success
                }
            }


# Función para ejecutar manualmente
async def run_quandl_test():
    """Ejecutar test completo de Nasdaq Data Link"""
    test_instance = TestQuandlIngestion()
    result = await test_instance.test_full_ingestion()
    return result


if __name__ == "__main__":
    result = asyncio.run(run_quandl_test())
    print("\n✅ Test Nasdaq Data Link completado")
    
    # Exit con código basado en éxito
    success = result['summary']['total_success'] > 0
    exit(0 if success else 1)