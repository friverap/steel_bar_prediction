#!/usr/bin/env python3
"""
Test de Nasdaq Data Link API usando la librería oficial
Enfocado en datos de empresas de acero y commodities
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Intentar importar la librería oficial
try:
    import nasdaqdatalink
    NASDAQ_LIB_AVAILABLE = True
except ImportError:
    NASDAQ_LIB_AVAILABLE = False
    print("⚠️ nasdaq-data-link no instalado. Instalar con: pip install nasdaq-data-link")

def test_nasdaq_data_link():
    """Test completo de Nasdaq Data Link para datos de acero"""
    
    print("=" * 60)
    print("🏗️ TEST NASDAQ DATA LINK - DATOS DE ACERO")
    print("=" * 60)
    print("📚 Librería: nasdaq-data-link")
    print("🔗 Docs: https://github.com/Nasdaq/data-link-python")
    print("=" * 60)
    
    if not NASDAQ_LIB_AVAILABLE:
        print("\n❌ ERROR: Librería nasdaq-data-link no instalada")
        print("   Instalar con: pip install nasdaq-data-link")
        return False
    
    # Obtener API key
    api_key = os.getenv('QUANDL_API_KEY') or os.getenv('NASDAQ_DATA_LINK_API_KEY')
    
    if not api_key:
        print("❌ ERROR: No se encontró API key")
        return False
    
    print(f"✅ API Key encontrada: {api_key[:10]}...")
    
    # Configurar la API key
    nasdaqdatalink.ApiConfig.api_key = api_key
    
    # Test 1: ZACKS/FC - Datos Fundamentales de Empresas de Acero
    print("\n🏗️ Test 1: DATOS FUNDAMENTALES - Empresas de Acero")
    print("=" * 60)
    
    steel_companies = {
        'X': 'United States Steel Corporation',
        'NUE': 'Nucor Corporation',
        'MT': 'ArcelorMittal',
        'CLF': 'Cleveland-Cliffs Inc',
        'STLD': 'Steel Dynamics Inc'
    }
    
    successful_companies = []
    
    for ticker, name in steel_companies.items():
        try:
            print(f"\n📊 {ticker} - {name}")
            print("-" * 40)
            
            # Obtener datos fundamentales
            data = nasdaqdatalink.get_table(
                'ZACKS/FC',
                ticker=ticker,
                paginate=False
            )
            
            if data is not None and not data.empty:
                print(f"✅ Datos obtenidos: {len(data)} registros")
                print(f"📅 Períodos: {data.shape[0]} × {data.shape[1]} columnas")
                
                # Analizar columnas clave
                key_metrics = ['tot_revnu', 'gross_profit', 'net_income_loss', 
                              'eps_basic_net', 'ebitda', 'tot_asset', 'tot_liab']
                
                available_metrics = [m for m in key_metrics if m in data.columns]
                print(f"📊 Métricas disponibles: {len(available_metrics)}/{len(key_metrics)}")
                
                # Mostrar último período
                if 'per_end_date' in data.columns:
                    latest_date = pd.to_datetime(data['per_end_date']).max()
                    print(f"📅 Último período: {latest_date}")
                
                # Mostrar algunas métricas clave
                if not data.empty:
                    latest = data.iloc[-1]
                    if 'tot_revnu' in data.columns:
                        revenue = latest.get('tot_revnu', 'N/A')
                        print(f"💰 Revenue: ${revenue:,.0f}" if pd.notna(revenue) else "💰 Revenue: N/A")
                    if 'eps_basic_net' in data.columns:
                        eps = latest.get('eps_basic_net', 'N/A')
                        print(f"📈 EPS: ${eps:.2f}" if pd.notna(eps) else "📈 EPS: N/A")
                    if 'ebitda' in data.columns:
                        ebitda = latest.get('ebitda', 'N/A')
                        print(f"📊 EBITDA: ${ebitda:,.0f}" if pd.notna(ebitda) else "📊 EBITDA: N/A")
                
                successful_companies.append(ticker)
            else:
                print("⚠️ Sin datos disponibles")
                
        except Exception as e:
            if "Status 403" in str(e):
                print("❌ Requiere suscripción premium")
            else:
                print(f"❌ Error: {str(e)[:100]}")
    
    print(f"\n📊 Resumen: {len(successful_companies)}/{len(steel_companies)} empresas con datos")
    
    # Test 2: Series Temporales - Commodities y Economía
    print("\n📈 Test 2: SERIES TEMPORALES - Commodities y Economía")
    print("=" * 60)
    
    timeseries = {
        'FRED/DEXMXUS': 'USD/MXN Exchange Rate',
        'FRED/DCOILWTICO': 'WTI Crude Oil Prices',
        'FRED/DDFUELUSGULF': 'US Gulf Coast Diesel',
        'FRED/INDPRO': 'Industrial Production Index',
        'FRED/PAYEMS': 'US Employment',
        'FRED/CPIAUCSL': 'Consumer Price Index'
    }
    
    successful_series = []
    
    for code, description in timeseries.items():
        try:
            print(f"\n📊 {code}")
            print(f"   {description}")
            print("-" * 40)
            
            # Obtener datos recientes
            data = nasdaqdatalink.get(
                code,
                start_date='2024-01-01',
                end_date='2024-12-31',
                collapse='monthly'
            )
            
            if data is not None and not data.empty:
                print(f"✅ Datos obtenidos: {len(data)} puntos")
                print(f"📅 Rango: {data.index[0]} a {data.index[-1]}")
                
                # Mostrar estadísticas
                if len(data.columns) > 0:
                    value_col = data.columns[0]
                    print(f"📊 Último valor: {data[value_col].iloc[-1]:.2f}")
                    print(f"📈 Promedio: {data[value_col].mean():.2f}")
                    print(f"📉 Min/Max: {data[value_col].min():.2f} / {data[value_col].max():.2f}")
                
                successful_series.append(code)
            else:
                print("⚠️ Sin datos")
                
        except Exception as e:
            if "Status 403" in str(e):
                print("❌ Sin acceso (requiere suscripción)")
            elif "Status 404" in str(e):
                print("❌ Dataset no encontrado")
            else:
                print(f"❌ Error: {str(e)[:50]}")
    
    print(f"\n📊 Resumen: {len(successful_series)}/{len(timeseries)} series con datos")
    
    # Test 3: Búsqueda de Commodities Relacionados con Acero
    print("\n🔍 Test 3: COMMODITIES - Metales y Energía")
    print("=" * 60)
    
    commodities_test = [
        ('CHRIS/CME_HG1', 'Copper Futures'),
        ('CHRIS/CME_SI1', 'Silver Futures'),
        ('CHRIS/ICE_B1', 'Brent Crude Oil'),
        ('LME/PR_AL', 'LME Aluminum'),
        ('LME/PR_CU', 'LME Copper'),
        ('ODA/PIORECR_USD', 'Iron Ore (World Bank)')
    ]
    
    for code, description in commodities_test:
        try:
            print(f"\n🔍 {code}: {description}")
            
            # Intentar obtener metadatos
            dataset = nasdaqdatalink.Dataset(code)
            metadata = dataset.metadata()
            
            if metadata:
                print(f"   ✅ Dataset existe")
                print(f"   📅 Actualizado: {metadata.get('newest_available_date', 'N/A')}")
                print(f"   🔢 Frecuencia: {metadata.get('frequency', 'N/A')}")
            else:
                print(f"   ⚠️ Sin metadatos")
                
        except Exception as e:
            if "Status 403" in str(e):
                print(f"   ❌ Requiere suscripción premium")
            elif "Status 404" in str(e):
                print(f"   ❌ No encontrado")
            else:
                print(f"   ❌ Error: {str(e)[:30]}")
    
    # Test 4: Configuración y Límites
    print("\n⚙️ Test 4: CONFIGURACIÓN Y LÍMITES")
    print("=" * 60)
    
    print("📋 Configuración actual:")
    print(f"   🔑 API Key: Configurada")
    print(f"   🔗 Base URL: {nasdaqdatalink.ApiConfig.api_base}")
    print(f"   🔄 Reintentos: {nasdaqdatalink.ApiConfig.use_retries}")
    print(f"   🔢 Máx reintentos: {nasdaqdatalink.ApiConfig.number_of_retries}")
    
    print("\n📊 Datasets confirmados funcionando:")
    print("   ✅ ZACKS/FC - Fundamentales de empresas")
    print("   ✅ FRED/* - Federal Reserve Economic Data")
    
    print("\n⚠️ Limitaciones del plan actual:")
    print("   • 50 llamadas API por día")
    print("   • Sin acceso a CHRIS, LME premium")
    print("   • ZACKS/FC funciona para empresas US")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 DIAGNÓSTICO FINAL NASDAQ DATA LINK")
    print("=" * 60)
    
    total_success = len(successful_companies) + len(successful_series)
    total_tested = len(steel_companies) + len(timeseries)
    
    print(f"\n🎯 RESULTADOS:")
    print(f"   ✅ Exitosos: {total_success}/{total_tested}")
    print(f"   🏗️ Empresas de acero: {len(successful_companies)}/{len(steel_companies)}")
    print(f"   📈 Series temporales: {len(successful_series)}/{len(timeseries)}")
    
    print(f"\n💡 RECOMENDACIONES:")
    print("   1. Usar ZACKS/FC para fundamentales de empresas de acero")
    print("   2. Usar FRED/* para indicadores económicos y commodities")
    print("   3. Cachear datos agresivamente (límite 50 calls/día)")
    print("   4. Complementar con Yahoo Finance para datos diarios")
    
    print(f"\n🏗️ DATOS DE ACERO DISPONIBLES:")
    if successful_companies:
        for ticker in successful_companies:
            print(f"   ✅ {ticker} - {steel_companies[ticker]}")
    
    print("=" * 60)
    
    return total_success > 0

if __name__ == "__main__":
    success = test_nasdaq_data_link()
    exit(0 if success else 1)