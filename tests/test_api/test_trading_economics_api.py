#!/usr/bin/env python3
"""
Test directo de Trading Economics API
Usando la librería oficial: https://github.com/tradingeconomics/tradingeconomics-python
"""

import os
import sys
from datetime import datetime

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

try:
    import tradingeconomics as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("⚠️ tradingeconomics no instalado. Instalar con: pip install tradingeconomics")

def test_trading_economics_api():
    """Test directo de Trading Economics API con librería oficial"""
    
    print("=" * 60)
    print("📈 TEST TRADING ECONOMICS API - LIBRERÍA OFICIAL")
    print("=" * 60)
    print("📚 Basado en: https://github.com/tradingeconomics/tradingeconomics-python")
    print("=" * 60)
    
    if not TE_AVAILABLE:
        print("❌ Librería tradingeconomics no disponible")
        return False
    
    api_key = os.getenv('TRADING_ECONOMICS_API_KEY')
    
    if not api_key:
        print("❌ ERROR: No se encontró TRADING_ECONOMICS_API_KEY en .env")
        return False
    
    print(f"✅ API Key encontrada: {api_key[:15]}...")
    
    # Inicializar librería
    try:
        te.login(api_key)
        print("✅ Login exitoso en Trading Economics")
    except Exception as e:
        print(f"❌ Error en login: {e}")
        return False
    
    # Test 1: Indicadores de México
    print("\n📊 Test 1: Indicadores Económicos de México")
    print("-" * 40)
    
    try:
        # Obtener lista de indicadores para México
        indicators = te.getIndicatorData(country='mexico', output_type='df')
        
        if indicators is not None and not indicators.empty:
            print(f"✅ Indicadores disponibles: {len(indicators)} indicadores")
            print("\n📈 Primeros 5 indicadores:")
            for i, row in indicators.head().iterrows():
                if 'Category' in indicators.columns and 'LatestValue' in indicators.columns:
                    print(f"   • {row.get('Category', 'N/A')}: {row.get('LatestValue', 'N/A')}")
        else:
            print("⚠️ Sin datos de indicadores")
            
    except Exception as e:
        print(f"❌ Error obteniendo indicadores: {str(e)[:100]}")
    
    # Test 2: Datos históricos
    print("\n📊 Test 2: Datos Históricos - Inflación México")
    print("-" * 40)
    
    try:
        # Obtener datos históricos de inflación
        historical = te.getHistoricalData(
            country='mexico',
            indicator='inflation rate',
            initDate='2020-01-01',
            endDate=datetime.now().strftime('%Y-%m-%d'),
            output_type='df'
        )
        
        if historical is not None and not historical.empty:
            print(f"✅ Datos históricos obtenidos: {len(historical)} puntos")
            print(f"   📅 Rango: {historical.index.min()} a {historical.index.max()}")
            print("\n   Últimos 3 valores:")
            for date, row in historical.tail(3).iterrows():
                print(f"      {date}: {row.get('Value', 'N/A')}")
        else:
            print("⚠️ Sin datos históricos")
            
    except Exception as e:
        print(f"❌ Error obteniendo históricos: {str(e)[:100]}")
    
    # Test 3: Mercados - Índices
    print("\n📊 Test 3: Mercados - Índice Bursátil México")
    print("-" * 40)
    
    try:
        # Obtener datos de mercados
        markets = te.getMarketsData(marketsField='index', output_type='df')
        
        if markets is not None and not markets.empty:
            # Filtrar por México
            mexico_markets = markets[markets['Country'].str.contains('Mexico', case=False, na=False)] if 'Country' in markets.columns else pd.DataFrame()
            
            if not mexico_markets.empty:
                print(f"✅ Índices mexicanos encontrados: {len(mexico_markets)}")
                for _, row in mexico_markets.iterrows():
                    print(f"   • {row.get('Symbol', 'N/A')}: {row.get('Last', 'N/A')}")
            else:
                print("⚠️ No se encontraron índices mexicanos")
                print(f"   Total de índices disponibles: {len(markets)}")
        else:
            print("⚠️ Sin datos de mercados")
            
    except Exception as e:
        print(f"❌ Error obteniendo mercados: {str(e)[:100]}")
    
    # Test 4: Calendario económico
    print("\n📊 Test 4: Calendario Económico")
    print("-" * 40)
    
    try:
        # Obtener calendario económico
        calendar = te.getCalendarData(
            country='mexico',
            initDate=datetime.now().strftime('%Y-%m-%d'),
            endDate=datetime.now().strftime('%Y-%m-%d'),
            output_type='df'
        )
        
        if calendar is not None and not calendar.empty:
            print(f"✅ Eventos del calendario: {len(calendar)} eventos")
            for _, event in calendar.head(3).iterrows():
                print(f"   • {event.get('Event', 'N/A')}: {event.get('Date', 'N/A')}")
        else:
            print("⚠️ Sin eventos en el calendario para hoy")
            
    except Exception as e:
        print(f"❌ Error obteniendo calendario: {str(e)[:100]}")
    
    # Test 5: Commodities
    print("\n📊 Test 5: Commodities")
    print("-" * 40)
    
    try:
        # Obtener datos de commodities
        commodities = te.getMarketsData(marketsField='commodities', output_type='df')
        
        if commodities is not None and not commodities.empty:
            print(f"✅ Commodities disponibles: {len(commodities)}")
            
            # Buscar metales relevantes
            metals = ['Copper', 'Steel', 'Iron', 'Aluminum']
            print("\n🏗️ Metales relevantes para acero:")
            for metal in metals:
                metal_data = commodities[commodities['Name'].str.contains(metal, case=False, na=False)] if 'Name' in commodities.columns else pd.DataFrame()
                if not metal_data.empty:
                    for _, row in metal_data.iterrows():
                        print(f"   • {row.get('Symbol', 'N/A')}: {row.get('Last', 'N/A')} {row.get('Unit', '')}")
        else:
            print("⚠️ Sin datos de commodities")
            
    except Exception as e:
        print(f"❌ Error obteniendo commodities: {str(e)[:100]}")
    
    # Test 6: Noticias
    print("\n📊 Test 6: Noticias Económicas")
    print("-" * 40)
    
    try:
        # Obtener noticias
        news = te.getNews(country='mexico', output_type='df')
        
        if news is not None and not news.empty:
            print(f"✅ Noticias disponibles: {len(news)} artículos")
            for _, article in news.head(2).iterrows():
                print(f"   • {article.get('title', 'N/A')[:60]}...")
                print(f"     {article.get('date', 'N/A')}")
        else:
            print("⚠️ Sin noticias disponibles")
            
    except Exception as e:
        print(f"❌ Error obteniendo noticias: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNÓSTICO TRADING ECONOMICS:")
    print("-" * 40)
    print("✅ VENTAJAS de la librería oficial:")
    print("  • Acceso simplificado a todos los endpoints")
    print("  • DataFrames de pandas integrados")
    print("  • Manejo automático de autenticación")
    print("  • Soporte para calendario y noticias")
    print("")
    print("⚠️ LIMITACIONES del plan gratuito:")
    print("  • Países limitados (México incluido ✅)")
    print("  • Límite de requests por día")
    print("  • Algunos commodities pueden no estar disponibles")
    print("")
    print("💡 RECOMENDACIONES:")
    print("  • Usar para datos económicos de México")
    print("  • Complementar commodities con Yahoo Finance")
    print("  • Cachear datos para evitar límites")
    print("=" * 60)

if __name__ == "__main__":
    import pandas as pd  # Importar aquí para evitar error si no se usa
    test_trading_economics_api()