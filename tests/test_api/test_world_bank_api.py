#!/usr/bin/env python3
"""
Test de World Bank API usando la librería oficial wbgapi
Basado en: https://pypi.org/project/wbgapi/
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    import wbgapi as wb
    WBGAPI_AVAILABLE = True
except ImportError:
    WBGAPI_AVAILABLE = False
    print("⚠️ wbgapi no instalado. Instalar con: pip install wbgapi")

def test_world_bank_api():
    """Test de World Bank API usando wbgapi"""
    
    print("=" * 60)
    print("🌍 TEST WORLD BANK API con wbgapi")
    print("=" * 60)
    print("📚 Usando librería oficial: wbgapi")
    print("🔗 Docs: https://pypi.org/project/wbgapi/")
    print("=" * 60)
    
    if not WBGAPI_AVAILABLE:
        print("❌ wbgapi no disponible. Saltando tests.")
        return
    
    # Test 1: Información básica y economías
    print("\n📊 Test 1: Información de México")
    print("-" * 40)
    
    try:
        # Obtener información de México
        for economy in wb.economy.list('MEX'):
            print(f"✅ País: {economy['value']}")
            print(f"   Código ISO: {economy['id']}")
            print(f"   Región: {economy.get('region', 'N/A')}")
            print(f"   Nivel de ingreso: {economy.get('incomeLevel', 'N/A')}")
            print(f"   Capital: {economy.get('capitalCity', 'N/A')}")
            print(f"   Latitud: {economy.get('latitude', 'N/A')}")
            print(f"   Longitud: {economy.get('longitude', 'N/A')}")
            break
    except Exception as e:
        print(f"❌ Error obteniendo info de México: {str(e)[:100]}")
    
    # Test 2: Indicadores económicos para México
    print("\n📊 Test 2: Indicadores Económicos de México")
    print("-" * 40)
    
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (%)',
        'NV.IND.TOTL.ZS': 'Industry value added (% of GDP)',
        'NV.IND.MANF.ZS': 'Manufacturing value added (% of GDP)',
        'PA.NUS.FCRF': 'Official exchange rate (LCU per US$)'
    }
    
    for indicator_code, name in indicators.items():
        try:
            print(f"\n🔍 {name}")
            print(f"   📌 Código: {indicator_code}")
            
            # Usar wbgapi para obtener datos
            data = list(wb.data.fetch(
                indicator_code,
                'MEX',
                time=range(2020, 2024)
            ))
            
            if data:
                print(f"   ✅ Datos obtenidos: {len(data)} puntos")
                # Mostrar últimos valores
                for record in data[-3:]:
                    if record.get('value') is not None:
                        print(f"      {record['time']}: {record['value']:.2f}")
            else:
                print(f"   ⚠️ Sin datos disponibles")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
    
    # Test 3: Limitaciones de Commodities
    print("\n📊 Test 3: Commodities en World Bank")
    print("-" * 40)
    
    print("⚠️ IMPORTANTE: Los datos de commodities del World Bank:")
    print("   • NO están disponibles en la API pública")
    print("   • Requieren suscripción al Pink Sheet/GEM")
    print("   • Database 15 (GEM) no es accesible públicamente")
    
    print("\n✅ ALTERNATIVAS RECOMENDADAS para commodities:")
    print("   1. Yahoo Finance (ya configurado):")
    print("      • HG=F: Copper Futures (diario)")
    print("      • CL=F: Crude Oil Futures (diario)")
    print("      • GC=F: Gold Futures (diario)")
    print("   2. FRED API (Federal Reserve):")
    print("      • DCOILWTICO: WTI Crude Oil")
    print("      • DHHNGSP: Natural Gas")
    print("   3. LME (London Metal Exchange) - via web scraping")
    
    # Verificar qué bases de datos están disponibles
    print("\n📊 Bases de datos disponibles:")
    count = 0
    for source in wb.source.list():
        if 'commodity' in source.get('name', '').lower() or 'price' in source.get('name', '').lower():
            print(f"   • DB {source['id']}: {source['name'][:50]}")
            count += 1
        if count >= 5:
            break
    
    if count == 0:
        print("   ❌ No hay bases de datos de commodities públicas")
    
    # Test 4: Búsqueda de indicadores
    print("\n📊 Test 4: Búsqueda de Indicadores")
    print("-" * 40)
    
    search_terms = ['steel', 'iron', 'metal', 'construction']
    
    for term in search_terms:
        try:
            print(f"\n🔍 Buscando: '{term}'")
            # Buscar indicadores
            count = 0
            found = False
            for series in wb.series.list(q=term):
                if count == 0:
                    print(f"   ✅ Indicadores encontrados:")
                    found = True
                if count < 3:
                    print(f"      {series['id']}: {series['value'][:60]}")
                    count += 1
                else:
                    break
            
            if not found:
                print(f"   ⚠️ Sin resultados")
                
        except Exception as e:
            print(f"   ❌ Error en búsqueda: {str(e)[:50]}")
    
    # Test 5: Bases de datos disponibles
    print("\n📊 Test 5: Bases de Datos Disponibles")
    print("-" * 40)
    
    try:
        count = 0
        relevant_dbs = ['Commodity', 'GEM', 'Pink Sheet', 'Global Economic']
        for source in wb.source.list():
            count += 1
            name = source.get('value', '')
            if any(term.lower() in name.lower() for term in relevant_dbs):
                print(f"   📌 DB {source['id']}: {name[:60]}")
        
        print(f"✅ Total de bases de datos: {count}")
                
    except Exception as e:
        print(f"❌ Error listando bases de datos: {str(e)}")
    
    # Test 6: Frecuencia de datos
    print("\n📊 Test 6: Frecuencia de Datos")
    print("-" * 40)
    
    try:
        # Probar obtener datos mensuales
        print("🔍 Probando datos mensuales de inflación...")
        
        # Usar MRV (Most Recent Value) para obtener el último valor
        data = list(wb.data.fetch(
            'FP.CPI.TOTL.ZG',
            'MEX',
            mrv=12  # Últimos 12 valores disponibles
        ))
        
        if data:
            print(f"✅ Datos obtenidos: {len(data)} puntos")
            for record in data[:3]:
                if record.get('value'):
                    print(f"   {record['time']}: {record['value']:.2f}%")
        else:
            print("⚠️ Sin datos mensuales disponibles")
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNÓSTICO WORLD BANK con wbgapi:")
    print("-" * 40)
    print("✅ VENTAJAS:")
    print("  • API oficial de Python del Banco Mundial")
    print("  • Fácil acceso a indicadores económicos")
    print("  • Búsqueda integrada de series")
    print("  • Soporte para pandas DataFrames")
    print("")
    print("⚠️ LIMITACIONES:")
    print("  • Datos principalmente anuales")
    print("  • Commodities en base de datos separada (GEM)")
    print("  • Algunos indicadores tienen retraso de 1-2 años")
    print("")
    print("💡 RECOMENDACIONES:")
    print("  • Usar para datos macroeconómicos anuales")
    print("  • Para commodities diarios, usar otras fuentes")
    print("  • Combinar con interpolación para series diarias")
    print("=" * 60)

if __name__ == "__main__":
    test_world_bank_api()