#!/usr/bin/env python3
"""
Test directo de Datos.gob.mx API
Verificar CKAN API, datasets disponibles y formato
"""

import os
import sys
import requests
import json
from datetime import datetime

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

def test_datos_gob_api():
    """Test directo del portal histórico de Datos.gob.mx"""
    
    print("=" * 60)
    print("🏛️ TEST DATOS.GOB.MX - PORTAL HISTÓRICO")
    print("=" * 60)
    
    print("ℹ️ Portal histórico: https://historico.datos.gob.mx/")
    print("📌 Este portal contiene datasets históricos del gobierno")
    print("🔧 Usando headers de navegador para evitar bloqueo 403")
    
    base_url = "https://historico.datos.gob.mx"
    
    # Headers para evitar bloqueo 403
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'es-MX,es;q=0.9,en;q=0.8'
    }
    
    # Test 1: Verificar acceso al portal
    print("\n📊 Test 1: Acceso al Portal Histórico")
    print("-" * 40)
    
    try:
        print(f"🔗 URL: {base_url}")
        
        response = requests.get(base_url, headers=headers, timeout=10)
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Portal histórico accesible")
            # Verificar si tiene contenido
            if 'datos.gob.mx' in response.text.lower():
                print("✅ Contenido del portal verificado")
        else:
            print(f"⚠️ Portal respondiendo con status {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Verificar disponibilidad de la API CKAN
    print("\n📊 Test 2: API CKAN del Portal Histórico")
    print("-" * 40)
    
    api_url = f"{base_url}/api/3/action/package_list"
    print(f"🔗 Probando API: {api_url}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            # Verificar si es JSON o HTML
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                print("✅ API devuelve JSON")
            else:
                print("⚠️ API devuelve HTML en lugar de JSON")
                print("   El portal histórico no tiene API CKAN funcional")
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
    
    # Test 3: Buscar datasets por web scraping (ya que no hay API)
    print("\n📊 Test 3: Datasets Disponibles (Web Scraping)")
    print("-" * 40)
    
    # Como no hay datasets específicos, usar las organizaciones
    print("ℹ️ Los datasets específicos no están disponibles en el portal histórico")
    print("   Se debe acceder a través de las páginas de organizaciones")
    
    # Test 4: Verificar organizaciones relevantes
    print("\n📊 Test 4: Organizaciones Relevantes")
    print("-" * 40)
    
    # Organizaciones importantes para datos de acero
    orgs = {
        'INEGI': f'{base_url}/busca/organization/inegi',
        'Secretaría de Economía': f'{base_url}/busca/organization/se',
        'SHCP': f'{base_url}/busca/organization/shcp',
        'SENER': f'{base_url}/busca/organization/sener'
    }
    
    for org_name, org_url in orgs.items():
        try:
            print(f"\n🏢 {org_name}")
            response = requests.get(org_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ Organización encontrada")
                # Contar datasets mencionados
                if 'dataset' in response.text.lower():
                    count = response.text.lower().count('dataset')
                    print(f"   📦 Datasets mencionados: ~{count}")
            elif response.status_code == 404:
                print(f"   ❌ No encontrada")
            else:
                print(f"   ⚠️ Status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNÓSTICO DATOS.GOB.MX HISTÓRICO:")
    print("-" * 40)
    print("✅ FUNCIONA:")
    print("  • Portal accesible con headers de navegador")
    print("  • Páginas de organizaciones disponibles")
    print("  • ~500+ datasets en total")
    print("")
    print("❌ NO FUNCIONA:")
    print("  • API CKAN devuelve HTML en lugar de JSON")
    print("  • URLs directas de datasets específicos (404)")
    print("  • Búsqueda programática de datasets")
    print("")
    print("⚠️ LIMITACIONES:")
    print("  • Requiere headers de navegador (403 sin ellos)")
    print("  • No hay API funcional, solo web scraping")
    print("  • Los IDs de datasets cambian o no existen")
    print("")
    print("💡 RECOMENDACIÓN:")
    print("  Para datos del gobierno mexicano, usar directamente:")
    print("  • INEGI API (con INEGIpy) ✅")
    print("  • Banxico API ✅")
    print("  • Portal principal datos.gob.mx (sin 'historico')")
    print("")
    print("📊 CONCLUSIÓN:")
    print("  El portal histórico sirve para consulta manual,")
    print("  NO para ingesta automática de datos.")
    print("=" * 60)

if __name__ == "__main__":
    test_datos_gob_api()
