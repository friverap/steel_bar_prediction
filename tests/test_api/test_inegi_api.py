#!/usr/bin/env python3
"""
Test directo de INEGI API usando INEGIpy
Basado en: https://github.com/andreslomeliv/DatosMex/tree/master/INEGIpy
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

def test_inegi_api():
    """Test de INEGI API usando INEGIpy"""
    
    print("=" * 60)
    print("🏛️ TEST INEGI API con INEGIpy")
    print("=" * 60)
    print("📚 Librería: INEGIpy")
    print("🔗 Repo: https://github.com/andreslomeliv/DatosMex")
    print("=" * 60)
    
    # Obtener API token
    api_token = os.getenv('INEGI_API_TOKEN') or os.getenv('INEGI_API_KEY')
    
    if not api_token:
        print("❌ ERROR: No se encontró INEGI_API_TOKEN en .env")
        print("   Obtener token en: https://www.inegi.org.mx/app/api/denue/v1/tokenVerify.aspx")
        return False
    
    print(f"✅ API Token encontrada: {api_token[:8]}...{api_token[-8:]}")
    
    # Test 1: Verificar instalación de INEGIpy
    print("\n📊 Test 1: Verificación de INEGIpy")
    print("-" * 40)
    
    try:
        from INEGIpy import Indicadores
        print("✅ INEGIpy importado correctamente")
        INEGIPY_AVAILABLE = True
    except ImportError:
        print("❌ INEGIpy no instalado")
        print("   Instalar con:")
        print("   pip install git+https://github.com/andreslomeliv/DatosMex.git#subdirectory=INEGIpy")
        INEGIPY_AVAILABLE = False
        return False
    
    if not INEGIPY_AVAILABLE:
        return False
    
    # Test 2: Inicializar cliente INEGIpy
    print("\n📊 Test 2: Inicialización del Cliente")
    print("-" * 40)
    
    try:
        inegi = Indicadores(api_token)
        print("✅ Cliente INEGIpy inicializado")
    except Exception as e:
        print(f"❌ Error inicializando cliente: {str(e)}")
        return False
    
    # Test 3: Obtener catálogo de indicadores BIE
    print("\n📊 Test 3: Catálogo de Indicadores BIE")
    print("-" * 40)
    
    try:
        print("🔍 Obteniendo catálogo BIE...")
        catalogo = inegi.catalogo_indicadores('BIE')
        
        if catalogo is not None and not catalogo.empty:
            print(f"✅ Catálogo obtenido: {len(catalogo)} indicadores")
            print(f"📊 Columnas: {list(catalogo.columns)[:5]}")
            
            # Buscar indicadores relevantes
            keywords = ['INPC', 'INPP', 'producción', 'industrial', 'construcción', 'metalúrgica']
            for keyword in keywords:
                mask = catalogo['Nombre'].str.contains(keyword, case=False, na=False)
                count = mask.sum()
                if count > 0:
                    print(f"   📌 Indicadores con '{keyword}': {count}")
        else:
            print("⚠️ Catálogo vacío o no disponible")
            
    except Exception as e:
        print(f"❌ Error obteniendo catálogo: {str(e)}")
    
    # Test 4: Obtener indicador específico - INPC
    print("\n📊 Test 4: Indicador INPC (Inflación)")
    print("-" * 40)
    
    try:
        print("🔍 Obteniendo INPC General (ID: 628194)...")
        
        # Obtener datos de INPC
        df_inpc = inegi.obtener_df(
            indicadores=['628194'],
            nombres=['INPC General'],
            inicio='2020',
            fin='2024'
        )
        
        if df_inpc is not None and not df_inpc.empty:
            print(f"✅ Datos obtenidos: {len(df_inpc)} filas")
            print(f"📅 Rango: {df_inpc.index[0]} a {df_inpc.index[-1]}")
            print(f"📊 Columnas: {list(df_inpc.columns)}")
            
            # Mostrar últimos valores
            print(f"\n📈 Últimos 5 valores:")
            print(df_inpc.tail())
            
            # Calcular inflación anualizada
            if 'INPC General' in df_inpc.columns:
                inflacion = df_inpc['INPC General'].pct_change(12) * 100
                print(f"\n💹 Inflación anualizada actual: {inflacion.iloc[-1]:.2f}%")
        else:
            print("⚠️ Sin datos de INPC")
            
    except Exception as e:
        print(f"❌ Error obteniendo INPC: {str(e)}")
    
    # Test 5: Obtener múltiples indicadores
    print("\n📊 Test 5: Múltiples Indicadores")
    print("-" * 40)
    
    indicadores_test = {
        '628194': 'INPC General',
        '628229': 'INPP Construcción',
        '444570': 'Producción Industrial - Construcción',
        '91634': 'Producción Industrial General',
        '383152': 'IGAE - Actividad Económica'
    }
    
    try:
        print(f"🔍 Obteniendo {len(indicadores_test)} indicadores...")
        
        ids = list(indicadores_test.keys())
        nombres = list(indicadores_test.values())
        
        df_multi = inegi.obtener_df(
            indicadores=ids,
            nombres=nombres,
            inicio='2023',
            fin='2024'
        )
        
        if df_multi is not None and not df_multi.empty:
            print(f"✅ DataFrame combinado: {df_multi.shape}")
            print(f"📊 Indicadores obtenidos:")
            
            for col in df_multi.columns:
                if col in nombres:
                    datos_validos = df_multi[col].notna().sum()
                    print(f"   ✅ {col}: {datos_validos} puntos")
        else:
            print("⚠️ Sin datos múltiples")
            
    except Exception as e:
        print(f"❌ Error obteniendo múltiples indicadores: {str(e)}")
    
    # Test 6: Indicadores del sector metalúrgico
    print("\n📊 Test 6: Indicadores Sector Metalúrgico")
    print("-" * 40)
    
    metal_indicators = {
        '444612': 'Índice de producción - Industrias metálicas básicas',
        '383152': 'Producción de hierro y acero',
        '383153': 'Producción de productos de hierro y acero'
    }
    
    for ind_id, desc in metal_indicators.items():
        try:
            print(f"\n🏗️ {desc} (ID: {ind_id})")
            
            df_metal = inegi.obtener_df(
                indicadores=[ind_id],
                nombres=[desc],
                inicio='2020',
                fin='2024'
            )
            
            if df_metal is not None and not df_metal.empty:
                print(f"   ✅ Disponible: {len(df_metal)} puntos")
                if desc in df_metal.columns:
                    ultimo_valor = df_metal[desc].iloc[-1]
                    print(f"   📊 Último valor: {ultimo_valor:.2f}")
            else:
                print(f"   ⚠️ Sin datos")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
    
    # Test 7: Verificar estructura de respuesta
    print("\n📊 Test 7: Estructura de Datos INEGIpy")
    print("-" * 40)
    
    try:
        # Obtener un indicador simple para analizar estructura
        df_test = inegi.obtener_df(
            indicadores=['628194'],
            nombres=['Test'],
            inicio='2024',
            fin='2024'
        )
        
        if df_test is not None:
            print(f"✅ Tipo de retorno: {type(df_test)}")
            print(f"📊 Shape: {df_test.shape}")
            print(f"📅 Índice: {type(df_test.index).__name__}")
            print(f"📊 Columnas: {df_test.columns.tolist()}")
            
            if not df_test.empty:
                print(f"\n📋 Info del DataFrame:")
                print(f"   - Fecha inicial: {df_test.index[0]}")
                print(f"   - Fecha final: {df_test.index[-1]}")
                print(f"   - Frecuencia detectada: {pd.infer_freq(df_test.index) or 'Irregular'}")
                print(f"   - Valores nulos: {df_test.isna().sum().sum()}")
        else:
            print("⚠️ DataFrame None")
            
    except Exception as e:
        print(f"❌ Error analizando estructura: {str(e)}")
    
    # Test 8: Métodos adicionales de INEGIpy
    print("\n📊 Test 8: Métodos Adicionales")
    print("-" * 40)
    
    try:
        # Verificar métodos disponibles
        metodos = [m for m in dir(inegi) if not m.startswith('_')]
        print(f"📋 Métodos disponibles en INEGIpy:")
        for metodo in metodos[:10]:
            print(f"   - {metodo}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("📋 DIAGNÓSTICO INEGI con INEGIpy:")
    print("-" * 40)
    print("✅ Ventajas de INEGIpy:")
    print("  • Manejo automático de autenticación")
    print("  • Conversión directa a DataFrame")
    print("  • Soporte para múltiples indicadores")
    print("  • Manejo de fechas automático")
    print("  • Caché de respuestas")
    print("")
    print("📌 Indicadores clave para acero:")
    print("  • INPC/INPP - Inflación y precios")
    print("  • Producción Industrial")
    print("  • Producción Metalúrgica")
    print("  • IGAE - Actividad económica")
    print("  • PIB Trimestral")
    print("")
    print("🔧 Instalación:")
    print("  pip install git+https://github.com/andreslomeliv/DatosMex.git#subdirectory=INEGIpy")
    print("=" * 60)

if __name__ == "__main__":
    test_inegi_api()