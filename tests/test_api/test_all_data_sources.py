#!/usr/bin/env python3
"""
Test comprehensivo para todas las fuentes de datos
Verifica el funcionamiento de todas las APIs con las keys actualizadas
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

from src.data_ingestion.master_collector import collect_all_steel_data
from datetime import datetime
import json


class TestAllDataSources:
    """Test class para todas las fuentes de datos"""
    
    @pytest.mark.asyncio
    async def test_all_apis_comprehensive(self):
        """Test comprehensivo de todas las APIs con keys actualizadas"""
        
        print("🧪 TEST COMPREHENSIVO DE TODAS LAS APIs")
        print("=" * 50)
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verificar que las API keys estén configuradas
        api_keys_status = {
            'BANXICO_API_TOKEN': os.getenv('BANXICO_API_TOKEN'),
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'TRADING_ECONOMICS_API_KEY': os.getenv('TRADING_ECONOMICS_API_KEY'),
            'QUANDL_API_KEY': os.getenv('QUANDL_API_KEY'),
            'INEGI_API_TOKEN': os.getenv('INEGI_API_TOKEN')
        }
        
        print("\\n🔑 VERIFICACIÓN DE API KEYS:")
        print("-" * 30)
        
        configured_keys = 0
        for key_name, key_value in api_keys_status.items():
            if key_value:
                print(f"✅ {key_name}: {key_value[:10]}...{key_value[-5:]}")
                configured_keys += 1
            else:
                print(f"❌ {key_name}: NO CONFIGURADA")
        
        print(f"\\n📊 API Keys configuradas: {configured_keys}/6")
        
        # Test de recopilación completa
        print("\\n🔄 RECOPILACIÓN COMPLETA DE DATOS:")
        print("-" * 35)
        
        try:
            collection_result = await collect_all_steel_data(
                start_date="2020-01-01",
                end_date="2025-09-25",
                sources=None,  # Todas las fuentes
                save_raw=False,  # No guardar para test
                verify_quality=True
            )
            
            assert collection_result is not None, "Collection result should not be None"
            assert 'data' in collection_result, "Collection result should have 'data' key"
            
            print("✅ Recopilación completada exitosamente")
            
            # Analizar resultados por fuente
            summary = collection_result['collection_summary']
            
            print(f"\\n📊 RESULTADOS GENERALES:")
            print(f"   ⏱️ Tiempo total: {summary['total_collection_time_seconds']:.1f}s")
            print(f"   ✅ Fuentes exitosas: {summary['sources_successful']}/{summary['sources_requested']}")
            print(f"   📊 Total series: {summary['totals']['total_series']}")
            print(f"   📈 Total puntos: {summary['totals']['total_data_points']:,}")
            print(f"   🔗 APIs reales: {summary['totals']['total_api_sources']}")
            
            # Verificar cada fuente
            print("\\n🔍 ANÁLISIS POR FUENTE:")
            print("-" * 25)
            
            critical_sources = ['banxico', 'fred', 'lme']
            critical_working = 0
            
            for source_name, stats in summary['by_source'].items():
                status_icon = "✅" if stats['success'] else "❌"
                priority_icon = "⭐" if source_name in critical_sources else "📊"
                
                print(f"{status_icon} {priority_icon} {source_name.upper()}:")
                
                if stats['success']:
                    print(f"   📊 Series: {stats.get('total_series', 0)}")
                    print(f"   📈 Puntos: {stats.get('total_data_points', 0):,}")
                    print(f"   🔗 APIs reales: {stats.get('api_sources', 0)}")
                    print(f"   ⏱️ Tiempo: {stats.get('collection_time_seconds', 0):.1f}s")
                    
                    if source_name in critical_sources:
                        critical_working += 1
                else:
                    print(f"   ❌ Error: {stats.get('error', 'Unknown')}")
            
            # Verificar calidad
            if 'quality_report' in collection_result and collection_result['quality_report']:
                quality = collection_result['quality_report']['overall_quality']
                print(f"\\n📊 CALIDAD GENERAL:")
                print(f"   Score: {quality['overall_quality_score']:.1f}%")
                print(f"   Estado: {quality['status'].upper()}")
                print(f"   Completitud: {quality['overall_completeness_pct']:.1f}%")
                print(f"   Cobertura API: {quality['overall_api_coverage_pct']:.1f}%")
            
            # Assertions para el test
            assert summary['sources_successful'] >= 8, f"Should have at least 8 working sources, got {summary['sources_successful']}"
            assert critical_working >= 3, f"Should have at least 3 critical sources working, got {critical_working}"
            assert summary['totals']['total_api_sources'] >= 20, f"Should have at least 20 real API sources, got {summary['totals']['total_api_sources']}"
            
            print("\\n🎉 TODAS LAS ASSERTIONS PASARON")
            
            return collection_result
            
        except Exception as e:
            print(f"❌ Error en recopilación: {str(e)}")
            pytest.fail(f"Collection failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_quandl_with_new_key(self):
        """Test específico para Quandl con la nueva API key"""
        
        print("\\n🔍 TEST ESPECÍFICO QUANDL - NUEVA API KEY")
        print("=" * 45)
        
        api_key = os.getenv('QUANDL_API_KEY')
        print(f"🔑 Nueva API Key: {api_key}")
        
        assert api_key is not None, "Quandl API key should be configured"
        assert api_key == "fNm4yDms8hGqyZTbLCpM", f"Expected new API key, got {api_key}"
        
        # Test directo con la nueva key
        import aiohttp
        
        try:
            # Probar endpoint básico
            url = "https://data.nasdaq.com/api/v3/datasets.json"
            params = {'api_key': api_key, 'per_page': 1}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    print(f"📡 Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print("✅ Nueva API key de Quandl FUNCIONA!")
                        
                        if 'datasets' in data:
                            print(f"📊 Acceso a datasets: {len(data['datasets'])}")
                        
                        return True
                        
                    elif response.status == 403:
                        error_text = await response.text()
                        print(f"❌ Nueva API key sin permisos: {error_text[:100]}...")
                        return False
                    else:
                        error_text = await response.text()
                        print(f"❌ Error {response.status}: {error_text[:100]}...")
                        return False
                        
        except Exception as e:
            print(f"❌ Error probando nueva key: {str(e)}")
            return False
    
    @pytest.mark.asyncio
    async def test_critical_sources_only(self):
        """Test enfocado solo en fuentes críticas"""
        
        print("\\n🎯 TEST FUENTES CRÍTICAS")
        print("=" * 30)
        
        critical_sources = ['banxico', 'fred', 'lme', 'trading_economics']
        
        try:
            collection_result = await collect_all_steel_data(
                start_date="2024-01-01",  # Período más corto para test rápido
                end_date="2025-09-25",
                sources=critical_sources,
                save_raw=False,
                verify_quality=False
            )
            
            summary = collection_result['collection_summary']
            
            print(f"✅ Fuentes críticas: {summary['sources_successful']}/{len(critical_sources)}")
            
            # Verificar que al menos 4 de 5 fuentes críticas funcionen
            assert summary['sources_successful'] >= 4, f"Should have at least 4/5 critical sources working"
            
            # Verificar que tengamos suficientes APIs reales
            assert summary['totals']['total_api_sources'] >= 15, f"Should have at least 15 real APIs in critical sources"
            
            print("🎉 FUENTES CRÍTICAS FUNCIONANDO CORRECTAMENTE")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en fuentes críticas: {str(e)}")
            pytest.fail(f"Critical sources test failed: {str(e)}")


# Función para ejecutar tests manualmente
async def run_manual_tests():
    """Ejecutar tests manualmente sin pytest"""
    
    print("🚀 EJECUCIÓN MANUAL DE TESTS")
    print("=" * 35)
    
    test_instance = TestAllDataSources()
    
    try:
        # Test 1: Comprehensivo
        print("\\n🧪 TEST 1: COMPREHENSIVO")
        await test_instance.test_all_apis_comprehensive()
        
        # Test 2: Quandl específico
        print("\\n🧪 TEST 2: QUANDL NUEVA KEY")
        quandl_success = await test_instance.test_quandl_with_new_key()
        
        # Test 3: Fuentes críticas
        print("\\n🧪 TEST 3: FUENTES CRÍTICAS")
        await test_instance.test_critical_sources_only()
        
        print("\\n🎉 TODOS LOS TESTS COMPLETADOS")
        
        if quandl_success:
            print("✅ QUANDL: Nueva API key funcionando")
        else:
            print("❌ QUANDL: Problemas con nueva API key")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en tests: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_manual_tests())
    exit(0 if success else 1)
