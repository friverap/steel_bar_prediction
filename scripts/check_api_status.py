#!/usr/bin/env python3
"""
Script para verificar el estado de todas las APIs
Útil para diagnóstico y monitoreo
"""

import asyncio
import sys
import os
from datetime import datetime

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

import aiohttp


async def check_api_status():
    """Verificar estado de todas las APIs"""
    
    print("🔍 VERIFICACIÓN DE ESTADO DE APIs")
    print("=" * 40)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # APIs a verificar
    apis_to_check = {
        'BANXICO': {
            'url': 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43718/datos/oportuno',
            'headers': {'Bmx-Token': os.getenv('BANXICO_API_TOKEN')},
            'key_required': True
        },
        'FRED': {
            'url': 'https://api.stlouisfed.org/fred/series/observations',
            'params': {
                'series_id': 'DHHNGSP',
                'api_key': os.getenv('FRED_API_KEY'),
                'file_type': 'json',
                'limit': 1
            },
            'key_required': True
        },
        'Alpha_Vantage': {
            'url': 'https://www.alphavantage.co/query',
            'params': {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'AAPL',
                'apikey': os.getenv('ALPHA_VANTAGE_API_KEY')
            },
            'key_required': True
        },
        'Yahoo_Finance': {
            'url': 'https://finance.yahoo.com',
            'key_required': False
        },
        'World_Bank': {
            'url': 'https://api.worldbank.org/v2/country/all/indicator/PIORECR',
            'params': {'format': 'json', 'per_page': 1},
            'key_required': False
        }
    }
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for api_name, api_config in apis_to_check.items():
            print(f"\\n🔍 Verificando {api_name}...")
            
            try:
                # Verificar API key si es requerida
                if api_config['key_required']:
                    if api_name == 'BANXICO':
                        key = os.getenv('BANXICO_API_TOKEN')
                    elif api_name == 'FRED':
                        key = os.getenv('FRED_API_KEY')
                    elif api_name == 'Alpha_Vantage':
                        key = os.getenv('ALPHA_VANTAGE_API_KEY')
                    
                    if not key:
                        results[api_name] = {
                            'status': 'no_key',
                            'message': 'API key no configurada'
                        }
                        print(f"   ❌ API key no configurada")
                        continue
                    else:
                        print(f"   🔑 API key: {key[:10]}...{key[-5:]}")
                
                # Hacer request de prueba
                url = api_config['url']
                params = api_config.get('params', {})
                headers = api_config.get('headers', {})
                
                async with session.get(url, params=params, headers=headers, timeout=10) as response:
                    print(f"   📡 Status: {response.status}")
                    
                    if response.status == 200:
                        # Verificar contenido
                        try:
                            if 'json' in response.headers.get('content-type', ''):
                                data = await response.json()
                                print(f"   ✅ JSON válido recibido")
                                
                                # Verificar contenido específico por API
                                if api_name == 'BANXICO' and 'bmx' in data:
                                    print(f"   ✅ Datos BANXICO válidos")
                                elif api_name == 'FRED' and 'observations' in data:
                                    print(f"   ✅ Datos FRED válidos")
                                elif api_name == 'Alpha_Vantage':
                                    if 'Information' in data and 'rate limit' in data['Information']:
                                        print(f"   ⚠️ Rate limit alcanzado")
                                        results[api_name] = {
                                            'status': 'rate_limited',
                                            'message': 'Rate limit alcanzado'
                                        }
                                    elif 'Time Series (Daily)' in data:
                                        print(f"   ✅ Datos Alpha Vantage válidos")
                                        results[api_name] = {
                                            'status': 'working',
                                            'message': 'API funcionando correctamente'
                                        }
                                    else:
                                        print(f"   ⚠️ Respuesta inesperada: {list(data.keys())}")
                                elif api_name == 'World_Bank' and isinstance(data, list):
                                    print(f"   ✅ Datos World Bank válidos")
                                
                                if api_name not in results:
                                    results[api_name] = {
                                        'status': 'working',
                                        'message': 'API funcionando correctamente'
                                    }
                            else:
                                print(f"   ✅ Respuesta válida (no JSON)")
                                results[api_name] = {
                                    'status': 'working',
                                    'message': 'API responde correctamente'
                                }
                                
                        except Exception as e:
                            print(f"   ⚠️ Error procesando respuesta: {str(e)}")
                            results[api_name] = {
                                'status': 'response_error',
                                'message': f'Error procesando respuesta: {str(e)}'
                            }
                    
                    elif response.status == 403:
                        print(f"   ❌ Forbidden - API key inválida o sin permisos")
                        results[api_name] = {
                            'status': 'forbidden',
                            'message': 'API key inválida o sin permisos'
                        }
                    
                    elif response.status == 404:
                        print(f"   ❌ Not Found - Endpoint no existe")
                        results[api_name] = {
                            'status': 'not_found',
                            'message': 'Endpoint no encontrado'
                        }
                    
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error {response.status}: {error_text[:100]}...")
                        results[api_name] = {
                            'status': 'error',
                            'message': f'HTTP {response.status}'
                        }
                
            except asyncio.TimeoutError:
                print(f"   ❌ Timeout - API no responde")
                results[api_name] = {
                    'status': 'timeout',
                    'message': 'API no responde (timeout)'
                }
            
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[api_name] = {
                    'status': 'error',
                    'message': str(e)
                }
    
    # Resumen final
    print(f"\\n📊 RESUMEN DE ESTADO DE APIs:")
    print("=" * 35)
    
    working_apis = [api for api, result in results.items() if result['status'] == 'working']
    rate_limited_apis = [api for api, result in results.items() if result['status'] == 'rate_limited']
    error_apis = [api for api, result in results.items() if result['status'] not in ['working', 'rate_limited']]
    
    print(f"✅ APIs funcionando: {len(working_apis)} - {working_apis}")
    print(f"⚠️ APIs con rate limit: {len(rate_limited_apis)} - {rate_limited_apis}")
    print(f"❌ APIs con problemas: {len(error_apis)} - {error_apis}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(check_api_status())
    print("\\n🎉 Verificación de APIs completada")
