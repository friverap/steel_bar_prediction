#!/usr/bin/env python3
"""
Script de Actualización Diaria de Datos
Para uso regular - actualiza solo los datos más recientes
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.data_ingestion.banxico_collector import collect_banxico_data
from src.data_ingestion.fred_collector import collect_fred_data
from src.data_ingestion.lme_collector import collect_lme_data
from src.data_ingestion.ahmsa_collector import collect_ahmsa_data

import json


async def daily_update():
    """Actualización diaria de datos críticos"""
    
    print("📅 ACTUALIZACIÓN DIARIA DE DATOS")
    print("=" * 35)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Solo últimos 7 días para actualización rápida
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    print(f"🔄 Actualizando últimos 7 días: {start_date} a {end_date}")
    
    # Solo fuentes críticas para actualización diaria
    critical_sources = ['banxico', 'fred', 'lme', 'ahmsa']
    
    results = {}
    
    for source_name in critical_sources:
        print(f"\\n🔄 Actualizando {source_name.upper()}...")
        
        try:
            if source_name == 'banxico':
                data = await collect_banxico_data(start_date=start_date, end_date=end_date, save_raw=True)
            elif source_name == 'fred':
                data = await collect_fred_data(start_date=start_date, end_date=end_date, save_raw=True)
            elif source_name == 'lme':
                data = await collect_lme_data(start_date=start_date, end_date=end_date, save_raw=True)
            elif source_name == 'ahmsa':
                data = await collect_ahmsa_data(start_date=start_date, end_date=end_date, save_raw=True)
            
            if data and 'summary' in data:
                summary = data['summary']
                series_count = summary.get('total_series', summary.get('total_companies', 0))
                api_count = summary.get('api_sources', 0)
                
                results[source_name] = {
                    'status': 'success',
                    'series': series_count,
                    'api_sources': api_count
                }
                
                print(f"   ✅ {series_count} series, {api_count} APIs reales")
            else:
                results[source_name] = {'status': 'no_data'}
                print(f"   ❌ Sin datos")
                
        except Exception as e:
            results[source_name] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Error: {str(e)}")
    
    # Resumen
    successful = len([r for r in results.values() if r['status'] == 'success'])
    total_series = sum(r.get('series', 0) for r in results.values() if r['status'] == 'success')
    total_apis = sum(r.get('api_sources', 0) for r in results.values() if r['status'] == 'success')
    
    print(f"\\n📊 RESUMEN ACTUALIZACIÓN DIARIA:")
    print(f"✅ Fuentes actualizadas: {successful}/{len(critical_sources)}")
    print(f"📊 Series actualizadas: {total_series}")
    print(f"🔗 APIs reales: {total_apis}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(daily_update())
    print("\\n🎉 Actualización diaria completada")
