#!/usr/bin/env python3
"""
Script Optimizado de Ingesta de Datos - DeAcero Steel Price Predictor V2
Ingesta SOLO las variables necesarias para los modelos V2

VARIABLES DIARIAS REQUERIDAS:
- precio_varilla_lme (LME)
- iron (Raw Materials/Yahoo)  
- coking (Raw Materials/Yahoo)
- commodities (Yahoo Finance)
- VIX (Yahoo Finance)
- steel (LME/Yahoo)
- sp500 (Yahoo Finance)
- tasa_interes_banxico (Banxico)

VARIABLES MENSUALES REQUERIDAS:
- tasa_fed_usa (FRED)
- indice_precios_productor_metales_usa (FRED)
- produccion_acero_usa (FRED)
- precio_chatarra_acero_usa (FRED)
- produccion_industrial_usa (FRED)
- gasto_construccion_usa (FRED)
- produccion_metalurgica_mexico (INEGI)
- produccion_construccion_mexico (INEGI)
- inflacion_mensual_mexico (Banxico)

Fecha: 28 de Septiembre de 2025
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import Dict, Any, List

# Configurar paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# ========== IMPORTAR SOLO COLECTORES NECESARIOS ==========
from src.data_ingestion.banxico_collector import collect_banxico_data
from src.data_ingestion.fred_collector import collect_fred_data
from src.data_ingestion.lme_collector import collect_lme_data
from src.data_ingestion.yahoo_finance import collect_yahoo_finance_data
from src.data_ingestion.raw_materials_collector import collect_raw_materials_data
from src.data_ingestion.inegi_collector import collect_inegi_data

# ========== COMENTADO: Fuentes no necesarias para modelos V2 ==========
# from src.data_ingestion.ahmsa_collector import collect_ahmsa_data
# from src.data_ingestion.world_bank_collector import collect_world_bank_data
# from src.data_ingestion.world_bank_monthly_collector import collect_world_bank_monthly_data
# from src.data_ingestion.trading_economics_collector import collect_trading_economics_data
# from src.data_ingestion.quandl_collector import collect_quandl_data
# from src.data_ingestion.datos_gob_collector import collect_datos_gob_data


class OptimizedDataIngestionMaster:
    """Clase optimizada para ingesta SOLO de variables necesarias para modelos V2"""
    
    def __init__(self):
        self.start_date = "2020-01-01"
        # IMPORTANTE: Usar fecha actual del sistema, NO del .env
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.raw_data_dir = 'data/raw'
        self.processed_data_dir = 'data/processed'
        
        print(f"📅 FECHAS DINÁMICAS OPTIMIZADAS:")
        print(f"   Start: {self.start_date} (fijo)")
        print(f"   End: {self.end_date} (DINÁMICO - fecha actual: {datetime.now().strftime('%Y-%m-%d')})")
        print(f"   ⚠️ Ignorando DATA_END_DATE del .env para asegurar datos actualizados")
        
        # Crear directorios necesarios
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Variables específicas requeridas por modelos V2
        self.required_variables = {
            'daily': [
                'precio_varilla_lme',      # Variable objetivo (LME)
                'iron',                    # Mineral de hierro (Raw Materials)
                'coking',                  # Carbón de coque (Raw Materials)
                'commodities',             # Índice commodities (Yahoo)
                'VIX',                     # Volatilidad (Yahoo)
                'steel',                   # Acero general (LME/Yahoo)
                'sp500',                   # S&P 500 (Yahoo)
                'tasa_interes_banxico'     # Tasa Banxico (Banxico)
            ],
            'monthly': [
                'tasa_fed_usa',                           # FRED
                'indice_precios_productor_metales_usa',   # FRED
                'produccion_acero_usa',                   # FRED
                'precio_chatarra_acero_usa',              # FRED
                'produccion_industrial_usa',              # FRED
                'gasto_construccion_usa',                 # FRED
                'produccion_metalurgica_mexico',          # INEGI
                'produccion_construccion_mexico',         # INEGI
                'inflacion_mensual_mexico'                # Banxico
            ]
        }
        
        # Configuración OPTIMIZADA de fuentes (solo las necesarias)
        self.sources_config = {
            # ========== FUENTES CRÍTICAS PARA MODELOS V2 ==========
            'lme': {
                'collector_func': collect_lme_data,
                'description': 'LME - precio_varilla_lme, steel',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'variables_needed': ['precio_varilla_lme', 'steel'],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            },
            'raw_materials': {
                'collector_func': collect_raw_materials_data,
                'description': 'Raw Materials - iron, coking',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'variables_needed': ['iron', 'coking'],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            },
            'yahoo_finance': {
                'collector_func': collect_yahoo_finance_data,
                'description': 'Yahoo Finance - commodities, VIX, sp500',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'variables_needed': ['commodities', 'VIX', 'sp500'],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'period': '5y'
                }
            },
            'banxico': {
                'collector_func': collect_banxico_data,
                'description': 'Banxico - tasa_interes_banxico, inflacion_mensual_mexico',
                'priority': 'critical',
                'frequency': 'daily_and_monthly',
                'status': 'functional',
                'variables_needed': ['tasa_interes_banxico', 'inflacion_mensual_mexico'],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            },
            'fred': {
                'collector_func': collect_fred_data,
                'description': 'FRED - Indicadores mensuales US',
                'priority': 'critical',
                'frequency': 'monthly',
                'status': 'functional',
                'variables_needed': [
                    'tasa_fed_usa', 
                    'indice_precios_productor_metales_usa',
                    'produccion_acero_usa', 
                    'precio_chatarra_acero_usa',
                    'produccion_industrial_usa', 
                    'gasto_construccion_usa'
                ],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            },
            'inegi': {
                'collector_func': collect_inegi_data,
                'description': 'INEGI - Indicadores mensuales México',
                'priority': 'critical',
                'frequency': 'monthly',
                'status': 'functional',
                'variables_needed': [
                    'produccion_metalurgica_mexico',
                    'produccion_construccion_mexico'
                ],
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            }
            
            # ========== COMENTADO: Fuentes no necesarias para modelos V2 ==========
            # 'ahmsa': {
            #     'collector_func': collect_ahmsa_data,
            #     'description': 'AHMSA - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # },
            # 'world_bank': {
            #     'collector_func': collect_world_bank_data,
            #     'description': 'World Bank - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # },
            # 'world_bank_monthly': {
            #     'collector_func': collect_world_bank_monthly_data,
            #     'description': 'World Bank Monthly - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # },
            # 'trading_economics': {
            #     'collector_func': collect_trading_economics_data,
            #     'description': 'Trading Economics - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # },
            # 'quandl': {
            #     'collector_func': collect_quandl_data,
            #     'description': 'Quandl - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # },
            # 'datos_gob': {
            #     'collector_func': collect_datos_gob_data,
            #     'description': 'Datos.gob.mx - No necesario para modelos V2',
            #     'priority': 'disabled',
            #     'status': 'disabled'
            # }
        }
    
    async def ingest_optimized_sources(self) -> Dict[str, Any]:
        """Ejecutar ingesta optimizada SOLO de variables necesarias para modelos V2"""
        
        print("=" * 70)
        print("🚀 INGESTA OPTIMIZADA - DEACERO STEEL PRICE PREDICTOR V2")
        print("=" * 70)
        print(f"📅 Período: {self.start_date} a {self.end_date}")
        print(f"🎯 Solo variables necesarias para modelos V2")
        print("=" * 70)
        
        # Mostrar variables objetivo
        print(f"\n📊 VARIABLES DIARIAS OBJETIVO:")
        for var in self.required_variables['daily']:
            print(f"   • {var}")
        
        print(f"\n📊 VARIABLES MENSUALES OBJETIVO:")
        for var in self.required_variables['monthly']:
            print(f"   • {var}")
        
        # Clasificar fuentes activas
        active_sources = [k for k, v in self.sources_config.items() 
                         if v.get('status') == 'functional']
        
        print(f"\n📊 FUENTES ACTIVAS: {len(active_sources)}")
        for source in active_sources:
            config = self.sources_config[source]
            print(f"   ✅ {source}: {config['description']}")
        
        # Resultados de ingesta
        results = {}
        successful = []
        failed = []
        
        # Procesar solo fuentes activas
        for source_name in active_sources:
            source_config = self.sources_config[source_name]
            
            print(f"\n📊 {source_name.upper()}")
            print(f"   {source_config['description']}")
            print(f"   Variables objetivo: {', '.join(source_config['variables_needed'])}")
            print("-" * 50)
            
            try:
                # Ejecutar colector
                collector_func = source_config['collector_func']
                params = source_config['params']
                
                print(f"   ⏳ Ejecutando ingesta optimizada...")
                source_data = await collector_func(**params)
                
                # Procesar resultados
                if source_data:
                    results[source_name] = source_data
                    
                    # Contar series/datos obtenidos
                    data_count = self._count_data_points(source_data)
                    
                    print(f"   ✅ ÉXITO: {data_count['series']} series, {data_count['points']} puntos")
                    successful.append(source_name)
                    
                    # Guardar resumen
                    await self._save_source_summary(source_name, source_data)
                else:
                    print(f"   ⚠️ Sin datos obtenidos")
                    failed.append(source_name)
                    
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)[:100]}")
                failed.append(source_name)
                continue
        
        # Resumen final
        print(f"\n{'=' * 70}")
        print("📋 RESUMEN DE INGESTA OPTIMIZADA")
        print(f"{'=' * 70}")
        
        total_series = sum(self._count_data_points(r)['series'] for r in results.values())
        total_points = sum(self._count_data_points(r)['points'] for r in results.values())
        
        print(f"\n✅ EXITOSAS: {len(successful)}/{len(active_sources)}")
        for source in successful:
            config = self.sources_config[source]
            variables = ', '.join(config['variables_needed'])
            print(f"   • {source}: {variables}")
        
        if failed:
            print(f"\n❌ FALLIDAS: {len(failed)}")
            for source in failed:
                print(f"   • {source}")
        
        print(f"\n📊 DATOS TOTALES (OPTIMIZADOS):")
        print(f"   • Series temporales: {total_series}")
        print(f"   • Puntos de datos: {total_points:,}")
        print(f"   • Reducción estimada: ~70% menos datos vs ingesta completa")
        
        # Guardar resumen maestro
        summary = {
            'timestamp': datetime.now().isoformat(),
            'optimization': 'variables_v2_only',
            'period': {'start': self.start_date, 'end': self.end_date},
            'required_variables': self.required_variables,
            'sources': {
                'total_active': len(active_sources),
                'successful': len(successful),
                'failed': len(failed)
            },
            'data': {
                'total_series': total_series,
                'total_points': total_points
            },
            'details': {
                'successful': successful,
                'failed': failed
            }
        }
        
        # Guardar resumen en archivo
        summary_file = os.path.join(self.processed_data_dir, 'ingestion_summary_optimized.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Resumen guardado en: {summary_file}")
        print(f"{'=' * 70}")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def _count_data_points(self, source_data: Any) -> Dict[str, int]:
        """Contar puntos de datos en resultado de fuente"""
        series_count = 0
        points_count = 0
        
        if isinstance(source_data, dict):
            # Buscar DataFrames en la estructura
            for key, value in source_data.items():
                if isinstance(value, pd.DataFrame):
                    series_count += 1
                    points_count += len(value)
                elif isinstance(value, dict):
                    sub_counts = self._count_data_points(value)
                    series_count += sub_counts['series']
                    points_count += sub_counts['points']
        
        return {'series': series_count, 'points': points_count}
    
    async def _save_source_summary(self, source_name: str, data: Any):
        """Guardar resumen de fuente individual"""
        try:
            summary_dir = os.path.join(self.processed_data_dir, 'summaries')
            os.makedirs(summary_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{source_name}_summary_optimized_{timestamp}.json"
            filepath = os.path.join(summary_dir, filename)
            
            # Crear resumen compacto (sin datos completos)
            summary = {
                'source': source_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.sources_config[source_name],
                'data_counts': self._count_data_points(data),
                'optimization': 'v2_models_only'
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        except Exception as e:
            print(f"      ⚠️ No se pudo guardar resumen: {str(e)[:50]}")


async def main():
    """
    Función principal de ingesta optimizada
    """
    
    # Verificar configuración
    print("\n🔑 VERIFICACIÓN DE CONFIGURACIÓN")
    print("-" * 40)
    
    api_keys = {
        'BANXICO_API_TOKEN': os.getenv('BANXICO_API_TOKEN'),
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'INEGI_API_TOKEN': os.getenv('INEGI_API_TOKEN')
        # Comentado: API keys no necesarias
        # 'QUANDL_API_KEY': os.getenv('QUANDL_API_KEY'),
        # 'TRADING_ECONOMICS_API_KEY': os.getenv('TRADING_ECONOMICS_API_KEY')
    }
    
    configured = sum(1 for v in api_keys.values() if v)
    print(f"✅ API Keys configuradas: {configured}/{len(api_keys)}")
    
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"   ✅ {key_name}: {key_value[:10]}...")
        else:
            print(f"   ⚠️ {key_name}: No configurada")
    
    # Ejecutar ingesta optimizada
    print("\n" + "=" * 70)
    print("🚀 INICIANDO INGESTA OPTIMIZADA PARA MODELOS V2")
    print("=" * 70)
    
    master = OptimizedDataIngestionMaster()
    results = await master.ingest_optimized_sources()
    
    print("\n✅ INGESTA OPTIMIZADA COMPLETADA")
    print(f"   Fuentes exitosas: {results['summary']['sources']['successful']}")
    print(f"   Total de datos: {results['summary']['data']['total_points']:,} puntos")
    print(f"   🎯 Solo variables necesarias para modelos V2")
    
    return results


if __name__ == "__main__":
    # Ejecutar ingesta optimizada
    results = asyncio.run(main())
