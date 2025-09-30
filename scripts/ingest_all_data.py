#!/usr/bin/env python3
"""
Script Maestro de Ingesta de Datos - DeAcero Steel Price Predictor V2
Ingesta OPTIMIZADA solo de variables necesarias para modelos V2

VARIABLES OBJETIVO:
- Diarias: precio_varilla_lme, iron, coking, commodities, VIX, steel, sp500, tasa_interes_banxico
- Mensuales: tasa_fed_usa, indice_precios_productor_metales_usa, produccion_acero_usa, 
            precio_chatarra_acero_usa, produccion_industrial_usa, gasto_construccion_usa,
            produccion_metalurgica_mexico, produccion_construccion_mexico, inflacion_mensual_mexico
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

# ========== IMPORTAR COLECTORES FUNCIONALES (12 fuentes) ==========
# Fuentes CRÍTICAS (funcionando perfectamente)
from src.data_ingestion.banxico_collector import collect_banxico_data
from src.data_ingestion.fred_collector import collect_fred_data
from src.data_ingestion.lme_collector import collect_lme_data  # REACTIVADO: Necesario para cobre, zinc, aluminio
from src.data_ingestion.yahoo_finance import collect_yahoo_finance_data
from src.data_ingestion.raw_materials_collector import collect_raw_materials_data

# Fuentes NECESARIAS para modelos V2
from src.data_ingestion.inegi_collector import collect_inegi_data

# AGREGAR: Función para actualizar steel rebar real
import subprocess

# ========== COMENTADO: Fuentes no necesarias para modelos V2 ==========
# from src.data_ingestion.ahmsa_collector import collect_ahmsa_data
# from src.data_ingestion.world_bank_collector import collect_world_bank_data
# from src.data_ingestion.world_bank_monthly_collector import collect_world_bank_monthly_data
# from src.data_ingestion.trading_economics_collector import collect_trading_economics_data
# from src.data_ingestion.quandl_collector import collect_quandl_data
# from src.data_ingestion.datos_gob_collector import collect_datos_gob_data


class DataIngestionMaster:
    """Clase maestra para ingesta completa de datos"""
    
    def __init__(self):
        self.start_date = "2020-01-01"
        # IMPORTANTE: Usar fecha actual del sistema, NO del .env
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.raw_data_dir = 'data/raw'
        self.processed_data_dir = 'data/processed'
        
        print(f"📅 FECHAS DINÁMICAS:")
        print(f"   Start: {self.start_date} (fijo)")
        print(f"   End: {self.end_date} (DINÁMICO - fecha actual del sistema)")
        print(f"   ⚠️ Ignorando DATA_END_DATE del .env (si existe)")
        
        # Crear directorios necesarios
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Configuración de fuentes por prioridad
        self.sources_config = {
            # ========== STEEL REBAR REAL (PRIORIDAD MÁXIMA) ==========
            'steel_rebar_real': {
                'collector_func': self._update_steel_rebar_real,
                'description': 'Steel Rebar Real - Investing.com (USD/tonelada)',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 1,
                'params': {}
            },
            
            # ========== FUENTES CRÍTICAS (DIARIAS) ==========
            'yahoo_finance': {
                'collector_func': collect_yahoo_finance_data,
                'description': 'Yahoo Finance - Commodities, índices y acciones',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 15,
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'period': '5y'
                }
            },
            'raw_materials': {
                'collector_func': collect_raw_materials_data,
                'description': 'Materias Primas - Mineral de hierro y carbón de coque',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 10,
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            },
            'banxico': {
                'collector_func': collect_banxico_data,
                'description': 'Banxico - Tipo de cambio, TIIE, UDIS',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 6,
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            },
            'fred': {
                'collector_func': collect_fred_data,
                'description': 'FRED - Indicadores económicos US',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 8,
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            },
            'lme': {
                'collector_func': collect_lme_data,
                'description': 'LME - Metales (cobre, zinc, aluminio) SIN steel rebar',
                'priority': 'critical',
                'frequency': 'daily',
                'status': 'functional',
                'expected_series': 4,  # cobre, zinc, aluminio, + otros (sin steel rebar)
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            },
            
            # ========== FUENTES MENSUALES NECESARIAS PARA MODELOS V2 ==========
            'inegi': {
                'collector_func': collect_inegi_data,
                'description': 'INEGI - Indicadores económicos México',
                'priority': 'high',
                'frequency': 'monthly',
                'status': 'functional',
                'expected_series': 12,
                'params': {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'save_raw': True
                }
            },
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
    
    async def ingest_all_sources(self) -> Dict[str, Any]:
        """Ejecutar ingesta completa de todas las fuentes"""
        
        print("=" * 70)
        print("🚀 INGESTA OPTIMIZADA V2 - DEACERO STEEL PRICE PREDICTOR")
        print("=" * 70)
        print(f"📅 Período: {self.start_date} a {self.end_date}")
        print(f"🎯 Solo variables necesarias para modelos V2")
        print(f"📊 Fuentes activas: {len([s for s in self.sources_config.values() if s.get('status') == 'functional'])}")
        print("=" * 70)
        
        # Clasificar fuentes por estado
        functional_sources = [k for k, v in self.sources_config.items() 
                            if v['status'] == 'functional']
        limited_sources = [k for k, v in self.sources_config.items() 
                         if v['status'] == 'limited']
        non_functional = [k for k, v in self.sources_config.items() 
                         if v['status'] == 'non_functional']
        
        print(f"\n📊 ESTADO DE FUENTES:")
        print(f"   ✅ Funcionales: {len(functional_sources)} - {functional_sources}")
        print(f"   ⚠️ Limitadas: {len(limited_sources)} - {limited_sources}")
        print(f"   ❌ No funcionales: {len(non_functional)} - {non_functional}")
        
        # Resultados de ingesta
        results = {}
        successful = []
        failed = []
        
        # Procesar fuentes por prioridad
        for priority in ['critical', 'high', 'medium', 'low']:
            priority_sources = [k for k, v in self.sources_config.items() 
                              if v['priority'] == priority]
            
            if priority_sources:
                print(f"\n{'=' * 70}")
                print(f"⚡ PROCESANDO FUENTES {priority.upper()}")
                print(f"{'=' * 70}")
                
                for source_name in priority_sources:
                    source_config = self.sources_config[source_name]
                    
                    # Saltar fuentes no funcionales
                    if source_config['status'] == 'non_functional':
                        print(f"\n⏭️ Saltando {source_name} (no funcional)")
                        continue
                    
                    print(f"\n📊 {source_name.upper()}")
                    print(f"   {source_config['description']}")
                    print(f"   Estado: {source_config['status']}")
                    print(f"   Frecuencia: {source_config['frequency']}")
                    print("-" * 50)
                    
                    try:
                        # Ejecutar colector
                        collector_func = source_config['collector_func']
                        params = source_config['params']
                        
                        print(f"   ⏳ Ejecutando ingesta...")
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
        print("📋 RESUMEN DE INGESTA")
        print(f"{'=' * 70}")
        
        total_series = sum(self._count_data_points(r)['series'] for r in results.values())
        total_points = sum(self._count_data_points(r)['points'] for r in results.values())
        
        print(f"\n✅ EXITOSAS: {len(successful)}/{len(functional_sources)}")
        for source in successful:
            config = self.sources_config[source]
            print(f"   • {source}: {config['frequency']}")
        
        if failed:
            print(f"\n❌ FALLIDAS: {len(failed)}")
            for source in failed:
                print(f"   • {source}")
        
        print(f"\n📊 DATOS TOTALES:")
        print(f"   • Series temporales: {total_series}")
        print(f"   • Puntos de datos: {total_points:,}")
        
        # Guardar resumen maestro
        summary = {
            'timestamp': datetime.now().isoformat(),
            'period': {'start': self.start_date, 'end': self.end_date},
            'sources': {
                'total': len(self.sources_config),
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
        summary_file = os.path.join(self.processed_data_dir, 'ingestion_summary.json')
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
            filename = f"{source_name}_summary_{timestamp}.json"
            filepath = os.path.join(summary_dir, filename)
            
            # Crear resumen compacto (sin datos completos)
            summary = {
                'source': source_name,
                'timestamp': datetime.now().isoformat(),
                'config': self.sources_config[source_name],
                'data_counts': self._count_data_points(data)
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        except Exception as e:
            print(f"      ⚠️ No se pudo guardar resumen: {str(e)[:50]}")
    
    async def _update_steel_rebar_real(self, **params) -> Dict[str, Any]:
        """
        Actualizar datos de steel rebar real usando scraper de investing.com
        """
        print("\n🔄 ACTUALIZANDO STEEL REBAR REAL (INVESTING.COM)")
        print("-" * 50)
        
        try:
            # Ejecutar scraper de investing.com
            script_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data_ingestion', 'scraper_investing_real.py')
            
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Scraper ejecutado exitosamente")
                
                # Verificar que el archivo se creó
                steel_file = os.path.join(self.raw_data_dir, 'Investing_steel_rebar_real.csv')
                
                if os.path.exists(steel_file):
                    df = pd.read_csv(steel_file)
                    
                    return {
                        'steel_rebar_real': {
                            'data': df,
                            'count': len(df),
                            'latest_date': df['fecha'].max() if 'fecha' in df.columns else 'N/A',
                            'latest_price': df['valor'].iloc[-1] if 'valor' in df.columns else 'N/A',
                            'source': 'investing.com',
                            'unit': 'USD/tonelada'
                        }
                    }
                else:
                    raise FileNotFoundError("Archivo de steel rebar no se creó")
            else:
                raise Exception(f"Scraper falló: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error actualizando steel rebar: {str(e)}")
            return {}


async def main():
    """Función principal de ingesta"""
    
    # Verificar configuración
    print("\n🔑 VERIFICACIÓN DE CONFIGURACIÓN")
    print("-" * 40)
    
    # Solo API keys necesarias para modelos V2
    api_keys = {
        'BANXICO_API_TOKEN': os.getenv('BANXICO_API_TOKEN'),
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'INEGI_API_TOKEN': os.getenv('INEGI_API_TOKEN')
        # Comentado: API keys no necesarias para modelos V2
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
    
    # Auto-continuar para ejecución automática
    print("\n" + "=" * 70)
    print("🚀 INICIANDO INGESTA AUTOMÁTICA (incluye steel rebar real)")
    print("=" * 70)
    
    # Ejecutar ingesta
    print("\n🚀 Iniciando ingesta...")
    master = DataIngestionMaster()
    results = await master.ingest_all_sources()
    
    print("\n✅ INGESTA COMPLETADA")
    print(f"   Total de fuentes exitosas: {results['summary']['sources']['successful']}")
    print(f"   Total de datos: {results['summary']['data']['total_points']:,} puntos")
    
    return results


if __name__ == "__main__":
    # Ejecutar ingesta maestra
    results = asyncio.run(main())