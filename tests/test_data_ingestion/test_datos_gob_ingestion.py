#!/usr/bin/env python3
"""
Test de ingesta específico para datos.gob.mx
Analiza granularidad temporal de datasets gubernamentales
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

from src.data_ingestion.datos_gob_collector import DatosGobCollector
from datetime import datetime, timedelta
import pandas as pd


class TestDatosGobIngestion:
    """Test class para ingesta de datos.gob.mx"""
    
    @pytest.mark.asyncio
    async def test_datos_gob_data_coverage(self):
        """Test cobertura y granularidad de datos datos.gob.mx desde 2020-01-01"""
        
        print("📋 TEST DATOS.GOB.MX - DATASETS GUBERNAMENTALES")
        print("=" * 50)
        
        async with DatosGobCollector() as collector:
            print(f"✅ Colector datos.gob.mx inicializado")
            print(f"🆓 Portal público - Sin API key requerida")
            print(f"🎯 Enfoque: Obras públicas y contratos gubernamentales")
            
            # Analizar cada dataset individualmente
            datasets_analysis = {}
            
            for dataset_key, config in collector.datasets_config.items():
                print(f"\\n📋 Analizando dataset: {dataset_key}")
                print(f"   ID: {config['dataset_id']}")
                print(f"   Nombre: {config['name']}")
                print(f"   Descripción: {config['description']}")
                print(f"   Frecuencia esperada: {config['frequency']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_dataset_data(
                        dataset_key, 
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_datos_gob_temporal_granularity(df, dataset_key, config)
                        
                        datasets_analysis[dataset_key] = {
                            'config': config,
                            'result': result,
                            'temporal_analysis': date_analysis,
                            'status': 'success'
                        }
                        
                        print(f"   ✅ Datos obtenidos: {len(df)} puntos")
                        print(f"   📅 Rango: {df['fecha'].min()} a {df['fecha'].max()}")
                        print(f"   🔗 Fuente: {result['source']}")
                        print(f"   📊 Granularidad: {date_analysis['detected_frequency']}")
                        print(f"   ⏱️ Días promedio: {date_analysis['avg_days_between']:.1f}")
                        print(f"   🎯 Para predicción diaria: {'INTERPOLACIÓN' if not date_analysis['suitable_for_daily'] else 'DIRECTO'}")
                        
                        # Información específica para obras públicas
                        if config['category'] == 'construction':
                            print(f"   🏗️ OBRAS PÚBLICAS - Indicador de demanda gubernamental")
                        
                    else:
                        datasets_analysis[dataset_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos o dataset no encontrado'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        
                except Exception as e:
                    datasets_analysis[dataset_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Resumen de datos.gob.mx
            successful_datasets = len([d for d in datasets_analysis.values() if d['status'] == 'success'])
            irregular_datasets = len([d for d in datasets_analysis.values() 
                                    if d['status'] == 'success' and d['temporal_analysis']['detected_frequency'] == 'irregular'])
            api_real_datasets = len([d for d in datasets_analysis.values() 
                                   if d['status'] == 'success' and d['result']['source'] == 'datos_gob_api'])
            
            print(f"\\n🎯 RESUMEN DATOS.GOB.MX:")
            print(f"   📊 Datasets exitosos: {successful_datasets}/{len(collector.datasets_config)}")
            print(f"   📅 Datasets irregulares: {irregular_datasets}")
            print(f"   🔗 APIs reales: {api_real_datasets}")
            print(f"   💡 Nota: Datos gubernamentales típicamente irregulares")
            
            return datasets_analysis
    
    def _analyze_datos_gob_temporal_granularity(self, df: pd.DataFrame, dataset_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de datos.gob.mx"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'suitable_for_daily': False,
                'avg_days_between': 0,
                'total_points': 0,
                'date_range_days': 0,
                'missing_data_pct': 100
            }
        
        # Convertir fechas y ordenar
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha')
        
        # Calcular diferencias entre fechas
        date_diffs = df['fecha'].diff().dropna()
        avg_days = date_diffs.dt.days.mean()
        
        # Datos gubernamentales típicamente irregulares
        if avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False
        elif avg_days <= 95:
            detected_freq = 'quarterly'
            is_daily = False
            suitable_for_daily = False
        elif avg_days <= 370:
            detected_freq = 'annual'
            is_daily = False
            suitable_for_daily = False
        else:
            detected_freq = 'irregular'
            is_daily = False
            suitable_for_daily = False
        
        # Calcular estadísticas
        date_range = (df['fecha'].max() - df['fecha'].min()).days
        
        # Verificar datos faltantes
        missing_values = df['valor'].isna().sum()
        missing_pct = (missing_values / len(df)) * 100
        
        # Verificar cobertura desde 2020
        coverage_from_2020 = df['fecha'].min() <= pd.Timestamp('2020-01-01')
        
        return {
            'detected_frequency': detected_freq,
            'is_daily': is_daily,
            'suitable_for_daily': suitable_for_daily,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df['fecha'].min(),
            'end_date': df['fecha'].max(),
            'coverage_from_2020': coverage_from_2020,
            'missing_data_pct': missing_pct,
            'data_density': (len(df) / date_range) * 100 if date_range > 0 else 0,
            'interpolation_required': True  # Datos gubernamentales requieren interpolación
        }


# Función para ejecutar manualmente
async def run_datos_gob_test():
    """Ejecutar test de datos.gob.mx manualmente"""
    
    test_instance = TestDatosGobIngestion()
    result = await test_instance.test_datos_gob_data_coverage()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_datos_gob_test())
    print("\\n✅ Test datos.gob.mx completado")
