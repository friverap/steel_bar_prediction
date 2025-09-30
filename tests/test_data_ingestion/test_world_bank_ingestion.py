#!/usr/bin/env python3
"""
Test de ingesta específico para World Bank
Analiza granularidad temporal de commodities del Banco Mundial
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

from src.data_ingestion.world_bank_collector import WorldBankCollector
from datetime import datetime, timedelta
import pandas as pd


class TestWorldBankIngestion:
    """Test class para ingesta de World Bank"""
    
    @pytest.mark.asyncio
    async def test_world_bank_data_coverage(self):
        """Test cobertura y granularidad de datos World Bank desde 2020-01-01"""
        
        print("🏛️ TEST WORLD BANK - COMMODITY PRICE DATA")
        print("=" * 45)
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with WorldBankCollector() as collector:
            print(f"✅ Colector World Bank inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            print(f"🆓 API pública - Sin API key requerida")
            print(f"📚 Usando librería wbgapi")
            
            # Analizar cada indicador individualmente
            indicators_analysis = {}
            
            for indicator_key, config in collector.indicators_config.items():
                print(f"\n🏛️ Analizando indicador: {indicator_key}")
                print(f"   Código: {config['code']}")
                print(f"   Nombre: {config['name']}")
                print(f"   Frecuencia esperada: {config['frequency']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                print(f"   Unidad: {config['unit']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_indicator_data(
                        indicator_key, 
                        start_date, 
                        end_date, 
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_wb_temporal_granularity(df, indicator_key, config)
                        
                        indicators_analysis[indicator_key] = {
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
                        print(f"   🎯 Para predicción diaria: {'INTERPOLACIÓN REQUERIDA' if not date_analysis['suitable_for_daily'] else 'DIRECTO'}")
                        
                        # Información específica para indicadores críticos
                        if config['importance'] == 'critical':
                            print(f"   ⭐ INDICADOR CRÍTICO para modelo de acero")
                        
                    else:
                        indicators_analysis[indicator_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos o problema de API'
                        }
                        print(f"   ❌ Sin datos obtenidos")
                        
                except Exception as e:
                    indicators_analysis[indicator_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Resumen de World Bank
            successful_indicators = len([c for c in indicators_analysis.values() if c['status'] == 'success'])
            annual_indicators = len([c for c in indicators_analysis.values() 
                                     if c['status'] == 'success' and c['temporal_analysis']['detected_frequency'] == 'annual'])
            api_real_indicators = len([c for c in indicators_analysis.values() 
                                      if c['status'] == 'success' and c['result']['source'] == 'world_bank_wbgapi'])
            critical_indicators = len([c for c in indicators_analysis.values() 
                                      if c['status'] == 'success' and c['config']['importance'] == 'critical'])
            
            print(f"\n🎯 RESUMEN WORLD BANK:")
            print(f"   📊 Indicadores exitosos: {successful_indicators}/{len(collector.indicators_config)}")
            print(f"   📅 Indicadores anuales: {annual_indicators}")
            print(f"   🔗 APIs reales: {api_real_indicators}")
            print(f"   ⭐ Indicadores críticos: {critical_indicators}")
            print(f"   💡 Nota: World Bank principalmente publica datos anuales")
            print(f"   ⚠️ Commodities NO disponibles en API pública")
            
            return indicators_analysis
    
    def _analyze_wb_temporal_granularity(self, df: pd.DataFrame, commodity_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de commodities World Bank"""
        
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
        
        # World Bank típicamente publica datos mensuales
        if avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False  # Requiere interpolación
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
            'interpolation_required': True  # World Bank siempre requiere interpolación para datos diarios
        }


# Función para ejecutar manualmente
async def run_world_bank_test():
    """Ejecutar test de World Bank manualmente"""
    
    test_instance = TestWorldBankIngestion()
    result = await test_instance.test_world_bank_data_coverage()
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_world_bank_test())
    print("\\n✅ Test World Bank completado")
