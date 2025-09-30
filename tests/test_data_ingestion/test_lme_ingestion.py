#!/usr/bin/env python3
"""
Test de ingesta específico para LME/Yahoo Finance
Analiza granularidad temporal de precios de metales críticos para predicción de acero
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

from src.data_ingestion.lme_collector import LMECollector
from datetime import datetime, timedelta
import pandas as pd


class TestLMEIngestion:
    """Test class para ingesta de LME/Yahoo Finance"""
    
    @pytest.mark.asyncio
    async def test_lme_data_coverage(self):
        """Test cobertura y granularidad de datos LME desde 2020-01-01"""
        
        print("🥇 TEST LME/YAHOO FINANCE - GRANULARIDAD TEMPORAL")
        print("=" * 55)
        
        start_date = "2020-01-01"
        end_date = "2025-09-25"
        
        async with LMECollector() as collector:
            print(f"✅ Colector LME inicializado")
            print(f"📅 Período: {start_date} a {end_date}")
            
            # Obtener información de metales
            metals_info = collector.get_metals_info()
            print(f"📊 Total metales configurados: {metals_info['total_metals']}")
            print(f"⭐ Metales críticos: {len(metals_info['critical_importance'])}")
            print(f"📈 Metales alta importancia: {len(metals_info['high_importance'])}")
            
            # Analizar cada metal individualmente
            metals_analysis = {}
            
            for metal_key, config in collector.metals_config.items():
                print(f"\\n🥇 Analizando metal: {metal_key}")
                print(f"   Símbolo: {config['symbol']}")
                print(f"   Nombre: {config['name']}")
                print(f"   Importancia: {config['importance']}")
                print(f"   Categoría: {config['category']}")
                print(f"   Unidad: {config['unit']}")
                
                try:
                    # Obtener datos sin guardar
                    result = await collector.get_metal_data(
                        metal_key, 
                        start_date, 
                        end_date, 
                        period="5y",  # 5 años para asegurar cobertura
                        save_raw=False
                    )
                    
                    if result and result['data'] is not None and not result['data'].empty:
                        df = result['data']
                        
                        # Análisis temporal detallado
                        date_analysis = self._analyze_metal_temporal_granularity(df, metal_key, config)
                        
                        metals_analysis[metal_key] = {
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
                        print(f"   🎯 Para predicción diaria: {'EXCELENTE' if date_analysis['suitable_for_daily'] else 'REQUIERE INTERPOLACIÓN'}")
                        
                        # Información específica para metales críticos
                        if config['importance'] == 'critical':
                            print(f"   ⭐ METAL CRÍTICO - Prioridad máxima para modelo")
                            if date_analysis['trading_days_only']:
                                print(f"   📅 Solo días laborables - Típico para commodities")
                        
                    else:
                        metals_analysis[metal_key] = {
                            'config': config,
                            'result': result,
                            'status': 'no_data',
                            'error': 'Sin datos obtenidos o símbolo delisted'
                        }
                        print(f"   ❌ Sin datos obtenidos (posible delisting)")
                        
                except Exception as e:
                    metals_analysis[metal_key] = {
                        'config': config,
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"   ❌ Error: {str(e)}")
            
            # Analizar empresas siderúrgicas también
            print(f"\\n🏭 ANALIZANDO EMPRESAS SIDERÚRGICAS:")
            print("-" * 35)
            
            companies_data = await collector.get_steel_companies_data(start_date, end_date, "5y")
            
            for company_key, company_result in companies_data.items():
                if company_result and company_result['data'] is not None:
                    df = company_result['data']
                    date_analysis = self._analyze_metal_temporal_granularity(df, company_key, {'symbol': company_result['symbol']})
                    
                    print(f"   🏭 {company_key}: {company_result['company_name']}")
                    print(f"      Símbolo: {company_result['symbol']}")
                    print(f"      Puntos: {len(df)}")
                    print(f"      Granularidad: {date_analysis['detected_frequency']}")
                    print(f"      Apto diario: {'SÍ' if date_analysis['suitable_for_daily'] else 'NO'}")
            
            # Resumen de LME
            successful_metals = len([m for m in metals_analysis.values() if m['status'] == 'success'])
            daily_metals = len([m for m in metals_analysis.values() 
                              if m['status'] == 'success' and m['temporal_analysis']['suitable_for_daily']])
            critical_metals = len([m for m in metals_analysis.values() 
                                 if m['status'] == 'success' and m['config']['importance'] == 'critical'])
            
            print(f"\\n🎯 RESUMEN LME:")
            print(f"   📊 Metales exitosos: {successful_metals}/{len(collector.metals_config)}")
            print(f"   🎯 Metales aptos para predicción diaria: {daily_metals}")
            print(f"   ⭐ Metales críticos funcionando: {critical_metals}")
            print(f"   🏭 Empresas funcionando: {len(companies_data)}")
            
            # Assertions
            assert successful_metals >= 6, f"LME should have at least 6 working metals"
            assert daily_metals >= 4, f"LME should have at least 4 daily metals"
            assert critical_metals >= 1, f"LME should have at least 1 critical metal working"
            
            return metals_analysis
    
    def _analyze_metal_temporal_granularity(self, df: pd.DataFrame, metal_name: str, config: dict) -> dict:
        """Analizar granularidad temporal específica de metales/commodities"""
        
        if df.empty or len(df) < 2:
            return {
                'detected_frequency': 'unknown',
                'is_daily': False,
                'suitable_for_daily': False,
                'trading_days_only': False,
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
        
        # Verificar si son solo días laborables
        trading_days_only = self._check_trading_days_pattern(df['fecha'])
        
        # Determinar frecuencia para commodities/metales
        if avg_days <= 1.2:
            detected_freq = 'daily'
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 1.8 and trading_days_only:
            detected_freq = 'business_daily'  # Solo días laborables
            is_daily = True
            suitable_for_daily = True
        elif avg_days <= 8:
            detected_freq = 'weekly'
            is_daily = False
            suitable_for_daily = True
        elif avg_days <= 35:
            detected_freq = 'monthly'
            is_daily = False
            suitable_for_daily = False
        else:
            detected_freq = 'irregular'
            is_daily = False
            suitable_for_daily = False
        
        # Calcular estadísticas
        date_range = (df['fecha'].max() - df['fecha'].min()).days
        
        # Verificar datos faltantes en valor
        missing_values = df['valor'].isna().sum()
        missing_pct = (missing_values / len(df)) * 100
        
        # Verificar cobertura desde 2020
        coverage_from_2020 = df['fecha'].min() <= pd.Timestamp('2020-01-01')
        
        return {
            'detected_frequency': detected_freq,
            'is_daily': is_daily,
            'suitable_for_daily': suitable_for_daily,
            'trading_days_only': trading_days_only,
            'avg_days_between': avg_days,
            'total_points': len(df),
            'date_range_days': date_range,
            'start_date': df['fecha'].min(),
            'end_date': df['fecha'].max(),
            'coverage_from_2020': coverage_from_2020,
            'missing_data_pct': missing_pct,
            'data_density': (len(df) / date_range) * 100 if date_range > 0 else 0
        }
    
    def _check_trading_days_pattern(self, dates: pd.Series) -> bool:
        """Verificar si las fechas siguen patrón de días laborables"""
        
        if len(dates) < 10:
            return False
        
        # Contar días por día de la semana
        weekdays = dates.dt.dayofweek
        weekend_count = len(weekdays[(weekdays == 5) | (weekdays == 6)])  # Sábado y domingo
        
        # Si menos del 5% son fines de semana, probablemente son solo días laborables
        weekend_pct = (weekend_count / len(dates)) * 100
        
        return weekend_pct < 5


# Función para ejecutar manualmente
async def run_lme_test():
    """Ejecutar test de LME manualmente"""
    
    test_instance = TestLMEIngestion()
    result = await test_instance.test_lme_data_coverage()
    
    print("\\n📋 ANÁLISIS DETALLADO LME:")
    print("=" * 30)
    
    critical_daily = []
    high_daily = []
    non_daily = []
    
    for metal_key, analysis in result.items():
        if analysis['status'] == 'success':
            temp_analysis = analysis['temporal_analysis']
            config = analysis['config']
            
            if temp_analysis['suitable_for_daily']:
                if config['importance'] == 'critical':
                    critical_daily.append(metal_key)
                else:
                    high_daily.append(metal_key)
            else:
                non_daily.append(metal_key)
    
    print(f"\\n🎯 METALES POR IMPORTANCIA Y GRANULARIDAD:")
    print(f"   ⭐ Críticos diarios: {len(critical_daily)} - {critical_daily}")
    print(f"   📈 Altos diarios: {len(high_daily)} - {high_daily}")
    print(f"   📅 No diarios: {len(non_daily)} - {non_daily}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(run_lme_test())
    print("\\n✅ Test LME completado")
