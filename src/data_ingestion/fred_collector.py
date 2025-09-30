"""
FRED (Federal Reserve Economic Data) Collector
Colector especializado para datos de la Reserva Federal de EE.UU.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

class FREDCollector:
    """
    Colector especializado para datos de FRED (Federal Reserve Economic Data)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Usar API key desde variables de entorno
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.session = None
        
        # Series importantes para predicción de precios de acero
        self.series_config = {
            'natural_gas': {
                'id': 'DHHNGSP',  # Natural Gas Price (como en BLM_forecasts)
                'name': 'Henry Hub Natural Gas Spot Price',
                'frequency': 'daily',
                'importance': 'high',
                'category': 'energy',
                'unit': 'USD/MMBtu'
            },
            'industrial_production': {
                'id': 'INDPRO',  # Industrial Production Index
                'name': 'Industrial Production Total Index',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'production',
                'unit': 'Index 2017=100'
            },
            'ppi_metals': {
                'id': 'WPU101',  # Producer Price Index: Metals and metal products
                'name': 'PPI: Metals and Metal Products',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'prices',
                'unit': 'Index 1982=100'
            },
            'dxy_index': {
                'id': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index
                'name': 'Trade Weighted US Dollar Index',
                'frequency': 'daily',
                'importance': 'high',
                'category': 'currency',
                'unit': 'Index'
            },
            'federal_funds_rate': {
                'id': 'FEDFUNDS',  # Federal Funds Rate
                'name': 'Federal Funds Effective Rate',
                'frequency': 'monthly',
                'importance': 'medium',
                'category': 'monetary',
                'unit': 'Percent'
            },
            'construction_spending': {
                'id': 'TTLCONS',  # Total Construction Spending
                'name': 'Total Construction Spending',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'construction',
                'unit': 'Million USD'
            },
            'steel_production': {
                'id': 'IPG3311A2N',  # Steel Production (corregido)
                'name': 'Industrial Production: Steel',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'steel',
                'unit': 'Index 2017=100'
            },
            'iron_steel_scrap': {
                'id': 'WPU101707',  # Iron and steel scrap
                'name': 'PPI: Iron and Steel Scrap',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'raw_materials',
                'unit': 'Index 1982=100'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_series_data(
        self, 
        series_key: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de una serie específica de FRED
        
        Args:
            series_key: Clave de la serie en series_config
            start_date: Fecha de inicio en formato YYYY-MM-DD
            end_date: Fecha de fin en formato YYYY-MM-DD
            save_raw: Guardar datos raw en archivo
            
        Returns:
            Diccionario con los datos de la serie
        """
        if series_key not in self.series_config:
            raise ValueError(f"Serie '{series_key}' no encontrada en configuración")
        
        series_info = self.series_config[series_key]
        series_id = series_info['id']
        
        logger.info(f"Obteniendo datos FRED para {series_key} ({series_id})")
        
        try:
            # Construir parámetros (exactamente como en BLM_forecasts)
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._process_fred_response(data, series_key)
                    
                    # Guardar datos raw si se solicita
                    if save_raw and result['data'] is not None:
                        await self._save_raw_data(result, series_key)
                    
                    return result
                else:
                    logger.error(f"Error FRED API para {series_key}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Excepción obteniendo datos FRED para {series_key}: {str(e)}")
            return None
    
    def _process_fred_response(self, response_data: Dict, series_key: str) -> Dict[str, Any]:
        """Procesar respuesta de la API de FRED (exactamente como en BLM_forecasts)"""
        try:
            if 'observations' not in response_data:
                raise ValueError("No hay observaciones en la respuesta")
            
            observations = response_data['observations']
            
            if not observations:
                logger.warning(f"No hay datos disponibles para {series_key}")
                return None
            
            # Convertir a DataFrame (como en BLM_forecasts)
            df = pd.DataFrame(observations)
            
            if 'date' in df.columns and 'value' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Renombrar columnas (como en BLM_forecasts)
                series_id = self.series_config[series_key]['id']
                df = df[['date', 'value']].rename(columns={'date': 'Date', 'value': series_id})
                df.dropna(inplace=True)
                df = df.sort_values('Date').reset_index(drop=True)
                
                series_info = self.series_config[series_key]
                
                return {
                    'series_key': series_key,
                    'series_name': series_info['name'],
                    'series_id': series_id,
                    'data': df,
                    'latest_value': df[series_id].iloc[-1] if not df.empty else np.nan,
                    'latest_date': df['Date'].iloc[-1] if not df.empty else None,
                    'count': len(df),
                    'frequency': series_info['frequency'],
                    'importance': series_info['importance'],
                    'category': series_info['category'],
                    'unit': series_info['unit'],
                    'source': 'fred_api'
                }
            else:
                raise ValueError("Columnas 'date' o 'value' no encontradas")
                
        except Exception as e:
            logger.error(f"Error procesando respuesta FRED: {str(e)}")
            return None
    
    async def _save_raw_data(self, series_result: Dict[str, Any], series_key: str):
        """Guardar datos raw en archivo con nombre descriptivo"""
        try:
            # Crear directorio raw si no existe
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            # Nombre descriptivo del archivo
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # Mapear series a nombres descriptivos
            name_mapping = {
                'fed_funds_rate': 'TasaFED',
                'treasury_10y': 'BonoUS10Y',
                'treasury_2y': 'BonoUS2Y',
                'industrial_production': 'ProduccionIndustrial',
                'unemployment': 'Desempleo',
                'cpi': 'InflacionUS',
                'gdp': 'PIBUS',
                'construction_spending': 'GastoConstruccion'
            }
            
            variable_name = name_mapping.get(series_key, series_key)
            filename = f"FRED_{variable_name}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            # Guardar DataFrame
            if series_result['data'] is not None and not series_result['data'].empty:
                series_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw guardados: {filepath}")
                
                # Guardar metadata
                metadata = {
                    'series_key': series_key,
                    'series_name': series_result['series_name'],
                    'series_id': series_result['series_id'],
                    'source': 'fred_api',
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': series_result['count'],
                    'latest_date': series_result['latest_date'].isoformat() if series_result['latest_date'] else None,
                    'latest_value': series_result['latest_value'],
                    'frequency': series_result['frequency'],
                    'importance': series_result['importance'],
                    'category': series_result['category'],
                    'unit': series_result['unit']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw: {str(e)}")

    async def get_all_series(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener todas las series configuradas de FRED
        
        Returns:
            Diccionario con todas las series de datos
        """
        logger.info("Recopilando todas las series de FRED...")
        
        results = {}
        
        # Procesar series en paralelo
        tasks = []
        for series_key in self.series_config.keys():
            task = self.get_series_data(series_key, start_date, end_date, save_raw)
            tasks.append(task)
        
        series_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(series_results):
            series_key = list(self.series_config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo serie FRED {series_key}: {str(result)}")
                # No agregar series que fallen - solo datos reales
                continue
            elif result is not None:
                results[series_key] = result
            else:
                logger.warning(f"Serie FRED {series_key} no devolvió datos - omitida")
        
        # Estadísticas generales
        total_points = sum(r['count'] for r in results.values())
        api_sources = sum(1 for r in results.values() if r['source'] == 'fred_api')
        
        summary = {
            'total_series': len(results),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'dummy_sources': len(results) - api_sources,
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'categories': self._get_categories_summary(results)
        }
        
        return {
            'series_data': results,
            'summary': summary
        }
    
    def _get_categories_summary(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Obtener resumen por categorías"""
        categories = {}
        for series_key, result in results.items():
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(series_key)
        return categories
    
    def get_series_info(self) -> Dict[str, Any]:
        """Obtener información sobre las series disponibles"""
        categories = {}
        for key, info in self.series_config.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        return {
            'available_series': self.series_config,
            'total_series': len(self.series_config),
            'categories': categories,
            'critical_importance': [k for k, v in self.series_config.items() if v['importance'] == 'critical'],
            'high_importance': [k for k, v in self.series_config.items() if v['importance'] == 'high'],
            'daily_series': [k for k, v in self.series_config.items() if v['frequency'] == 'daily'],
            'monthly_series': [k for k, v in self.series_config.items() if v['frequency'] == 'monthly']
        }

# Función de conveniencia para uso directo
async def collect_fred_data(
    api_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos de FRED
    
    Args:
        api_key: API key de FRED
        start_date: Fecha de inicio (YYYY-MM-DD)
        end_date: Fecha de fin (YYYY-MM-DD)
        save_raw: Guardar datos raw en archivos
    
    Returns:
        Diccionario con todos los datos recopilados
    """
    async with FREDCollector(api_key) as collector:
        return await collector.get_all_series(start_date, end_date, save_raw)
