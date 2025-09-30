"""
BANXICO Data Collector
Colector especializado para datos del Banco de México
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

class BANXICOCollector:
    """
    Colector especializado para datos del Banco de México (BANXICO)
    """
    
    def __init__(self, api_token: Optional[str] = None):
        # Usar API token desde variables de entorno
        self.api_token = api_token or os.getenv('BANXICO_API_TOKEN')
        self.base_url = "https://www.banxico.org.mx/SieAPIRest/service/v1"
        self.session = None
        
        # Series más importantes para predicción de precios de acero
        self.series_config = {
            'usd_mxn': {
                'id': 'SF43718',  # Tipo de cambio USD/MXN
                'name': 'Tipo de Cambio USD/MXN',
                'frequency': 'daily',
                'importance': 'high'
            },
            'udis': {
                'id': 'SP68257',  # Valor de UDIS diario
                'name': 'UDIS - Unidades de Inversión',
                'frequency': 'daily',
                'importance': 'critical'
            },
            'tiie_28': {
                'id': 'SF43878',  # TIIE a 28 días
                'name': 'TIIE 28 días',
                'frequency': 'daily',
                'importance': 'critical'
            },
            'tiie_91': {
                'id': 'SF43927',  # TIIE a 91 días
                'name': 'TIIE 91 días',
                'frequency': 'daily',
                'importance': 'high'
            },
            'inflation_monthly': {
                'id': 'SP74625',  # INPC General
                'name': 'Inflación Mensual (INPC)',
                'frequency': 'monthly',
                'importance': 'high'
            },
            'interest_rate': {
                'id': 'SF43783',  # Tasa de interés interbancaria
                'name': 'Tasa de Interés Interbancaria',
                'frequency': 'daily',
                'importance': 'high'
            },
            'industrial_production': {
                'id': 'SP74653',  # Producción industrial
                'name': 'Índice de Producción Industrial',
                'frequency': 'monthly',
                'importance': 'medium'
            },
            'manufacturing_index': {
                'id': 'SP74654',  # Industrias manufactureras
                'name': 'Índice de Industrias Manufactureras',
                'frequency': 'monthly',
                'importance': 'medium'
            },
            'construction_materials_price': {
                'id': 'SP74688',  # Precios de materiales de construcción
                'name': 'Índice de Precios de Materiales de Construcción',
                'frequency': 'monthly',
                'importance': 'high'
            },
            'steel_imports': {
                'id': 'SE43707',  # Importaciones de productos siderúrgicos
                'name': 'Importaciones de Productos Siderúrgicos',
                'frequency': 'monthly',
                'importance': 'high'
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
        Obtener datos de una serie específica de BANXICO
        
        Args:
            series_key: Clave de la serie en series_config
            start_date: Fecha de inicio en formato YYYY-MM-DD
            end_date: Fecha de fin en formato YYYY-MM-DD
            
        Returns:
            Diccionario con los datos de la serie
        """
        if series_key not in self.series_config:
            raise ValueError(f"Serie '{series_key}' no encontrada en configuración")
        
        series_info = self.series_config[series_key]
        series_id = series_info['id']
        
        if not self.api_token:
            logger.error(f"No hay token de BANXICO para {series_key}")
            return None
        
        try:
            # Si no se proporciona end_date, usar la fecha actual
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Construir URL
            if start_date:
                url = f"{self.base_url}/series/{series_id}/datos/{start_date}/{end_date}"
            else:
                # Si no hay start_date, obtener datos oportunos (más recientes)
                url = f"{self.base_url}/series/{series_id}/datos/oportuno"
            
            headers = {'Bmx-Token': self.api_token}
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    result = self._process_banxico_response(data, series_key)
                    
                    # Guardar datos raw si se solicita
                    if save_raw and result['data'] is not None:
                        await self._save_raw_data(result, series_key)
                    
                    return result
                else:
                    logger.error(f"Error BANXICO API para {series_key}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Excepción obteniendo datos BANXICO para {series_key}: {str(e)}")
            return None
    
    def _process_banxico_response(self, response_data: Dict, series_key: str) -> Dict[str, Any]:
        """Procesar respuesta de la API de BANXICO"""
        try:
            if 'bmx' not in response_data or 'series' not in response_data['bmx']:
                raise ValueError("Formato de respuesta inválido")
            
            series_data = response_data['bmx']['series'][0]
            datos = series_data.get('datos', [])
            
            if not datos:
                logger.warning(f"No hay datos disponibles para {series_key}")
                return None
            
            # Convertir a DataFrame
            df_data = []
            for punto in datos:
                try:
                    fecha = datetime.strptime(punto['fecha'], '%d/%m/%Y')
                    valor = float(punto['dato']) if punto['dato'] != 'N/E' else np.nan
                    df_data.append({'fecha': fecha, 'valor': valor})
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error procesando punto de datos: {e}")
                    continue
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('fecha')
            
            # Información de la serie
            series_info = self.series_config[series_key]
            
            return {
                'series_key': series_key,
                'series_name': series_info['name'],
                'data': df,
                'latest_value': df['valor'].iloc[-1] if not df.empty else np.nan,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'frequency': series_info['frequency'],
                'importance': series_info['importance'],
                'source': 'banxico_api'
            }
            
        except Exception as e:
            logger.error(f"Error procesando respuesta BANXICO: {str(e)}")
            return None
    
    async def _save_raw_data(self, series_result: Dict[str, Any], series_key: str):
        """Guardar datos raw en archivo con nombre descriptivo"""
        try:
            # Crear directorio raw si no existe
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            # Nombre descriptivo del archivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"banxico_{series_key}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            # Guardar DataFrame
            if series_result['data'] is not None and not series_result['data'].empty:
                series_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw BANXICO guardados: {filepath}")
                
                # Guardar metadata
                metadata = {
                    'series_key': series_key,
                    'series_name': series_result['series_name'],
                    'source': 'banxico_api',
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': series_result['count'],
                    'latest_date': series_result['latest_date'].isoformat() if series_result['latest_date'] else None,
                    'latest_value': series_result['latest_value'],
                    'frequency': series_result['frequency'],
                    'importance': series_result['importance']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw BANXICO: {str(e)}")

    async def get_all_series(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener todas las series configuradas
        
        Returns:
            Diccionario con todas las series de datos
        """
        logger.info("Recopilando todas las series de BANXICO...")
        
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
                logger.error(f"Error obteniendo serie {series_key}: {str(result)}")
                # No agregar series que fallen - solo datos reales
                continue
            elif result is not None:
                results[series_key] = result
            else:
                logger.warning(f"Serie {series_key} no devolvió datos - omitida")
        
        # Estadísticas generales
        total_points = sum(r['count'] for r in results.values())
        api_sources = sum(1 for r in results.values() if r['source'] == 'banxico_api')
        
        summary = {
            'total_series': len(results),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'dummy_sources': len(results) - api_sources,
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
        
        return {
            'series_data': results,
            'summary': summary
        }
    
    def get_series_info(self) -> Dict[str, Any]:
        """Obtener información sobre las series disponibles"""
        return {
            'available_series': self.series_config,
            'total_series': len(self.series_config),
            'high_importance': [k for k, v in self.series_config.items() if v['importance'] == 'high'],
            'medium_importance': [k for k, v in self.series_config.items() if v['importance'] == 'medium'],
            'daily_series': [k for k, v in self.series_config.items() if v['frequency'] == 'daily'],
            'monthly_series': [k for k, v in self.series_config.items() if v['frequency'] == 'monthly']
        }

# Función de conveniencia para uso directo
async def collect_banxico_data(
    api_token: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos de BANXICO
    
    Args:
        api_token: Token de la API de BANXICO
        start_date: Fecha de inicio (YYYY-MM-DD)
        end_date: Fecha de fin (YYYY-MM-DD)
    
    Returns:
        Diccionario con todos los datos recopilados
    """
    async with BANXICOCollector(api_token) as collector:
        return await collector.get_all_series(start_date, end_date)
