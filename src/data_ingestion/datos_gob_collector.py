"""
Datos.gob.mx Collector
Colector para Portal de Datos Abiertos del Gobierno de México
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

class DatosGobCollector:
    """
    Colector para datos del Portal de Datos Abiertos del Gobierno de México
    """
    
    def __init__(self):
        # Usar portal histórico que tiene datos más estables
        self.base_url = "https://historico.datos.gob.mx"
        self.api_url = "https://historico.datos.gob.mx/api/3/action"
        self.session = None
        
        # Headers para evitar bloqueo 403 del portal histórico
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'es-MX,es;q=0.9,en;q=0.8'
        }
        
        # Datasets relevantes para construcción e industria
        self.datasets_config = {
            'public_works': {
                'dataset_id': 'obras-publicas-federales',
                'name': 'Obras Públicas Federales',
                'description': 'Información sobre obras públicas que demandan acero',
                'frequency': 'irregular',
                'importance': 'high',
                'category': 'construction',
                'unit': 'MXN'
            },
            'government_contracts': {
                'dataset_id': 'contratos-gobierno-federal',
                'name': 'Contratos del Gobierno Federal',
                'description': 'Contratos gubernamentales relacionados con construcción',
                'frequency': 'irregular',
                'importance': 'medium',
                'category': 'government',
                'unit': 'MXN'
            },
            'infrastructure_investment': {
                'dataset_id': 'inversion-infraestructura',
                'name': 'Inversión en Infraestructura',
                'description': 'Inversión pública en infraestructura',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'investment',
                'unit': 'MXN'
            },
            'industrial_permits': {
                'dataset_id': 'permisos-industriales',
                'name': 'Permisos Industriales',
                'description': 'Permisos para actividades industriales',
                'frequency': 'monthly',
                'importance': 'medium',
                'category': 'permits',
                'unit': 'Count'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_dataset_data(
        self, 
        dataset_key: str, 
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de dataset específico de datos.gob.mx
        """
        if dataset_key not in self.datasets_config:
            raise ValueError(f"Dataset '{dataset_key}' no encontrado en configuración")
        
        dataset_info = self.datasets_config[dataset_key]
        dataset_id = dataset_info['dataset_id']
        
        logger.info(f"Obteniendo datos datos.gob.mx para {dataset_key}")
        
        try:
            # Buscar dataset en datos.gob.mx
            search_url = f"{self.api_url}/package_search"
            search_params = {
                'q': dataset_id,
                'rows': 1
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    search_data = await response.json()
                    
                    if search_data['success'] and search_data['result']['count'] > 0:
                        # Obtener primer dataset encontrado
                        dataset = search_data['result']['results'][0]
                        
                        # Buscar recursos CSV
                        csv_resources = [r for r in dataset['resources'] if r['format'].upper() == 'CSV']
                        
                        if csv_resources:
                            # Usar primer recurso CSV
                            resource_url = csv_resources[0]['url']
                            result = await self._download_csv_data(resource_url, dataset_key)
                            
                            # Guardar datos raw si se solicita
                            if save_raw and result['data'] is not None:
                                await self._save_raw_data(result, dataset_key)
                            
                            return result
                        else:
                            logger.warning(f"No se encontraron recursos CSV para {dataset_key}")
                            return None
                    else:
                        logger.warning(f"Dataset {dataset_key} no encontrado en datos.gob.mx")
                        return None
                else:
                    logger.error(f"Error datos.gob.mx API para {dataset_key}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Excepción obteniendo datos datos.gob.mx para {dataset_key}: {str(e)}")
            return None
    
    async def _download_csv_data(self, csv_url: str, dataset_key: str) -> Dict[str, Any]:
        """Descargar y procesar datos CSV"""
        try:
            async with self.session.get(csv_url) as response:
                if response.status == 200:
                    csv_content = await response.text()
                    
                    # Procesar CSV
                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_content))
                    
                    # Intentar identificar columnas de fecha y valor
                    date_columns = ['fecha', 'Fecha', 'Date', 'date', 'año', 'Año']
                    value_columns = ['valor', 'Valor', 'monto', 'Monto', 'cantidad', 'Cantidad']
                    
                    date_col = None
                    value_col = None
                    
                    for col in date_columns:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    for col in value_columns:
                        if col in df.columns:
                            value_col = col
                            break
                    
                    if date_col and value_col:
                        # Procesar datos encontrados
                        df_clean = df[[date_col, value_col]].copy()
                        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
                        
                        df_clean = df_clean.rename(columns={date_col: 'fecha', value_col: 'valor'})
                        df_clean = df_clean.dropna().sort_values('fecha')
                        
                        dataset_info = self.datasets_config[dataset_key]
                        
                        return {
                            'dataset_key': dataset_key,
                            'dataset_name': dataset_info['name'],
                            'dataset_id': dataset_info['dataset_id'],
                            'data': df_clean,
                            'latest_value': df_clean['valor'].iloc[-1] if not df_clean.empty else np.nan,
                            'latest_date': df_clean['fecha'].iloc[-1] if not df_clean.empty else None,
                            'count': len(df_clean),
                            'frequency': dataset_info['frequency'],
                            'importance': dataset_info['importance'],
                            'category': dataset_info['category'],
                            'unit': dataset_info['unit'],
                            'source': 'datos_gob_api',
                            'csv_url': csv_url
                        }
                    else:
                        logger.warning(f"No se encontraron columnas de fecha/valor en {dataset_key}")
                        return None
                else:
                    logger.error(f"Error descargando CSV para {dataset_key}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error procesando CSV para {dataset_key}: {str(e)}")
            return None

    async def _save_raw_data(self, dataset_result: Dict[str, Any], dataset_key: str):
        """Guardar datos raw en archivo"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"datos_gob_{dataset_key}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if dataset_result['data'] is not None and not dataset_result['data'].empty:
                dataset_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw datos.gob.mx guardados: {filepath}")
                
                metadata = {
                    'dataset_key': dataset_key,
                    'dataset_name': dataset_result['dataset_name'],
                    'dataset_id': dataset_result['dataset_id'],
                    'source': dataset_result['source'],
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': dataset_result['count'],
                    'latest_date': dataset_result['latest_date'].isoformat() if dataset_result['latest_date'] else None,
                    'latest_value': dataset_result['latest_value'],
                    'frequency': dataset_result['frequency'],
                    'importance': dataset_result['importance'],
                    'category': dataset_result['category'],
                    'unit': dataset_result['unit'],
                    'csv_url': dataset_result.get('csv_url', 'N/A')
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw datos.gob.mx: {str(e)}")
    
    async def get_all_datasets(
        self, 
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """Obtener todos los datasets configurados"""
        logger.info("Recopilando datos de datos.gob.mx...")
        
        results = {}
        
        tasks = []
        for dataset_key in self.datasets_config.keys():
            task = self.get_dataset_data(dataset_key, save_raw)
            tasks.append(task)
        
        dataset_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(dataset_results):
            dataset_key = list(self.datasets_config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo dataset datos.gob.mx {dataset_key}: {str(result)}")
                continue
            else:
                results[dataset_key] = result
        
        total_points = sum(r['count'] for r in results.values())
        api_sources = sum(1 for r in results.values() if r['source'] == 'datos_gob_api')
        
        summary = {
            'total_datasets': len(results),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'dummy_sources': len(results) - api_sources,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return {
            'datasets_data': results,
            'summary': summary
        }

# Función de conveniencia
async def collect_datos_gob_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """Función de conveniencia para recopilar datos de datos.gob.mx"""
    async with DatosGobCollector() as collector:
        return await collector.get_all_datasets(save_raw)
