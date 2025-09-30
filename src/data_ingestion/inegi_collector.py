"""
INEGI Data Collector
Colector especializado para datos del Instituto Nacional de Estadística y Geografía
Usando librería INEGIpy de https://github.com/andreslomeliv/DatosMex
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import json

logger = logging.getLogger(__name__)

# Intentar importar INEGIpy
try:
    from INEGIpy import Indicadores
    INEGIPY_AVAILABLE = True
    logger.info("INEGIpy importado correctamente")
except ImportError:
    INEGIPY_AVAILABLE = False
    logger.warning("INEGIpy no está instalado. Instalar con: pip install git+https://github.com/andreslomeliv/DatosMex.git#subdirectory=INEGIpy")

class INEGICollector:
    """
    Colector especializado para datos del INEGI usando INEGIpy
    Basado en: https://github.com/andreslomeliv/DatosMex/tree/master/INEGIpy
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Inicializar colector INEGI
        
        Args:
            api_token: Token de API de INEGI (requerido)
        """
        # Usar API token desde variables de entorno si no se proporciona
        self.api_token = api_token or os.getenv('INEGI_API_TOKEN') or os.getenv('INEGI_API_KEY')
        
        if not self.api_token:
            logger.warning("No se encontró token de INEGI. Configurar INEGI_API_TOKEN en .env")
        
        # Inicializar cliente de INEGIpy si está disponible
        self.inegi_client = None
        if INEGIPY_AVAILABLE and self.api_token:
            try:
                self.inegi_client = Indicadores(self.api_token)
                logger.info("Cliente INEGIpy inicializado correctamente")
            except Exception as e:
                logger.error(f"Error inicializando INEGIpy: {str(e)}")
        
        # Configuración de indicadores relevantes para predicción de acero
        # IDs tomados del catálogo BIE de INEGI
        self.indicators_config = {
            # Indicadores de Inflación (INPC)
            'inpc_general': {
                'id': '628194',
                'name': 'INPC General',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'inflation'
            },
            'inpc_subyacente': {
                'id': '628195',
                'name': 'INPC Subyacente',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'inflation'
            },
            
            # Indicadores de Producción Industrial
            'produccion_industrial': {
                'id': '91634',  # Índice de Volumen Físico de Producción Industrial
                'name': 'Producción Industrial General',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'production'
            },
            'produccion_construccion': {
                'id': '444570',  # Producción Industrial - Construcción
                'name': 'Producción Industrial - Construcción',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'production'
            },
            'produccion_manufactura': {
                'id': '91646',  # Producción Manufacturera
                'name': 'Producción Manufacturera',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'production'
            },
            
            # Indicadores de Precios Productor (INPP)
            'inpp_general': {
                'id': '628228',  # INPP General sin petróleo
                'name': 'INPP General sin Petróleo',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'producer_prices'
            },
            'inpp_construccion': {
                'id': '628229',  # INPP Construcción
                'name': 'INPP Materiales de Construcción',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'producer_prices'
            },
            
            # Indicadores de Actividad Económica
            'igae': {
                'id': '383152',  # IGAE - Indicador Global de Actividad Económica
                'name': 'IGAE - Actividad Económica',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'economic_activity'
            },
            'pib_trimestral': {
                'id': '493911',  # PIB Trimestral
                'name': 'PIB Trimestral',
                'frequency': 'quarterly',
                'importance': 'medium',
                'category': 'economic_activity'
            },
            
            # Indicadores del Sector Metalúrgico
            'produccion_metalurgica': {
                'id': '444612',  # Índice de producción - Industrias metálicas básicas
                'name': 'Producción Metalúrgica',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'steel_sector'
            },
            
            # Indicadores Financieros DIARIOS - Críticos para predicción
            # Nota: Estos indicadores están en Banxico con granularidad diaria
            # INEGI los tiene pero con menor frecuencia
            'udis': {
                'id': '628693',  # Valor de UDIS (mensual en INEGI)
                'name': 'UDIS - Unidades de Inversión',
                'frequency': 'monthly',  # En INEGI es mensual
                'importance': 'critical',
                'category': 'financial_monthly',
                'note': 'Para datos diarios usar Banxico SP68257'
            },
            'tiie_28': {
                'id': '628223',  # TIIE a 28 días
                'name': 'TIIE 28 días',
                'frequency': 'monthly',  # En INEGI es mensual
                'importance': 'critical',
                'category': 'financial_monthly',
                'note': 'Para datos diarios usar Banxico SF43878'
            },
            'tiie_91': {
                'id': '628224',  # TIIE a 91 días
                'name': 'TIIE 91 días',
                'frequency': 'monthly',  # En INEGI es mensual
                'importance': 'high',
                'category': 'financial_monthly',
                'note': 'Para datos diarios usar Banxico SF43927'
            },
            'tipo_cambio': {
                'id': '628208',  # Tipo de cambio promedio
                'name': 'Tipo de Cambio USD/MXN Promedio',
                'frequency': 'monthly',  # En INEGI es mensual
                'importance': 'critical',
                'category': 'financial_monthly',
                'note': 'Para datos diarios usar Banxico SF43718'
            },
            'cetes_28': {
                'id': '628211',  # SF43936 - CETES 28 días
                'name': 'CETES 28 días',
                'frequency': 'weekly',
                'importance': 'high',
                'category': 'financial_weekly'
            },
            'cetes_91': {
                'id': '628212',  # SF43939 - CETES 91 días
                'name': 'CETES 91 días',
                'frequency': 'weekly',
                'importance': 'medium',
                'category': 'financial_weekly'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    def _get_data_with_inegipy(
        self,
        indicator_ids: List[str],
        names: List[str],
        start_year: str = '2020',
        end_year: str = None
    ) -> pd.DataFrame:
        """
        Obtener datos usando INEGIpy
        
        Args:
            indicator_ids: Lista de IDs de indicadores
            names: Lista de nombres para las columnas
            start_year: Año inicial
            end_year: Año final (si es None, usa el año actual)
            
        Returns:
            DataFrame con los datos
        """
        if not self.inegi_client:
            logger.error("Cliente INEGIpy no inicializado")
            return pd.DataFrame()
        
        try:
            # Si no se especifica end_year, usar el año actual
            if end_year is None:
                end_year = str(datetime.now().year)
            
            # Obtener DataFrame usando INEGIpy
            df = self.inegi_client.obtener_df(
                indicadores=indicator_ids,
                nombres=names,
                inicio=start_year,
                fin=end_year
            )
            
            if df is not None and not df.empty:
                logger.info(f"Datos obtenidos con INEGIpy: {len(df)} filas")
                return df
            else:
                logger.warning("INEGIpy no devolvió datos")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error obteniendo datos con INEGIpy: {str(e)}")
            return pd.DataFrame()
    
    async def get_indicator_data(
        self,
        indicator_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de un indicador específico de INEGI
        
        Args:
            indicator_key: Clave del indicador en indicators_config
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            save_raw: Si guardar datos crudos
            
        Returns:
            Diccionario con los datos del indicador
        """
        if indicator_key not in self.indicators_config:
            raise ValueError(f"Indicador '{indicator_key}' no encontrado")
        
        indicator_info = self.indicators_config[indicator_key]
        indicator_id = indicator_info['id']
        
        logger.info(f"Obteniendo datos INEGI para {indicator_key} (ID: {indicator_id})")
        
        # Determinar años de inicio y fin
        start_year = '2020' if not start_date else start_date[:4]
        end_year = str(datetime.now().year) if not end_date else end_date[:4]
        
        # Intentar obtener datos con INEGIpy
        if self.inegi_client:
            try:
                df = self._get_data_with_inegipy(
                    indicator_ids=[indicator_id],
                    names=[indicator_info['name']],
                    start_year=start_year,
                    end_year=end_year
                )
                
                if not df.empty:
                    # Procesar DataFrame
                    df_processed = self._process_inegipy_dataframe(df, indicator_info)
                    
                    if not df_processed.empty:
                        result = {
                            'indicator_key': indicator_key,
                            'indicator_name': indicator_info['name'],
                            'indicator_id': indicator_id,
                            'data': df_processed,
                            'latest_value': df_processed['valor'].iloc[-1] if not df_processed.empty else np.nan,
                            'latest_date': df_processed['fecha'].iloc[-1] if not df_processed.empty else None,
                            'count': len(df_processed),
                            'frequency': indicator_info['frequency'],
                            'importance': indicator_info['importance'],
                            'category': indicator_info['category'],
                            'source': 'inegi_inegipy'
                        }
                        
                        # Guardar datos raw si se solicita
                        if save_raw:
                            await self._save_raw_data(result, indicator_key)
                        
                        return result
                        
            except Exception as e:
                logger.error(f"Error obteniendo indicador {indicator_key}: {str(e)}")
        
        # Si no hay datos, retornar estructura vacía
        logger.warning(f"No se pudieron obtener datos para {indicator_key}")
        return {
            'indicator_key': indicator_key,
            'indicator_name': indicator_info['name'],
            'indicator_id': indicator_id,
            'data': pd.DataFrame({'fecha': [], 'valor': []}),
            'latest_value': np.nan,
            'latest_date': None,
            'count': 0,
            'frequency': indicator_info['frequency'],
            'importance': indicator_info['importance'],
            'category': indicator_info['category'],
            'source': 'no_data'
        }
    
    async def _save_raw_data(self, data_result: Dict[str, Any], indicator_key: str):
        """Guardar datos raw en archivo CSV"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # Mapear indicadores a nombres descriptivos
            name_mapping = {
                'inpc': 'INPC',
                'inpp': 'INPP',
                'produccion_construccion': 'ProduccionConstruccion',
                'produccion_industrial': 'ProduccionIndustrial',
                'produccion_manufactura': 'ProduccionManufactura',
                'produccion_metalurgia': 'ProduccionMetalurgia',
                'igae': 'IGAE',
                'pib': 'PIB',
                'udis': 'UDIS',
                'tiie_28': 'TIIE28',
                'tiie_91': 'TIIE91',
                'tipo_cambio': 'TipoCambio',
                'cetes_28': 'CETES28',
                'cetes_91': 'CETES91'
            }
            
            variable_name = name_mapping.get(indicator_key, indicator_key)
            filename = f"INEGI_{variable_name}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if 'data' in data_result and data_result['data'] is not None:
                df = data_result['data']
                if not df.empty:
                    df.to_csv(filepath, index=False)
                    logger.info(f"Datos raw guardados: {filepath}")
                    
                    # Guardar metadata
                    metadata = {
                        'indicator_key': indicator_key,
                        'indicator_name': data_result.get('indicator_name', ''),
                        'source': 'inegi',
                        'collection_timestamp': datetime.now().isoformat(),
                        'count': len(df),
                        'latest_date': str(data_result.get('latest_date', '')),
                        'latest_value': data_result.get('latest_value', None),
                        'frequency': data_result.get('frequency', '')
                    }
                    
                    metadata_file = filepath.replace('.csv', '_metadata.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                        
        except Exception as e:
            logger.error(f"Error guardando datos raw INEGI: {str(e)}")
    
    def _process_inegipy_dataframe(self, df: pd.DataFrame, indicator_info: Dict) -> pd.DataFrame:
        """
        Procesar DataFrame de INEGIpy a formato estándar
        
        Args:
            df: DataFrame de INEGIpy
            indicator_info: Información del indicador
            
        Returns:
            DataFrame procesado con columnas 'fecha' y 'valor'
        """
        try:
            # INEGIpy devuelve DataFrame con índice de fechas
            df_processed = pd.DataFrame()
            
            # El DataFrame de INEGIpy tiene el indicador como columna
            if indicator_info['name'] in df.columns:
                df_processed['valor'] = df[indicator_info['name']]
            elif len(df.columns) > 0:
                # Usar la primera columna si no encontramos el nombre exacto
                df_processed['valor'] = df.iloc[:, 0]
            else:
                return pd.DataFrame({'fecha': [], 'valor': []})
            
            # Convertir índice a columna de fecha
            df_processed['fecha'] = pd.to_datetime(df.index)
            
            # Limpiar valores nulos
            df_processed = df_processed.dropna(subset=['valor'])
            
            # Ordenar por fecha
            df_processed = df_processed.sort_values('fecha')
            
            # Reset índice
            df_processed = df_processed.reset_index(drop=True)
            
            logger.info(f"DataFrame procesado: {len(df_processed)} filas válidas")
            
            return df_processed[['fecha', 'valor']]
            
        except Exception as e:
            logger.error(f"Error procesando DataFrame: {str(e)}")
            return pd.DataFrame({'fecha': [], 'valor': []})
    
    async def get_all_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener todos los indicadores configurados
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            save_raw: Si guardar datos crudos
            
        Returns:
            Diccionario con todos los indicadores
        """
        logger.info("Recopilando todos los indicadores de INEGI...")
        
        results = {}
        
        # Si tenemos INEGIpy, intentar obtener todos de una vez
        if self.inegi_client:
            try:
                # Preparar listas de IDs y nombres
                indicator_ids = [info['id'] for info in self.indicators_config.values()]
                indicator_names = [info['name'] for info in self.indicators_config.values()]
                
                start_year = '2020' if not start_date else start_date[:4]
                end_year = str(datetime.now().year) if not end_date else end_date[:4]
                
                # Obtener todos los indicadores de una vez
                logger.info(f"Obteniendo {len(indicator_ids)} indicadores con INEGIpy...")
                df_all = self._get_data_with_inegipy(
                    indicator_ids=indicator_ids,
                    names=indicator_names,
                    start_year=start_year,
                    end_year=end_year
                )
                
                if not df_all.empty:
                    # Procesar cada indicador del DataFrame combinado
                    for indicator_key, info in self.indicators_config.items():
                        if info['name'] in df_all.columns:
                            df_indicator = pd.DataFrame({
                                'fecha': pd.to_datetime(df_all.index),
                                'valor': df_all[info['name']]
                            }).dropna()
                            
                            results[indicator_key] = {
                                'indicator_key': indicator_key,
                                'indicator_name': info['name'],
                                'indicator_id': info['id'],
                                'data': df_indicator,
                                'latest_value': df_indicator['valor'].iloc[-1] if not df_indicator.empty else np.nan,
                                'latest_date': df_indicator['fecha'].iloc[-1] if not df_indicator.empty else None,
                                'count': len(df_indicator),
                                'frequency': info['frequency'],
                                'importance': info['importance'],
                                'category': info['category'],
                                'source': 'inegi_inegipy_batch'
                            }
                            logger.info(f"✅ {indicator_key}: {len(df_indicator)} puntos")
                        else:
                            logger.warning(f"❌ {indicator_key}: No encontrado en respuesta")
                            
            except Exception as e:
                logger.error(f"Error obteniendo indicadores en batch: {str(e)}")
        
        # Si no obtuvimos todos, intentar uno por uno
        if len(results) < len(self.indicators_config):
            logger.info("Obteniendo indicadores individualmente...")
            
            tasks = []
            for indicator_key in self.indicators_config.keys():
                if indicator_key not in results:
                    task = self.get_indicator_data(indicator_key, start_date, end_date, save_raw)
                    tasks.append(task)
            
            if tasks:
                indicator_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in indicator_results:
                    if not isinstance(result, Exception) and result:
                        results[result['indicator_key']] = result
        
        # Generar resumen
        categories = {}
        for result in results.values():
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result['indicator_key'])
        
        summary = {
            'total_indicators': len(results),
            'successful_indicators': len([r for r in results.values() if r['count'] > 0]),
            'total_data_points': sum(r['count'] for r in results.values()),
            'categories': categories,
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'using_inegipy': INEGIPY_AVAILABLE and self.inegi_client is not None
        }
        
        return {
            'indicators_data': results,
            'summary': summary
        }
    
    def get_indicators_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre los indicadores disponibles
        
        Returns:
            Diccionario con información de indicadores
        """
        categories = {}
        for key, info in self.indicators_config.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        return {
            'available_indicators': self.indicators_config,
            'total_indicators': len(self.indicators_config),
            'categories': categories,
            'critical_importance': [k for k, v in self.indicators_config.items() if v['importance'] == 'critical'],
            'high_importance': [k for k, v in self.indicators_config.items() if v['importance'] == 'high'],
            'medium_importance': [k for k, v in self.indicators_config.items() if v['importance'] == 'medium'],
            'frequencies': {
                'daily': [k for k, v in self.indicators_config.items() if v['frequency'] == 'daily'],
                'weekly': [k for k, v in self.indicators_config.items() if v['frequency'] == 'weekly'],
                'monthly': [k for k, v in self.indicators_config.items() if v['frequency'] == 'monthly'],
                'quarterly': [k for k, v in self.indicators_config.items() if v['frequency'] == 'quarterly']
            },
            'inegipy_available': INEGIPY_AVAILABLE,
            'api_token_configured': bool(self.api_token)
        }

# Función de conveniencia
async def collect_inegi_data(
    api_token: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos de INEGI
    
    Args:
        api_token: Token de API de INEGI
        start_date: Fecha de inicio
        end_date: Fecha de fin
        save_raw: Si guardar datos crudos
        
    Returns:
        Diccionario con datos de indicadores
    """
    async with INEGICollector(api_token) as collector:
        return await collector.get_all_indicators(start_date, end_date, save_raw)