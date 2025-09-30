"""
World Bank Data Collector
Colector para datos de commodities del Banco Mundial usando wbgapi
Basado en: https://pypi.org/project/wbgapi/
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os

try:
    import wbgapi as wb
    WBGAPI_AVAILABLE = True
except ImportError:
    WBGAPI_AVAILABLE = False
    logging.warning("wbgapi no está instalado. Instalar con: pip install wbgapi")

logger = logging.getLogger(__name__)

class WorldBankCollector:
    """
    Colector para datos del World Bank usando la librería oficial wbgapi
    """
    
    def __init__(self):
        self.wbgapi_available = WBGAPI_AVAILABLE
        
        # Indicadores económicos relevantes para predicción de acero
        self.indicators_config = {
            # Indicadores macroeconómicos de México
            'gdp_mexico': {
                'code': 'NY.GDP.MKTP.CD',
                'name': 'GDP México (current US$)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'macroeconomic',
                'unit': 'USD'
            },
            'inflation_mexico': {
                'code': 'FP.CPI.TOTL.ZG',
                'name': 'Inflación México (%)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'macroeconomic',
                'unit': '%'
            },
            'industry_value_mexico': {
                'code': 'NV.IND.TOTL.ZS',
                'name': 'Industry value added México (% of GDP)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'industry',
                'unit': '%'
            },
            'manufacturing_mexico': {
                'code': 'NV.IND.MANF.ZS',
                'name': 'Manufacturing value added México (% of GDP)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'industry',
                'unit': '%'
            },
            'exchange_rate_mexico': {
                'code': 'PA.NUS.FCRF',
                'name': 'Tipo de cambio oficial (MXN per US$)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'critical',
                'category': 'currency',
                'unit': 'MXN/USD'
            },
            # Indicadores adicionales de construcción e industria
            'construction_value': {
                'code': 'NV.IND.TOTL.CD',
                'name': 'Industry (including construction) value added (USD)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'high',
                'category': 'construction',
                'unit': 'USD'
            },
            'imports_mexico': {
                'code': 'TM.VAL.MMTL.ZS.UN',
                'name': 'Ores and metals imports (% of merchandise imports)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'medium',
                'category': 'trade',
                'unit': '%'
            },
            'exports_mexico': {
                'code': 'TX.VAL.MMTL.ZS.UN',
                'name': 'Ores and metals exports (% of merchandise exports)',
                'economy': 'MEX',
                'frequency': 'annual',
                'importance': 'medium',
                'category': 'trade',
                'unit': '%'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def get_indicator_data(
        self, 
        indicator_key: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de indicador específico usando wbgapi
        
        Args:
            indicator_key: Clave del indicador en indicators_config
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            save_raw: Guardar datos raw en archivo
            
        Returns:
            Diccionario con los datos del indicador
        """
        if not self.wbgapi_available:
            logger.error("wbgapi no está disponible")
            return None
            
        if indicator_key not in self.indicators_config:
            raise ValueError(f"Indicador '{indicator_key}' no encontrado en configuración")
        
        indicator_info = self.indicators_config[indicator_key]
        indicator_code = indicator_info['code']
        economy = indicator_info['economy']
        
        logger.info(f"Obteniendo datos World Bank para {indicator_key} ({indicator_code})")
        
        try:
            # Convertir fechas a años
            if start_date:
                start_year = datetime.strptime(start_date, '%Y-%m-%d').year
            else:
                start_year = 2000
            
            if end_date:
                end_year = datetime.strptime(end_date, '%Y-%m-%d').year
            else:
                end_year = datetime.now().year
            
            # Configurar base de datos si es commodity
            db = indicator_info.get('database', 2)  # Default es WDI (2)
            
            # Obtener datos usando wbgapi
            # Para datos más recientes, usar mrv (Most Recent Values)
            if indicator_info['frequency'] == 'monthly':
                # Para datos mensuales, intentar obtener más puntos
                data = wb.data.fetch(
                    indicator_code,
                    economy,
                    mrv=60,  # Últimos 60 valores (5 años mensuales)
                    db=db
                )
            else:
                # Para datos anuales
                data = wb.data.fetch(
                    indicator_code,
                    economy,
                    time=range(start_year, end_year + 1),
                    db=db
                )
            
            if data:
                # Convertir a DataFrame
                df_data = []
                for record in data:
                    if record.get('value') is not None:
                        # Parsear fecha según el formato
                        time_str = str(record.get('time', ''))
                        
                        # wbgapi devuelve 'YR2024' para años
                        if time_str.startswith('YR'):
                            year = int(time_str[2:])  # Quitar 'YR' del inicio
                            fecha = datetime(year, 1, 1)
                        elif 'M' in time_str:  # Formato mensual YYYYmMM
                            year = int(time_str[:4])
                            month = int(time_str.split('M')[1])
                            fecha = datetime(year, month, 1)
                        else:  # Intentar parsear como año directo
                            try:
                                fecha = datetime(int(time_str), 1, 1)
                            except:
                                continue  # Saltar si no se puede parsear
                        
                        df_data.append({
                            'fecha': fecha,
                            'valor': float(record['value'])
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('fecha')
                    
                    result = {
                        'indicator_key': indicator_key,
                        'indicator_name': indicator_info['name'],
                        'indicator_code': indicator_code,
                        'data': df,
                        'latest_value': df['valor'].iloc[-1] if not df.empty else np.nan,
                        'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                        'count': len(df),
                        'frequency': indicator_info['frequency'],
                        'importance': indicator_info['importance'],
                        'category': indicator_info['category'],
                        'unit': indicator_info['unit'],
                        'source': 'world_bank_wbgapi'
                    }
                    
                    # Guardar datos raw si se solicita
                    if save_raw and result['data'] is not None:
                        await self._save_raw_data(result, indicator_key)
                    
                    return result
                else:
                    logger.warning(f"No hay datos válidos para {indicator_key}")
                    return None
            else:
                logger.warning(f"No se obtuvieron datos para {indicator_key}")
                return None
                    
        except Exception as e:
            logger.error(f"Error obteniendo datos World Bank para {indicator_key}: {str(e)}")
            return None
    
    async def _save_raw_data(self, indicator_result: Dict[str, Any], indicator_key: str):
        """Guardar datos raw en archivo"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"world_bank_{indicator_key}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if indicator_result['data'] is not None and not indicator_result['data'].empty:
                indicator_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw World Bank guardados: {filepath}")
                
                # Guardar metadata
                metadata = {
                    'indicator_key': indicator_key,
                    'indicator_name': indicator_result['indicator_name'],
                    'indicator_code': indicator_result['indicator_code'],
                    'source': indicator_result['source'],
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': indicator_result['count'],
                    'latest_date': indicator_result['latest_date'].isoformat() if indicator_result['latest_date'] else None,
                    'latest_value': float(indicator_result['latest_value']) if not pd.isna(indicator_result['latest_value']) else None,
                    'frequency': indicator_result['frequency'],
                    'importance': indicator_result['importance'],
                    'category': indicator_result['category'],
                    'unit': indicator_result['unit']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw World Bank: {str(e)}")
    
    async def get_all_indicators(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de todos los indicadores configurados
        
        Returns:
            Diccionario con todos los datos de indicadores
        """
        if not self.wbgapi_available:
            logger.error("wbgapi no está disponible. Instalar con: pip install wbgapi")
            return {
                'indicators_data': {},
                'summary': {
                    'error': 'wbgapi not available',
                    'total_indicators': 0,
                    'collection_timestamp': datetime.now().isoformat()
                }
            }
        
        logger.info("Recopilando datos del World Bank con wbgapi...")
        
        results = {}
        
        # Procesar indicadores secuencialmente (wbgapi no es async)
        for indicator_key in self.indicators_config.keys():
            result = await self.get_indicator_data(indicator_key, start_date, end_date, save_raw)
            if result:
                results[indicator_key] = result
            await asyncio.sleep(0.5)  # Pequeña pausa entre requests
        
        # Estadísticas generales
        total_points = sum(r['count'] for r in results.values() if r)
        api_sources = sum(1 for r in results.values() if r and r['source'] == 'world_bank_wbgapi')
        
        # Categorías
        categories = {}
        for result in results.values():
            if result:
                category = result['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(result['indicator_key'])
        
        summary = {
            'total_indicators': len(results),
            'successful_indicators': len([r for r in results.values() if r]),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'categories': categories,
            'collection_timestamp': datetime.now().isoformat(),
            'library': 'wbgapi'
        }
        
        return {
            'indicators_data': results,
            'summary': summary
        }
    
    def search_indicators(self, search_term: str) -> List[Dict[str, str]]:
        """
        Buscar indicadores por término usando wbgapi
        
        Args:
            search_term: Término de búsqueda
            
        Returns:
            Lista de indicadores encontrados
        """
        if not self.wbgapi_available:
            logger.error("wbgapi no está disponible")
            return []
        
        try:
            # Buscar indicadores
            results = wb.series.info(q=search_term)
            
            indicators = []
            for code, info in results.items():
                indicators.append({
                    'code': code,
                    'name': info['value'],
                    'source': info.get('source', {}).get('value', 'N/A')
                })
            
            return indicators[:20]  # Limitar a 20 resultados
            
        except Exception as e:
            logger.error(f"Error buscando indicadores: {str(e)}")
            return []

# Función de conveniencia
async def collect_world_bank_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos del World Bank
    """
    async with WorldBankCollector() as collector:
        return await collector.get_all_indicators(start_date, end_date, save_raw)