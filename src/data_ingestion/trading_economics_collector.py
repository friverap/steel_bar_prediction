"""
Trading Economics Data Collector
Colector para indicadores de commodities de Trading Economics
Usando la librería oficial: https://github.com/tradingeconomics/tradingeconomics-python
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
    import tradingeconomics as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    logging.warning("tradingeconomics no está instalado. Instalar con: pip install tradingeconomics")

logger = logging.getLogger(__name__)

class TradingEconomicsCollector:
    """
    Colector para datos de Trading Economics
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TRADING_ECONOMICS_API_KEY')
        
        # Inicializar librería Trading Economics
        if TE_AVAILABLE and self.api_key:
            try:
                te.login(self.api_key)
                self.te_client = te
                logger.info("Trading Economics API inicializada correctamente")
            except Exception as e:
                logger.error(f"Error inicializando Trading Economics: {e}")
                self.te_client = None
        else:
            self.te_client = None
            
        # Indicadores relevantes para predicción de acero
        # Enfoque en México e indicadores globales de commodities
        self.indicators_config = {
            'mexico_stock_market': {
                'country': 'mexico',
                'indicator': 'stock-market',
                'name': 'México Stock Market Index (IPC)',
                'frequency': 'daily',
                'importance': 'high',
                'category': 'financial',
                'unit': 'Index Points'
            },
            'mexico_inflation': {
                'country': 'mexico',
                'indicator': 'inflation-cpi',
                'name': 'México Inflation Rate',
                'frequency': 'monthly',
                'importance': 'critical',
                'category': 'inflation',
                'unit': 'Percent'
            },
            'mexico_interest_rate': {
                'country': 'mexico',
                'indicator': 'interest-rate',
                'name': 'México Interest Rate',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'monetary',
                'unit': 'Percent'
            },
            'mexico_gdp': {
                'country': 'mexico',
                'indicator': 'gdp-growth-annual',
                'name': 'México GDP Growth',
                'frequency': 'quarterly',
                'importance': 'high',
                'category': 'economic',
                'unit': 'Percent'
            },
            'mexico_manufacturing': {
                'country': 'mexico',
                'indicator': 'manufacturing-production',
                'name': 'México Manufacturing Production',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'manufacturing',
                'unit': 'Index'
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
        Obtener datos de indicador específico usando la librería oficial de Trading Economics
        """
        if not self.te_client:
            logger.error("Trading Economics client no está disponible")
            return None
            
        if indicator_key not in self.indicators_config:
            raise ValueError(f"Indicador '{indicator_key}' no encontrado en configuración")
        
        indicator_info = self.indicators_config[indicator_key]
        country = indicator_info['country']
        indicator = indicator_info['indicator']
        
        logger.info(f"Obteniendo datos Trading Economics para {indicator_key}")
        
        try:
            # Usar la librería oficial de Trading Economics
            # Obtener datos históricos según el tipo de indicador
            if indicator == 'stock-market':
                # Para índices bursátiles - obtener todos y filtrar por país
                all_indices = self.te_client.getMarketsData(
                    marketsField='index',
                    output_type='df'
                )
                # Filtrar por México
                if all_indices is not None and not all_indices.empty and 'Country' in all_indices.columns:
                    data = all_indices[all_indices['Country'].str.contains('Mexico', case=False, na=False)]
                else:
                    data = pd.DataFrame()
            else:
                # Para indicadores económicos
                data = self.te_client.getHistoricalData(
                    country=country,
                    indicator=indicator,
                    initDate=start_date if start_date else '2020-01-01',
                    endDate=end_date if end_date else datetime.now().strftime('%Y-%m-%d'),
                    output_type='df'
                )
            
            if data is not None and not data.empty:
                result = self._process_te_library_response(data, indicator_key)
                
                # Guardar datos raw si se solicita
                if save_raw and result and result['data'] is not None:
                    await self._save_raw_data(result, indicator_key)
                
                return result
            else:
                logger.warning(f"Sin datos para {indicator_key}")
                return None
                    
        except Exception as e:
            logger.error(f"Error obteniendo datos Trading Economics para {indicator_key}: {str(e)}")
            return None
    
    def _process_te_library_response(self, data: pd.DataFrame, indicator_key: str) -> Dict[str, Any]:
        """Procesar respuesta de la librería oficial de Trading Economics"""
        try:
            if data.empty:
                raise ValueError("DataFrame vacío")
            
            # La librería devuelve un DataFrame con diferentes estructuras según el tipo
            # Normalizar a estructura común
            df = pd.DataFrame()
            
            # Identificar columnas de fecha y valor
            if 'DateTime' in data.columns:
                df['fecha'] = pd.to_datetime(data['DateTime'])
            elif 'Date' in data.columns:
                df['fecha'] = pd.to_datetime(data['Date'])
            elif data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex):
                df['fecha'] = pd.to_datetime(data.index)
            else:
                # Intentar con la primera columna que parezca fecha
                for col in data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        df['fecha'] = pd.to_datetime(data[col])
                        break
            
            # Identificar columna de valor
            if 'Value' in data.columns:
                df['valor'] = pd.to_numeric(data['Value'], errors='coerce')
            elif 'Close' in data.columns:
                df['valor'] = pd.to_numeric(data['Close'], errors='coerce')
            elif 'Last' in data.columns:
                df['valor'] = pd.to_numeric(data['Last'], errors='coerce')
            else:
                # Usar la primera columna numérica
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['valor'] = pd.to_numeric(data[numeric_cols[0]], errors='coerce')
            
            if df.empty or 'fecha' not in df.columns or 'valor' not in df.columns:
                raise ValueError("No se pudieron identificar columnas de fecha y valor")
            
            # Limpiar y ordenar
            df = df.dropna()
            df = df.sort_values('fecha')
            df = df.reset_index(drop=True)
            
            indicator_info = self.indicators_config[indicator_key]
            
            return {
                'indicator_key': indicator_key,
                'indicator_name': indicator_info['name'],
                'country': indicator_info['country'],
                'indicator': indicator_info['indicator'],
                'data': df,
                'latest_value': df['valor'].iloc[-1] if not df.empty else np.nan,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'frequency': indicator_info['frequency'],
                'importance': indicator_info['importance'],
                'category': indicator_info['category'],
                'unit': indicator_info['unit'],
                'source': 'trading_economics_library'
            }
            
        except Exception as e:
            logger.error(f"Error procesando respuesta TE library: {str(e)}")
            return None
    
    async def _save_raw_data(self, indicator_result: Dict[str, Any], indicator_key: str):
        """Guardar datos raw en archivo"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_economics_{indicator_key}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if indicator_result['data'] is not None and not indicator_result['data'].empty:
                indicator_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw Trading Economics guardados: {filepath}")
                
                # Guardar metadata
                metadata = {
                    'indicator_key': indicator_key,
                    'indicator_name': indicator_result['indicator_name'],
                    'country': indicator_result['country'],
                    'indicator': indicator_result['indicator'],
                    'source': 'trading_economics_api',
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': indicator_result['count'],
                    'latest_date': indicator_result['latest_date'].isoformat() if indicator_result['latest_date'] else None,
                    'latest_value': indicator_result['latest_value'],
                    'frequency': indicator_result['frequency'],
                    'importance': indicator_result['importance'],
                    'category': indicator_result['category'],
                    'unit': indicator_result['unit']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw Trading Economics: {str(e)}")

    async def get_all_indicators(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """Obtener todos los indicadores configurados"""
        logger.info("Recopilando datos de Trading Economics...")
        
        results = {}
        
        tasks = []
        for indicator_key in self.indicators_config.keys():
            task = self.get_indicator_data(indicator_key, start_date, end_date, save_raw)
            tasks.append(task)
        
        indicator_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(indicator_results):
            indicator_key = list(self.indicators_config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo indicador Trading Economics {indicator_key}: {str(result)}")
                # No agregar indicadores que fallen - solo datos reales
                continue
            elif result is not None:
                results[indicator_key] = result
            else:
                logger.warning(f"Indicador {indicator_key} no devolvió datos - omitido")
        
        total_points = sum(r['count'] for r in results.values())
        api_sources = sum(1 for r in results.values() if r['source'] == 'trading_economics_api')
        
        summary = {
            'total_indicators': len(results),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'dummy_sources': len(results) - api_sources,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return {
            'indicators_data': results,
            'summary': summary
        }

# Función de conveniencia
async def collect_trading_economics_data(
    api_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """Función de conveniencia para recopilar datos de Trading Economics"""
    async with TradingEconomicsCollector(api_key) as collector:
        return await collector.get_all_indicators(start_date, end_date, save_raw)
