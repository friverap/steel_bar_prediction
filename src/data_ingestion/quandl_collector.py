"""
Nasdaq Data Link (Quandl) Collector
Colector para datos fundamentales de empresas de acero y commodities
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os

logger = logging.getLogger(__name__)

# Intentar importar la librería oficial
try:
    import nasdaqdatalink
    NASDAQ_LIB_AVAILABLE = True
except ImportError:
    NASDAQ_LIB_AVAILABLE = False
    logger.warning("nasdaq-data-link no instalado. Instalar con: pip install nasdaq-data-link")

class QuandlCollector:
    """
    Colector para datos de Nasdaq Data Link (anteriormente Quandl)
    Usa la librería oficial nasdaq-data-link
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('QUANDL_API_KEY') or os.getenv('NASDAQ_DATA_LINK_API_KEY')
        
        if NASDAQ_LIB_AVAILABLE and self.api_key:
            nasdaqdatalink.ApiConfig.api_key = self.api_key
            logger.info(f"Nasdaq Data Link configurado con API key: {self.api_key[:8]}...")
        else:
            logger.warning("Nasdaq Data Link no disponible o sin API key")
        
        # Configuración de empresas de acero para ZACKS/FC
        self.steel_companies = {
            'us_steel': {
                'ticker': 'X',
                'name': 'United States Steel Corporation',
                'importance': 'critical'
            },
            'nucor': {
                'ticker': 'NUE',
                'name': 'Nucor Corporation',
                'importance': 'critical'
            },
            'arcelormittal': {
                'ticker': 'MT',
                'name': 'ArcelorMittal',
                'importance': 'critical'
            },
            'cleveland_cliffs': {
                'ticker': 'CLF',
                'name': 'Cleveland-Cliffs Inc',
                'importance': 'high'
            },
            'steel_dynamics': {
                'ticker': 'STLD',
                'name': 'Steel Dynamics Inc',
                'importance': 'high'
            }
        }
        
        # Datasets de series temporales disponibles
        self.timeseries_datasets = {
            'usd_mxn': {
                'code': 'FRED/DEXMXUS',
                'name': 'USD/MXN Exchange Rate',
                'frequency': 'daily',
                'importance': 'critical',
                'category': 'currency'
            },
            'wti_oil': {
                'code': 'FRED/DCOILWTICO',
                'name': 'WTI Crude Oil Prices',
                'frequency': 'daily',
                'importance': 'high',
                'category': 'energy'
            },
            'diesel_prices': {
                'code': 'FRED/DDFUELUSGULF',
                'name': 'US Gulf Coast Diesel Prices',
                'frequency': 'daily',
                'importance': 'medium',
                'category': 'energy'
            },
            'industrial_production': {
                'code': 'FRED/INDPRO',
                'name': 'US Industrial Production Index',
                'frequency': 'monthly',
                'importance': 'high',
                'category': 'economic'
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    async def get_company_fundamentals(
        self, 
        company_key: str,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos fundamentales de empresas de acero usando ZACKS/FC
        """
        if not NASDAQ_LIB_AVAILABLE:
            logger.error("Librería nasdaq-data-link no disponible")
            return None
            
        if company_key not in self.steel_companies:
            logger.error(f"Empresa {company_key} no encontrada")
            return None
        
        company_info = self.steel_companies[company_key]
        ticker = company_info['ticker']
        
        logger.info(f"Obteniendo datos fundamentales para {ticker} - {company_info['name']}")
        
        try:
            # Obtener datos de ZACKS/FC (funciona según el test)
            data = nasdaqdatalink.get_table(
                'ZACKS/FC',
                ticker=ticker,
                paginate=False
            )
            
            if data is not None and not data.empty:
                # Procesar datos fundamentales
                result = self._process_fundamentals(data, company_key, company_info)
                
                if save_raw and result:
                    await self._save_raw_data(result, f"fundamentals_{company_key}")
                
                return result
            else:
                logger.warning(f"Sin datos para {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo fundamentales de {ticker}: {str(e)}")
            return None
    
    def _process_fundamentals(self, df: pd.DataFrame, company_key: str, company_info: dict) -> Dict[str, Any]:
        """Procesar datos fundamentales de ZACKS/FC"""
        try:
            # Seleccionar columnas clave para análisis
            key_columns = [
                'per_end_date',      # Fecha del período
                'tot_revnu',         # Revenue total
                'gross_profit',      # Ganancia bruta
                'oper_income',       # Income operacional
                'net_income_loss',   # Income neto
                'eps_basic_net',     # EPS básico
                'eps_diluted_net',   # EPS diluido
                'tot_asset',         # Assets totales
                'tot_liab',          # Liabilities totales
                'tot_share_holder_equity',  # Equity
                'cash_flow_oper_activity',  # Cash flow operacional
                'ebitda',            # EBITDA
                'tot_curr_asset',    # Current assets
                'tot_curr_liab',     # Current liabilities
                'tot_lterm_debt'     # Long term debt
            ]
            
            # Filtrar columnas disponibles
            available_cols = [col for col in key_columns if col in df.columns]
            df_filtered = df[available_cols].copy()
            
            # Convertir fechas
            if 'per_end_date' in df_filtered.columns:
                df_filtered['fecha'] = pd.to_datetime(df_filtered['per_end_date'])
                df_filtered = df_filtered.sort_values('fecha')
            
            # Calcular métricas derivadas
            metrics = {}
            
            if not df_filtered.empty:
                latest_row = df_filtered.iloc[-1]
                
                # Métricas básicas
                metrics['revenue'] = float(latest_row.get('tot_revnu', 0)) if 'tot_revnu' in latest_row else None
                metrics['gross_profit'] = float(latest_row.get('gross_profit', 0)) if 'gross_profit' in latest_row else None
                metrics['net_income'] = float(latest_row.get('net_income_loss', 0)) if 'net_income_loss' in latest_row else None
                metrics['eps_basic'] = float(latest_row.get('eps_basic_net', 0)) if 'eps_basic_net' in latest_row else None
                metrics['ebitda'] = float(latest_row.get('ebitda', 0)) if 'ebitda' in latest_row else None
                
                # Ratios financieros
                if 'tot_curr_asset' in latest_row and 'tot_curr_liab' in latest_row:
                    curr_assets = float(latest_row['tot_curr_asset'])
                    curr_liab = float(latest_row['tot_curr_liab'])
                    if curr_liab > 0:
                        metrics['current_ratio'] = curr_assets / curr_liab
                
                if 'tot_lterm_debt' in latest_row and 'tot_share_holder_equity' in latest_row:
                    debt = float(latest_row['tot_lterm_debt'])
                    equity = float(latest_row['tot_share_holder_equity'])
                    if equity > 0:
                        metrics['debt_to_equity'] = debt / equity
                
                if 'gross_profit' in latest_row and 'tot_revnu' in latest_row:
                    gross = float(latest_row['gross_profit'])
                    revenue = float(latest_row['tot_revnu'])
                    if revenue > 0:
                        metrics['gross_margin'] = (gross / revenue) * 100
            
            return {
                'company_key': company_key,
                'ticker': company_info['ticker'],
                'name': company_info['name'],
                'data': df_filtered,
                'metrics': metrics,
                'latest_date': df_filtered['fecha'].max() if 'fecha' in df_filtered.columns else None,
                'count': len(df_filtered),
                'importance': company_info['importance'],
                'source': 'nasdaq_data_link'
            }
            
        except Exception as e:
            logger.error(f"Error procesando fundamentales: {str(e)}")
            return None
    
    async def get_timeseries_data(
        self,
        dataset_key: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de series temporales (commodities, tipos de cambio, etc.)
        """
        if not NASDAQ_LIB_AVAILABLE:
            logger.error("Librería nasdaq-data-link no disponible")
            return None
            
        if dataset_key not in self.timeseries_datasets:
            logger.error(f"Dataset {dataset_key} no encontrado")
            return None
        
        dataset_info = self.timeseries_datasets[dataset_key]
        code = dataset_info['code']
        
        logger.info(f"Obteniendo serie temporal {code} - {dataset_info['name']}")
        
        try:
            # Obtener datos usando la API
            data = nasdaqdatalink.get(
                code,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and not data.empty:
                # Procesar serie temporal
                result = self._process_timeseries(data, dataset_key, dataset_info)
                
                if save_raw and result:
                    await self._save_raw_data(result, f"timeseries_{dataset_key}")
                
                return result
            else:
                logger.warning(f"Sin datos para {code}")
                return None
                
        except Exception as e:
            if "Status 403" in str(e):
                logger.error(f"Sin acceso a {code} (requiere suscripción)")
            elif "Status 404" in str(e):
                logger.error(f"Dataset {code} no encontrado")
            else:
                logger.error(f"Error obteniendo {code}: {str(e)}")
            return None
    
    def _process_timeseries(self, data: pd.DataFrame, dataset_key: str, dataset_info: dict) -> Dict[str, Any]:
        """Procesar datos de series temporales"""
        try:
            # El DataFrame ya viene indexado por fecha
            df = data.reset_index()
            
            # Renombrar columnas
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'fecha'})
            elif 'index' in df.columns:
                df = df.rename(columns={'index': 'fecha'})
            
            # Tomar la primera columna numérica como valor
            value_col = None
            for col in df.columns:
                if col != 'fecha' and pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col
                    break
            
            if value_col:
                df = df.rename(columns={value_col: 'valor'})
                df = df[['fecha', 'valor']].dropna()
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.sort_values('fecha')
                
                return {
                    'dataset_key': dataset_key,
                    'code': dataset_info['code'],
                    'name': dataset_info['name'],
                    'data': df,
                    'latest_value': df['valor'].iloc[-1] if not df.empty else None,
                    'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                    'count': len(df),
                    'frequency': dataset_info['frequency'],
                    'importance': dataset_info['importance'],
                    'category': dataset_info['category'],
                    'source': 'nasdaq_data_link'
                }
            else:
                logger.error(f"No se encontró columna de valores en {dataset_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error procesando serie temporal: {str(e)}")
            return None
    
    async def get_all_steel_companies(self, save_raw: bool = True) -> Dict[str, Any]:
        """Obtener datos fundamentales de todas las empresas de acero"""
        logger.info("Recopilando datos fundamentales de empresas de acero...")
        
        results = {}
        
        for company_key in self.steel_companies.keys():
            result = await self.get_company_fundamentals(company_key, save_raw)
            if result:
                results[company_key] = result
                logger.info(f"✅ {company_key}: Datos obtenidos")
            else:
                logger.warning(f"❌ {company_key}: Sin datos")
            
            # Pequeña pausa para no sobrecargar la API
            await asyncio.sleep(0.5)
        
        return {
            'companies_data': results,
            'summary': {
                'total_companies': len(results),
                'successful': len(results),
                'failed': len(self.steel_companies) - len(results),
                'source': 'nasdaq_data_link',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def get_all_timeseries(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """Obtener todas las series temporales configuradas"""
        logger.info("Recopilando series temporales...")
        
        results = {}
        
        for dataset_key in self.timeseries_datasets.keys():
            result = await self.get_timeseries_data(dataset_key, start_date, end_date, save_raw)
            if result:
                results[dataset_key] = result
                logger.info(f"✅ {dataset_key}: {result['count']} puntos")
            else:
                logger.warning(f"❌ {dataset_key}: Sin datos")
            
            await asyncio.sleep(0.5)
        
        return {
            'timeseries_data': results,
            'summary': {
                'total_series': len(results),
                'total_points': sum(r['count'] for r in results.values()),
                'source': 'nasdaq_data_link',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def _save_raw_data(self, data: Dict[str, Any], prefix: str):
        """Guardar datos raw en archivo"""
        try:
            # Crear directorio si no existe
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            # Nombre del archivo con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_{timestamp}.json"
            filepath = os.path.join(raw_dir, filename)
            
            # Convertir DataFrames a dict para JSON
            save_data = data.copy()
            if 'data' in save_data and isinstance(save_data['data'], pd.DataFrame):
                save_data['data'] = save_data['data'].to_dict('records')
            
            # Guardar JSON
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Datos guardados en {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando datos: {str(e)}")

# Función de conveniencia
async def collect_quandl_data(
    api_key: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """Función de conveniencia para recopilar todos los datos de Nasdaq Data Link"""
    async with QuandlCollector(api_key) as collector:
        # Obtener datos fundamentales de empresas
        companies = await collector.get_all_steel_companies(save_raw)
        
        # Obtener series temporales
        timeseries = await collector.get_all_timeseries(start_date, end_date, save_raw)
        
        return {
            'companies': companies,
            'timeseries': timeseries,
            'collection_timestamp': datetime.now().isoformat()
        }