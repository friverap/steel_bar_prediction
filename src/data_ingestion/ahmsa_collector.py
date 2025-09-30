"""
AHMSA Data Collector
Colector para datos de empresas siderúrgicas mexicanas (AHMSA, Ternium, DeAcero)
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
import yfinance as yf

logger = logging.getLogger(__name__)

class AHMSACollector:
    """
    Colector para datos de empresas siderúrgicas mexicanas
    """
    
    def __init__(self):
        self.session = None
        
        # Empresas siderúrgicas mexicanas y sus proxies financieros
        self.companies_config = {
            'ahmsa': {
                'symbol': 'AHMSA.MX',  # NOTA: AHMSA cesó operaciones en 2023
                'name': 'Altos Hornos de México (Cesó operaciones 2023)',
                'website': 'https://www.ahmsa.com/',
                'importance': 'low',  # Reducida por cierre
                'category': 'steel_company',
                'country': 'mexico',
                'status': 'inactive'
            },
            'ternium_mexico': {
                'symbol': 'TX',  # Ternium cotiza en NYSE
                'name': 'Ternium México',
                'website': 'https://mx.ternium.com/',
                'importance': 'critical',
                'category': 'steel_company',
                'country': 'mexico'
            },
            'deacero': {
                'symbol': None,  # Empresa privada, no cotiza
                'name': 'DeAcero',
                'website': 'https://www.deacero.com/',
                'importance': 'critical',
                'category': 'steel_company',
                'country': 'mexico'
            },
            # Proxies internacionales
            'arcelormittal': {
                'symbol': 'MT',
                'name': 'ArcelorMittal',
                'importance': 'high',
                'category': 'steel_company',
                'country': 'global'
            },
            'nucor': {
                'symbol': 'NUE',
                'name': 'Nucor Corporation',
                'importance': 'medium',
                'category': 'steel_company',
                'country': 'usa'
            },
            # ETFs como empresas para simplificar
            'steel_etf': {
                'symbol': 'SLX',  # VanEck Vectors Steel ETF
                'name': 'Steel Industry ETF',
                'importance': 'high',
                'category': 'industry_index',
                'country': 'usa'
            },
            'materials_etf': {
                'symbol': 'XLB',  # Materials Select Sector SPDR ETF
                'name': 'Materials Sector ETF',
                'importance': 'high',
                'category': 'industry_index',
                'country': 'usa'
            },
            'emerging_markets_etf': {
                'symbol': 'EEM',  # iShares MSCI Emerging Markets ETF
                'name': 'Emerging Markets ETF',
                'importance': 'medium',
                'category': 'market_index',
                'country': 'usa'
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
    
    async def get_company_data(
        self, 
        company_key: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y",
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos financieros de empresa siderúrgica
        """
        if company_key not in self.companies_config:
            raise ValueError(f"Empresa/Indicador '{company_key}' no encontrada en configuración")
        
        company_info = self.companies_config[company_key]
        symbol = company_info.get('symbol')
        
        logger.info(f"Obteniendo datos de empresa {company_key}")
        
        if not symbol:
            logger.info(f"Empresa {company_key} no cotiza públicamente, generando datos dummy")
            return None
        
        try:
            # Usar yfinance para obtener datos
            ticker = yf.Ticker(symbol)
            
            # Si no se proporciona end_date, usar la fecha actual
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date:
                hist = ticker.history(start=start_date, end=end_date)
            else:
                # Si no hay start_date, usar period pero con end_date actual
                hist = ticker.history(period=period, end=end_date)
            
            if hist.empty:
                logger.warning(f"No hay datos disponibles para {symbol}")
                return None
            
            # Procesar datos
            df = hist.reset_index()
            df['fecha'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remover timezone
            df['valor'] = df['Close']
            
            # Calcular métricas financieras
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['market_cap_proxy'] = df['Close'] * df['Volume']  # Proxy simple
            
            result = {
                'company_key': company_key,
                'company_name': company_info['name'],
                'symbol': symbol,
                'data': df[['fecha', 'valor', 'Close', 'Volume', 'returns', 'volatility', 'market_cap_proxy']],
                'latest_value': df['Close'].iloc[-1] if not df.empty else np.nan,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'importance': company_info['importance'],
                'category': company_info['category'],
                'country': company_info['country'],
                'source': 'yahoo_finance',
                'financial_metrics': {
                    'current_price': df['Close'].iloc[-1],
                    'change_1d': df['returns'].iloc[-1],
                    'volatility_20d': df['volatility'].iloc[-1],
                    'avg_volume': df['Volume'].mean()
                }
            }
            
            # Guardar datos raw si se solicita
            if save_raw:
                await self._save_raw_data(result, company_key)
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de empresa {company_key}: {str(e)}")
            return None

    async def _save_raw_data(self, company_result: Dict[str, Any], company_key: str):
        """Guardar datos raw en archivo"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ahmsa_{company_key}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if company_result['data'] is not None and not company_result['data'].empty:
                company_result['data'].to_csv(filepath, index=False)
                logger.info(f"Datos raw AHMSA guardados: {filepath}")
                
                metadata = {
                    'company_key': company_key,
                    'company_name': company_result['company_name'],
                    'symbol': company_result['symbol'],
                    'source': company_result['source'],
                    'collection_timestamp': datetime.now().isoformat(),
                    'count': company_result['count'],
                    'latest_date': company_result['latest_date'].isoformat() if company_result['latest_date'] else None,
                    'latest_value': company_result['latest_value'],
                    'importance': company_result['importance'],
                    'category': company_result['category'],
                    'country': company_result['country'],
                    'financial_metrics': company_result['financial_metrics']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos raw AHMSA: {str(e)}")
    
    async def get_all_companies(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y",
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """Obtener datos de todas las empresas configuradas"""
        logger.info("Recopilando datos de empresas siderúrgicas...")
        
        results = {}
        
        tasks = []
        for company_key in self.companies_config.keys():
            task = self.get_company_data(company_key, start_date, end_date, period, save_raw)
            tasks.append(task)
        
        item_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(item_results):
            company_key = list(self.companies_config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo empresa {company_key}: {str(result)}")
                # No agregar empresas que fallen - solo datos reales
                continue
            elif result is not None:
                results[company_key] = result
            else:
                logger.warning(f"Empresa {company_key} no devolvió datos - omitida")
        
        # Filtrar resultados None
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        total_points = sum(r['count'] for r in valid_results.values())
        api_sources = sum(1 for r in valid_results.values() if r['source'] == 'yahoo_finance')
        
        summary = {
            'total_companies': len(valid_results),
            'total_data_points': total_points,
            'api_sources': api_sources,
            'failed_companies': len(results) - len(valid_results),
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return {
            'companies_data': valid_results,
            'summary': summary
        }

# Función de conveniencia
async def collect_ahmsa_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y",
    save_raw: bool = True
) -> Dict[str, Any]:
    """Función de conveniencia para recopilar datos de empresas siderúrgicas"""
    async with AHMSACollector() as collector:
        return await collector.get_all_companies(start_date, end_date, period, save_raw)
