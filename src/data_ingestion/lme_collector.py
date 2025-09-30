"""
LME (London Metal Exchange) Data Collector
Colector para datos de precios de metales del London Metal Exchange
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
import json
import yfinance as yf

logger = logging.getLogger(__name__)

class LMECollector:
    """
    Colector para datos del London Metal Exchange y commodities relacionados
    Nota: LME no tiene API pública gratuita, usaremos proxies como Yahoo Finance
    """
    
    def __init__(self):
        self.session = None
        
        # Símbolos y proxies para datos de metales
        self.metals_config = {
            # 'steel_rebar': {  # COMENTADO: RB=F es gasolina, usamos datos reales de Investing.com
            #     'symbol': 'RB=F',  # Steel Rebar Futures (INCORRECTO - es gasolina)
            #     'name': 'Steel Rebar Futures',
            #     'unit': 'USD/ton',
            #     'importance': 'critical',
            #     'category': 'steel'
            # },
            'iron_ore': {
                'symbol': 'VALE',  # Vale como proxy de mineral de hierro
                'name': 'Iron Ore (Vale Proxy)',
                'unit': 'USD/share',
                'importance': 'critical',
                'category': 'raw_materials'
            },
            'coking_coal': {
                'symbol': 'BTU',  # Peabody Energy como proxy
                'name': 'Coking Coal (BTU Proxy)',
                'unit': 'USD/share',
                'importance': 'high',
                'category': 'raw_materials'
            },
            'copper': {
                'symbol': 'HG=F',  # Copper Futures
                'name': 'Copper Futures',
                'unit': 'USD/lb',
                'importance': 'high',
                'category': 'metals'
            },
            'aluminum': {
                'symbol': 'ALI=F',  # Aluminum Futures
                'name': 'Aluminum Futures',
                'unit': 'USD/lb',
                'importance': 'medium',
                'category': 'metals'
            },
            'nickel': {
                'symbol': 'NI=F',  # Nickel Futures
                'name': 'Nickel Futures',
                'unit': 'USD/lb',
                'importance': 'medium',
                'category': 'metals'
            },
            'zinc': {
                'symbol': 'ZN=F',  # Zinc Futures
                'name': 'Zinc Futures',
                'unit': 'USD/lb',
                'importance': 'medium',
                'category': 'metals'
            },
            'steel_etf': {
                'symbol': 'SLX',  # VanEck Vectors Steel ETF
                'name': 'Steel Industry ETF',
                'unit': 'USD/share',
                'importance': 'high',
                'category': 'steel'
            }
        }
        
        # Empresas siderúrgicas como indicadores
        self.steel_companies = {
            'us_steel': {
                'symbol': 'X',
                'name': 'United States Steel Corporation',
                'importance': 'high'
            },
            'arcelormittal': {
                'symbol': 'MT',
                'name': 'ArcelorMittal',
                'importance': 'high'
            },
            'steel_dynamics': {
                'symbol': 'STLD',
                'name': 'Steel Dynamics Inc',
                'importance': 'medium'
            },
            'cleveland_cliffs': {
                'symbol': 'CLF',
                'name': 'Cleveland-Cliffs Inc',
                'importance': 'medium'
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
    
    async def get_metal_data(
        self, 
        metal_key: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y",
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de un metal específico usando Yahoo Finance
        
        Args:
            metal_key: Clave del metal en metals_config
            start_date: Fecha de inicio
            end_date: Fecha de fin
            period: Período para yfinance (1y, 2y, 5y, etc.)
        """
        if metal_key not in self.metals_config:
            raise ValueError(f"Metal '{metal_key}' no encontrado en configuración")
        
        metal_info = self.metals_config[metal_key]
        symbol = metal_info['symbol']
        
        try:
            logger.info(f"Obteniendo datos de {metal_key} ({symbol})")
            
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
            
            # Procesar datos históricos
            df = hist.reset_index()
            df['fecha'] = pd.to_datetime(df['Date'])
            
            # Normalizar timezone - remover timezone para evitar errores de comparación
            if df['fecha'].dt.tz is not None:
                df['fecha'] = df['fecha'].dt.tz_localize(None)
            
            # Usar precio de cierre como valor principal
            df['valor'] = df['Close']
            
            # Calcular métricas adicionales
            df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            result = {
                'metal_key': metal_key,
                'metal_name': metal_info['name'],
                'symbol': symbol,
                'data': df[['fecha', 'valor', 'Close', 'Volume', 'volatility', 'ma_20', 'ma_50', 'rsi']],
                'latest_value': df['valor'].iloc[-1] if not df.empty else np.nan,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'unit': metal_info['unit'],
                'importance': metal_info['importance'],
                'category': metal_info['category'],
                'source': 'yahoo_finance',
                'metrics': {
                    'current_price': df['Close'].iloc[-1],
                    'change_1d': df['Close'].pct_change().iloc[-1],
                    'change_1w': (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else np.nan,
                    'change_1m': (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) if len(df) >= 22 else np.nan,
                    'volatility_20d': df['volatility'].iloc[-1],
                    'volume_avg': df['Volume'].mean()
                }
            }
            
            # Guardar datos raw si se solicita
            if save_raw and not df.empty:
                await self._save_raw_data(result, metal_key)
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo datos para {metal_key}: {str(e)}")
            return None
    
    async def _save_raw_data(self, data_result: Dict[str, Any], metal_key: str):
        """Guardar datos raw en archivo CSV"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # Mapear metales a nombres descriptivos
            name_mapping = {
                'copper': 'Cobre',
                'aluminum': 'Aluminio',
                'zinc': 'Zinc',
                'lead': 'Plomo',
                'nickel': 'Niquel',
                'tin': 'Estano',
                'gold': 'Oro',
                'silver': 'Plata'
            }
            
            variable_name = name_mapping.get(metal_key, metal_key)
            filename = f"LME_{variable_name}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if 'data' in data_result and data_result['data'] is not None:
                df = data_result['data']
                if not df.empty:
                    df.to_csv(filepath, index=False)
                    logger.info(f"Datos raw guardados: {filepath}")
                    
                    # Guardar metadata
                    metadata = {
                        'metal_key': metal_key,
                        'symbol': data_result.get('symbol', ''),
                        'source': 'lme',
                        'collection_timestamp': datetime.now().isoformat(),
                        'count': len(df),
                        'latest_date': str(data_result.get('latest_date', '')),
                        'latest_value': data_result.get('latest_value', None)
                    }
                    
                    metadata_file = filepath.replace('.csv', '_metadata.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                        
        except Exception as e:
            logger.error(f"Error guardando datos raw LME: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcular RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def get_steel_companies_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> Dict[str, Any]:
        """Obtener datos de empresas siderúrgicas"""
        
        logger.info("Obteniendo datos de empresas siderúrgicas...")
        
        results = {}
        
        for company_key, company_info in self.steel_companies.items():
            try:
                symbol = company_info['symbol']
                ticker = yf.Ticker(symbol)
                
                # Si no se proporciona end_date, usar la fecha actual
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                if start_date:
                    hist = ticker.history(start=start_date, end=end_date)
                else:
                    # Si no hay start_date, usar period pero con end_date actual
                    hist = ticker.history(period=period, end=end_date)
                
                if not hist.empty:
                    df = hist.reset_index()
                    df['fecha'] = pd.to_datetime(df['Date'])
                    
                    # Normalizar timezone
                    if df['fecha'].dt.tz is not None:
                        df['fecha'] = df['fecha'].dt.tz_localize(None)
                    
                    df['valor'] = df['Close']
                    
                    results[company_key] = {
                        'company_key': company_key,
                        'company_name': company_info['name'],
                        'symbol': symbol,
                        'data': df[['fecha', 'valor', 'Close', 'Volume']],
                        'latest_value': df['Close'].iloc[-1],
                        'latest_date': df['fecha'].iloc[-1],
                        'count': len(df),
                        'importance': company_info['importance'],
                        'source': 'yahoo_finance'
                    }
                else:
                    logger.warning(f"No hay datos para {company_key}")
                    
            except Exception as e:
                logger.error(f"Error obteniendo datos de {company_key}: {str(e)}")
                continue
        
        return results
    
    async def get_all_metals_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> Dict[str, Any]:
        """
        Obtener datos de todos los metales configurados
        """
        logger.info("Recopilando datos de todos los metales...")
        
        results = {}
        
        # Procesar metales en paralelo
        tasks = []
        for metal_key in self.metals_config.keys():
            task = self.get_metal_data(metal_key, start_date, end_date, period)
            tasks.append(task)
        
        metal_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(metal_results):
            metal_key = list(self.metals_config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo metal {metal_key}: {str(result)}")
                # No agregar metales que fallen - solo datos reales
                continue
            elif result is not None:
                results[metal_key] = result
            else:
                logger.warning(f"Metal {metal_key} no devolvió datos - omitido")
        
        # Obtener datos de empresas siderúrgicas
        companies_data = await self.get_steel_companies_data(start_date, end_date, period)
        
        # Calcular correlaciones entre metales
        correlations = self._calculate_metal_correlations(results)
        
        summary = {
            'total_metals': len(results),
            'total_companies': len(companies_data),
            'total_data_points': sum(r['count'] for r in results.values()),
            'api_sources': sum(1 for r in results.values() if r['source'] == 'yahoo_finance'),
            'dummy_sources': sum(1 for r in results.values() if r['source'] == 'dummy_data'),
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date,
                'period': period
            },
            'correlations': correlations
        }
        
        return {
            'metals_data': results,
            'companies_data': companies_data,
            'summary': summary
        }
    
    def _calculate_metal_correlations(self, metals_data: Dict[str, Any]) -> Dict[str, float]:
        """Calcular correlaciones entre precios de metales"""
        correlations = {}
        
        try:
            # Obtener precios de metales críticos
            steel_prices = metals_data.get('steel_rebar', {}).get('data', pd.DataFrame())
            iron_prices = metals_data.get('iron_ore', {}).get('data', pd.DataFrame())
            copper_prices = metals_data.get('copper', {}).get('data', pd.DataFrame())
            
            if not steel_prices.empty and not iron_prices.empty:
                # Alinear fechas
                steel_values = steel_prices.set_index('fecha')['valor']
                iron_values = iron_prices.set_index('fecha')['valor']
                
                aligned_data = pd.concat([steel_values, iron_values], axis=1, join='inner')
                if len(aligned_data) > 10:
                    correlations['steel_iron'] = aligned_data.corr().iloc[0, 1]
            
            if not steel_prices.empty and not copper_prices.empty:
                steel_values = steel_prices.set_index('fecha')['valor']
                copper_values = copper_prices.set_index('fecha')['valor']
                
                aligned_data = pd.concat([steel_values, copper_values], axis=1, join='inner')
                if len(aligned_data) > 10:
                    correlations['steel_copper'] = aligned_data.corr().iloc[0, 1]
                    
        except Exception as e:
            logger.warning(f"Error calculando correlaciones: {str(e)}")
        
        return correlations
    
    def get_metals_info(self) -> Dict[str, Any]:
        """Información sobre metales disponibles"""
        categories = {}
        for key, info in self.metals_config.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        return {
            'available_metals': self.metals_config,
            'steel_companies': self.steel_companies,
            'total_metals': len(self.metals_config),
            'categories': categories,
            'critical_importance': [k for k, v in self.metals_config.items() if v['importance'] == 'critical'],
            'high_importance': [k for k, v in self.metals_config.items() if v['importance'] == 'high']
        }

# Función de conveniencia
async def collect_lme_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y"
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos de metales
    """
    async with LMECollector() as collector:
        return await collector.get_all_metals_data(start_date, end_date, period)
