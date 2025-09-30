"""
Yahoo Finance Data Collector
Colector para datos financieros y económicos adicionales via Yahoo Finance
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import yfinance as yf
import os
import json

logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """
    Colector para datos financieros complementarios via Yahoo Finance
    """
    
    def __init__(self):
        # Índices y ETFs relevantes para predicción de precios de acero
        self.financial_instruments = {
            'oil_wti': {
                'symbol': 'CL=F',
                'name': 'WTI Crude Oil Futures',
                'category': 'energy',
                'importance': 'high',
                'unit': 'USD/barrel'
            },
            'oil_brent': {
                'symbol': 'BZ=F',
                'name': 'Brent Crude Oil Futures',
                'category': 'energy',
                'importance': 'high',
                'unit': 'USD/barrel'
            },
            'natural_gas': {
                'symbol': 'NG=F',
                'name': 'Natural Gas Futures',
                'category': 'energy',
                'importance': 'high',
                'unit': 'USD/MMBtu'
            },
            'dxy_index': {
                'symbol': 'DX-Y.NYB',
                'name': 'US Dollar Index',
                'category': 'currency',
                'importance': 'critical',
                'unit': 'Index'
            },
            'vix': {
                'symbol': '^VIX',
                'name': 'Volatility Index',
                'category': 'volatility',
                'importance': 'medium',
                'unit': 'Index'
            },
            'sp500': {
                'symbol': '^GSPC',
                'name': 'S&P 500 Index',
                'category': 'equity',
                'importance': 'medium',
                'unit': 'Index'
            },
            'treasury_10y': {
                'symbol': '^TNX',
                'name': '10-Year Treasury Yield',
                'category': 'bonds',
                'importance': 'medium',
                'unit': 'Yield %'
            },
            'commodities_etf': {
                'symbol': 'DJP',
                'name': 'iPath Bloomberg Commodity ETF',
                'category': 'commodities',
                'importance': 'high',
                'unit': 'USD/share'
            },
            'materials_etf': {
                'symbol': 'XLB',
                'name': 'Materials Select Sector SPDR ETF',
                'category': 'materials',
                'importance': 'high',
                'unit': 'USD/share'
            },
            'emerging_markets': {
                'symbol': 'EEM',
                'name': 'iShares MSCI Emerging Markets ETF',
                'category': 'equity',
                'importance': 'medium',
                'unit': 'USD/share'
            },
            'china_etf': {
                'symbol': 'FXI',
                'name': 'iShares China Large-Cap ETF',
                'category': 'equity',
                'importance': 'high',
                'unit': 'USD/share'
            },
            'infrastructure_etf': {
                'symbol': 'IFRA',
                'name': 'iShares U.S. Infrastructure ETF',
                'category': 'infrastructure',
                'importance': 'high',
                'unit': 'USD/share'
            }
        }
        
        # Criptomonedas como indicadores de riesgo
        self.crypto_instruments = {
            'bitcoin': {
                'symbol': 'BTC-USD',
                'name': 'Bitcoin',
                'category': 'crypto',
                'importance': 'low',
                'unit': 'USD'
            },
            'ethereum': {
                'symbol': 'ETH-USD',
                'name': 'Ethereum',
                'category': 'crypto',
                'importance': 'low',
                'unit': 'USD'
            }
        }
    
    async def get_instrument_data(
        self, 
        instrument_key: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y",
        include_crypto: bool = False,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de un instrumento financiero específico
        
        Args:
            instrument_key: Clave del instrumento
            start_date: Fecha de inicio
            end_date: Fecha de fin
            period: Período para yfinance
            include_crypto: Incluir criptomonedas
        """
        # Seleccionar configuración apropiada
        config = self.financial_instruments.copy()
        if include_crypto:
            config.update(self.crypto_instruments)
        
        if instrument_key not in config:
            raise ValueError(f"Instrumento '{instrument_key}' no encontrado")
        
        instrument_info = config[instrument_key]
        symbol = instrument_info['symbol']
        
        try:
            logger.info(f"Obteniendo datos de {instrument_key} ({symbol})")
            
            # Usar yf.download para obtener datos (más robusto)
            # Si no se proporciona end_date, usar la fecha actual
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date:
                hist = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            else:
                # Si no hay start_date, usar período pero con end_date actual
                hist = yf.download(symbol, period=period, end=end_date, progress=False, auto_adjust=False)
            
            if hist.empty:
                logger.warning(f"No hay datos disponibles para {symbol}")
                return None
            
            # Procesar datos - el índice ya es la fecha
            df = hist.reset_index()
            # Normalizar timezone para evitar errores
            df['fecha'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df['valor'] = df['Close']
            
            # Calcular indicadores técnicos
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['ma_5'] = df['Close'].rolling(window=5).mean()
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Métricas de performance - asegurar valores escalares
            close_first = float(df['Close'].iloc[0]) if len(df) > 0 else 1.0
            close_last = float(df['Close'].iloc[-1]) if len(df) > 0 else 1.0
            total_return = (close_last / close_first - 1) if len(df) > 1 else 0
            annualized_return = ((1 + total_return) ** (252 / len(df)) - 1) if len(df) > 1 else 0
            max_drawdown = float(self._calculate_max_drawdown(df['Close']))
            
            # Asegurar que las métricas son valores escalares
            volatility_value = float(df['volatility'].iloc[-1]) if not df.empty and pd.notna(df['volatility'].iloc[-1]) else 0.0
            rsi_value = float(df['rsi'].iloc[-1]) if not df.empty and pd.notna(df['rsi'].iloc[-1]) else 0.0
            
            result = {
                'instrument_key': instrument_key,
                'instrument_name': instrument_info['name'],
                'symbol': symbol,
                'data': df[['fecha', 'valor', 'Close', 'Volume', 'returns', 'volatility', 
                          'ma_5', 'ma_20', 'ma_50', 'rsi', 'bollinger_upper', 'bollinger_lower']],
                'latest_value': df['Close'].iloc[-1] if not df.empty else np.nan,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'category': instrument_info['category'],
                'importance': instrument_info['importance'],
                'unit': instrument_info['unit'],
                'source': 'yahoo_finance',
                'performance_metrics': {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility_value,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': annualized_return / (volatility_value + 1e-6) if volatility_value > 0 else 0.0,
                    'current_rsi': rsi_value
                }
            }
            
            # Guardar datos raw si se solicita
            if save_raw and not df.empty:
                await self._save_raw_data(result, instrument_key)
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo datos para {instrument_key}: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2):
        """Calcular Bandas de Bollinger"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calcular máximo drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    async def _save_raw_data(self, data_result: Dict[str, Any], instrument_key: str):
        """Guardar datos raw en archivo CSV"""
        try:
            raw_dir = 'data/raw'
            os.makedirs(raw_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # Mapear instrumentos a nombres descriptivos
            name_mapping = {
                'copper_futures': 'Cobre_Futuros',
                'aluminum_futures': 'Aluminio_Futuros',
                'gold_futures': 'Oro_Futuros',
                'silver_futures': 'Plata_Futuros',
                'oil_wti': 'Petroleo_WTI',
                'oil_brent': 'Petroleo_Brent',
                'natural_gas': 'GasNatural',
                'sp500': 'SP500',
                'nasdaq': 'NASDAQ',
                'dxy': 'DolarIndex',
                'vix': 'VIX_Volatilidad',
                'us_steel': 'USSteel',
                'nucor': 'Nucor',
                'arcelormittal': 'ArcelorMittal',
                'cleveland_cliffs': 'ClevelandCliffs',
                'steel_dynamics': 'SteelDynamics'
            }
            
            variable_name = name_mapping.get(instrument_key, instrument_key)
            filename = f"YahooFinance_{variable_name}_{timestamp}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            if 'data' in data_result and data_result['data'] is not None:
                df = data_result['data']
                if not df.empty:
                    df.to_csv(filepath, index=False)
                    logger.info(f"Datos raw guardados: {filepath}")
                    
                    # Guardar metadata
                    metadata = {
                        'instrument_key': instrument_key,
                        'symbol': data_result.get('symbol', ''),
                        'source': 'yahoo_finance',
                        'collection_timestamp': datetime.now().isoformat(),
                        'count': len(df),
                        'latest_date': str(data_result.get('latest_date', '')),
                        'latest_value': data_result.get('latest_value', None)
                    }
                    
                    metadata_file = filepath.replace('.csv', '_metadata.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                        
        except Exception as e:
            logger.error(f"Error guardando datos raw Yahoo: {str(e)}")

    async def get_all_financial_data(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y",
        include_crypto: bool = False,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de todos los instrumentos financieros
        """
        logger.info("Recopilando datos financieros de Yahoo Finance...")
        
        config = self.financial_instruments.copy()
        if include_crypto:
            config.update(self.crypto_instruments)
        
        results = {}
        
        # Procesar instrumentos en paralelo
        tasks = []
        for instrument_key in config.keys():
            task = self.get_instrument_data(instrument_key, start_date, end_date, period, include_crypto, save_raw)
            tasks.append(task)
        
        instrument_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(instrument_results):
            instrument_key = list(config.keys())[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error obteniendo instrumento {instrument_key}: {str(result)}")
                continue
            else:
                results[instrument_key] = result
        
        # Análisis de correlaciones
        correlations = self._calculate_correlations(results)
        
        # Estadísticas por categoría
        categories_stats = {}
        for result in results.values():
            category = result['category']
            if category not in categories_stats:
                categories_stats[category] = {
                    'count': 0,
                    'avg_return': 0,
                    'avg_volatility': 0
                }
            categories_stats[category]['count'] += 1
            if 'performance_metrics' in result:
                categories_stats[category]['avg_return'] += result['performance_metrics'].get('annualized_return', 0)
                categories_stats[category]['avg_volatility'] += result['performance_metrics'].get('volatility', 0)
        
        # Promediar estadísticas
        for category in categories_stats:
            count = categories_stats[category]['count']
            categories_stats[category]['avg_return'] /= count
            categories_stats[category]['avg_volatility'] /= count
        
        summary = {
            'total_instruments': len(results),
            'total_data_points': sum(r['count'] for r in results.values()),
            'api_sources': sum(1 for r in results.values() if r['source'] == 'yahoo_finance'),
            'dummy_sources': sum(1 for r in results.values() if r['source'] == 'dummy_data'),
            'categories_stats': categories_stats,
            'correlations': correlations,
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date,
                'period': period
            }
        }
        
        return {
            'financial_data': results,
            'summary': summary
        }
    
    def _calculate_correlations(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """Calcular correlaciones entre instrumentos clave"""
        correlations = {}
        
        try:
            # Obtener datos de instrumentos clave
            oil_data = financial_data.get('oil_wti', {}).get('data', pd.DataFrame())
            dxy_data = financial_data.get('dxy_index', {}).get('data', pd.DataFrame())
            materials_data = financial_data.get('materials_etf', {}).get('data', pd.DataFrame())
            
            if not oil_data.empty and not materials_data.empty:
                oil_values = oil_data.set_index('fecha')['valor']
                materials_values = materials_data.set_index('fecha')['valor']
                
                aligned_data = pd.concat([oil_values, materials_values], axis=1, join='inner')
                if len(aligned_data) > 10:
                    correlations['oil_materials'] = aligned_data.corr().iloc[0, 1]
            
            if not dxy_data.empty and not materials_data.empty:
                dxy_values = dxy_data.set_index('fecha')['valor']
                materials_values = materials_data.set_index('fecha')['valor']
                
                aligned_data = pd.concat([dxy_values, materials_values], axis=1, join='inner')
                if len(aligned_data) > 10:
                    correlations['dxy_materials'] = aligned_data.corr().iloc[0, 1]
                    
        except Exception as e:
            logger.warning(f"Error calculando correlaciones financieras: {str(e)}")
        
        return correlations
    
    def get_instruments_info(self) -> Dict[str, Any]:
        """Información sobre instrumentos disponibles"""
        all_instruments = {**self.financial_instruments, **self.crypto_instruments}
        
        categories = {}
        for key, info in all_instruments.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        return {
            'financial_instruments': self.financial_instruments,
            'crypto_instruments': self.crypto_instruments,
            'total_instruments': len(all_instruments),
            'categories': categories,
            'critical_importance': [k for k, v in all_instruments.items() if v['importance'] == 'critical'],
            'high_importance': [k for k, v in all_instruments.items() if v['importance'] == 'high']
        }

# Función de conveniencia
async def collect_yahoo_finance_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y",
    include_crypto: bool = False
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos financieros
    """
    collector = YahooFinanceCollector()
    return await collector.get_all_financial_data(start_date, end_date, period, include_crypto)
