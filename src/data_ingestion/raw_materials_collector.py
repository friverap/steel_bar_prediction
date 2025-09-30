"""
Colector de Materias Primas del Acero
Mineral de Hierro y Carbón de Coque - Insumos críticos para producción de acero
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
from .utils import save_raw_data, get_variable_name

logger = logging.getLogger(__name__)

class RawMaterialsCollector:
    """
    Colector para datos de materias primas del acero:
    - Mineral de Hierro (Iron Ore)
    - Carbón de Coque (Coking Coal)
    """
    
    def __init__(self):
        self.session = None
        
        # Configuración de empresas mineras como proxy para precios
        self.mining_companies = {
            'vale': {
                'symbol': 'VALE',
                'name': 'Vale S.A.',
                'description': 'Mayor productor mundial de mineral de hierro',
                'commodity': 'iron_ore',
                'importance': 'critical'
            },
            'rio_tinto': {
                'symbol': 'RIO',
                'name': 'Rio Tinto Group',
                'description': 'Segundo mayor productor de mineral de hierro',
                'commodity': 'iron_ore',
                'importance': 'critical'
            },
            'bhp': {
                'symbol': 'BHP',
                'name': 'BHP Group',
                'description': 'Productor de mineral de hierro y carbón metalúrgico',
                'commodity': 'iron_ore_coal',
                'importance': 'critical'
            },
            'anglo_american': {
                'symbol': 'AAL.L',
                'name': 'Anglo American',
                'description': 'Productor diversificado - hierro y carbón',
                'commodity': 'iron_ore_coal',
                'importance': 'high'
            },
            'teck': {
                'symbol': 'TECK',
                'name': 'Teck Resources',
                'description': 'Mayor productor de carbón metalúrgico en América',
                'commodity': 'coking_coal',
                'importance': 'high'
            },
        }
        
        # ETFs relacionados con materias primas
        self.commodity_etfs = {
            'steel_etf': {
                'symbol': 'SLX',
                'name': 'VanEck Steel ETF',
                'description': 'ETF del sector acero',
                'importance': 'high'
            },
            'metals_miners': {
                'symbol': 'XME',
                'name': 'SPDR S&P Metals & Mining ETF',
                'description': 'ETF de minería y metales',
                'importance': 'high'
            },
            'materials': {
                'symbol': 'XLB',
                'name': 'Materials Select Sector SPDR',
                'description': 'ETF del sector materiales',
                'importance': 'medium'
            }
        }
        
        # Índices de referencia para commodities
        self.commodity_indices = {
            'dxy': {
                'symbol': 'DX-Y.NYB',
                'name': 'US Dollar Index',
                'description': 'Índice del dólar (inverso a commodities)',
                'importance': 'high'
            },
            'commodity_index': {
                'symbol': 'DJP',
                'name': 'iPath Bloomberg Commodity Index',
                'description': 'Índice general de commodities',
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
    
    async def get_mining_company_data(
        self,
        company_key: str,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de empresas mineras como proxy para precios de commodities
        """
        if company_key not in self.mining_companies:
            logger.error(f"Empresa {company_key} no encontrada")
            return None
        
        company_info = self.mining_companies[company_key]
        symbol = company_info['symbol']
        
        logger.info(f"Obteniendo datos de {symbol} - {company_info['name']}")
        
        try:
            # Usar yfinance para obtener datos
            ticker = yf.Ticker(symbol)
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Obtener datos históricos
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"Sin datos para {symbol}")
                return None
            
            # Procesar datos
            df = pd.DataFrame({
                'fecha': hist.index,
                'precio_cierre': hist['Close'],
                'precio_apertura': hist['Open'],
                'precio_max': hist['High'],
                'precio_min': hist['Low'],
                'volumen': hist['Volume']
            }).reset_index(drop=True)
            
            # Calcular indicadores técnicos
            df['sma_20'] = df['precio_cierre'].rolling(window=20).mean()
            df['sma_50'] = df['precio_cierre'].rolling(window=50).mean()
            df['volatilidad_20'] = df['precio_cierre'].pct_change().rolling(window=20).std()
            
            # Calcular correlación con el commodity
            correlation_factor = self._calculate_commodity_correlation(company_info['commodity'])
            
            result = {
                'company_key': company_key,
                'symbol': symbol,
                'name': company_info['name'],
                'commodity': company_info['commodity'],
                'data': df,
                'latest_price': df['precio_cierre'].iloc[-1] if not df.empty else None,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df),
                'price_change_pct': ((df['precio_cierre'].iloc[-1] / df['precio_cierre'].iloc[0]) - 1) * 100 if len(df) > 1 else 0,
                'avg_volume': df['volumen'].mean(),
                'volatility': df['volatilidad_20'].iloc[-1] if not df['volatilidad_20'].isna().all() else None,
                'correlation_factor': correlation_factor,
                'importance': company_info['importance']
            }
            
            # Guardar datos raw si se solicita
            if save_raw and result['data'] is not None and not result['data'].empty:
                variable_name = get_variable_name('raw_materials', company_key)
                save_raw_data(
                    data=result['data'],
                    source='RawMaterials',
                    variable=variable_name,
                    metadata={
                        'symbol': symbol,
                        'commodity': company_info['commodity'],
                        'latest_price': result['latest_price'],
                        'price_change_pct': result['price_change_pct']
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de {symbol}: {str(e)}")
            return None
    
    
    def _calculate_commodity_correlation(self, commodity_type: str) -> float:
        """
        Calcular factor de correlación estimado entre empresa y commodity
        """
        correlation_factors = {
            'iron_ore': 0.85,  # Alta correlación con mineral de hierro
            'coking_coal': 0.75,  # Alta correlación con carbón de coque
            'iron_ore_coal': 0.80,  # Correlación mixta
        }
        return correlation_factors.get(commodity_type, 0.70)
    
    async def get_all_mining_companies(
        self,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de todas las empresas mineras
        """
        logger.info("Recopilando datos de empresas mineras...")
        
        results = {}
        iron_ore_companies = []
        coal_companies = []
        
        for company_key in self.mining_companies.keys():
            result = await self.get_mining_company_data(company_key, start_date, end_date, save_raw)
            
            if result:
                results[company_key] = result
                
                # Clasificar por commodity
                commodity = result['commodity']
                if 'iron_ore' in commodity:
                    iron_ore_companies.append(company_key)
                if 'coal' in commodity:
                    coal_companies.append(company_key)
                
                logger.info(f"✅ {company_key}: {result['count']} datos")
            else:
                logger.warning(f"❌ {company_key}: Sin datos")
            
            # Pequeña pausa para no sobrecargar
            await asyncio.sleep(0.5)
        
        # Calcular índices proxy para commodities
        iron_ore_proxy = self._calculate_commodity_proxy(results, iron_ore_companies)
        coal_proxy = self._calculate_commodity_proxy(results, coal_companies)
        
        return {
            'companies_data': results,
            'iron_ore_proxy': iron_ore_proxy,
            'coal_proxy': coal_proxy,
            'summary': {
                'total_companies': len(results),
                'iron_ore_companies': len(iron_ore_companies),
                'coal_companies': len(coal_companies),
                'data_points': sum(r['count'] for r in results.values()),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_commodity_proxy(
        self,
        companies_data: Dict[str, Any],
        company_keys: List[str]
    ) -> pd.DataFrame:
        """
        Calcular índice proxy para un commodity basado en empresas relacionadas
        """
        if not company_keys:
            return pd.DataFrame()
        
        # Combinar datos de empresas relacionadas
        dfs = []
        weights = []
        
        for key in company_keys:
            if key in companies_data:
                data = companies_data[key]
                df = data['data'][['fecha', 'precio_cierre']].copy()
                df = df.rename(columns={'precio_cierre': f'precio_{key}'})
                df = df.set_index('fecha')
                dfs.append(df)
                
                # Peso basado en importancia
                weight = 1.0 if data['importance'] == 'critical' else 0.7
                weights.append(weight)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combinar todos los DataFrames
        combined = pd.concat(dfs, axis=1, join='outer')
        
        # Calcular índice ponderado
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        proxy_values = []
        for idx, row in combined.iterrows():
            weighted_sum = 0
            valid_weights = 0
            
            for i, col in enumerate(combined.columns):
                if not pd.isna(row[col]):
                    weighted_sum += row[col] * normalized_weights[i]
                    valid_weights += normalized_weights[i]
            
            if valid_weights > 0:
                proxy_values.append(weighted_sum / valid_weights)
            else:
                proxy_values.append(np.nan)
        
        proxy_df = pd.DataFrame({
            'fecha': combined.index,
            'valor_proxy': proxy_values
        }).reset_index(drop=True)
        
        # Normalizar a base 100
        if not proxy_df.empty and not proxy_df['valor_proxy'].isna().all():
            first_valid = proxy_df['valor_proxy'].first_valid_index()
            if first_valid is not None:
                base_value = proxy_df.loc[first_valid, 'valor_proxy']
                proxy_df['indice'] = (proxy_df['valor_proxy'] / base_value) * 100
        
        return proxy_df
    
    async def get_etf_data(
        self,
        etf_key: str,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        save_raw: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener datos de ETFs relacionados con materias primas
        """
        if etf_key not in self.commodity_etfs:
            logger.error(f"ETF {etf_key} no encontrado")
            return None
        
        etf_info = self.commodity_etfs[etf_key]
        symbol = etf_info['symbol']
        
        logger.info(f"Obteniendo datos de ETF {symbol} - {etf_info['name']}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            df = pd.DataFrame({
                'fecha': hist.index,
                'valor': hist['Close'],
                'volumen': hist['Volume']
            }).reset_index(drop=True)
            
            result = {
                'etf_key': etf_key,
                'symbol': symbol,
                'name': etf_info['name'],
                'data': df,
                'latest_value': df['valor'].iloc[-1] if not df.empty else None,
                'latest_date': df['fecha'].iloc[-1] if not df.empty else None,
                'count': len(df)
            }
            
            # Guardar datos raw si se solicita
            if save_raw and result['data'] is not None and not result['data'].empty:
                # Mapeo simplificado para ETFs
                etf_names = {
                    'steel_etf': 'slx',
                    'metals_miners': 'xme', 
                    'materials': 'xlb'
                }
                variable_name = get_variable_name('raw_materials', etf_names.get(etf_key, etf_key))
                save_raw_data(
                    data=result['data'],
                    source='RawMaterials',
                    variable=variable_name,
                    metadata={
                        'symbol': symbol,
                        'name': etf_info['name'],
                        'latest_value': result['latest_value']
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo ETF {symbol}: {str(e)}")
            return None


# Función de conveniencia
async def collect_raw_materials_data(
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia para recopilar datos de materias primas
    """
    async with RawMaterialsCollector() as collector:
        # Obtener datos de empresas mineras
        mining_data = await collector.get_all_mining_companies(start_date, end_date, save_raw)
        
        # Obtener ETFs
        etfs = {}
        for etf_key in collector.commodity_etfs.keys():
            etf_data = await collector.get_etf_data(etf_key, start_date, end_date, save_raw)
            if etf_data:
                etfs[etf_key] = etf_data
        
        return {
            'mining_companies': mining_data,
            'etfs': etfs,
            'iron_ore_proxy': mining_data.get('iron_ore_proxy'),
            'coal_proxy': mining_data.get('coal_proxy'),
            'collection_timestamp': datetime.now().isoformat()
        }
