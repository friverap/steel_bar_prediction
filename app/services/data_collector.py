"""
Data Collection Service
Collects data from various external APIs and sources
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
import logging
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Service for collecting data from external sources
    """
    
    def __init__(self):
        self.session = None
        self.data_cache = {}
        self.last_update = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_latest_data(self) -> Dict[str, Any]:
        """
        Collect latest data from all sources
        
        Returns:
            Dictionary with latest market data
        """
        logger.info("Collecting latest market data from all sources")
        
        # Initialize session if not exists
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Collect data from multiple sources concurrently
            tasks = [
                self._get_banxico_data(),
                self._get_lme_data(),
                self._get_yahoo_finance_data(),
                self._get_fred_data(),
                self._get_inegi_data()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine all data
            combined_data = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Error collecting data from source {i}: {str(result)}")
                else:
                    combined_data.update(result)
            
            # Add timestamp
            combined_data['timestamp'] = datetime.utcnow().isoformat()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error collecting latest data: {str(e)}")
            return self._get_fallback_data()
    
    async def get_historical_data(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Collect historical data for model training/prediction
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dictionary with historical market data
        """
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=settings.HISTORICAL_DATA_DAYS)
        
        logger.info(f"Collecting historical data from {start_date} to {end_date}")
        
        # For now, return dummy historical data
        # In production, this would fetch actual historical data
        return self._generate_dummy_historical_data(start_date, end_date)
    
    async def _get_banxico_data(self) -> Dict[str, Any]:
        """
        Collect data from Banco de México API
        
        Returns:
            Dictionary with BANXICO data
        """
        try:
            if not settings.BANXICO_API_TOKEN:
                logger.warning("BANXICO API token not configured")
                return self._get_dummy_banxico_data()
            
            # BANXICO API endpoints
            endpoints = {
                'usd_mxn': 'SF43718',  # USD/MXN exchange rate
                'inflation': 'SP74625',  # Inflation rate
                'interest_rate': 'SF43783'  # Interest rate
            }
            
            data = {}
            for key, series_id in endpoints.items():
                url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos/oportuno"
                headers = {'Bmx-Token': settings.BANXICO_API_TOKEN}
                
                try:
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Extract latest value
                            if 'bmx' in result and 'series' in result['bmx']:
                                series_data = result['bmx']['series'][0]['datos']
                                if series_data:
                                    data[key] = float(series_data[0]['dato'])
                        else:
                            logger.warning(f"BANXICO API error for {key}: {response.status}")
                except Exception as e:
                    logger.warning(f"Error fetching BANXICO data for {key}: {str(e)}")
            
            return {'banxico': data} if data else self._get_dummy_banxico_data()
            
        except Exception as e:
            logger.error(f"Error in BANXICO data collection: {str(e)}")
            return self._get_dummy_banxico_data()
    
    async def _get_lme_data(self) -> Dict[str, Any]:
        """
        Collect data from London Metal Exchange (via web scraping or alternative APIs)
        
        Returns:
            Dictionary with LME data
        """
        try:
            # LME doesn't have a free public API, so we'll use dummy data
            # In production, you might use paid services or web scraping
            return self._get_dummy_lme_data()
            
        except Exception as e:
            logger.error(f"Error in LME data collection: {str(e)}")
            return self._get_dummy_lme_data()
    
    async def _get_yahoo_finance_data(self) -> Dict[str, Any]:
        """
        Collect data from Yahoo Finance (steel-related stocks and commodities)
        
        Returns:
            Dictionary with Yahoo Finance data
        """
        try:
            # Yahoo Finance symbols for steel industry
            symbols = [
                'X',     # United States Steel Corporation
                'MT',    # ArcelorMittal
                'STLD',  # Steel Dynamics
                'CLF',   # Cleveland-Cliffs Inc
                'CL=F'   # Crude Oil Futures
            ]
            
            # For now, return dummy data
            # In production, use yfinance library or Yahoo Finance API
            return self._get_dummy_yahoo_data()
            
        except Exception as e:
            logger.error(f"Error in Yahoo Finance data collection: {str(e)}")
            return self._get_dummy_yahoo_data()
    
    async def _get_fred_data(self) -> Dict[str, Any]:
        """
        Collect data from FRED (Federal Reserve Economic Data)
        
        Returns:
            Dictionary with FRED data
        """
        try:
            if not settings.FRED_API_KEY:
                logger.warning("FRED API key not configured")
                return self._get_dummy_fred_data()
            
            # FRED series IDs for relevant economic indicators
            series_ids = {
                'industrial_production': 'INDPRO',
                'ppi_metals': 'WPU101',
                'dxy_index': 'DTWEXBGS'
            }
            
            data = {}
            for key, series_id in series_ids.items():
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': settings.FRED_API_KEY,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                try:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'observations' in result and result['observations']:
                                data[key] = float(result['observations'][0]['value'])
                except Exception as e:
                    logger.warning(f"Error fetching FRED data for {key}: {str(e)}")
            
            return {'fred': data} if data else self._get_dummy_fred_data()
            
        except Exception as e:
            logger.error(f"Error in FRED data collection: {str(e)}")
            return self._get_dummy_fred_data()
    
    async def _get_inegi_data(self) -> Dict[str, Any]:
        """
        Collect data from INEGI (Instituto Nacional de Estadística y Geografía)
        
        Returns:
            Dictionary with INEGI data
        """
        try:
            # INEGI API is complex and requires specific indicators
            # For now, return dummy data
            return self._get_dummy_inegi_data()
            
        except Exception as e:
            logger.error(f"Error in INEGI data collection: {str(e)}")
            return self._get_dummy_inegi_data()
    
    def _get_dummy_banxico_data(self) -> Dict[str, Any]:
        """Generate dummy BANXICO data for testing"""
        return {
            'banxico': {
                'usd_mxn': 18.45 + np.random.normal(0, 0.5),
                'inflation': 4.2 + np.random.normal(0, 0.3),
                'interest_rate': 11.0 + np.random.normal(0, 0.2)
            }
        }
    
    def _get_dummy_lme_data(self) -> Dict[str, Any]:
        """Generate dummy LME data for testing"""
        return {
            'lme': {
                'iron_ore_62': 120.50 + np.random.normal(0, 5),
                'coking_coal': 180.30 + np.random.normal(0, 8),
                'steel_rebar': 750.00 + np.random.normal(0, 25),
                'copper': 8500.00 + np.random.normal(0, 200)
            }
        }
    
    def _get_dummy_yahoo_data(self) -> Dict[str, Any]:
        """Generate dummy Yahoo Finance data for testing"""
        return {
            'yahoo_finance': {
                'us_steel_price': 25.30 + np.random.normal(0, 1),
                'arcelormittal_price': 28.45 + np.random.normal(0, 1.2),
                'oil_price': 75.20 + np.random.normal(0, 2),
                'steel_etf': 45.80 + np.random.normal(0, 1.5)
            }
        }
    
    def _get_dummy_fred_data(self) -> Dict[str, Any]:
        """Generate dummy FRED data for testing"""
        return {
            'fred': {
                'industrial_production': 105.8 + np.random.normal(0, 2),
                'ppi_metals': 110.2 + np.random.normal(0, 3),
                'dxy_index': 102.5 + np.random.normal(0, 1)
            }
        }
    
    def _get_dummy_inegi_data(self) -> Dict[str, Any]:
        """Generate dummy INEGI data for testing"""
        return {
            'inegi': {
                'construction_activity': 98.5 + np.random.normal(0, 2),
                'manufacturing_index': 102.1 + np.random.normal(0, 1.5),
                'gdp_construction': 95.8 + np.random.normal(0, 1)
            }
        }
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Generate fallback data when all sources fail"""
        logger.warning("Using fallback data - all sources failed")
        
        fallback = {}
        fallback.update(self._get_dummy_banxico_data())
        fallback.update(self._get_dummy_lme_data())
        fallback.update(self._get_dummy_yahoo_data())
        fallback.update(self._get_dummy_fred_data())
        fallback.update(self._get_dummy_inegi_data())
        fallback['timestamp'] = datetime.utcnow().isoformat()
        fallback['source'] = 'fallback'
        
        return fallback
    
    def _generate_dummy_historical_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate dummy historical data for testing"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic time series data
        n_days = len(dates)
        base_price = 750.0
        
        # Create trending data with noise
        trend = np.linspace(0, 50, n_days)  # Upward trend
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Yearly seasonality
        noise = np.random.normal(0, 15, n_days)
        
        steel_prices = base_price + trend + seasonal + noise
        
        historical_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'steel_rebar_prices': steel_prices.tolist(),
            'usd_mxn_rates': (18.5 + np.random.normal(0, 0.5, n_days)).tolist(),
            'iron_ore_prices': (120 + np.random.normal(0, 8, n_days)).tolist(),
            'oil_prices': (75 + np.random.normal(0, 3, n_days)).tolist(),
            'source': 'dummy_historical'
        }
        
        return historical_data
