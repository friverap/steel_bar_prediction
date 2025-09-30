"""
Feature Engineering Service
Creates features for machine learning models from raw market data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Service for creating engineered features from raw market data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    async def create_features(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Create feature vector from raw market data
        
        Args:
            raw_data: Dictionary with raw market data
            
        Returns:
            Numpy array with engineered features
        """
        try:
            features = []
            feature_names = []
            
            # Extract and engineer features from different data sources
            features.extend(self._extract_price_features(raw_data))
            feature_names.extend(self._get_price_feature_names())
            
            features.extend(self._extract_economic_features(raw_data))
            feature_names.extend(self._get_economic_feature_names())
            
            features.extend(self._extract_technical_features(raw_data))
            feature_names.extend(self._get_technical_feature_names())
            
            features.extend(self._extract_temporal_features())
            feature_names.extend(self._get_temporal_feature_names())
            
            # Store feature names for reference
            self.feature_names = feature_names
            
            # Convert to numpy array
            feature_array = np.array(features, dtype=np.float32)
            
            # Handle missing values
            feature_array = self._handle_missing_values(feature_array)
            
            logger.info(f"Created {len(features)} features from raw data")
            return feature_array
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            # Return dummy feature vector
            return np.random.random(20)
    
    def _extract_price_features(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract price-related features"""
        features = []
        
        try:
            # LME prices
            lme_data = raw_data.get('lme', {})
            features.append(lme_data.get('iron_ore_62', 120.0))
            features.append(lme_data.get('coking_coal', 180.0))
            features.append(lme_data.get('steel_rebar', 750.0))
            features.append(lme_data.get('copper', 8500.0))
            
            # Oil prices
            yahoo_data = raw_data.get('yahoo_finance', {})
            oil_price = yahoo_data.get('oil_price', 75.0)
            features.append(oil_price)
            
            # Steel company stock prices
            features.append(yahoo_data.get('us_steel_price', 25.0))
            features.append(yahoo_data.get('arcelormittal_price', 28.0))
            features.append(yahoo_data.get('steel_etf', 45.0))
            
            # Price ratios (important for steel pricing)
            iron_ore_price = lme_data.get('iron_ore_62', 120.0)
            steel_price = lme_data.get('steel_rebar', 750.0)
            features.append(steel_price / iron_ore_price if iron_ore_price > 0 else 6.25)
            features.append(steel_price / oil_price if oil_price > 0 else 10.0)
            
        except Exception as e:
            logger.warning(f"Error extracting price features: {str(e)}")
            # Return default values
            features.extend([120.0, 180.0, 750.0, 8500.0, 75.0, 25.0, 28.0, 45.0, 6.25, 10.0])
        
        return features
    
    def _extract_economic_features(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract economic indicator features"""
        features = []
        
        try:
            # BANXICO data
            banxico_data = raw_data.get('banxico', {})
            usd_mxn = banxico_data.get('usd_mxn', 18.5)
            inflation = banxico_data.get('inflation', 4.2)
            interest_rate = banxico_data.get('interest_rate', 11.0)
            
            features.extend([usd_mxn, inflation, interest_rate])
            
            # FRED data
            fred_data = raw_data.get('fred', {})
            features.append(fred_data.get('industrial_production', 105.0))
            features.append(fred_data.get('ppi_metals', 110.0))
            features.append(fred_data.get('dxy_index', 102.0))
            
            # INEGI data
            inegi_data = raw_data.get('inegi', {})
            features.append(inegi_data.get('construction_activity', 98.0))
            features.append(inegi_data.get('manufacturing_index', 102.0))
            
            # Economic ratios
            features.append(inflation / interest_rate if interest_rate > 0 else 0.38)
            
        except Exception as e:
            logger.warning(f"Error extracting economic features: {str(e)}")
            # Return default values
            features.extend([18.5, 4.2, 11.0, 105.0, 110.0, 102.0, 98.0, 102.0, 0.38])
        
        return features
    
    def _extract_technical_features(self, raw_data: Dict[str, Any]) -> List[float]:
        """Extract technical analysis features"""
        features = []
        
        try:
            # For now, generate dummy technical indicators
            # In production, these would be calculated from historical price data
            
            # Moving averages (dummy values)
            features.append(750.0)  # MA_5
            features.append(745.0)  # MA_20
            features.append(740.0)  # MA_50
            
            # Technical indicators (dummy values)
            features.append(65.0)   # RSI
            features.append(0.5)    # MACD signal
            features.append(2.1)    # Bollinger Band position
            
            # Volatility measures
            features.append(15.5)   # Historical volatility
            
        except Exception as e:
            logger.warning(f"Error extracting technical features: {str(e)}")
            features.extend([750.0, 745.0, 740.0, 65.0, 0.5, 2.1, 15.5])
        
        return features
    
    def _extract_temporal_features(self) -> List[float]:
        """Extract time-based features"""
        features = []
        
        try:
            now = datetime.now()
            
            # Day of week (0-6)
            features.append(float(now.weekday()))
            
            # Month (1-12)
            features.append(float(now.month))
            
            # Quarter (1-4)
            features.append(float((now.month - 1) // 3 + 1))
            
            # Day of year (1-365/366)
            features.append(float(now.timetuple().tm_yday))
            
            # Cyclical features (sine/cosine encoding)
            features.append(np.sin(2 * np.pi * now.month / 12))  # Monthly cycle
            features.append(np.cos(2 * np.pi * now.month / 12))
            
            features.append(np.sin(2 * np.pi * now.weekday() / 7))  # Weekly cycle
            features.append(np.cos(2 * np.pi * now.weekday() / 7))
            
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {str(e)}")
            features.extend([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        
        return features
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing or invalid values in features"""
        try:
            # Replace NaN with median values or defaults
            nan_mask = np.isnan(features)
            if np.any(nan_mask):
                logger.warning(f"Found {np.sum(nan_mask)} NaN values in features")
                # Simple imputation with median of non-NaN values
                for i in range(len(features)):
                    if np.isnan(features[i]):
                        # Use a reasonable default based on feature position
                        features[i] = self._get_default_feature_value(i)
            
            # Replace infinite values
            inf_mask = np.isinf(features)
            if np.any(inf_mask):
                logger.warning(f"Found {np.sum(inf_mask)} infinite values in features")
                features[inf_mask] = self._get_default_feature_value(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return features
    
    def _get_default_feature_value(self, feature_index: int) -> float:
        """Get default value for a feature based on its index"""
        # Default values for different feature types
        defaults = {
            0: 120.0,   # Iron ore price
            1: 180.0,   # Coking coal
            2: 750.0,   # Steel rebar
            3: 8500.0,  # Copper
            4: 75.0,    # Oil price
            5: 25.0,    # US Steel stock
            6: 28.0,    # ArcelorMittal stock
            7: 45.0,    # Steel ETF
            8: 6.25,    # Steel/Iron ratio
            9: 10.0,    # Steel/Oil ratio
            10: 18.5,   # USD/MXN
            11: 4.2,    # Inflation
            12: 11.0,   # Interest rate
            13: 105.0,  # Industrial production
            14: 110.0,  # PPI metals
            15: 102.0,  # DXY index
            16: 98.0,   # Construction activity
            17: 102.0,  # Manufacturing index
            18: 0.38,   # Inflation/Interest ratio
        }
        
        return defaults.get(feature_index, 1.0)
    
    def _get_price_feature_names(self) -> List[str]:
        """Get names for price features"""
        return [
            'iron_ore_62_price', 'coking_coal_price', 'steel_rebar_price', 'copper_price',
            'oil_price', 'us_steel_stock', 'arcelormittal_stock', 'steel_etf',
            'steel_iron_ratio', 'steel_oil_ratio'
        ]
    
    def _get_economic_feature_names(self) -> List[str]:
        """Get names for economic features"""
        return [
            'usd_mxn_rate', 'mexico_inflation', 'mexico_interest_rate',
            'us_industrial_production', 'ppi_metals', 'dxy_index',
            'mexico_construction_activity', 'mexico_manufacturing_index',
            'inflation_interest_ratio'
        ]
    
    def _get_technical_feature_names(self) -> List[str]:
        """Get names for technical features"""
        return [
            'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd_signal', 
            'bollinger_position', 'historical_volatility'
        ]
    
    def _get_temporal_feature_names(self) -> List[str]:
        """Get names for temporal features"""
        return [
            'day_of_week', 'month', 'quarter', 'day_of_year',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (dummy implementation)
        In production, this would come from the trained model
        """
        if not self.feature_names:
            return {}
        
        # Dummy importance scores
        importance_scores = np.random.random(len(self.feature_names))
        importance_scores = importance_scores / importance_scores.sum()  # Normalize
        
        return dict(zip(self.feature_names, importance_scores))
