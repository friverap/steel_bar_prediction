"""
World Bank Monthly Commodities Collector (Pink Sheet)
Colector para datos MENSUALES de commodities del World Bank Pink Sheet
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class WorldBankMonthlyCollector:
    """
    Colector para datos mensuales de commodities del World Bank Pink Sheet
    """
    
    def __init__(self):
        self.pink_sheet_url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"
        self.temp_file = "temp_wb_commodities.xlsx"
        
        # Mapeo de columnas a nombres estandarizados
        self.commodity_mapping = {
            'iron_ore': 'Iron ore, cfr spot',
            'coal_australian': 'Coal, Australian',
            'aluminum': 'Aluminum',
            'copper': 'Copper',
            'zinc': 'Zinc',
            'nickel': 'Nickel',
            'steel_rebar': 'Steel rebar',
            'steel_wire': 'Steel wire rod',
            'crude_oil_brent': 'Crude oil, Brent',
            'natural_gas_us': 'Natural gas, US',
            'natural_gas_europe': 'Natural gas, Europe'
        }
    
    def download_pink_sheet(self) -> bool:
        """
        Descarga el archivo Excel de Pink Sheet
        """
        try:
            logger.info("Descargando World Bank Pink Sheet (datos mensuales)...")
            response = requests.get(self.pink_sheet_url, timeout=60)
            
            if response.status_code == 200:
                with open(self.temp_file, 'wb') as f:
                    f.write(response.content)
                logger.info("✅ Pink Sheet descargado exitosamente")
                return True
            else:
                logger.error(f"Error HTTP {response.status_code} descargando Pink Sheet")
                return False
                
        except Exception as e:
            logger.error(f"Error descargando Pink Sheet: {str(e)}")
            return False
    
    def load_monthly_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga y procesa los datos mensuales de commodities
        """
        try:
            # Descargar si no existe
            if not os.path.exists(self.temp_file):
                if not self.download_pink_sheet():
                    return {}
            
            # Leer Excel - los nombres de columnas están en fila 5 (skiprows=4)
            df = pd.read_excel(self.temp_file, sheet_name='Monthly Prices', skiprows=4)
            
            # Eliminar la primera fila que contiene las unidades
            df = df.iloc[1:].reset_index(drop=True)
            
            # La primera columna contiene las fechas en formato YYYYMM
            date_col = df.columns[0]
            
            # Convertir fechas del formato "2024M01" a datetime
            df[date_col] = df[date_col].astype(str).str.replace('M', '-')
            df[date_col] = pd.to_datetime(df[date_col] + '-01', format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.rename(columns={date_col: 'fecha'})
            
            # Filtrar por rango de fechas si se especifica
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['fecha'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['fecha'] <= end]
            
            logger.info(f"Datos cargados: {len(df)} meses desde {df['fecha'].min()} hasta {df['fecha'].max()}")
            
            results = {}
            
            # Procesar cada commodity
            for key, col_name in self.commodity_mapping.items():
                if col_name in df.columns:
                    # Crear DataFrame para esta serie
                    commodity_df = pd.DataFrame({
                        'fecha': df['fecha'],
                        'valor': df[col_name]
                    })
                    
                    # Limpiar valores nulos
                    commodity_df = commodity_df.dropna(subset=['valor'])
                    
                    if not commodity_df.empty:
                        results[key] = {
                            'name': col_name,
                            'data': commodity_df,
                            'count': len(commodity_df),
                            'latest_value': commodity_df['valor'].iloc[-1],
                            'latest_date': commodity_df['fecha'].iloc[-1],
                            'frequency': 'monthly',
                            'unit': 'USD/mt' if 'ore' in key or 'metal' in key.lower() else 'USD/bbl',
                            'source': 'world_bank_pink_sheet'
                        }
                        
                        logger.info(f"✅ {col_name}: {len(commodity_df)} datos mensuales")
                    else:
                        logger.warning(f"⚠️ {col_name}: Sin datos en el período especificado")
                else:
                    logger.warning(f"⚠️ Columna '{col_name}' no encontrada")
            
            # Limpiar archivo temporal
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            
            return {
                'commodities_data': results,
                'summary': {
                    'total_commodities': len(results),
                    'total_data_points': sum(r['count'] for r in results.values()),
                    'frequency': 'monthly',
                    'source': 'World Bank Pink Sheet',
                    'latest_update': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error procesando datos mensuales: {str(e)}")
            # Limpiar archivo temporal en caso de error
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            return {}
    
    def save_raw_data(self, data: Dict[str, Any], output_dir: str = 'data/raw'):
        """
        Guarda los datos en formato CSV
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        
        for key, commodity_data in data.get('commodities_data', {}).items():
            if 'data' in commodity_data and not commodity_data['data'].empty:
                filename = f"world_bank_monthly_{key}_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                
                commodity_data['data'].to_csv(filepath, index=False)
                logger.info(f"Datos guardados: {filepath}")
                
                # Guardar metadata
                metadata = {
                    'commodity': commodity_data['name'],
                    'frequency': 'monthly',
                    'source': 'World Bank Pink Sheet',
                    'unit': commodity_data['unit'],
                    'count': commodity_data['count'],
                    'latest_date': str(commodity_data['latest_date']),
                    'latest_value': commodity_data['latest_value']
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

async def collect_world_bank_monthly_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_raw: bool = True
) -> Dict[str, Any]:
    """
    Función de conveniencia asíncrona para recopilar datos mensuales de World Bank
    """
    collector = WorldBankMonthlyCollector()
    data = collector.load_monthly_data(start_date, end_date)
    
    if save_raw and data:
        collector.save_raw_data(data)
    
    return data
