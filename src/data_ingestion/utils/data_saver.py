"""
Utilidad para guardar datos de forma estandarizada
Formato: FuenteDatos_NombreVariable_Fecha.csv
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_raw_data(
    data: pd.DataFrame,
    source: str,
    variable: str,
    metadata: Optional[Dict[str, Any]] = None,
    raw_dir: str = 'data/raw'
) -> bool:
    """
    Guardar datos raw con nombre estandarizado
    
    Args:
        data: DataFrame con los datos
        source: Nombre de la fuente (ej: 'Banxico', 'FRED', 'Yahoo')
        variable: Nombre de la variable (ej: 'TIIE28', 'USD_MXN', 'Cobre')
        metadata: Metadata adicional opcional
        raw_dir: Directorio donde guardar los archivos
        
    Returns:
        bool: True si se guardó correctamente
    """
    try:
        # Crear directorio si no existe
        os.makedirs(raw_dir, exist_ok=True)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # Limpiar nombres para evitar caracteres problemáticos
        source_clean = source.replace(' ', '').replace('/', '_').replace('\\', '_')
        variable_clean = variable.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Formato: FuenteDatos_NombreVariable_Fecha.csv
        filename = f"{source_clean}_{variable_clean}_{timestamp}.csv"
        filepath = os.path.join(raw_dir, filename)
        
        # Guardar datos
        if data is not None and not data.empty:
            data.to_csv(filepath, index=False)
            logger.info(f"✅ Datos guardados: {filepath}")
            
            # Guardar metadata si se proporciona
            if metadata:
                metadata_enhanced = {
                    'source': source,
                    'variable': variable,
                    'filename': filename,
                    'collection_timestamp': datetime.now().isoformat(),
                    'rows': len(data),
                    'columns': len(data.columns),
                    **metadata  # Agregar metadata adicional
                }
                
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_enhanced, f, indent=2, default=str)
                logger.info(f"📋 Metadata guardada: {metadata_file}")
            
            return True
        else:
            logger.warning(f"⚠️ No hay datos para guardar: {source}_{variable}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error guardando datos {source}_{variable}: {str(e)}")
        return False


# Mapeos de nombres para cada fuente
SOURCE_NAME_MAPPINGS = {
    'banxico': {
        'usd_mxn': 'USD_MXN',
        'tiie_28': 'TIIE_28dias',
        'tiie_91': 'TIIE_91dias',
        'udis': 'UDIS',
        'cetes_28': 'CETES_28dias',
        'cetes_91': 'CETES_91dias',
        'inflation': 'Inflacion',
        'igae': 'IGAE'
    },
    'fred': {
        'fed_funds_rate': 'TasaFED',
        'treasury_10y': 'BonoUS_10Y',
        'treasury_2y': 'BonoUS_2Y',
        'industrial_production': 'ProduccionIndustrial',
        'unemployment': 'Desempleo',
        'cpi': 'InflacionUS',
        'gdp': 'PIB_US',
        'construction_spending': 'GastoConstruccion'
    },
    'yahoo': {
        'copper': 'Cobre_Futuros',
        'aluminum': 'Aluminio_Futuros',
        'gold': 'Oro_Futuros',
        'silver': 'Plata_Futuros',
        'oil_wti': 'Petroleo_WTI',
        'oil_brent': 'Petroleo_Brent',
        'natural_gas': 'GasNatural',
        'sp500': 'SP500',
        'nasdaq': 'NASDAQ',
        'dxy': 'DolarIndex',
        'vix': 'VIX_Volatilidad'
    },
    'lme': {
        'copper': 'Cobre_LME',
        'aluminum': 'Aluminio_LME',
        'zinc': 'Zinc_LME',
        'lead': 'Plomo_LME',
        'nickel': 'Niquel_LME',
        'tin': 'Estano_LME'
    },
    'ahmsa': {
        'varilla_corrugada': 'VarillaCorrugada',
        'alambre_recocido': 'AlambreRecocido',
        'alambre_galvanizado': 'AlambreGalvanizado',
        'clavo': 'Clavo',
        'malla_electrosoldada': 'MallaElectrosoldada',
        'perfil_ptr': 'PerfilPTR',
        'perfil_comercial': 'PerfilComercial'
    },
    'inegi': {
        'inpc': 'INPC',
        'inpp': 'INPP',
        'produccion_industrial': 'ProduccionIndustrial',
        'produccion_construccion': 'ProduccionConstruccion',
        'produccion_manufactura': 'ProduccionManufactura',
        'produccion_metalurgia': 'ProduccionMetalurgia',
        'igae': 'IGAE',
        'pib': 'PIB_Mexico'
    },
    'raw_materials': {
        'vale': 'MineralHierro_VALE',
        'rio_tinto': 'MineralHierro_RIO',
        'rio': 'MineralHierro_RIO',
        'bhp': 'MineralHierro_BHP',
        'teck': 'CarbonCoque_TECK',
        'anglo_american': 'CarbonCoque_AAL',
        'aal': 'CarbonCoque_AAL',
        'slx': 'ETF_Acero_SLX',
        'xme': 'ETF_Mineria_XME',
        'xlb': 'ETF_Materiales_XLB',
        'steel_etf': 'ETF_Acero_SLX',
        'metals_miners': 'ETF_Mineria_XME',
        'materials': 'ETF_Materiales_XLB'
    },
    'world_bank': {
        'gdp_mexico': 'PIB_Mexico_Anual',
        'gdp_growth': 'CrecimientoPIB',
        'inflation': 'Inflacion_Anual',
        'industry_value': 'ValorAgregadoIndustrial',
        'manufacturing': 'Manufactura',
        'exports': 'Exportaciones',
        'imports': 'Importaciones',
        'exchange_rate': 'TipoCambio_Promedio'
    }
}


def get_variable_name(source: str, key: str) -> str:
    """
    Obtener nombre descriptivo de variable basado en fuente y clave
    
    Args:
        source: Nombre de la fuente en minúsculas
        key: Clave de la variable
        
    Returns:
        str: Nombre descriptivo de la variable
    """
    source_lower = source.lower()
    key_lower = key.lower()
    
    if source_lower in SOURCE_NAME_MAPPINGS:
        mapping = SOURCE_NAME_MAPPINGS[source_lower]
        return mapping.get(key_lower, key)
    
    return key
