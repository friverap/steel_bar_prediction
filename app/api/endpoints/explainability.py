"""
Explainability endpoints for steel rebar price prediction
Endpoints para análisis de explicabilidad y factores causales
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any

from app.models.prediction import FeatureImportanceResponse, CausalFactorsResponse
from app.services.predictor import SteelPricePredictor
from app.core.security import check_rate_limit, verify_api_key
from app.core.logging import api_logger
from src.ml_pipeline.prediction_cache import PredictionCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize predictor service
predictor = SteelPricePredictor()


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    request: Request,
    model_name: Optional[str] = Query(None, description="Modelo específico (XGBoost_V2_regime, MIDAS_V2_hibrida)"),
    top_n: int = Query(20, ge=5, le=50, description="Número de features más importantes"),
    api_key: str = Depends(verify_api_key)
) -> FeatureImportanceResponse:
    """
    Obtener análisis de importancia de features y factores causales
    
    Este endpoint explica qué factores tienen mayor influencia en la predicción
    del precio de la varilla corrugada, basado en los modelos V2 entrenados.
    
    Args:
        model_name: Modelo específico a analizar (opcional)
        top_n: Número de features más importantes a retornar
        
    Returns:
        FeatureImportanceResponse: Análisis de importancia con categorización
    """
    try:
        # Log request
        api_logger.log_request("GET", "/explainability/feature-importance", request.client.host, api_key)
        
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        logger.info(f"📊 Analizando feature importance (modelo: {model_name or 'todos'}, top: {top_n})")
        
        # RESPUESTA INSTANTÁNEA desde cache pre-calculado
        cache = PredictionCache()
        cached_analysis = await cache.get_cached_feature_importance()
        
        if cached_analysis and not cached_analysis.get('is_default', False):
            logger.info("⚡ Retornando feature importance desde cache (respuesta instantánea)")
            
            # Filtrar por modelo específico si se solicita
            if model_name:
                if model_name not in cached_analysis.get('models_analyzed', []):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Modelo {model_name} no encontrado en análisis cacheado"
                    )
                
                # Filtrar factores
                filtered_factors = []
                for factor in cached_analysis.get('top_factors', []):
                    if model_name in factor.get('models', []):
                        filtered_factors.append(factor)
                
                top_factors = filtered_factors[:top_n]
            else:
                top_factors = cached_analysis.get('top_factors', [])[:top_n]
            
            # Preparar respuesta desde cache
            response = FeatureImportanceResponse(
                models_analyzed=cached_analysis.get('models_analyzed', []),
                total_factors_analyzed=cached_analysis.get('total_factors_analyzed', 0),
                top_factors=top_factors,
                factors_by_category=cached_analysis.get('factors_by_category', {}),
                analysis_timestamp=cached_analysis.get('analysis_timestamp', datetime.now().isoformat()),
                requested_model=model_name,
                top_n_requested=top_n
            )
            
            return response
        
        # FALLBACK: Generar análisis completo (más lento)
        logger.info("🔄 Cache no disponible - generando análisis completo...")
        
        # Obtener análisis completo del predictor
        analysis = await predictor.get_feature_importance_analysis()
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=500,
                detail=f"Error en análisis: {analysis['error']}"
            )
        
        # Filtrar por modelo específico si se solicita
        if model_name:
            if model_name not in analysis.get('models_analyzed', []):
                raise HTTPException(
                    status_code=404,
                    detail=f"Modelo {model_name} no encontrado o no analizado"
                )
            
            # Filtrar factores solo de ese modelo
            filtered_factors = []
            for factor in analysis.get('top_factors', []):
                if model_name in factor.get('models', []):
                    filtered_factors.append(factor)
            
            top_factors = filtered_factors[:top_n]
        else:
            top_factors = analysis.get('top_factors', [])[:top_n]
        
        # Preparar respuesta
        response = FeatureImportanceResponse(
            models_analyzed=analysis.get('models_analyzed', []),
            total_factors_analyzed=len(analysis.get('top_factors', [])),
            top_factors=top_factors,
            factors_by_category=analysis.get('factors_by_category', {}),
            analysis_timestamp=analysis.get('analysis_timestamp', datetime.now().isoformat()),
            requested_model=model_name,
            top_n_requested=top_n
        )
        
        # Log successful analysis
        api_logger.log_request("SUCCESS", f"feature_importance_analysis_{model_name or 'all'}", request.client.host, api_key)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis de feature importance: {str(e)}", exc_info=True)
        api_logger.log_error(e, "feature_importance_analysis")
        
        raise HTTPException(
            status_code=500,
            detail="Error generando análisis de importancia de features"
        )


@router.get("/causal-factors", response_model=CausalFactorsResponse)
async def get_causal_factors(
    request: Request,
    category: Optional[str] = Query(None, description="Categoría específica (Autorregresivo, Materias Primas, etc.)"),
    min_importance: float = Query(0.01, ge=0.0, le=1.0, description="Importancia mínima (0.0-1.0)"),
    api_key: str = Depends(verify_api_key)
) -> CausalFactorsResponse:
    """
    Obtener factores causales que influyen en el precio de la varilla
    
    Este endpoint proporciona una explicación detallada de los factores económicos
    y técnicos que causan cambios en el precio de la varilla corrugada.
    
    Args:
        category: Filtrar por categoría específica
        min_importance: Importancia mínima para incluir factores
        
    Returns:
        CausalFactorsResponse: Factores causales con explicaciones económicas
    """
    try:
        # Log request
        api_logger.log_request("GET", "/explainability/causal-factors", request.client.host, api_key)
        
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        logger.info(f"🔍 Analizando factores causales (categoría: {category}, min_importance: {min_importance})")
        
        # Obtener análisis completo
        analysis = await predictor.get_feature_importance_analysis()
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=500,
                detail=f"Error en análisis: {analysis['error']}"
            )
        
        # Filtrar factores según criterios
        all_factors = analysis.get('top_factors', [])
        
        # Filtrar por importancia mínima
        filtered_factors = [f for f in all_factors if f.get('average_importance', 0) >= min_importance]
        
        # Filtrar por categoría si se especifica
        if category:
            filtered_factors = [f for f in filtered_factors if f.get('category') == category]
        
        # Obtener categorías disponibles
        factors_by_category = analysis.get('factors_by_category', {})
        available_categories = list(factors_by_category.keys())
        
        # Crear explicaciones causales detalladas
        causal_explanations = []
        
        for factor in filtered_factors[:15]:  # Top 15 factores causales
            explanation = _create_causal_explanation(factor)
            causal_explanations.append(explanation)
        
        # Preparar respuesta
        response = CausalFactorsResponse(
            total_factors_analyzed=len(all_factors),
            factors_returned=len(causal_explanations),
            causal_factors=causal_explanations,
            available_categories=available_categories,
            filter_applied={
                'category': category,
                'min_importance': min_importance
            },
            economic_context=_get_economic_context(),
            analysis_timestamp=analysis.get('analysis_timestamp', datetime.now().isoformat())
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis de factores causales: {str(e)}", exc_info=True)
        api_logger.log_error(e, "causal_factors_analysis")
        
        raise HTTPException(
            status_code=500,
            detail="Error generando análisis de factores causales"
        )


def _create_causal_explanation(factor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crear explicación causal detallada para un factor
    """
    feature_name = factor['feature']
    importance = factor['average_importance']
    category = factor['category']
    
    # Explicaciones causales específicas
    causal_explanations = {
        'precio_varilla_lme_lag_1': {
            'mechanism': 'Inercia de precios',
            'economic_rationale': 'Los precios tienden a seguir tendencias debido a contratos a plazo y expectativas de mercado',
            'impact_direction': 'Positiva (precios altos tienden a mantenerse)',
            'time_horizon': 'Inmediato (1 día)'
        },
        'iron': {
            'mechanism': 'Costo de materia prima',
            'economic_rationale': 'El mineral de hierro representa ~70% del costo de producción del acero',
            'impact_direction': 'Positiva (mayor costo de hierro → mayor precio de acero)',
            'time_horizon': 'Corto plazo (1-5 días)'
        },
        'coking': {
            'mechanism': 'Costo de materia prima',
            'economic_rationale': 'El carbón de coque es esencial para la producción de acero en altos hornos',
            'impact_direction': 'Positiva (mayor costo de carbón → mayor precio de acero)',
            'time_horizon': 'Corto plazo (1-5 días)'
        },
        'VIX': {
            'mechanism': 'Aversión al riesgo',
            'economic_rationale': 'Alta volatilidad reduce demanda de commodities como refugio seguro',
            'impact_direction': 'Negativa (mayor VIX → menor precio de commodities)',
            'time_horizon': 'Inmediato (mismo día)'
        },
        'sp500': {
            'mechanism': 'Condiciones macroeconómicas',
            'economic_rationale': 'Mercados alcistas indican crecimiento económico y mayor demanda de acero',
            'impact_direction': 'Positiva (mayor S&P 500 → mayor demanda de acero)',
            'time_horizon': 'Mediano plazo (5-20 días)'
        },
        'tipo_cambio_usdmxn': {
            'mechanism': 'Competitividad comercial',
            'economic_rationale': 'Peso débil hace más caro importar materias primas pero más competitivo exportar',
            'impact_direction': 'Compleja (depende del balance importación/exportación)',
            'time_horizon': 'Corto plazo (1-3 días)'
        }
    }
    
    # Obtener explicación específica o crear genérica
    if feature_name in causal_explanations:
        causal_info = causal_explanations[feature_name]
    else:
        # Explicación genérica basada en categoría
        causal_info = _get_generic_causal_explanation(category)
    
    return {
        'factor_name': feature_name,
        'importance_score': importance,
        'category': category,
        'causal_mechanism': causal_info['mechanism'],
        'economic_rationale': causal_info['economic_rationale'],
        'impact_direction': causal_info['impact_direction'],
        'time_horizon': causal_info['time_horizon'],
        'description': factor.get('description', feature_name)
    }


def _get_generic_causal_explanation(category: str) -> Dict[str, str]:
    """Obtener explicación causal genérica por categoría"""
    
    generic_explanations = {
        'Autorregresivo': {
            'mechanism': 'Persistencia temporal',
            'economic_rationale': 'Los precios muestran inercia debido a contratos, expectativas y costos de ajuste',
            'impact_direction': 'Positiva (continuidad de tendencias)',
            'time_horizon': 'Muy corto plazo (1-2 días)'
        },
        'Materias Primas': {
            'mechanism': 'Transmisión de costos',
            'economic_rationale': 'Los costos de materias primas se transmiten directamente al precio final',
            'impact_direction': 'Positiva (mayor costo → mayor precio)',
            'time_horizon': 'Corto plazo (1-10 días)'
        },
        'Mercados Financieros': {
            'mechanism': 'Sentimiento y liquidez',
            'economic_rationale': 'Los mercados financieros reflejan expectativas y disponibilidad de capital',
            'impact_direction': 'Variable (depende del contexto)',
            'time_horizon': 'Inmediato a corto plazo'
        },
        'Tipo de Cambio': {
            'mechanism': 'Competitividad y costos',
            'economic_rationale': 'Afecta costos de importación y competitividad exportadora',
            'impact_direction': 'Compleja (balance comercial)',
            'time_horizon': 'Corto plazo (1-5 días)'
        },
        'Tasas de Interés': {
            'mechanism': 'Costo de capital',
            'economic_rationale': 'Afecta costos de financiamiento y demanda de inversión',
            'impact_direction': 'Negativa (mayor tasa → menor demanda)',
            'time_horizon': 'Mediano plazo (10-30 días)'
        },
        'Indicadores Técnicos': {
            'mechanism': 'Señales de mercado',
            'economic_rationale': 'Capturan patrones técnicos y momentum de precios',
            'impact_direction': 'Variable (según indicador)',
            'time_horizon': 'Muy corto plazo (1-5 días)'
        }
    }
    
    return generic_explanations.get(category, {
        'mechanism': 'Factor específico',
        'economic_rationale': 'Influencia particular en el mercado de acero',
        'impact_direction': 'Variable',
        'time_horizon': 'Variable'
    })


def _get_economic_context() -> Dict[str, str]:
    """Obtener contexto económico general para la explicación"""
    return {
        'market_structure': 'El mercado de varilla corrugada está influenciado por costos de materias primas, condiciones macroeconómicas y factores técnicos de mercado',
        'price_formation': 'Los precios se forman a través de la interacción entre costos de producción (hierro, carbón), demanda (construcción), y condiciones financieras globales',
        'key_relationships': 'Relaciones causales principales: Materias Primas → Costos → Precios; Macro → Demanda → Precios; Técnico → Momentum → Precios',
        'prediction_horizon': 'Los modelos están optimizados para predicción del precio de cierre del día siguiente (t+1)',
        'model_approach': 'Combinación de modelos MIDAS (frecuencias mixtas) y XGBoost (no-linealidades) para capturar diferentes aspectos causales'
    }


@router.get("/causal-chain")
async def get_causal_chain(
    request: Request,
    factor: str = Query(..., description="Factor específico a analizar"),
    api_key: str = Depends(verify_api_key)
):
    """
    Obtener cadena causal detallada para un factor específico
    
    Explica cómo un factor específico influye en el precio de la varilla
    a través de mecanismos económicos específicos.
    
    Args:
        factor: Nombre del factor a analizar
        
    Returns:
        Cadena causal detallada con mecanismos de transmisión
    """
    try:
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        logger.info(f"🔗 Analizando cadena causal para: {factor}")
        
        # Obtener análisis de feature importance
        analysis = await predictor.get_feature_importance_analysis()
        
        # Buscar el factor específico
        factor_info = None
        for f in analysis.get('top_factors', []):
            if f['feature'].lower() == factor.lower():
                factor_info = f
                break
        
        if not factor_info:
            raise HTTPException(
                status_code=404,
                detail=f"Factor '{factor}' no encontrado en el análisis"
            )
        
        # Crear cadena causal detallada
        causal_chain = _create_detailed_causal_chain(factor_info)
        
        return {
            'factor': factor,
            'importance_score': factor_info['average_importance'],
            'category': factor_info['category'],
            'causal_chain': causal_chain,
            'models_using_factor': factor_info.get('models', []),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis de cadena causal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error analizando cadena causal"
        )


def _create_detailed_causal_chain(factor_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Crear cadena causal detallada para un factor
    """
    factor_name = factor_info['feature']
    category = factor_info['category']
    
    # Cadenas causales específicas por factor
    causal_chains = {
        'iron': [
            {'step': 1, 'mechanism': 'Costo de Materia Prima', 'description': 'Aumento en precio del mineral de hierro'},
            {'step': 2, 'mechanism': 'Transmisión de Costos', 'description': 'Incremento en costos de producción de acero'},
            {'step': 3, 'mechanism': 'Ajuste de Precios', 'description': 'Productores ajustan precios para mantener márgenes'},
            {'step': 4, 'mechanism': 'Impacto Final', 'description': 'Aumento en precio de varilla corrugada'}
        ],
        'coking': [
            {'step': 1, 'mechanism': 'Insumo Crítico', 'description': 'Cambio en precio del carbón de coque'},
            {'step': 2, 'mechanism': 'Proceso Siderúrgico', 'description': 'Impacto en costos de producción en altos hornos'},
            {'step': 3, 'mechanism': 'Estructura de Costos', 'description': 'Ajuste en estructura de costos de acería'},
            {'step': 4, 'mechanism': 'Precio Final', 'description': 'Reflejo en precio de productos de acero'}
        ],
        'VIX': [
            {'step': 1, 'mechanism': 'Aversión al Riesgo', 'description': 'Aumento en volatilidad implícita del mercado'},
            {'step': 2, 'mechanism': 'Flight to Quality', 'description': 'Inversores buscan activos seguros'},
            {'step': 3, 'mechanism': 'Reducción de Demanda', 'description': 'Menor demanda especulativa de commodities'},
            {'step': 4, 'mechanism': 'Presión Bajista', 'description': 'Presión a la baja en precios de materias primas'}
        ],
        'sp500': [
            {'step': 1, 'mechanism': 'Indicador Macroeconómico', 'description': 'S&P 500 refleja salud económica general'},
            {'step': 2, 'mechanism': 'Expectativas de Crecimiento', 'description': 'Mercados alcistas indican crecimiento esperado'},
            {'step': 3, 'mechanism': 'Demanda de Infraestructura', 'description': 'Mayor crecimiento → mayor demanda de construcción'},
            {'step': 4, 'mechanism': 'Demanda de Acero', 'description': 'Incremento en demanda de productos de acero'}
        ]
    }
    
    # Usar cadena específica o crear genérica
    if factor_name in causal_chains:
        return causal_chains[factor_name]
    else:
        return _create_generic_causal_chain(category, factor_name)


def _create_generic_causal_chain(category: str, factor_name: str) -> List[Dict[str, str]]:
    """Crear cadena causal genérica basada en categoría"""
    
    generic_chains = {
        'Autorregresivo': [
            {'step': 1, 'mechanism': 'Persistencia', 'description': f'{factor_name} muestra continuidad temporal'},
            {'step': 2, 'mechanism': 'Expectativas', 'description': 'Formación de expectativas basadas en historia reciente'},
            {'step': 3, 'mechanism': 'Comportamiento', 'description': 'Agentes actúan según expectativas formadas'},
            {'step': 4, 'mechanism': 'Autorealización', 'description': 'Las expectativas se vuelven realidad'}
        ],
        'Materias Primas': [
            {'step': 1, 'mechanism': 'Cambio de Costo', 'description': f'Variación en costo de {factor_name}'},
            {'step': 2, 'mechanism': 'Transmisión', 'description': 'Transmisión a través de la cadena productiva'},
            {'step': 3, 'mechanism': 'Ajuste de Márgenes', 'description': 'Productores ajustan precios para mantener rentabilidad'},
            {'step': 4, 'mechanism': 'Precio Final', 'description': 'Impacto en precio final de varilla'}
        ],
        'Mercados Financieros': [
            {'step': 1, 'mechanism': 'Señal de Mercado', 'description': f'{factor_name} envía señal sobre condiciones'},
            {'step': 2, 'mechanism': 'Sentimiento', 'description': 'Cambio en sentimiento de inversores'},
            {'step': 3, 'mechanism': 'Flujos de Capital', 'description': 'Reasignación de capital entre activos'},
            {'step': 4, 'mechanism': 'Impacto en Commodities', 'description': 'Efecto en precios de materias primas'}
        ]
    }
    
    return generic_chains.get(category, [
        {'step': 1, 'mechanism': 'Factor Específico', 'description': f'Influencia de {factor_name}'},
        {'step': 2, 'mechanism': 'Transmisión', 'description': 'Mecanismo de transmisión al mercado'},
        {'step': 3, 'mechanism': 'Ajuste de Mercado', 'description': 'Respuesta del mercado al cambio'},
        {'step': 4, 'mechanism': 'Impacto Final', 'description': 'Efecto en precio de varilla'}
    ])


@router.get("/model-comparison")
async def get_model_comparison(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Comparar performance y características de los modelos V2 disponibles
    
    Returns:
        Comparación detallada entre XGBoost_V2_regime y MIDAS_V2_hibrida
    """
    try:
        # Check rate limiting
        await check_rate_limit(request, api_key)
        
        logger.info("📊 Generando comparación de modelos V2...")
        
        # Obtener información de modelos
        model_factory = predictor.production_predictor.model_factory
        available_models = model_factory.list_available_models()
        
        if not available_models:
            raise HTTPException(
                status_code=404,
                detail="No hay modelos V2 disponibles para comparar"
            )
        
        # Comparar modelos
        model_comparison = {}
        
        for model_name in available_models:
            model_info = model_factory.get_model_info(model_name)
            feature_importance = model_factory.get_feature_importance(model_name)
            
            model_comparison[model_name] = {
                'description': model_info.get('description', ''),
                'type': model_info.get('type', ''),
                'variables_used': model_info.get('variables', []),
                'test_metrics': model_info.get('test_metrics', {}),
                'top_5_features': feature_importance[:5],
                'strengths': _get_model_strengths(model_name),
                'use_cases': _get_model_use_cases(model_name)
            }
        
        return {
            'models_compared': list(model_comparison.keys()),
            'comparison': model_comparison,
            'recommendation': _get_model_recommendation(model_comparison),
            'comparison_timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en comparación de modelos: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error generando comparación de modelos"
        )


def _get_model_strengths(model_name: str) -> List[str]:
    """Obtener fortalezas específicas de cada modelo"""
    strengths = {
        'XGBoost_V2_regime': [
            'Excelente para capturar relaciones no-lineales',
            'Robusto a cambios de régimen de mercado',
            'Manejo automático de interacciones entre variables',
            'Alta precisión en condiciones de mercado volátiles'
        ],
        'MIDAS_V2_hibrida': [
            'Combina variables autorregresivas y fundamentales',
            'Excelente para capturar tendencias de mediano plazo',
            'Manejo sofisticado de frecuencias mixtas',
            'Alta estabilidad en predicciones'
        ]
    }
    
    return strengths.get(model_name, ['Modelo especializado para predicción de precios'])


def _get_model_use_cases(model_name: str) -> List[str]:
    """Obtener casos de uso recomendados para cada modelo"""
    use_cases = {
        'XGBoost_V2_regime': [
            'Períodos de alta volatilidad de mercado',
            'Cambios súbitos en condiciones macroeconómicas',
            'Detección de cambios de régimen',
            'Predicciones en mercados no-lineales'
        ],
        'MIDAS_V2_hibrida': [
            'Condiciones de mercado estables',
            'Predicciones de tendencia a mediano plazo',
            'Integración de datos de múltiples frecuencias',
            'Análisis fundamental de mercado'
        ]
    }
    
    return use_cases.get(model_name, ['Predicción general de precios'])


def _get_model_recommendation(model_comparison: Dict[str, Any]) -> Dict[str, str]:
    """Generar recomendación de uso de modelos"""
    
    # Comparar métricas si están disponibles
    best_mape = float('inf')
    best_r2 = -1
    best_model_mape = None
    best_model_r2 = None
    
    for model_name, info in model_comparison.items():
        metrics = info.get('test_metrics', {})
        
        mape = metrics.get('mape', float('inf'))
        r2 = metrics.get('r2', -1)
        
        if mape < best_mape:
            best_mape = mape
            best_model_mape = model_name
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_r2 = model_name
    
    return {
        'primary_recommendation': best_model_r2 or 'XGBoost_V2_regime',
        'reasoning': f'Mejor R² ({best_r2:.3f}) y MAPE ({best_mape:.2f}%)',
        'alternative': 'Usar ambos modelos y promediar predicciones para mayor robustez',
        'context_dependent': 'XGBoost para mercados volátiles, MIDAS para condiciones estables'
    }
