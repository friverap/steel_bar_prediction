#!/usr/bin/env python3
"""
Model Selector - DeAcero Steel Price Predictor V2
Sistema inteligente para seleccionar automáticamente el mejor modelo
basado en múltiples métricas y criterios de performance

Este módulo:
1. Compara modelos XGBoost_V2_regime vs MIDAS_V2_hibrida
2. Usa scoring multi-criterio para selección
3. Considera estabilidad temporal y robustez
4. Proporciona explicación de la selección
5. Maneja fallbacks en caso de fallas

Fecha: 28 de Septiembre de 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Selector inteligente de modelos basado en performance multi-criterio
    """
    
    def __init__(self):
        self.selection_criteria = {
            'r2_weight': 0.40,           # Capacidad explicativa
            'mape_weight': 0.30,         # Error porcentual
            'hit_rate_weight': 0.20,     # Precisión práctica ±2%
            'directional_accuracy_weight': 0.10  # Dirección correcta
        }
        
        self.performance_thresholds = {
            'min_r2': 0.5,              # R² mínimo aceptable
            'max_mape': 10.0,           # MAPE máximo aceptable
            'min_hit_rate': 50.0,       # Hit rate mínimo
            'min_directional_acc': 45.0  # Directional accuracy mínimo
        }
        
        self.model_preferences = [
            'MIDAS_V2_hibrida',         # Preferido por estabilidad
            'XGBoost_V2_regime'         # Backup por robustez
        ]
        
        logger.info("🏆 ModelSelector inicializado con criterios multi-métrica")
    
    def select_best_model(self, models_performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Seleccionar el mejor modelo basado en métricas de performance
        
        Args:
            models_performance: Dict con métricas de cada modelo
            
        Returns:
            Información del modelo seleccionado y justificación
        """
        logger.info("🔍 Analizando modelos disponibles para selección...")
        
        if not models_performance:
            return self._get_fallback_selection()
        
        # Calcular score para cada modelo
        model_scores = {}
        detailed_analysis = {}
        
        for model_name, metrics in models_performance.items():
            score, analysis = self._calculate_model_score(model_name, metrics)
            model_scores[model_name] = score
            detailed_analysis[model_name] = analysis
        
        # Seleccionar mejor modelo
        best_model = max(model_scores.keys(), key=lambda x: model_scores[x])
        best_score = model_scores[best_model]
        
        # Verificar que cumple umbrales mínimos
        best_metrics = models_performance[best_model]
        meets_thresholds = self._validate_thresholds(best_metrics)
        
        if not meets_thresholds['valid']:
            logger.warning(f"⚠️ Mejor modelo {best_model} no cumple umbrales mínimos")
            return self._get_emergency_fallback(best_model, meets_thresholds)
        
        # Análisis de confianza
        confidence_analysis = self._analyze_confidence(best_model, best_metrics, model_scores)
        
        selection_result = {
            'selected_model': best_model,
            'selection_score': float(best_score),
            'model_metrics': best_metrics,
            'confidence_analysis': confidence_analysis,
            'detailed_analysis': detailed_analysis,
            'all_scores': model_scores,
            'selection_timestamp': datetime.now().isoformat(),
            'selection_criteria': self.selection_criteria,
            'meets_thresholds': meets_thresholds,
            'justification': self._generate_selection_justification(best_model, best_metrics, best_score)
        }
        
        logger.info(f"🏆 Modelo seleccionado: {best_model}")
        logger.info(f"📊 Score: {best_score:.3f}")
        logger.info(f"💡 Justificación: {selection_result['justification']['primary_reason']}")
        
        return selection_result
    
    def _calculate_model_score(self, model_name: str, metrics: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Calcular score multi-criterio para un modelo
        """
        # Extraer métricas
        r2 = metrics.get('r2', 0.0)
        mape = metrics.get('mape', 100.0)
        hit_rate = metrics.get('hit_rate_2pct', 0.0)
        directional_acc = metrics.get('directional_accuracy', 50.0)
        
        # Normalizar métricas (0-1)
        r2_norm = max(0, min(1, r2))  # R² ya está en 0-1
        mape_norm = max(0, min(1, (100 - min(mape, 100)) / 100))  # Invertir MAPE (menor es mejor)
        hit_rate_norm = max(0, min(1, hit_rate / 100))
        directional_norm = max(0, min(1, directional_acc / 100))
        
        # Calcular score ponderado
        score = (
            r2_norm * self.selection_criteria['r2_weight'] +
            mape_norm * self.selection_criteria['mape_weight'] +
            hit_rate_norm * self.selection_criteria['hit_rate_weight'] +
            directional_norm * self.selection_criteria['directional_accuracy_weight']
        )
        
        # Análisis detallado
        analysis = {
            'normalized_metrics': {
                'r2_norm': r2_norm,
                'mape_norm': mape_norm,
                'hit_rate_norm': hit_rate_norm,
                'directional_norm': directional_norm
            },
            'weighted_contributions': {
                'r2_contribution': r2_norm * self.selection_criteria['r2_weight'],
                'mape_contribution': mape_norm * self.selection_criteria['mape_weight'],
                'hit_rate_contribution': hit_rate_norm * self.selection_criteria['hit_rate_weight'],
                'directional_contribution': directional_norm * self.selection_criteria['directional_accuracy_weight']
            },
            'raw_metrics': metrics,
            'total_score': score,
            'grade': self._assign_performance_grade(score)
        }
        
        return score, analysis
    
    def _validate_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Validar que el modelo cumple umbrales mínimos
        """
        validations = {
            'r2_valid': metrics.get('r2', 0) >= self.performance_thresholds['min_r2'],
            'mape_valid': metrics.get('mape', 100) <= self.performance_thresholds['max_mape'],
            'hit_rate_valid': metrics.get('hit_rate_2pct', 0) >= self.performance_thresholds['min_hit_rate'],
            'directional_valid': metrics.get('directional_accuracy', 0) >= self.performance_thresholds['min_directional_acc']
        }
        
        all_valid = all(validations.values())
        
        failed_criteria = [k for k, v in validations.items() if not v]
        
        return {
            'valid': all_valid,
            'individual_validations': validations,
            'failed_criteria': failed_criteria,
            'thresholds_used': self.performance_thresholds
        }
    
    def _analyze_confidence(self, model_name: str, metrics: Dict[str, float], all_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Analizar confianza en la selección
        """
        best_score = all_scores[model_name]
        other_scores = [score for name, score in all_scores.items() if name != model_name]
        
        if other_scores:
            score_margin = best_score - max(other_scores)
            relative_advantage = (score_margin / best_score) * 100 if best_score > 0 else 0
        else:
            score_margin = 0
            relative_advantage = 100
        
        # Determinar nivel de confianza
        if relative_advantage > 20:
            confidence_level = "ALTA"
        elif relative_advantage > 10:
            confidence_level = "MEDIA"
        elif relative_advantage > 5:
            confidence_level = "BAJA"
        else:
            confidence_level = "MARGINAL"
        
        return {
            'confidence_level': confidence_level,
            'score_margin': float(score_margin),
            'relative_advantage_pct': float(relative_advantage),
            'recommendation': self._get_confidence_recommendation(confidence_level),
            'alternative_models': [name for name in all_scores.keys() if name != model_name]
        }
    
    def _generate_selection_justification(self, model_name: str, metrics: Dict[str, float], score: float) -> Dict[str, str]:
        """
        Generar justificación detallada de la selección
        """
        primary_strengths = []
        
        # Identificar fortalezas principales
        if metrics.get('r2', 0) > 0.9:
            primary_strengths.append("Excelente capacidad explicativa (R² > 0.9)")
        elif metrics.get('r2', 0) > 0.8:
            primary_strengths.append("Buena capacidad explicativa (R² > 0.8)")
        
        if metrics.get('mape', 100) < 2.0:
            primary_strengths.append("Error muy bajo (MAPE < 2%)")
        elif metrics.get('mape', 100) < 5.0:
            primary_strengths.append("Error aceptable (MAPE < 5%)")
        
        if metrics.get('hit_rate_2pct', 0) > 80:
            primary_strengths.append("Alta precisión práctica (Hit Rate > 80%)")
        elif metrics.get('hit_rate_2pct', 0) > 70:
            primary_strengths.append("Buena precisión práctica (Hit Rate > 70%)")
        
        # Justificación específica por modelo
        model_specific_reasons = {
            'MIDAS_V2_hibrida': {
                'primary_reason': 'Combina variables autorregresivas con fundamentales para máxima precisión',
                'secondary_reason': 'Excelente para capturar tendencias de mediano plazo',
                'use_case': 'Óptimo para condiciones de mercado estables'
            },
            'XGBoost_V2_regime': {
                'primary_reason': 'Captura relaciones no-lineales y cambios de régimen',
                'secondary_reason': 'Robusto ante volatilidad y cambios súbitos',
                'use_case': 'Óptimo para mercados volátiles y no-lineales'
            }
        }
        
        specific = model_specific_reasons.get(model_name, {
            'primary_reason': 'Mejor performance según métricas multi-criterio',
            'secondary_reason': 'Cumple todos los umbrales de calidad',
            'use_case': 'Predicción general de precios'
        })
        
        return {
            'primary_reason': specific['primary_reason'],
            'secondary_reason': specific['secondary_reason'],
            'use_case': specific['use_case'],
            'strengths': primary_strengths,
            'overall_score': f"{score:.3f} puntos (escala 0-1)"
        }
    
    def _assign_performance_grade(self, score: float) -> str:
        """Asignar grado de performance"""
        if score >= 0.9:
            return "EXCELENTE"
        elif score >= 0.8:
            return "BUENO"
        elif score >= 0.7:
            return "ACEPTABLE"
        elif score >= 0.6:
            return "REGULAR"
        else:
            return "DEFICIENTE"
    
    def _get_confidence_recommendation(self, confidence_level: str) -> str:
        """Obtener recomendación basada en confianza"""
        recommendations = {
            'ALTA': 'Usar este modelo con confianza total',
            'MEDIA': 'Modelo recomendado, monitorear performance',
            'BAJA': 'Usar con precaución, considerar ensemble',
            'MARGINAL': 'Diferencia mínima, usar ensemble o modelo por defecto'
        }
        
        return recommendations.get(confidence_level, 'Evaluar manualmente')
    
    def _get_fallback_selection(self) -> Dict[str, Any]:
        """Selección de fallback cuando no hay modelos disponibles"""
        logger.warning("⚠️ No hay modelos disponibles - usando fallback")
        
        return {
            'selected_model': 'fallback_simple',
            'selection_score': 0.0,
            'model_metrics': {},
            'confidence_analysis': {
                'confidence_level': 'NINGUNA',
                'recommendation': 'Entrenar modelos inmediatamente'
            },
            'justification': {
                'primary_reason': 'No hay modelos V2 disponibles',
                'secondary_reason': 'Usando predicción de fallback conservadora',
                'use_case': 'Solo para emergencias'
            },
            'is_fallback': True,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def _get_emergency_fallback(self, model_name: str, threshold_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback cuando el mejor modelo no cumple umbrales"""
        logger.error(f"🚨 Modelo {model_name} no cumple umbrales mínimos")
        
        return {
            'selected_model': model_name,  # Usar de todas formas
            'selection_score': 0.5,
            'confidence_analysis': {
                'confidence_level': 'CRÍTICA',
                'recommendation': 'Revisar datos y reentrenar inmediatamente'
            },
            'justification': {
                'primary_reason': 'Mejor modelo disponible pero no cumple estándares',
                'secondary_reason': f"Falló en: {', '.join(threshold_analysis['failed_criteria'])}",
                'use_case': 'Solo para emergencias - requiere atención inmediata'
            },
            'is_emergency': True,
            'failed_thresholds': threshold_analysis,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def get_model_comparison_for_user(self, models_performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generar comparación de modelos para mostrar al usuario final
        """
        if not models_performance:
            return {'error': 'No hay modelos disponibles para comparar'}
        
        # Seleccionar mejor modelo
        selection = self.select_best_model(models_performance)
        
        # Crear comparación detallada
        comparison_table = []
        
        for model_name, metrics in models_performance.items():
            score, analysis = self._calculate_model_score(model_name, metrics)
            
            comparison_table.append({
                'model_name': model_name,
                'score': round(score, 3),
                'grade': analysis['grade'],
                'mape': round(metrics.get('mape', 0), 2),
                'r2': round(metrics.get('r2', 0), 3),
                'hit_rate': round(metrics.get('hit_rate_2pct', 0), 1),
                'directional_accuracy': round(metrics.get('directional_accuracy', 0), 1),
                'is_selected': model_name == selection['selected_model']
            })
        
        # Ordenar por score
        comparison_table.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'selected_model': selection['selected_model'],
            'selection_justification': selection['justification'],
            'confidence_level': selection['confidence_analysis']['confidence_level'],
            'models_comparison': comparison_table,
            'selection_criteria_weights': self.selection_criteria,
            'recommendation_for_user': self._get_user_recommendation(selection)
        }
    
    def _get_user_recommendation(self, selection: Dict[str, Any]) -> str:
        """
        Generar recomendación específica para el usuario final
        """
        model_name = selection['selected_model']
        confidence = selection['confidence_analysis']['confidence_level']
        
        if selection.get('is_fallback'):
            return "⚠️ Sistema en modo de emergencia - contactar soporte técnico"
        elif selection.get('is_emergency'):
            return "🚨 Predicción con calidad reducida - usar con precaución"
        elif confidence == 'ALTA':
            return f"✅ Predicción confiable usando {model_name} con alta precisión"
        elif confidence == 'MEDIA':
            return f"📊 Predicción estándar usando {model_name} - monitorear resultados"
        else:
            return f"⚠️ Predicción con confianza limitada - considerar factores adicionales"


class ProductionModelManager:
    """
    Gestor de modelos para producción que integra selección automática
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.selector = ModelSelector()
        self.current_selection = None
        self.selection_history = []
        
    async def get_best_prediction(self, models_performance: Dict[str, Dict[str, float]], 
                                models_predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Obtener la mejor predicción para mostrar al usuario final
        
        Args:
            models_performance: Métricas de performance de cada modelo
            models_predictions: Predicciones de cada modelo
            
        Returns:
            Predicción del mejor modelo con justificación completa
        """
        logger.info("🎯 Seleccionando mejor predicción para usuario final...")
        
        # Seleccionar mejor modelo
        selection = self.selector.select_best_model(models_performance)
        best_model = selection['selected_model']
        
        # Obtener predicción del mejor modelo
        if best_model in models_predictions:
            best_prediction = models_predictions[best_model]
        else:
            # Fallback a cualquier predicción disponible
            if models_predictions:
                best_model = list(models_predictions.keys())[0]
                best_prediction = models_predictions[best_model]
                logger.warning(f"⚠️ Usando predicción de fallback: {best_model}")
            else:
                raise ValueError("No hay predicciones disponibles")
        
        # Guardar selección actual
        self.current_selection = selection
        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'selected_model': best_model,
            'score': selection['selection_score'],
            'prediction': best_prediction
        })
        
        # Mantener solo últimas 100 selecciones
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
        
        # Preparar respuesta para usuario final
        # Manejar casos donde model_metrics puede no existir (fallback/emergency)
        model_metrics = selection.get('model_metrics', {})
        detailed_analysis = selection.get('detailed_analysis', {})
        
        user_response = {
            'prediction_value': float(best_prediction),
            'model_used': best_model,
            'confidence_level': selection['confidence_analysis']['confidence_level'],
            'model_confidence_score': model_metrics.get('r2', 0.5),
            'selection_justification': selection['justification']['primary_reason'],
            'performance_grade': detailed_analysis.get(best_model, {}).get('grade', 'N/A'),
            'recommendation': self.selector._get_user_recommendation(selection),
            'model_metrics_summary': {
                'mape': round(model_metrics.get('mape', 999), 2),
                'r2': round(model_metrics.get('r2', 0), 3),
                'hit_rate': round(model_metrics.get('hit_rate_2pct', 0), 1)
            },
            'is_emergency': selection.get('is_emergency', False),
            'is_fallback': selection.get('is_fallback', False)
        }
        
        logger.info(f"🏆 Predicción final para usuario: ${best_prediction:.2f} ({best_model})")
        
        return user_response
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo actualmente seleccionado
        """
        if not self.current_selection:
            return {'error': 'No hay modelo seleccionado actualmente'}
        
        return {
            'current_model': self.current_selection['selected_model'],
            'selection_timestamp': self.current_selection['selection_timestamp'],
            'performance_metrics': self.current_selection['model_metrics'],
            'confidence_analysis': self.current_selection['confidence_analysis'],
            'justification': self.current_selection['justification']
        }
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de selecciones de modelo
        """
        return self.selection_history[-limit:] if self.selection_history else []


# Función de utilidad para integración con API
def select_best_model_for_api(models_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función helper para seleccionar mejor modelo en endpoints de API
    
    Args:
        models_data: Datos de modelos con predicciones y métricas
        
    Returns:
        Información del mejor modelo para respuesta de API
    """
    selector = ModelSelector()
    
    # Extraer métricas de performance
    models_performance = {}
    models_predictions = {}
    
    for model_name, data in models_data.items():
        if 'test_metrics' in data:
            models_performance[model_name] = data['test_metrics']
        
        if 'prediction' in data:
            models_predictions[model_name] = data['prediction']
    
    # Seleccionar mejor modelo
    selection = selector.select_best_model(models_performance)
    best_model = selection['selected_model']
    
    # Preparar respuesta
    return {
        'best_model': best_model,
        'prediction': models_predictions.get(best_model, 0.0),
        'confidence': selection['model_metrics'].get('r2', 0.8),
        'justification': selection['justification']['primary_reason'],
        'performance_grade': selection['detailed_analysis'][best_model]['grade'],
        'all_models_comparison': selection['detailed_analysis']
    }
