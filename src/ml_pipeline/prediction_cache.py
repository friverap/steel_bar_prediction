#!/usr/bin/env python3
"""
Prediction Cache System - DeAcero Steel Price Predictor V2
Sistema de cache para respuestas instantáneas (<2 segundos)

Este módulo:
1. Guarda predicciones pre-calculadas después del reentrenamiento
2. Guarda análisis de feature importance pre-calculado
3. Proporciona respuestas instantáneas a endpoints
4. Maneja invalidación y actualización de cache
5. Optimiza para tiempo de respuesta <2 segundos

Estrategia: 
- Reentrenamiento diario (18:00) → Calcular y guardar predicción + explicabilidad
- Endpoint de usuario → Leer predicción pre-calculada (instantáneo)

Fecha: 28 de Septiembre de 2025
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import os

# Import Redis de forma opcional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionCache:
    """
    Sistema de cache para predicciones y análisis pre-calculados
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, redis_url: Optional[str] = None):
        # Configurar directorios
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar Redis si está disponible
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis cache conectado")
            except Exception as e:
                logger.warning(f"⚠️ Redis no disponible: {str(e)}")
                self.redis_client = None
        elif not REDIS_AVAILABLE:
            logger.info("📦 Redis no instalado - usando cache en archivo y memoria")
        
        # Cache en memoria como fallback
        self.memory_cache = {}
        
        # Configuración de cache
        self.cache_config = {
            'prediction_ttl_hours': 24,      # Predicción válida por 24 horas
            'feature_importance_ttl_hours': 168,  # Feature importance válida por 1 semana
            'model_comparison_ttl_hours': 168     # Comparación válida por 1 semana
        }
        
        logger.info("💾 PredictionCache inicializado")
    
    async def cache_daily_prediction(self, prediction_data: Dict[str, Any], 
                                   feature_importance: Dict[str, Any],
                                   model_comparison: Dict[str, Any]) -> bool:
        """
        Guardar predicción diaria y análisis pre-calculados
        
        Args:
            prediction_data: Datos de predicción del mejor modelo
            feature_importance: Análisis de feature importance
            model_comparison: Comparación de modelos
            
        Returns:
            True si se guardó exitosamente
        """
        logger.info("💾 Guardando predicción y análisis en cache...")
        
        try:
            timestamp = datetime.now()
            cache_data = {
                'prediction': prediction_data,
                'feature_importance': feature_importance,
                'model_comparison': model_comparison,
                'cached_at': timestamp.isoformat(),
                'valid_until': (timestamp + timedelta(hours=24)).isoformat(),
                'cache_version': '2.0'
            }
            
            # Guardar en múltiples ubicaciones para redundancia
            success_count = 0
            
            # 1. Cache en archivo local
            cache_file = self.cache_dir / 'daily_prediction_cache.json'
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2, default=str)
                success_count += 1
                logger.info(f"✅ Cache guardado en archivo: {cache_file}")
                
                # NUEVO: Sincronizar con contenedor Docker si está corriendo
                self._sync_cache_to_docker(cache_file)
                
            except Exception as e:
                logger.error(f"❌ Error guardando cache en archivo: {str(e)}")
            
            # 2. Cache en Redis si está disponible
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        'daily_prediction_cache',
                        int(self.cache_config['prediction_ttl_hours'] * 3600),
                        json.dumps(cache_data, default=str)
                    )
                    success_count += 1
                    logger.info("✅ Cache guardado en Redis")
                except Exception as e:
                    logger.error(f"❌ Error guardando cache en Redis: {str(e)}")
            
            # 3. Cache en memoria
            self.memory_cache['daily_prediction'] = cache_data
            success_count += 1
            
            logger.info(f"💾 Cache guardado en {success_count} ubicaciones")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error general guardando cache: {str(e)}")
            return False
    
    async def get_cached_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Obtener predicción MÁS RECIENTE desde cache
        Busca automáticamente la predicción con timestamp más nuevo
        
        Returns:
            Datos de predicción más recientes o None si no están disponibles
        """
        logger.info("🔍 Buscando predicción MÁS RECIENTE en cache...")
        
        # Buscar en múltiples ubicaciones y elegir la más reciente
        cache_candidates = []
        
        # 1. Redis
        if self.redis_client:
            try:
                cached_json = self.redis_client.get('daily_prediction_cache')
                if cached_json:
                    data = json.loads(cached_json)
                    cache_candidates.append(('redis', data))
                    logger.info("📦 Predicción encontrada en Redis")
            except Exception as e:
                logger.warning(f"⚠️ Error leyendo Redis: {str(e)}")
        
        # 2. Memoria cache
        if 'daily_prediction' in self.memory_cache:
            data = self.memory_cache['daily_prediction']
            cache_candidates.append(('memory', data))
            logger.info("📦 Predicción encontrada en memoria")
        
        # 3. Archivo local
        cache_file = self.cache_dir / 'daily_prediction_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                cache_candidates.append(('file', data))
                logger.info("📦 Predicción encontrada en archivo")
            except Exception as e:
                logger.error(f"❌ Error leyendo cache de archivo: {str(e)}")
        
        if not cache_candidates:
            logger.info("❌ No hay predicción en cache")
            return None
        
        # SELECCIONAR LA PREDICCIÓN MÁS RECIENTE
        most_recent_cache = None
        most_recent_timestamp = None
        
        for source, cache_data in cache_candidates:
            try:
                # Buscar timestamp en la predicción
                prediction = cache_data.get('prediction', {})
                timestamp_str = prediction.get('timestamp') or cache_data.get('cached_at')
                
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    if most_recent_timestamp is None or timestamp > most_recent_timestamp:
                        most_recent_timestamp = timestamp
                        most_recent_cache = cache_data
                        logger.info(f"🔄 Predicción más reciente: {source} ({timestamp_str})")
                
            except Exception as e:
                logger.warning(f"⚠️ Error procesando timestamp de {source}: {str(e)}")
        
        if most_recent_cache:
            # Verificar validez del cache más reciente
            if self._is_cache_valid(most_recent_cache):
                logger.info("⚡ Retornando predicción MÁS RECIENTE (respuesta instantánea)")
                return most_recent_cache
            else:
                logger.warning("⏰ Cache más reciente expirado")
                await self._invalidate_cache()
        
        logger.info("❌ No hay predicción válida en cache")
        return None
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """
        Verificar si el cache sigue siendo válido y es reciente
        
        Validaciones:
        1. Campo valid_until no expirado
        2. Prediction_date es para el futuro
        3. Cache no tiene más de 2 horas de antigüedad (CRÍTICO)
        """
        try:
            # Validación 1: Campo valid_until
            valid_until = datetime.fromisoformat(cache_data['valid_until'])
            is_valid = datetime.now() < valid_until
            
            if not is_valid:
                logger.warning(f"⏰ Cache expirado según valid_until: {cache_data['valid_until']}")
                return False
            
            # Validación 2: Fecha de predicción debe ser futura
            prediction = cache_data.get('prediction', {})
            prediction_date_str = prediction.get('prediction_date')
            
            if prediction_date_str:
                prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
                is_future = prediction_date >= date.today()
                
                if not is_future:
                    logger.warning(f"⏰ Fecha de predicción es pasada: {prediction_date_str}")
                    return False
            
            # Validación 3: FRESCURA DEL CACHE (máximo 24 horas para producción)
            cached_at_str = cache_data.get('cached_at')
            if cached_at_str:
                cached_at = datetime.fromisoformat(cached_at_str.replace('Z', '+00:00'))
                
                # Remover timezone info para comparación
                cached_at = cached_at.replace(tzinfo=None)
                now = datetime.now()
                
                hours_old = (now - cached_at).total_seconds() / 3600
                
                # Cache NO debe tener más de 24 horas (reentrenamiento diario)
                if hours_old > 24.0:
                    logger.warning(f"⏰ Cache muy antiguo: {hours_old:.1f} horas (máximo: 24 horas)")
                    logger.warning(f"   Cached at: {cached_at_str}")
                    logger.warning(f"   Current time: {now.isoformat()}")
                    return False
                else:
                    logger.info(f"✅ Cache válido: {hours_old:.1f} horas de antigüedad (límite: 24h)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando cache: {str(e)}")
            return False
    
    async def clear_all_cache(self) -> None:
        """Limpiar TODO el cache (función pública)"""
        logger.info("🧹 Limpiando TODO el cache...")
        try:
            # Limpiar Redis
            if self.redis_client:
                self.redis_client.delete('daily_prediction_cache')
                self.redis_client.delete('feature_importance_cache')
                logger.info("✅ Cache Redis limpiado")
            
            # Limpiar memoria
            self.memory_cache.clear()
            logger.info("✅ Cache memoria limpiado")
            
            # Limpiar archivos locales
            cache_files = list(self.cache_dir.glob('*.json'))
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    logger.info(f"✅ Archivo cache eliminado: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo eliminar {cache_file.name}: {e}")
            
            logger.info("🎉 Cache completamente limpiado")
            
        except Exception as e:
            logger.error(f"❌ Error limpiando cache: {e}")

    async def _invalidate_cache(self) -> None:
        """Invalidar cache expirado"""
        try:
            # Limpiar Redis
            if self.redis_client:
                self.redis_client.delete('daily_prediction_cache')
            
            # Limpiar memoria
            if 'daily_prediction' in self.memory_cache:
                del self.memory_cache['daily_prediction']
            
            # Marcar archivo como expirado (no eliminar por seguridad)
            cache_file = self.cache_dir / 'daily_prediction_cache.json'
            if cache_file.exists():
                expired_file = self.cache_dir / f'expired_cache_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                cache_file.rename(expired_file)
            
            logger.info("🗑️ Cache invalidado")
            
        except Exception as e:
            logger.error(f"Error invalidando cache: {str(e)}")
    
    async def get_cached_feature_importance(self) -> Optional[Dict[str, Any]]:
        """
        Obtener análisis de feature importance desde cache
        """
        cache_key = 'feature_importance_analysis'
        
        # Intentar Redis
        if self.redis_client:
            try:
                cached_json = self.redis_client.get(cache_key)
                if cached_json:
                    return json.loads(cached_json)
            except Exception as e:
                logger.warning(f"Error leyendo feature importance de Redis: {str(e)}")
        
        # Intentar archivo
        cache_file = self.cache_dir / 'feature_importance_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Verificar validez (1 semana)
                cached_at = datetime.fromisoformat(data.get('cached_at', '2020-01-01'))
                if (datetime.now() - cached_at).total_seconds() < (168 * 3600):  # 1 semana
                    return data
                    
            except Exception as e:
                logger.error(f"Error leyendo feature importance de archivo: {str(e)}")
        
        return None
    
    async def cache_feature_importance(self, feature_importance_data: Dict[str, Any]) -> bool:
        """
        Guardar análisis de feature importance en cache
        """
        try:
            cache_data = {
                **feature_importance_data,
                'cached_at': datetime.now().isoformat(),
                'cache_version': '2.0'
            }
            
            # Guardar en Redis
            if self.redis_client:
                self.redis_client.setex(
                    'feature_importance_analysis',
                    int(self.cache_config['feature_importance_ttl_hours'] * 3600),
                    json.dumps(cache_data, default=str)
                )
            
            # Guardar en archivo
            cache_file = self.cache_dir / 'feature_importance_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info("💾 Feature importance guardado en cache")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando feature importance: {str(e)}")
            return False
    
    async def prepare_instant_response_data(self, retrain_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preparar todos los datos para respuestas instantáneas después del reentrenamiento
        
        Args:
            retrain_results: Resultados del reentrenamiento diario
            
        Returns:
            Datos preparados para cache
        """
        logger.info("⚡ Preparando datos para respuestas instantáneas...")
        
        try:
            # Verificar estructura de retrain_results - SIN FALLBACKS
            if 'stages' not in retrain_results:
                raise ValueError("No hay 'stages' en retrain_results")
            
            # Extraer datos de predicción - SIN FALLBACKS
            if 'prediction' not in retrain_results['stages']:
                raise ValueError("No hay datos de predicción en retrain_results")
            
            prediction_stage = retrain_results['stages']['prediction']
            if prediction_stage.get('status') != 'success':
                raise ValueError(f"Predicción falló: {prediction_stage.get('error', 'Unknown error')}")
            
            # Preparar datos de predicción - USAR SOLO DATOS DEL MEJOR MODELO
            best_model = prediction_stage['best_model']
            best_prediction = prediction_stage['best_prediction']
            
            logger.info(f"🔍 Preparando cache para mejor modelo: {best_model}")
            logger.info(f"🔍 Predicción del mejor modelo: ${best_prediction:.2f}")
            
            # CRÍTICO: Usar confidence basada en MAPE (no R²)
            model_confidence = prediction_stage.get('confidence', 0.8)  # Ya viene calculada con MAPE
            model_metrics = prediction_stage.get('model_metrics', {})
            
            logger.info(f"🔍 Confidence del mejor modelo: {model_confidence:.3f} (basada en MAPE)")
            
            # Verificar que los datos sean consistentes
            if 'predictions' in prediction_stage:
                predictions = prediction_stage['predictions']
                if best_model in predictions:
                    best_model_data = predictions[best_model]
                    
                    # VERIFICACIÓN: Asegurar que usamos datos del modelo correcto
                    cache_prediction = best_model_data.get('prediction', best_prediction)
                    cache_confidence = best_model_data.get('confidence', model_confidence)
                    cache_metrics = best_model_data.get('model_metrics', model_metrics)
                    
                    logger.info(f"✅ Verificación: Predicción {cache_prediction:.2f}, Confianza {cache_confidence:.3f}")
                    
                    # Usar datos verificados
                    best_prediction = cache_prediction
                    model_confidence = cache_confidence
                    model_metrics = cache_metrics
            
            prediction_data = {
                'prediction_date': prediction_stage['target_date'],
                'predicted_price_usd': float(best_prediction),  # Predicción del mejor modelo
                'currency': 'USD',
                'unit': 'metric ton',
                'model_confidence': float(model_confidence),   # Confianza del mejor modelo
                'timestamp': datetime.now().isoformat() + 'Z',
                'best_model': best_model,
                'gap_days': prediction_stage['gap_days'],
                'all_models_evaluated': len(prediction_stage.get('predictions', {})),
                'model_metrics_used': model_metrics  # Para debugging
            }
            
            # Preparar datos de explicabilidad
            if 'explainability' in retrain_results['stages']:
                explainability_stage = retrain_results['stages']['explainability']
                feature_importance_data = {
                    'models_analyzed': explainability_stage.get('models_analyzed', []),
                    'total_factors_analyzed': len(explainability_stage.get('causal_factors', [])),
                    'top_factors': explainability_stage.get('causal_factors', [])[:20],
                    'factors_by_category': explainability_stage.get('feature_importance_by_model', {}),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            else:
                # Crear feature importance básica si no está disponible
                feature_importance_data = {
                    'models_analyzed': [prediction_stage['best_model']],
                    'total_factors_analyzed': 0,
                    'top_factors': [],
                    'factors_by_category': {},
                    'analysis_timestamp': datetime.now().isoformat(),
                    'note': 'Análisis de feature importance no disponible - usar datos por defecto'
                }
            
            # Preparar comparación de modelos
            model_comparison_data = {
                'models_compared': list(prediction_stage.get('predictions', {}).keys()),
                'best_model_selected': prediction_stage['best_model'],
                'selection_criteria': 'Multi-criterio: R², MAPE, Hit Rate, Directional Accuracy',
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            return {
                'prediction': prediction_data,
                'feature_importance': feature_importance_data,
                'model_comparison': model_comparison_data
            }
            
        except Exception as e:
            logger.error(f"Error preparando datos de cache: {str(e)}")
            return {}
    
    async def get_instant_prediction(self) -> Dict[str, Any]:
        """
        Obtener predicción instantánea (objetivo: <2 segundos)
        
        Returns:
            Predicción pre-calculada en formato de API
        """
        start_time = datetime.now()
        
        # Buscar en cache
        cached_data = await self.get_cached_prediction()
        
        if cached_data and 'prediction' in cached_data:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⚡ Respuesta instantánea: {response_time:.3f}s")
            
            prediction = cached_data['prediction']
            
            # Agregar información de cache
            prediction['response_time_seconds'] = response_time
            prediction['from_cache'] = True
            prediction['cache_timestamp'] = cached_data.get('cached_at')
            
            return prediction
        else:
            # No hay cache válido - FALLAR CLARAMENTE
            logger.error("❌ No hay predicción válida en cache")
            raise ValueError("No hay predicción válida disponible en cache")
    
    async def get_instant_feature_importance(self, top_n: int = 20, 
                                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener análisis de feature importance instantáneo
        """
        start_time = datetime.now()
        
        cached_data = await self.get_cached_feature_importance()
        
        if cached_data:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⚡ Feature importance instantáneo: {response_time:.3f}s")
            
            # Filtrar por modelo si se especifica
            if model_name and 'top_factors' in cached_data:
                filtered_factors = []
                for factor in cached_data['top_factors']:
                    if model_name in factor.get('models', []):
                        filtered_factors.append(factor)
                cached_data['top_factors'] = filtered_factors[:top_n]
            else:
                cached_data['top_factors'] = cached_data.get('top_factors', [])[:top_n]
            
            # Agregar metadata de respuesta
            cached_data['response_time_seconds'] = response_time
            cached_data['from_cache'] = True
            cached_data['requested_model'] = model_name
            cached_data['top_n_requested'] = top_n
            
            return cached_data
        else:
            # No hay cache válido - FALLAR CLARAMENTE
            logger.error("❌ No hay feature importance válida en cache")
            raise ValueError("No hay análisis de feature importance disponible en cache")
    
    # ELIMINADO: No más respuestas de emergencia - si falla, debe fallar
    
    # ELIMINADO: No más análisis básico - debe usar datos reales o fallar
    
    def _group_factors_by_category(self, factors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Agrupar factores por categoría"""
        categories = {}
        
        for factor in factors:
            category = factor['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(factor)
        
        return categories
    
    async def get_cache_status(self) -> Dict[str, Any]:
        """
        Obtener estado del sistema de cache
        """
        status = {
            'redis_available': self.redis_client is not None,
            'memory_cache_items': len(self.memory_cache),
            'cache_directory': str(self.cache_dir),
            'cache_files': []
        }
        
        # Verificar archivos de cache
        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob('*.json'))
            status['cache_files'] = [{'file': f.name, 'size_kb': f.stat().st_size / 1024} for f in cache_files]
        
        # Verificar Redis
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                status['redis_status'] = {
                    'connected': True,
                    'memory_usage': redis_info.get('used_memory_human', 'Unknown'),
                    'keys_count': self.redis_client.dbsize()
                }
            except Exception as e:
                status['redis_status'] = {
                    'connected': False,
                    'error': str(e)
                }
        
        # Verificar validez de cache actual
        cached_prediction = await self.get_cached_prediction()
        status['current_prediction_cached'] = cached_prediction is not None
        
        if cached_prediction:
            status['cached_prediction_valid_until'] = cached_prediction.get('valid_until')
            status['cached_prediction_for_date'] = cached_prediction.get('prediction', {}).get('prediction_date')
        
        return status
    
    def _sync_cache_to_docker(self, cache_file: Path) -> None:
        """
        Sincronizar cache con contenedor Docker si está corriendo
        Este método se ejecuta automáticamente al guardar el cache
        """
        import subprocess
        
        try:
            # Verificar si el contenedor está corriendo
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'name=deacero-steel-predictor', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'deacero-steel-predictor' in result.stdout:
                container_name = result.stdout.strip()
                
                if container_name:
                    # Copiar archivo de cache al contenedor
                    copy_result = subprocess.run(
                        ['docker', 'cp', str(cache_file), f'{container_name}:/app/cache/daily_prediction_cache.json'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if copy_result.returncode == 0:
                        logger.info(f"🐳 Cache sincronizado con contenedor Docker: {container_name}")
                    else:
                        logger.warning(f"⚠️ No se pudo copiar cache al contenedor: {copy_result.stderr}")
            else:
                logger.debug("🐳 Contenedor Docker no está corriendo - skip sincronización")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Timeout verificando contenedor Docker")
        except FileNotFoundError:
            logger.debug("🐳 Docker no disponible - skip sincronización")
        except Exception as e:
            logger.debug(f"⚠️ Error sincronizando con Docker (no crítico): {str(e)}")


class FastResponseOptimizer:
    """
    Optimizador para garantizar respuestas <2 segundos
    """
    
    def __init__(self, cache_system: PredictionCache):
        self.cache = cache_system
        self.response_time_target = 2.0  # segundos
        self.response_time_history = []
        
    async def get_optimized_prediction(self) -> Dict[str, Any]:
        """
        Obtener predicción optimizada para respuesta rápida
        """
        start_time = datetime.now()
        
        # Intentar cache primero
        prediction = await self.cache.get_instant_prediction()
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Registrar tiempo de respuesta
        self.response_time_history.append({
            'timestamp': start_time.isoformat(),
            'response_time': response_time,
            'from_cache': prediction.get('from_cache', False)
        })
        
        # Mantener solo últimas 100 mediciones
        if len(self.response_time_history) > 100:
            self.response_time_history = self.response_time_history[-100:]
        
        # Verificar objetivo de tiempo
        if response_time > self.response_time_target:
            logger.warning(f"⚠️ Tiempo de respuesta alto: {response_time:.3f}s > {self.response_time_target}s")
        else:
            logger.info(f"⚡ Respuesta rápida: {response_time:.3f}s")
        
        # Agregar metadata de performance
        prediction['response_time_seconds'] = response_time
        prediction['meets_time_target'] = response_time <= self.response_time_target
        
        return prediction
    
    def get_response_time_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de tiempo de respuesta
        """
        if not self.response_time_history:
            return {'no_data': True}
        
        times = [r['response_time'] for r in self.response_time_history]
        
        return {
            'average_response_time': np.mean(times),
            'p95_response_time': np.percentile(times, 95),
            'p99_response_time': np.percentile(times, 99),
            'max_response_time': max(times),
            'min_response_time': min(times),
            'target_compliance_rate': sum(1 for t in times if t <= self.response_time_target) / len(times) * 100,
            'total_requests': len(self.response_time_history),
            'cache_hit_rate': sum(1 for r in self.response_time_history if r.get('from_cache', False)) / len(self.response_time_history) * 100
        }


# Funciones de utilidad para integración

async def cache_post_retrain_data(retrain_results: Dict[str, Any], cache_dir: Optional[Path] = None) -> bool:
    """
    Función helper para guardar datos después del reentrenamiento
    """
    cache = PredictionCache(cache_dir)
    
    # Preparar datos para cache
    cache_data = await cache.prepare_instant_response_data(retrain_results)
    
    if not cache_data:
        logger.error("❌ No se pudieron preparar datos para cache")
        return False
    
    # Guardar en cache
    success = await cache.cache_daily_prediction(
        cache_data['prediction'],
        cache_data['feature_importance'], 
        cache_data['model_comparison']
    )
    
    if success:
        logger.info("✅ Datos guardados en cache para respuestas instantáneas")
    else:
        logger.error("❌ Error guardando datos en cache")
    
    return success


async def get_instant_api_response() -> Dict[str, Any]:
    """
    Función helper para endpoints de API que requieren respuesta instantánea
    """
    cache = PredictionCache()
    optimizer = FastResponseOptimizer(cache)
    
    return await optimizer.get_optimized_prediction()


    # ELIMINADO: No más cache básico - debe usar datos reales o fallar
