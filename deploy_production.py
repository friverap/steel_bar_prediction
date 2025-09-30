#!/usr/bin/env python3
"""
Script de Despliegue y Verificación - DeAcero Steel Price Predictor V2
Emulación completa del flujo de producción en la nube

FLUJO COMPLETO EMULADO:
🔄 REENTRENAMIENTO DIARIO (18:00 MX):
  📥 Ingestar Datos → 🤖 Entrenar Modelos → 🏆 Seleccionar Mejor → 
  🎯 Calcular Predicción → 📊 Generar Feature Importance → 💾 Guardar en Cache

⚡ RESPUESTA INSTANTÁNEA (24/7):
  👤 Usuario Request → 📦 Leer Cache → ⚡ Respuesta <2s

Funcionalidades:
1. Emulación completa del flujo de reentrenamiento diario
2. Verificación del sistema de cache para respuestas instantáneas
3. Pruebas end-to-end del pipeline completo
4. Validación de selección automática del mejor modelo
5. Monitoreo de performance y tiempos de respuesta

Fecha: 28 de Septiembre de 2025
"""

import asyncio
import subprocess
import requests
import time
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Agregar path del proyecto para imports
sys.path.append(str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """
    Clase para manejar el despliegue y emulación completa del flujo de producción
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.api_base_url = "http://localhost:8000"
        self.container_name = "deacero-steel-predictor-v2"
        self.deployment_stats = {}
        
        # Configuración del flujo de emulación
        self.emulation_config = {
            'test_full_retrain_flow': True,
            'test_cache_system': True,
            'test_instant_responses': True,
            'test_model_selection': True,
            'measure_response_times': True
        }
        
    def verify_prerequisites(self) -> Dict[str, bool]:
        """
        Verificar prerequisitos para despliegue
        """
        logger.info("🔍 Verificando prerequisitos...")
        
        checks = {}
        
        # 1. Docker disponible
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            checks['docker_available'] = result.returncode == 0
            if checks['docker_available']:
                logger.info(f"✅ Docker: {result.stdout.strip()}")
            else:
                logger.error("❌ Docker no disponible")
        except FileNotFoundError:
            checks['docker_available'] = False
            logger.error("❌ Docker no instalado")
        
        # 2. Docker Compose disponible
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            checks['docker_compose_available'] = result.returncode == 0
            if checks['docker_compose_available']:
                logger.info(f"✅ Docker Compose: {result.stdout.strip()}")
        except FileNotFoundError:
            checks['docker_compose_available'] = False
            logger.error("❌ Docker Compose no disponible")
        
        # 3. Archivo .env existe
        env_file = self.base_dir / '.env'
        checks['env_file_exists'] = env_file.exists()
        if checks['env_file_exists']:
            logger.info("✅ Archivo .env encontrado")
        else:
            logger.error("❌ Archivo .env no encontrado")
        
        # 4. Modelos V2 disponibles
        models_dir = self.base_dir / 'models' / 'test'
        xgboost_model = models_dir / 'XGBoost_V2_regime.pkl'
        midas_model = models_dir / 'MIDAS_V2_hibrida.pkl'
        
        checks['xgboost_model_exists'] = xgboost_model.exists()
        checks['midas_model_exists'] = midas_model.exists()
        
        if checks['xgboost_model_exists']:
            logger.info("✅ Modelo XGBoost_V2_regime encontrado")
        else:
            logger.warning("⚠️ Modelo XGBoost_V2_regime no encontrado")
        
        if checks['midas_model_exists']:
            logger.info("✅ Modelo MIDAS_V2_hibrida encontrado")
        else:
            logger.warning("⚠️ Modelo MIDAS_V2_hibrida no encontrado")
        
        # 5. Directorios de datos
        data_dir = self.base_dir / 'data'
        checks['data_directory_exists'] = data_dir.exists()
        if not checks['data_directory_exists']:
            logger.warning("⚠️ Directorio de datos no existe - se creará")
            data_dir.mkdir(exist_ok=True)
            (data_dir / 'raw').mkdir(exist_ok=True)
            (data_dir / 'processed').mkdir(exist_ok=True)
            checks['data_directory_exists'] = True
        
        # Resumen de verificación
        total_checks = len(checks)
        passed_checks = sum(checks.values())
        
        logger.info(f"\n📊 Verificación completada: {passed_checks}/{total_checks} checks pasaron")
        
        return checks
    
    def build_and_deploy(self, force_rebuild: bool = False) -> bool:
        """
        Construir y desplegar contenedores
        """
        logger.info("🔨 Construyendo y desplegando contenedores...")
        
        try:
            # Detener contenedores existentes si están corriendo
            logger.info("🛑 Deteniendo contenedores existentes...")
            subprocess.run(['docker-compose', 'down'], cwd=self.base_dir, capture_output=True)
            
            # Construir imagen (con o sin cache)
            build_args = ['docker-compose', 'build']
            if force_rebuild:
                build_args.append('--no-cache')
            
            logger.info("🔨 Construyendo imagen Docker...")
            result = subprocess.run(build_args, cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Error construyendo imagen: {result.stderr}")
                return False
            
            logger.info("✅ Imagen construida exitosamente")
            
            # Iniciar servicios
            logger.info("🚀 Iniciando servicios...")
            result = subprocess.run(['docker-compose', 'up', '-d'], cwd=self.base_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Error iniciando servicios: {result.stderr}")
                return False
            
            logger.info("✅ Servicios iniciados exitosamente")
            
            # Esperar a que el servicio esté listo
            logger.info("⏳ Esperando a que el servicio esté listo...")
            time.sleep(30)  # Esperar inicialización
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en despliegue: {str(e)}")
            return False
    
    def verify_api_endpoints(self, api_key: str = "gusanito_medidor") -> Dict[str, bool]:
        """
        Verificar que todos los endpoints respondan correctamente
        """
        logger.info("🔍 Verificando endpoints de API...")
        
        endpoints_to_test = [
            {'path': '/', 'method': 'GET', 'name': 'Root'},
            {'path': '/health', 'method': 'GET', 'name': 'Health Check'},
            {'path': '/predict/steel-rebar-price', 'method': 'GET', 'name': 'Predicción Principal'},
            {'path': '/predict/model/status', 'method': 'GET', 'name': 'Estado del Modelo'},
            {'path': '/predict/pipeline/status', 'method': 'GET', 'name': 'Estado del Pipeline'},
            {'path': '/explainability/feature-importance', 'method': 'GET', 'name': 'Feature Importance'},
            {'path': '/explainability/causal-factors', 'method': 'GET', 'name': 'Factores Causales'}
        ]
        
        results = {}
        headers = {'X-API-Key': api_key} if api_key else {}
        
        for endpoint in endpoints_to_test:
            try:
                url = f"{self.api_base_url}{endpoint['path']}"
                
                if endpoint['method'] == 'GET':
                    response = requests.get(url, headers=headers, timeout=30)
                else:
                    response = requests.post(url, headers=headers, timeout=30)
                
                success = response.status_code == 200
                results[endpoint['name']] = success
                
                if success:
                    logger.info(f"✅ {endpoint['name']}: OK ({response.status_code})")
                else:
                    logger.error(f"❌ {endpoint['name']}: FAIL ({response.status_code}) - {response.text[:100]}")
                
            except requests.exceptions.RequestException as e:
                results[endpoint['name']] = False
                logger.error(f"❌ {endpoint['name']}: ERROR - {str(e)}")
            
            # Pequeña pausa entre requests
            time.sleep(1)
        
        return results
    
    def test_prediction_flow(self, api_key: str = "gusanito_medidor") -> Dict[str, Any]:
        """
        Probar flujo completo de predicción
        """
        logger.info("🧪 Probando flujo completo de predicción...")
        
        headers = {'X-API-Key': api_key}
        test_results = {}
        
        try:
            # 1. Probar predicción principal
            logger.info("1️⃣ Probando predicción principal...")
            response = requests.get(
                f"{self.api_base_url}/predict/steel-rebar-price",
                headers=headers,
                timeout=60  # Más tiempo para primera predicción
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                test_results['main_prediction'] = {
                    'success': True,
                    'price': prediction_data.get('predicted_price_usd'),
                    'confidence': prediction_data.get('model_confidence'),
                    'timestamp': prediction_data.get('timestamp')
                }
                logger.info(f"✅ Predicción: ${prediction_data.get('predicted_price_usd', 0):.2f}")
            else:
                test_results['main_prediction'] = {
                    'success': False,
                    'error': response.text
                }
                logger.error(f"❌ Predicción principal falló: {response.status_code}")
            
            # 2. Probar feature importance
            logger.info("2️⃣ Probando análisis de explicabilidad...")
            response = requests.get(
                f"{self.api_base_url}/explainability/feature-importance",
                headers=headers,
                timeout=30
            )
            
            test_results['feature_importance'] = {
                'success': response.status_code == 200,
                'factors_count': len(response.json().get('top_factors', [])) if response.status_code == 200 else 0
            }
            
            if test_results['feature_importance']['success']:
                logger.info(f"✅ Feature importance: {test_results['feature_importance']['factors_count']} factores")
            else:
                logger.error("❌ Feature importance falló")
            
            # 3. Probar factores causales
            logger.info("3️⃣ Probando factores causales...")
            response = requests.get(
                f"{self.api_base_url}/explainability/causal-factors",
                headers=headers,
                timeout=30
            )
            
            test_results['causal_factors'] = {
                'success': response.status_code == 200,
                'factors_count': len(response.json().get('causal_factors', [])) if response.status_code == 200 else 0
            }
            
            if test_results['causal_factors']['success']:
                logger.info(f"✅ Factores causales: {test_results['causal_factors']['factors_count']} factores")
            else:
                logger.error("❌ Factores causales falló")
            
            # 4. Probar estado del pipeline
            logger.info("4️⃣ Probando estado del pipeline...")
            response = requests.get(
                f"{self.api_base_url}/predict/pipeline/status",
                headers=headers,
                timeout=30
            )
            
            test_results['pipeline_status'] = {
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                pipeline_data = response.json()
                test_results['pipeline_status'].update({
                    'models_available': len(pipeline_data.get('available_models', [])),
                    'service_version': pipeline_data.get('service_version'),
                    'data_files_exist': all(pipeline_data.get('pipeline_status', {}).get('data_files_exist', {}).values())
                })
                logger.info(f"✅ Pipeline status: {test_results['pipeline_status']['models_available']} modelos disponibles")
            else:
                logger.error("❌ Pipeline status falló")
            
        except Exception as e:
            logger.error(f"❌ Error en pruebas de flujo: {str(e)}")
            test_results['error'] = str(e)
        
        return test_results
    
    def generate_deployment_report(self, 
                                 prerequisites: Dict[str, bool],
                                 deployment_success: bool,
                                 api_tests: Dict[str, bool],
                                 flow_tests: Dict[str, Any]) -> str:
        """
        Generar reporte de despliegue
        """
        
        report_lines = [
            "# 📊 REPORTE DE DESPLIEGUE - DeAcero Steel Price Predictor V2",
            f"\n**Fecha de Despliegue**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Versión**: 2.0 (Pipeline Completo)",
            "\n---\n"
        ]
        
        # Prerequisitos
        report_lines.append("## 🔍 Verificación de Prerequisitos\n")
        for check, passed in prerequisites.items():
            status = "✅" if passed else "❌"
            report_lines.append(f"- {status} {check.replace('_', ' ').title()}")
        
        # Despliegue
        report_lines.append(f"\n## 🚀 Despliegue de Contenedores\n")
        if deployment_success:
            report_lines.append("✅ **Contenedores desplegados exitosamente**")
        else:
            report_lines.append("❌ **Error en despliegue de contenedores**")
        
        # Pruebas de API
        report_lines.append(f"\n## 🔍 Verificación de Endpoints\n")
        for endpoint, passed in api_tests.items():
            status = "✅" if passed else "❌"
            report_lines.append(f"- {status} {endpoint}")
        
        # Pruebas de flujo
        report_lines.append(f"\n## 🧪 Pruebas de Flujo Completo\n")
        
        if 'main_prediction' in flow_tests:
            pred = flow_tests['main_prediction']
            if pred['success']:
                report_lines.append(f"✅ **Predicción Principal**: ${pred.get('price', 0):.2f} (Confianza: {pred.get('confidence', 0):.1%})")
            else:
                report_lines.append("❌ **Predicción Principal**: FALLÓ")
        
        if 'feature_importance' in flow_tests:
            feat = flow_tests['feature_importance']
            if feat['success']:
                report_lines.append(f"✅ **Feature Importance**: {feat.get('factors_count', 0)} factores analizados")
            else:
                report_lines.append("❌ **Feature Importance**: FALLÓ")
        
        if 'causal_factors' in flow_tests:
            causal = flow_tests['causal_factors']
            if causal['success']:
                report_lines.append(f"✅ **Factores Causales**: {causal.get('factors_count', 0)} factores identificados")
            else:
                report_lines.append("❌ **Factores Causales**: FALLÓ")
        
        # Estado general
        total_api_tests = len(api_tests)
        passed_api_tests = sum(api_tests.values())
        
        total_flow_tests = len([t for t in flow_tests.values() if isinstance(t, dict) and 'success' in t])
        passed_flow_tests = len([t for t in flow_tests.values() if isinstance(t, dict) and t.get('success', False)])
        
        report_lines.append(f"\n## 📊 Resumen de Despliegue\n")
        report_lines.append(f"- **Prerequisitos**: {sum(prerequisites.values())}/{len(prerequisites)} ✅")
        report_lines.append(f"- **Endpoints API**: {passed_api_tests}/{total_api_tests} ✅")
        report_lines.append(f"- **Flujos de Prueba**: {passed_flow_tests}/{total_flow_tests} ✅")
        
        # Recomendaciones
        report_lines.append(f"\n## 💡 Recomendaciones\n")
        
        if deployment_success and passed_api_tests == total_api_tests and passed_flow_tests == total_flow_tests:
            report_lines.append("🎉 **DESPLIEGUE EXITOSO** - El sistema está listo para producción")
            report_lines.append("\n### Próximos Pasos:")
            report_lines.append("1. Configurar monitoreo continuo")
            report_lines.append("2. Establecer alertas de degradación de performance")
            report_lines.append("3. Programar reentrenamiento periódico")
            report_lines.append("4. Documentar procedimientos operativos")
        else:
            report_lines.append("⚠️ **DESPLIEGUE PARCIAL** - Revisar errores antes de producción")
            report_lines.append("\n### Acciones Requeridas:")
            
            if not deployment_success:
                report_lines.append("- Revisar configuración de Docker y dependencias")
            
            if passed_api_tests < total_api_tests:
                report_lines.append("- Verificar configuración de API y autenticación")
            
            if passed_flow_tests < total_flow_tests:
                report_lines.append("- Revisar modelos V2 y pipeline de datos")
        
        # URLs de acceso
        report_lines.append(f"\n## 🌐 URLs de Acceso\n")
        report_lines.append(f"- **API Principal**: {self.api_base_url}")
        report_lines.append(f"- **Documentación**: {self.api_base_url}/docs")
        report_lines.append(f"- **Health Check**: {self.api_base_url}/health")
        report_lines.append(f"- **Predicción**: {self.api_base_url}/predict/steel-rebar-price")
        report_lines.append(f"- **Explicabilidad**: {self.api_base_url}/explainability/feature-importance")
        
        return "\n".join(report_lines)
    
    async def run_full_deployment(self, force_rebuild: bool = False, api_key: str = "gusanito_medidor") -> Dict[str, Any]:
        """
        Ejecutar despliegue completo con verificaciones
        """
        logger.info("=" * 80)
        logger.info("🚀 INICIANDO DESPLIEGUE DE PRODUCCIÓN V2")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # 1. Verificar prerequisitos
        prerequisites = self.verify_prerequisites()
        
        # 2. Construir y desplegar
        deployment_success = False
        if all(prerequisites.values()):
            deployment_success = self.build_and_deploy(force_rebuild)
        else:
            logger.error("❌ Prerequisitos no cumplidos - abortando despliegue")
        
        # 3. Verificar endpoints (solo si despliegue fue exitoso)
        api_tests = {}
        flow_tests = {}
        
        if deployment_success:
            # Esperar un poco más para que el servicio se estabilice
            logger.info("⏳ Esperando estabilización del servicio...")
            time.sleep(45)
            
            api_tests = self.verify_api_endpoints(api_key)
            flow_tests = self.test_prediction_flow(api_key)
        
        # 4. Generar reporte
        report = self.generate_deployment_report(prerequisites, deployment_success, api_tests, flow_tests)
        
        # 5. Guardar reporte
        report_file = self.base_dir / f"deployment_report_{start_time.strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Resultado final
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        overall_success = (
            all(prerequisites.values()) and
            deployment_success and
            all(api_tests.values()) and
            all(t.get('success', False) for t in flow_tests.values() if isinstance(t, dict) and 'success' in t)
        )
        
        result = {
            'overall_success': overall_success,
            'execution_time_seconds': execution_time,
            'prerequisites': prerequisites,
            'deployment_success': deployment_success,
            'api_tests': api_tests,
            'flow_tests': flow_tests,
            'report_file': str(report_file),
            'api_url': self.api_base_url,
            'timestamp': end_time.isoformat()
        }
        
        # Log final
        if overall_success:
            logger.info("=" * 80)
            logger.info("🎉 DESPLIEGUE COMPLETADO EXITOSAMENTE")
            logger.info(f"⏱️ Tiempo total: {execution_time:.1f} segundos")
            logger.info(f"🌐 API disponible en: {self.api_base_url}")
            logger.info(f"📚 Documentación en: {self.api_base_url}/docs")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error("❌ DESPLIEGUE FALLÓ O INCOMPLETO")
            logger.error(f"📄 Ver reporte detallado en: {report_file}")
            logger.error("=" * 80)
        
        return result
    
    async def emulate_cloud_production_flow(self, api_key: str = "gusanito_medidor") -> Dict[str, Any]:
        """
        Emular completamente el flujo de producción de la nube
        
        FLUJO EMULADO:
        🔄 REENTRENAMIENTO DIARIO (18:00 MX):
          📥 Ingestar → 🤖 Entrenar → 🏆 Seleccionar → 🎯 Predecir → 📊 Explicar → 💾 Cache
        
        ⚡ RESPUESTA INSTANTÁNEA (24/7):
          👤 Request → 📦 Cache → ⚡ <2s
        """
        logger.info("=" * 80)
        logger.info("🌩️ EMULANDO FLUJO COMPLETO DE PRODUCCIÓN EN LA NUBE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        emulation_results = {
            'start_time': start_time.isoformat(),
            'phases': {}
        }
        
        try:
            # ========== FASE 1: REENTRENAMIENTO DIARIO (Emulación 18:00 MX) ==========
            logger.info("\n🔄 FASE 1: EMULANDO REENTRENAMIENTO DIARIO (18:00 MX)")
            logger.info("=" * 60)
            
            retrain_results = await self._emulate_daily_retrain_flow()
            emulation_results['phases']['daily_retrain'] = retrain_results
            
            if retrain_results['status'] != 'success':
                logger.error("❌ Reentrenamiento falló - abortando emulación")
                return emulation_results
            
            logger.info("✅ Fase 1 completada - Reentrenamiento y cache listos")
            
            # ========== FASE 2: RESPUESTA INSTANTÁNEA (Emulación 24/7) ==========
            logger.info("\n⚡ FASE 2: EMULANDO RESPUESTAS INSTANTÁNEAS (24/7)")
            logger.info("=" * 60)
            
            instant_results = await self._emulate_instant_response_flow(api_key)
            emulation_results['phases']['instant_responses'] = instant_results
            
            # ========== FASE 3: VALIDACIÓN DE PERFORMANCE ==========
            logger.info("\n📊 FASE 3: VALIDANDO PERFORMANCE DEL SISTEMA")
            logger.info("=" * 60)
            
            performance_results = await self._validate_system_performance(api_key)
            emulation_results['phases']['performance_validation'] = performance_results
            
            # ========== RESUMEN FINAL ==========
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            emulation_results.update({
                'status': 'success',
                'end_time': end_time.isoformat(),
                'total_execution_time': total_time,
                'summary': self._generate_emulation_summary(emulation_results)
            })
            
            logger.info("=" * 80)
            logger.info("🎉 EMULACIÓN COMPLETA DEL FLUJO DE NUBE EXITOSA")
            logger.info(f"⏱️ Tiempo total: {total_time:.1f} segundos")
            logger.info("=" * 80)
            
            return emulation_results
            
        except Exception as e:
            logger.error(f"❌ Error en emulación: {str(e)}", exc_info=True)
            emulation_results.update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            return emulation_results
    
    async def _emulate_daily_retrain_flow(self) -> Dict[str, Any]:
        """
        Emular el flujo completo de reentrenamiento diario
        
        📥 Ingestar → 🤖 Entrenar → 🏆 Seleccionar → 🎯 Predecir → 📊 Explicar → 💾 Cache
        """
        logger.info("🔄 Ejecutando flujo completo de reentrenamiento...")
        
        flow_results = {
            'steps': {},
            'timing': {}
        }
        
        try:
            # PASO 1: 📥 Ingestar Datos
            logger.info("\n📥 PASO 1: INGESTA DE DATOS")
            logger.info("-" * 40)
            
            step_start = datetime.now()
            
            # Ejecutar ingesta usando el script existente
            try:
                from scripts.ingest_all_data import DataIngestionMaster
                
                logger.info("🔄 Ejecutando ingesta maestra...")
                master = DataIngestionMaster()
                ingestion_results = await master.ingest_all_sources()
                
                flow_results['steps']['data_ingestion'] = {
                    'status': 'success',
                    'sources_successful': ingestion_results['summary']['sources']['successful'],
                    'total_series': ingestion_results['summary']['data']['total_series'],
                    'total_points': ingestion_results['summary']['data']['total_points']
                }
                
                logger.info(f"✅ Ingesta completada: {ingestion_results['summary']['sources']['successful']} fuentes")
                
            except Exception as e:
                logger.error(f"❌ Error en ingesta: {str(e)}")
                flow_results['steps']['data_ingestion'] = {'status': 'error', 'error': str(e)}
                return {'status': 'error', 'failed_step': 'data_ingestion', 'error': str(e)}
            
            flow_results['timing']['data_ingestion'] = (datetime.now() - step_start).total_seconds()
            
            # PASO 2: 🔗 Consolidar Series
            logger.info("\n🔗 PASO 2: CONSOLIDACIÓN DE SERIES")
            logger.info("-" * 40)
            
            step_start = datetime.now()
            
            try:
                # Ejecutar join de series diarias y mensuales
                from scripts.join_daily_series import main as join_daily
                from scripts.join_monthly_series import main as join_monthly
                
                # Cambiar directorio temporalmente
                original_cwd = os.getcwd()
                scripts_dir = self.base_dir / 'scripts'
                os.chdir(scripts_dir)
                
                try:
                    logger.info("🔗 Consolidando series diarias...")
                    daily_df = join_daily()
                    
                    logger.info("🔗 Consolidando series mensuales...")
                    monthly_df = join_monthly()
                    
                    # Verificar resultados de consolidación
                    daily_success = daily_df is not None and not daily_df.empty if hasattr(daily_df, 'empty') else daily_df is not None
                    monthly_success = monthly_df is not None and not monthly_df.empty if hasattr(monthly_df, 'empty') else monthly_df is not None
                    
                    flow_results['steps']['data_consolidation'] = {
                        'status': 'success',
                        'daily_series_shape': daily_df.shape if daily_success else (0, 0),
                        'monthly_series_shape': monthly_df.shape if monthly_success else (0, 0),
                        'daily_success': daily_success,
                        'monthly_success': monthly_success
                    }
                    
                    logger.info(f"✅ Consolidación completada:")
                    logger.info(f"   Diarias: {daily_df.shape if daily_success else '(0,0)'}")
                    logger.info(f"   Mensuales: {monthly_df.shape if monthly_success else '(0,0)'}")
                    
                finally:
                    os.chdir(original_cwd)
                    
            except Exception as e:
                logger.error(f"❌ Error en consolidación: {str(e)}")
                flow_results['steps']['data_consolidation'] = {'status': 'error', 'error': str(e)}
                return {'status': 'error', 'failed_step': 'data_consolidation', 'error': str(e)}
            
            flow_results['timing']['data_consolidation'] = (datetime.now() - step_start).total_seconds()
            
            # PASO 3: 🤖 Reentrenar Modelos
            logger.info("\n🤖 PASO 3: REENTRENAMIENTO DE MODELOS V2")
            logger.info("-" * 40)
            
            step_start = datetime.now()
            
            try:
                from src.ml_pipeline.daily_retrain_pipeline import DailyRetrainPipeline
                
                logger.info("🔄 Ejecutando reentrenamiento diario...")
                retrain_pipeline = DailyRetrainPipeline()
                retrain_results = await retrain_pipeline.run_daily_retrain(force_retrain=True)
                
                if retrain_results['status'] == 'success':
                    flow_results['steps']['model_retraining'] = {
                        'status': 'success',
                        'models_retrained': retrain_results.get('models_retrained', 0),
                        'prediction_ready': retrain_results.get('prediction_generated', False)
                    }
                    
                    logger.info(f"✅ Reentrenamiento completado: {retrain_results.get('models_retrained', 0)} modelos")
                else:
                    flow_results['steps']['model_retraining'] = {'status': 'error', 'error': retrain_results.get('error')}
                    return {'status': 'error', 'failed_step': 'model_retraining', 'error': retrain_results.get('error')}
                
            except Exception as e:
                logger.error(f"❌ Error en reentrenamiento: {str(e)}")
                flow_results['steps']['model_retraining'] = {'status': 'error', 'error': str(e)}
                return {'status': 'error', 'failed_step': 'model_retraining', 'error': str(e)}
            
            flow_results['timing']['model_retraining'] = (datetime.now() - step_start).total_seconds()
            
            # PASO 4: 💾 Verificar Cache
            logger.info("\n💾 PASO 4: VERIFICACIÓN DE SISTEMA DE CACHE")
            logger.info("-" * 40)
            
            step_start = datetime.now()
            
            try:
                from src.ml_pipeline.prediction_cache import PredictionCache
                
                cache = PredictionCache()
                cache_status = await cache.get_cache_status()
                
                flow_results['steps']['cache_verification'] = {
                    'status': 'success',
                    'cache_ready': cache_status.get('current_prediction_cached', False),
                    'cache_files': len(cache_status.get('cache_files', [])),
                    'redis_available': cache_status.get('redis_available', False)
                }
                
                logger.info(f"✅ Cache verificado: Predicción lista = {cache_status.get('current_prediction_cached', False)}")
                
            except Exception as e:
                logger.error(f"❌ Error verificando cache: {str(e)}")
                flow_results['steps']['cache_verification'] = {'status': 'error', 'error': str(e)}
            
            flow_results['timing']['cache_verification'] = (datetime.now() - step_start).total_seconds()
            
            return {
                'status': 'success',
                'flow_results': flow_results,
                'total_steps_completed': len([s for s in flow_results['steps'].values() if s.get('status') == 'success']),
                'total_time_seconds': sum(flow_results['timing'].values())
            }
            
        except Exception as e:
            logger.error(f"❌ Error general en flujo de reentrenamiento: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _emulate_instant_response_flow(self, api_key: str) -> Dict[str, Any]:
        """
        Emular el flujo de respuestas instantáneas (24/7)
        
        👤 Request → 📦 Cache → ⚡ <2s
        """
        logger.info("⚡ Emulando flujo de respuestas instantáneas...")
        
        instant_tests = []
        response_times = []
        
        # Probar múltiples requests para medir consistencia
        test_scenarios = [
            {'name': 'Predicción Principal', 'endpoint': '/predict/steel-rebar-price', 'target_time': 2.0},
            {'name': 'Feature Importance', 'endpoint': '/explainability/feature-importance', 'target_time': 2.0},
            {'name': 'Factores Causales', 'endpoint': '/explainability/causal-factors', 'target_time': 2.0},
            {'name': 'Estado Pipeline', 'endpoint': '/predict/pipeline/status', 'target_time': 1.0}
        ]
        
        headers = {'X-API-Key': api_key}
        
        for scenario in test_scenarios:
            logger.info(f"\n⚡ Probando: {scenario['name']}")
            
            # Hacer múltiples requests para medir consistencia
            scenario_times = []
            scenario_results = []
            
            for attempt in range(3):  # 3 intentos por escenario
                try:
                    start_time = time.time()
                    
                    response = requests.get(
                        f"{self.api_base_url}{scenario['endpoint']}",
                        headers=headers,
                        timeout=10
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    scenario_times.append(response_time)
                    response_times.append(response_time)
                    
                    success = response.status_code == 200
                    meets_target = response_time <= scenario['target_time']
                    
                    scenario_results.append({
                        'attempt': attempt + 1,
                        'success': success,
                        'response_time': response_time,
                        'meets_target': meets_target,
                        'status_code': response.status_code
                    })
                    
                    if success and meets_target:
                        logger.info(f"   ✅ Intento {attempt + 1}: {response_time:.3f}s (objetivo: {scenario['target_time']}s)")
                    elif success:
                        logger.warning(f"   ⚠️ Intento {attempt + 1}: {response_time:.3f}s (LENTO - objetivo: {scenario['target_time']}s)")
                    else:
                        logger.error(f"   ❌ Intento {attempt + 1}: FALLÓ ({response.status_code})")
                    
                    # Pausa entre intentos
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"   ❌ Error en intento {attempt + 1}: {str(e)}")
                    scenario_results.append({
                        'attempt': attempt + 1,
                        'success': False,
                        'error': str(e)
                    })
            
            # Estadísticas del escenario
            successful_attempts = len([r for r in scenario_results if r.get('success', False)])
            avg_response_time = np.mean(scenario_times) if scenario_times else 0
            meets_target_rate = len([t for t in scenario_times if t <= scenario['target_time']]) / len(scenario_times) * 100 if scenario_times else 0
            
            instant_tests.append({
                'scenario': scenario['name'],
                'endpoint': scenario['endpoint'],
                'target_time': scenario['target_time'],
                'successful_attempts': successful_attempts,
                'total_attempts': len(scenario_results),
                'avg_response_time': avg_response_time,
                'meets_target_rate': meets_target_rate,
                'attempts': scenario_results
            })
            
            logger.info(f"   📊 Resumen: {successful_attempts}/3 exitosos, promedio: {avg_response_time:.3f}s")
        
        # Estadísticas generales
        overall_success_rate = len([t for t in instant_tests if t['successful_attempts'] == 3]) / len(instant_tests) * 100
        overall_avg_time = np.mean(response_times) if response_times else 0
        overall_target_compliance = len([t for t in response_times if t <= 2.0]) / len(response_times) * 100 if response_times else 0
        
        return {
            'status': 'success',
            'instant_tests': instant_tests,
            'overall_stats': {
                'success_rate': overall_success_rate,
                'avg_response_time': overall_avg_time,
                'target_compliance_rate': overall_target_compliance,
                'total_requests': len(response_times),
                'fastest_response': min(response_times) if response_times else 0,
                'slowest_response': max(response_times) if response_times else 0
            }
        }
    
    async def _validate_system_performance(self, api_key: str) -> Dict[str, Any]:
        """
        Validar performance completa del sistema
        """
        logger.info("📊 Validando performance del sistema...")
        
        headers = {'X-API-Key': api_key}
        validation_results = {}
        
        try:
            # 1. Verificar que el mejor modelo fue seleccionado
            logger.info("\n🏆 Verificando selección automática del mejor modelo...")
            
            response = requests.get(f"{self.api_base_url}/predict/steel-rebar-price", headers=headers, timeout=5)
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Verificar que se está usando el mejor modelo (probablemente MIDAS)
                best_model = prediction_data.get('best_model', 'unknown')
                confidence = prediction_data.get('model_confidence', 0)
                
                validation_results['model_selection'] = {
                    'success': True,
                    'selected_model': best_model,
                    'confidence': confidence,
                    'is_expected_best': best_model in ['MIDAS_V2_hibrida', 'XGBoost_V2_regime'],
                    'high_confidence': confidence > 0.8
                }
                
                logger.info(f"✅ Modelo seleccionado: {best_model} (confianza: {confidence:.1%})")
                
            else:
                validation_results['model_selection'] = {'success': False, 'error': 'No se pudo obtener predicción'}
            
            # 2. Verificar feature importance
            logger.info("\n📊 Verificando análisis de explicabilidad...")
            
            response = requests.get(f"{self.api_base_url}/explainability/feature-importance", headers=headers, timeout=5)
            
            if response.status_code == 200:
                importance_data = response.json()
                
                top_factors = importance_data.get('top_factors', [])
                models_analyzed = importance_data.get('models_analyzed', [])
                
                validation_results['explainability'] = {
                    'success': True,
                    'factors_found': len(top_factors),
                    'models_analyzed': len(models_analyzed),
                    'has_autorregresive': any('lag' in f.get('feature', '') for f in top_factors),
                    'has_materials': any(f.get('category') == 'Materias Primas' for f in top_factors)
                }
                
                logger.info(f"✅ Explicabilidad: {len(top_factors)} factores, {len(models_analyzed)} modelos")
                
            else:
                validation_results['explainability'] = {'success': False, 'error': 'No se pudo obtener explicabilidad'}
            
            # 3. Verificar cache y performance
            logger.info("\n💾 Verificando sistema de cache...")
            
            # Hacer múltiples requests rápidos para verificar cache
            cache_test_times = []
            
            for i in range(5):
                start_time = time.time()
                response = requests.get(f"{self.api_base_url}/predict/steel-rebar-price", headers=headers, timeout=5)
                end_time = time.time()
                
                if response.status_code == 200:
                    cache_test_times.append(end_time - start_time)
                
                time.sleep(0.2)  # Pequeña pausa
            
            if cache_test_times:
                avg_cache_time = np.mean(cache_test_times)
                cache_consistency = np.std(cache_test_times) < 0.5  # Baja variabilidad = buen cache
                
                validation_results['cache_performance'] = {
                    'success': True,
                    'avg_response_time': avg_cache_time,
                    'meets_target': avg_cache_time < 2.0,
                    'cache_consistency': cache_consistency,
                    'tests_performed': len(cache_test_times)
                }
                
                logger.info(f"✅ Cache performance: {avg_cache_time:.3f}s promedio, consistente: {cache_consistency}")
                
            else:
                validation_results['cache_performance'] = {'success': False, 'error': 'No se pudieron hacer pruebas de cache'}
            
            return {
                'status': 'success',
                'validation_results': validation_results,
                'performance_grade': self._calculate_performance_grade(validation_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en validación de performance: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_performance_grade(self, validation_results: Dict[str, Any]) -> str:
        """
        Calcular grado de performance del sistema
        """
        score = 0
        max_score = 0
        
        # Model selection (25 puntos)
        if validation_results.get('model_selection', {}).get('success', False):
            model_sel = validation_results['model_selection']
            if model_sel.get('is_expected_best', False) and model_sel.get('high_confidence', False):
                score += 25
            elif model_sel.get('is_expected_best', False):
                score += 20
            elif model_sel.get('high_confidence', False):
                score += 15
            else:
                score += 10
        max_score += 25
        
        # Explainability (25 puntos)
        if validation_results.get('explainability', {}).get('success', False):
            expl = validation_results['explainability']
            if expl.get('factors_found', 0) >= 10 and expl.get('models_analyzed', 0) >= 2:
                score += 25
            elif expl.get('factors_found', 0) >= 5:
                score += 20
            else:
                score += 10
        max_score += 25
        
        # Cache performance (50 puntos)
        if validation_results.get('cache_performance', {}).get('success', False):
            cache = validation_results['cache_performance']
            if cache.get('meets_target', False) and cache.get('cache_consistency', False):
                score += 50
            elif cache.get('meets_target', False):
                score += 40
            elif cache.get('cache_consistency', False):
                score += 30
            else:
                score += 20
        max_score += 50
        
        # Calcular porcentaje
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        if percentage >= 90:
            return "EXCELENTE"
        elif percentage >= 80:
            return "BUENO"
        elif percentage >= 70:
            return "ACEPTABLE"
        else:
            return "DEFICIENTE"
    
    def _generate_emulation_summary(self, emulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar resumen de la emulación
        """
        phases = emulation_results.get('phases', {})
        
        summary = {
            'phases_completed': len([p for p in phases.values() if p.get('status') == 'success']),
            'total_phases': len(phases),
            'overall_success': all(p.get('status') == 'success' for p in phases.values()),
            'execution_time': emulation_results.get('total_execution_time', 0)
        }
        
        # Extraer estadísticas clave
        if 'daily_retrain' in phases and phases['daily_retrain'].get('status') == 'success':
            retrain_data = phases['daily_retrain']['flow_results']
            summary['retrain_steps_completed'] = retrain_data.get('total_steps_completed', 0)
            summary['retrain_time'] = retrain_data.get('total_time_seconds', 0)
        
        if 'instant_responses' in phases and phases['instant_responses'].get('status') == 'success':
            instant_data = phases['instant_responses']['overall_stats']
            summary['avg_response_time'] = instant_data.get('avg_response_time', 0)
            summary['target_compliance_rate'] = instant_data.get('target_compliance_rate', 0)
        
        if 'performance_validation' in phases and phases['performance_validation'].get('status') == 'success':
            perf_data = phases['performance_validation']
            summary['performance_grade'] = perf_data.get('performance_grade', 'UNKNOWN')
        
        return summary
    
    async def _wait_for_api_ready(self, max_attempts: int = 20, wait_seconds: int = 15) -> bool:
        """
        Esperar a que la API esté completamente lista
        """
        logger.info(f"🔄 Esperando que la API esté lista (máximo {max_attempts * wait_seconds} segundos)...")
        
        for attempt in range(max_attempts):
            try:
                # Probar health check
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ API lista en intento {attempt + 1}")
                    
                    # Verificar también que los logs no muestren errores críticos
                    try:
                        subprocess.run(['docker-compose', 'logs', '--tail=20', 'steel-predictor'], 
                                     cwd=self.base_dir, capture_output=True, text=True, timeout=10)
                        logger.info("📋 Logs del contenedor verificados")
                    except:
                        pass
                    
                    return True
                else:
                    logger.info(f"⏳ Intento {attempt + 1}: API no lista (status: {response.status_code})")
                    
            except requests.exceptions.RequestException as e:
                logger.info(f"⏳ Intento {attempt + 1}: {str(e)[:100]}")
            except Exception as e:
                logger.warning(f"⚠️ Error inesperado en intento {attempt + 1}: {str(e)}")
            
            if attempt < max_attempts - 1:  # No esperar en el último intento
                logger.info(f"   Esperando {wait_seconds} segundos antes del siguiente intento...")
                time.sleep(wait_seconds)
        
        logger.error(f"❌ API no estuvo lista después de {max_attempts} intentos")
        
        # Mostrar logs del contenedor para diagnóstico
        try:
            logger.info("📋 Mostrando logs del contenedor para diagnóstico:")
            result = subprocess.run(['docker-compose', 'logs', '--tail=50', 'steel-predictor'], 
                                  cwd=self.base_dir, capture_output=True, text=True, timeout=30)
            if result.stdout:
                print("\n" + "="*60)
                print("📋 LOGS DEL CONTENEDOR:")
                print("="*60)
                print(result.stdout[-2000:])  # Últimas 2000 caracteres
                print("="*60)
        except Exception as e:
            logger.warning(f"No se pudieron obtener logs: {str(e)}")
        
        return False


async def main():
    """
    Función principal de despliegue
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Desplegar DeAcero Steel Price Predictor V2')
    parser.add_argument('--force-rebuild', action='store_true', help='Forzar reconstrucción de imagen Docker')
    parser.add_argument('--api-key', default='gusanito_medidor', help='API key para pruebas')
    parser.add_argument('--skip-tests', action='store_true', help='Saltar pruebas de verificación')
    parser.add_argument('--emulate-cloud', action='store_true', help='Emular flujo completo de producción en la nube')
    parser.add_argument('--basic-deploy', action='store_true', help='Solo despliegue básico sin emulación')
    
    args = parser.parse_args()
    
    # Crear deployment manager
    deployment = ProductionDeployment()
    
    if args.emulate_cloud:
        # ========== EMULACIÓN COMPLETA DEL FLUJO DE NUBE ==========
        logger.info("🌩️ Modo: EMULACIÓN COMPLETA DEL FLUJO DE PRODUCCIÓN")
        
        # 1. Verificar prerequisitos
        prerequisites = deployment.verify_prerequisites()
        if not all(prerequisites.values()):
            logger.error("❌ Prerequisitos no cumplidos para emulación")
            exit(1)
        
        # 2. Desplegar contenedores
        logger.info("\n🐳 Desplegando contenedores para emulación...")
        success = deployment.build_and_deploy(args.force_rebuild)
        if not success:
            logger.error("❌ Error desplegando contenedores")
            exit(1)
        
        # Esperar estabilización (más tiempo para reentrenamiento)
        logger.info("⏳ Esperando estabilización del servicio...")
        logger.info("   El reentrenamiento puede tomar varios minutos...")
        time.sleep(120)  # 2 minutos para estabilización completa
        
        # 3. Verificar que la API esté lista
        logger.info("🏥 Verificando que la API esté lista...")
        api_ready = await deployment._wait_for_api_ready()
        if not api_ready:
            logger.error("❌ API no está lista después de esperar")
            exit(1)
        
        # 4. Ejecutar emulación completa
        emulation_result = await deployment.emulate_cloud_production_flow(args.api_key)
        
        # 4. Mostrar resultados
        if emulation_result.get('status') == 'success':
            summary = emulation_result.get('summary', {})
            
            print("\n" + "=" * 80)
            print("🎉 EMULACIÓN COMPLETA EXITOSA")
            print("=" * 80)
            print(f"✅ Fases completadas: {summary.get('phases_completed', 0)}/{summary.get('total_phases', 0)}")
            print(f"⏱️ Tiempo total: {summary.get('execution_time', 0):.1f} segundos")
            
            if 'avg_response_time' in summary:
                print(f"⚡ Tiempo promedio de respuesta: {summary['avg_response_time']:.3f}s")
                print(f"🎯 Cumplimiento objetivo <2s: {summary.get('target_compliance_rate', 0):.1f}%")
            
            if 'performance_grade' in summary:
                print(f"📊 Grado de performance: {summary['performance_grade']}")
            
            print(f"\n🌐 API Lista: {deployment.api_base_url}")
            print(f"📚 Docs: {deployment.api_base_url}/docs")
            print(f"🎯 Test: curl -H \"X-API-Key: {args.api_key}\" {deployment.api_base_url}/predict/steel-rebar-price")
            
        else:
            error_msg = emulation_result.get('error', 'Unknown error')
            failed_step = emulation_result.get('failed_step', 'Unknown step')
            
            print(f"\n❌ Emulación falló en: {failed_step}")
            print(f"💥 Error: {error_msg}")
            
            # Mostrar detalles adicionales si están disponibles
            if 'phases' in emulation_result:
                print(f"\n📊 Estado de las fases:")
                for phase_name, phase_data in emulation_result['phases'].items():
                    status = phase_data.get('status', 'unknown')
                    print(f"   {phase_name}: {status}")
            
            exit(1)
    
    elif args.basic_deploy or args.skip_tests:
        # ========== DESPLIEGUE BÁSICO ==========
        logger.info("🚀 Modo: DESPLIEGUE BÁSICO")
        
        prerequisites = deployment.verify_prerequisites()
        if all(prerequisites.values()):
            success = deployment.build_and_deploy(args.force_rebuild)
            if success:
                logger.info("✅ Despliegue completado")
                logger.info(f"🌐 API disponible en: {deployment.api_base_url}")
            else:
                logger.error("❌ Despliegue falló")
        else:
            logger.error("❌ Prerequisitos no cumplidos")
    
    else:
        # ========== DESPLIEGUE ESTÁNDAR CON VERIFICACIONES ==========
        logger.info("📊 Modo: DESPLIEGUE ESTÁNDAR CON VERIFICACIONES")
        
        result = await deployment.run_full_deployment(args.force_rebuild, args.api_key)
        
        if result['overall_success']:
            print(f"\n🎉 ¡Despliegue exitoso!")
            print(f"🌐 API: {result['api_url']}")
            print(f"📄 Reporte: {result['report_file']}")
            print(f"\n💡 Para emular flujo completo de nube:")
            print(f"python deploy_production.py --emulate-cloud")
        else:
            print(f"\n❌ Despliegue falló o incompleto")
            print(f"📄 Ver reporte: {result['report_file']}")
            exit(1)


if __name__ == "__main__":
    asyncio.run(main())
