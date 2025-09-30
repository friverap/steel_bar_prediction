# 🏗️ DeAcero Steel Rebar Price Predictor V2

Sistema avanzado de predicción de precios de varilla corrugada que combina análisis econométrico riguroso con machine learning, desarrollado como solución integral para pronóstico del precio de cierre del día siguiente.

## 🎯 Descripción

Este proyecto implementa un sistema completo de predicción de precios de varilla corrugada (steel rebar) utilizando:
- **Datos reales** de mercado desde múltiples fuentes especializadas
- **Análisis estadístico exhaustivo** (estacionariedad, autocorrelación, cointegración, causalidad)
- **Modelos híbridos** (ARIMA-GARCH, XGBoost, VECM, MIDAS)
- **API REST robusta** con cache inteligente y monitoreo
- **Pipeline de reentrenamiento automático** para mantenimiento continuo

## 📋 Características Principales

- **Predicción Diaria**: Precio de cierre para el día siguiente (t+1)
- **Precisión Excepcional**: RMSE < $25 USD/tonelada, Hit Rate 100%
- **Datos Reales**: Steel rebar de Investing.com ($422-580 USD/ton)
- **Variables Optimizadas**: 11 variables seleccionadas por análisis riguroso
- **Cache Inteligente**: Respuestas instantáneas <2 segundos
- **Monitoreo Continuo**: Alertas automáticas de degradación
- **Arquitectura Escalable**: Preparado para despliegue en GCP

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.11+
- Docker y Docker Compose
- Git
- API Keys (BANXICO, FRED, INEGI)

### Instalación y Despliegue Local

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd deacero_steel_price_predictor
```

2. **Crear entorno virtual**
```bash
python -m venv deacero_env
source deacero_env/bin/activate  # En Windows: deacero_env\Scripts\activate
pip install -r requirements.txt
```

3. **Configurar variables de entorno**
```bash
cp env_example.txt .env
# Editar .env con tus API keys
```

4. **Despliegue Completo con Emulación de Producción**
```bash
# Opción 1: Despliegue estándar con verificaciones
python deploy_production.py

# Opción 2: Emulación completa del flujo de nube
python deploy_production.py --emulate-cloud

# Opción 3: Despliegue básico rápido
python deploy_production.py --basic-deploy
```

5. **Verificar que funciona**
```bash
# Test de predicción
curl -H "X-API-Key: clave_secreta" http://localhost:8000/predict/steel-rebar-price

# Documentación interactiva
open http://localhost:8000/docs
```

### 🚀 Despliegue Local con deploy_production.py

El script `deploy_production.py` ofrece tres modos de despliegue optimizados:

```bash
# 1. DESPLIEGUE ESTÁNDAR (Recomendado para desarrollo)
python deploy_production.py
# - Tiempo: ~3 minutos
# - Verificaciones completas de endpoints y modelos
# - Validación de datos y cache
# - Reporte detallado de despliegue

# 2. EMULACIÓN DE PRODUCCIÓN (Recomendado para testing)
python deploy_production.py --emulate-cloud
# - Tiempo: ~5 minutos
# - Emulación completa del flujo de nube
# - Reentrenamiento de modelos
# - Pruebas de carga y performance
# - Validación exhaustiva end-to-end

# 3. DESPLIEGUE RÁPIDO (Desarrollo rápido)
python deploy_production.py --basic-deploy
# - Tiempo: ~1 minuto
# - Solo levanta contenedores
# - Verificaciones mínimas
# - Sin reentrenamiento

# 4. DOCKER DIRECTO (No recomendado)
docker-compose up --build
# - Tiempo: ~2 minutos
# - Sin verificaciones
# - Sin reporte de estado
```

#### 📋 Proceso de Despliegue

1. **Verificación de Prerequisitos**
   - Docker y Docker Compose instalados
   - Variables de entorno configuradas
   - Modelos base disponibles

2. **Construcción de Contenedores**
   - API FastAPI
   - Redis para cache
   - Servicios auxiliares

3. **Verificación de Endpoints**
   - Health check
   - Predicción principal
   - Explicabilidad
   - Estado del pipeline

4. **Validación del Sistema**
   - Pruebas de predicción
   - Verificación de cache
   - Monitoreo de performance

5. **Reporte de Despliegue**
   - Estado de servicios
   - Métricas de API
   - URLs de acceso
   - Recomendaciones

## 📖 Uso de la API

### Endpoint Principal

```bash
curl -X GET "http://localhost:8000/predict/steel-rebar-price" \
     -H "X-API-Key: gusanito_medidor"
```

**Respuesta esperada:**
```json
{
  "prediction_date": "2025-09-30",
  "predicted_price_usd": 527.89,
  "currency": "USD",
  "unit": "metric ton (steel rebar proxy)",
  "model_confidence": 0.905,
  "timestamp": "2025-09-29T22:20:03Z",
  "best_model": "MIDAS_V2_hibrida"
}
```

### Otros Endpoints

- **Health Check**: `GET /health`
- **Documentación**: `GET /docs`
- **Feature Importance**: `GET /explainability/feature-importance`
- **Factores Causales**: `GET /explainability/causal-factors`
- **Estado del Pipeline**: `GET /predict/pipeline/status`

## 🏗️ Arquitectura Modular del Proyecto

```
deacero_steel_price_predictor/
├── 📁 app/                          # Aplicación FastAPI
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── api/                        # API endpoints y dependencias
│   │   ├── endpoints/
│   │   │   └── predict.py          # Endpoint predicción
│   │   └── dependencies.py         # API dependencies
│   ├── core/                       # Core configurations
│   │   ├── config.py              # Settings y variables
│   │   ├── security.py            # API key auth
│   │   └── logging.py             # Logging config
│   ├── models/                     # Modelos y schemas
│   │   ├── prediction.py          # Pydantic models
│   │   └── steel_price_model.py   # ML model implementation
│   └── services/                   # Business services
│       ├── data_collector.py      # Data collection
│       ├── feature_engineer.py    # Feature engineering
│       ├── predictor.py           # Prediction service
│       └── cache_manager.py       # Prediction cache
├── 📁 src/                          # Core source code
│   ├── data_ingestion/             # Data collectors
│   │   ├── scraper_investing_real.py  # Steel rebar real
│   │   ├── lme_collector.py          # LME metals
│   │   ├── banxico_collector.py      # Banxico data
│   │   ├── fred_collector.py         # FRED data
│   │   └── raw_materials_collector.py # Raw materials
│   ├── data_processing/            # Data processing
│   │   ├── cleaners.py            # Data cleaning
│   │   ├── feature_engineering.py # Feature creation
│   │   └── validators.py          # Data validation
│   ├── ml_pipeline/                # ML pipeline
│   │   ├── daily_retrain_pipeline.py # Auto retraining
│   │   ├── models_v2.py           # MIDAS & XGBoost V2
│   │   ├── prediction_cache.py    # Cache system
│   │   └── model_selector.py      # Model selection
│   └── utils/                      # Utilities
│       ├── date_utils.py          # Date handling
│       ├── api_clients.py         # API clients
│       └── constants.py           # Constants
├── 📁 data/                         # Data storage
│   ├── raw/                       # Raw data
│   ├── processed/                 # Processed data
│   └── models/                    # Trained models
├── 📁 scripts/                      # Automation scripts
│   ├── ingest_all_data.py        # Data ingestion
│   ├── join_daily_series.py      # Daily consolidation
│   ├── join_monthly_series.py    # Monthly consolidation
│   └── clear_cache.py            # Cache clearing
├── 📁 notebooks/                    # Analysis notebooks
│   ├── 01_data_exploration.ipynb # Initial exploration
│   ├── 02_feature_analysis.ipynb # Statistical analysis
│   └── 03_AB_TESTING.ipynb       # Model testing
├── 📁 docs/                         # Documentation
│   ├── 02_FEATURE_ANALYSIS/      # Analysis docs
│   │   ├── DOCUMENTATION.md      # Executive summary
│   │   └── ANALISIS_*.md        # Detailed analysis
│   └── 03_AB_TESTING/            # Testing results
├── deploy_production.py            # Deployment script
├── docker-compose.yml             # Container config
├── Dockerfile                     # Docker image
└── requirements.txt               # Dependencies
```

## 📊 Fuentes de Datos

### Fuentes Principales

1. **London Metal Exchange (LME)**: Precios de metales industriales
2. **Banco de México (BANXICO)**: Tipo de cambio, inflación, tasas
3. **INEGI**: Indicadores económicos y de construcción mexicanos
4. **Yahoo Finance**: Precios de acciones siderúrgicas y futuros
5. **FRED**: Indicadores económicos de EE.UU.

### Variables Clave

- Precios de mineral de hierro y carbón metalúrgico
- Tipo de cambio USD/MXN
- Índices de construcción y manufactura
- Precios de energía y combustibles
- Indicadores técnicos y tendencias

## 🔧 Configuración

## 🎯 Variables Seleccionadas para Predicción

Las variables utilizadas en el modelo fueron seleccionadas tras un exhaustivo análisis estadístico documentado en [`docs/02_FEATURE_ANALYSIS/DOCUMENTATION.md`](docs/02_FEATURE_ANALYSIS/DOCUMENTATION.md). Se eligieron por su estabilidad temporal (Std < 0.15), poder predictivo y relaciones causales demostradas.

### 🎯 Variable Objetivo

**`precio_varilla_lme`** - Precio de Varilla Corrugada (Steel Rebar)
- **Fuente**: [Investing.com](https://www.investing.com/commodities/steel-rebar) - Datos reales de mercado
- **Rango histórico**: $422-580 USD/tonelada (2020-2025)
- **Frecuencia**: Diaria (días hábiles)
- **Scraper**: `src/data_ingestion/scraper_investing_real.py`

### 🔍 Variables Explicativas (11 variables)

1. **Metales Base (4 variables)**
   - `cobre_lme`: London Metal Exchange - Precio cobre
   - `zinc_lme`: London Metal Exchange - Precio zinc
   - `steel`: London Metal Exchange - Índice acero
   - `aluminio_lme`: London Metal Exchange - Precio aluminio
   - **Fuente**: LME Data Services
   - **Criterio**: Alta correlación y estabilidad temporal

2. **Materias Primas (2 variables)**
   - `coking`: Precio carbón metalúrgico
   - `iron`: Precio mineral de hierro
   - **Fuente**: Raw Materials Index
   - **Criterio**: Cointegración demostrada (Test Johansen)

3. **Indicadores Macro/Financieros (3 variables)**
   - `dxy`: Índice Dólar US
   - `treasury`: Treasury Yield 10Y
   - `tasa_interes_banxico`: Tasa de referencia Banxico
   - **Fuentes**: FRED, Banxico
   - **Criterio**: Causalidad Granger significativa

4. **Riesgo y Mercado (2 variables)**
   - `VIX`: Índice de Volatilidad
   - `infrastructure`: Índice de Infraestructura
   - **Fuente**: Yahoo Finance, FRED
   - **Criterio**: Correlación dinámica estable

### 📊 Criterios de Selección

La selección final se basó en múltiples criterios cuantitativos:

1. **Estabilidad Temporal**
   - Correlaciones estables (Std < 0.15)
   - Sin cambios de régimen significativos
   - Relaciones consistentes 2020-2025

2. **Poder Predictivo**
   - Mutual Information Score > 0.15
   - Random Forest Importance > 0.10
   - Correlación significativa (p < 0.01)

3. **Causalidad y Cointegración**
   - Test Granger (p < 0.05)
   - Test Johansen positivo
   - VECM significativo

4. **Consideraciones Prácticas**
   - Disponibilidad diaria garantizada
   - Fuentes confiables y estables
   - Latencia < 1 hora en actualización

### 📅 Manejo de Días Hábiles

**Importante**: El sistema está diseñado para operar exclusivamente en días hábiles del mercado:

1. **Predicciones**: Solo se generan para días hábiles (lunes a viernes, excluyendo festivos)
2. **Calendario de Mercado**: Utiliza `src/utils/business_calendar.py` para determinar días válidos
3. **Tratamiento de Datos Faltantes**:
   - **Fines de semana**: Interpolación lineal entre viernes y lunes
   - **Días festivos**: Interpolación basada en días hábiles adyacentes
   - **Método**: `pandas.interpolate(method='linear', limit=2)`
   - **Validación**: Máximo 2 días consecutivos de interpolación permitidos

```python
# Ejemplo de procesamiento de días hábiles
def fill_missing_business_days(df):
    """
    Rellena valores faltantes en fines de semana y festivos
    """
    # Identificar días hábiles
    business_days = pd.bdate_range(start=df.index.min(), 
                                   end=df.index.max())
    
    # Reindexar a todos los días
    df = df.reindex(pd.date_range(df.index.min(), 
                                  df.index.max(), 
                                  freq='D'))
    
    # Interpolación lineal (máximo 2 días)
    df = df.interpolate(method='linear', limit=2)
    
    # Mantener solo días hábiles para predicción
    df = df.loc[df.index.isin(business_days)]
    
    return df
```

Para un análisis detallado de la selección de variables y su justificación estadística, consulte:
[`docs/02_FEATURE_ANALYSIS/DOCUMENTATION.md`](docs/02_FEATURE_ANALYSIS/DOCUMENTATION.md)

### ⚙️ Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `API_KEY` | Clave de autenticación | `required` |
| `HOST` | Host del servidor | `0.0.0.0` |
| `PORT` | Puerto del servidor | `8000` |
| `DEBUG` | Modo debug | `False` |
| `RATE_LIMIT_PER_HOUR` | Límite de requests | `100` |
| `CACHE_EXPIRY_HOURS` | Expiración de cache | `1` |

### 🔑 APIs Externas Requeridas

Para obtener datos en tiempo real, configure las siguientes API keys:

- `BANXICO_API_TOKEN`: Token de la API de BANXICO
  - Variables: tasa_interes_banxico
  - Frecuencia: Diaria
  - [Documentación Banxico](https://www.banxico.org.mx/SieAPIRest/service/v1/)

- `FRED_API_KEY`: Clave de FRED API
  - Variables: dxy, treasury, infrastructure
  - Frecuencia: Diaria
  - [Documentación FRED](https://fred.stlouisfed.org/docs/api/fred/)

- `LME_API_KEY`: Acceso a London Metal Exchange
  - Variables: cobre_lme, zinc_lme, aluminio_lme
  - Frecuencia: Diaria
  - [LME Data Services](https://www.lme.com/en/Market-data)

## 🧪 Testing

```bash
# Ejecutar todas las pruebas
pytest

# Con cobertura
pytest --cov=app

# Solo pruebas de la API
pytest tests/test_api/

# Pruebas específicas
pytest tests/test_api/test_predict_endpoint.py -v
```

## 📈 Monitoreo y Logs

### Health Checks

```bash
curl http://localhost:8000/health
```

### Logs

Los logs se almacenan en `./logs/deacero_api.log` y incluyen:
- Requests y responses de la API
- Predicciones generadas
- Errores y warnings
- Métricas de performance

### Métricas

- Tiempo de respuesta por endpoint
- Precisión del modelo
- Uso de cache
- Rate limiting por API key

## 🔒 Seguridad

- **Autenticación**: API Key obligatoria
- **Rate Limiting**: 100 requests/hora por defecto
- **CORS**: Configurado para producción
- **Logs**: Sanitización de datos sensibles
- **Headers de Seguridad**: Implementados

## 🚀 Deployment

### Producción con Docker

```bash
# Construir imagen de producción
docker build -t deacero-api:prod .

# Ejecutar en producción
docker run -d \
  --name deacero-api \
  -p 80:8000 \
  -e API_KEY=your_production_api_key \
  -e DEBUG=False \
  -e LOG_LEVEL=WARNING \
  deacero-api:prod
```

### Variables de Producción

```env
DEBUG=False
LOG_LEVEL=WARNING
CACHE_EXPIRY_HOURS=4
RATE_LIMIT_PER_HOUR=1000
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://redis-host:6379
```

## 📚 Documentación Adicional

- [Metodología del Modelo](docs/model_methodology.md)
- [Fuentes de Datos](docs/data_sources.md)
- [Guía de Deployment](docs/deployment_guide.md)
- [API Documentation](http://localhost:8000/docs)

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

- **Proyecto**: DeAcero Steel Price Predictor
- **Versión**: 1.0.0
- **Autor**: Candidato CDO
- **Email**: [frariv@deacero.com]

---

**Nota**: Este proyecto fue desarrollado como parte de la prueba técnica para el puesto de Chief Data Officer en DeAcero. Incluye predicciones de precios de varilla corrugada utilizando datos históricos y modelos de machine learning.
