#!/bin/bash
# Docker entrypoint para Google Cloud Platform
# DeAcero Steel Price Predictor V2 - GCP Production

set -e

echo "☁️ Iniciando DeAcero Steel Price Predictor V2 en GCP..."
echo "📅 $(date)"
echo "🐳 Container: $(hostname)"
echo "🌍 Platform: GCP Cloud Run"
echo "🐍 Python: $(python --version)"

# Configurar autenticación con GCP (si está disponible)
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "🔑 Autenticación GCP configurada"
    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
else
    echo "🔑 Usando autenticación por defecto de GCP"
fi

# Verificar proyecto GCP
if [ ! -z "$GCP_PROJECT_ID" ]; then
    echo "☁️ Proyecto GCP: $GCP_PROJECT_ID"
    gcloud config set project $GCP_PROJECT_ID
fi

# Verificar acceso a Cloud Storage
if [ ! -z "$GCS_BUCKET_MODELS" ]; then
    echo "📦 Verificando acceso a Cloud Storage..."
    gsutil ls gs://$GCS_BUCKET_MODELS/ > /dev/null 2>&1 && echo "✅ Cloud Storage accesible" || echo "⚠️ Cloud Storage no accesible"
fi

# Verificar Secret Manager
echo "🔐 Verificando Secret Manager..."
if [ ! -z "$GCP_PROJECT_ID" ]; then
    gcloud secrets list --limit=1 > /dev/null 2>&1 && echo "✅ Secret Manager accesible" || echo "⚠️ Secret Manager no accesible"
fi

# Cargar API keys desde Secret Manager
echo "🔑 Cargando API keys desde Secret Manager..."

# Función helper para cargar secrets
load_secret() {
    local secret_name=$1
    local env_var=$2
    
    if [ ! -z "$GCP_PROJECT_ID" ]; then
        secret_value=$(gcloud secrets versions access latest --secret="$secret_name" 2>/dev/null)
        if [ $? -eq 0 ] && [ ! -z "$secret_value" ]; then
            export $env_var="$secret_value"
            echo "✅ $env_var cargado desde Secret Manager"
        else
            echo "⚠️ No se pudo cargar $secret_name"
        fi
    fi
}

# Cargar secrets
load_secret "banxico-api-token" "BANXICO_API_TOKEN"
load_secret "fred-api-key" "FRED_API_KEY"
load_secret "inegi-api-token" "INEGI_API_TOKEN"
load_secret "quandl-api-key" "QUANDL_API_KEY"
load_secret "trading-economics-key" "TRADING_ECONOMICS_API_KEY"
load_secret "deacero-api-key" "API_KEY"

# Verificar estructura de directorios
echo "📁 Verificando estructura de directorios..."
mkdir -p /app/data/raw /app/data/processed /app/logs /app/cache

# Sincronizar modelos desde Cloud Storage (si están disponibles)
if [ ! -z "$GCS_BUCKET_MODELS" ] && [ "$SYNC_MODELS_ON_START" = "true" ]; then
    echo "📥 Sincronizando modelos desde Cloud Storage..."
    gsutil -m rsync -r gs://$GCS_BUCKET_MODELS/latest/ /app/models/production/ 2>/dev/null || echo "⚠️ No se pudieron sincronizar modelos"
fi

# Verificar modelos V2 locales
echo "🤖 Verificando modelos V2..."
if [ -f "/app/models/test/XGBoost_V2_regime.pkl" ]; then
    echo "✅ XGBoost_V2_regime encontrado"
else
    echo "⚠️ XGBoost_V2_regime no encontrado"
fi

if [ -f "/app/models/test/MIDAS_V2_hibrida.pkl" ]; then
    echo "✅ MIDAS_V2_hibrida encontrado"
else
    echo "⚠️ MIDAS_V2_hibrida no encontrado"
fi

# Verificar conectividad a APIs externas
echo "🌐 Verificando conectividad a APIs externas..."
curl -s --connect-timeout 5 https://api.banxico.org.mx > /dev/null && echo "✅ Banxico accesible" || echo "⚠️ Banxico no accesible"
curl -s --connect-timeout 5 https://fred.stlouisfed.org > /dev/null && echo "✅ FRED accesible" || echo "⚠️ FRED no accesible"

# Configurar logging para GCP
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export GCP_LOGGING=${GCP_LOGGING:-true}

# Ejecutar inicialización si es necesario
if [ "$1" = "init" ]; then
    echo "🔄 Ejecutando inicialización del pipeline en GCP..."
    python -c "
import asyncio
import sys
sys.path.append('/app')
from src.ml_pipeline.production_pipeline import main
asyncio.run(main())
"
    echo "✅ Inicialización completada"
    exit 0
fi

# Ejecutar reentrenamiento si es Cloud Function
if [ "$1" = "retrain" ]; then
    echo "🤖 Ejecutando reentrenamiento diario..."
    python -c "
import asyncio
import sys
sys.path.append('/app')
from src.ml_pipeline.daily_retrain_pipeline import main
asyncio.run(main())
"
    exit 0
fi

# Cargar datos iniciales si es el primer despliegue
if [ "$INIT_DATA_ON_START" = "true" ]; then
    echo "📊 Cargando datos iniciales..."
    python -c "
import asyncio
import sys
sys.path.append('/app')
from src.ml_pipeline.production_pipeline import ProductionPipeline
async def init():
    pipeline = ProductionPipeline()
    result = await pipeline.run_full_pipeline(force_data_refresh=False)
    print(f'Pipeline inicial: {result[\"status\"]}')
asyncio.run(init())
" || echo "⚠️ Carga inicial de datos falló - continuando con API"
fi

# Verificar que la aplicación puede importarse
echo "🔍 Verificando imports de la aplicación..."
python -c "
import sys
sys.path.append('/app')
try:
    from app.main import app
    print('✅ Aplicación FastAPI importada correctamente')
    
    # Verificar componentes críticos
    from src.ml_pipeline.model_selector import ModelSelector
    print('✅ ModelSelector importado')
    
    from src.ml_pipeline.models_v2 import ProductionPredictor
    print('✅ ProductionPredictor importado')
    
except Exception as e:
    print(f'❌ Error importando aplicación: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Configurar Google Cloud Logging si está habilitado
if [ "$GCP_LOGGING" = "true" ] && [ ! -z "$GCP_PROJECT_ID" ]; then
    echo "📊 Configurando Google Cloud Logging..."
    python -c "
import google.cloud.logging
client = google.cloud.logging.Client()
client.setup_logging()
print('✅ Google Cloud Logging configurado')
" 2>/dev/null || echo "⚠️ Google Cloud Logging no disponible"
fi

echo "✅ Inicialización GCP completada - iniciando aplicación..."
echo "🌐 API estará disponible en Cloud Run URL"
echo "📚 Documentación en: [CLOUD_RUN_URL]/docs"
echo "🎯 Endpoint principal: [CLOUD_RUN_URL]/predict/steel-rebar-price"

# Ejecutar comando pasado como argumento
exec "$@"
