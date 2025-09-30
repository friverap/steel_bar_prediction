#!/bin/bash
# Docker entrypoint SIMPLIFICADO para Google Cloud Platform
# Los secrets ya vienen como env vars desde Cloud Run

set -e

echo "☁️ Iniciando DeAcero Steel Price Predictor V2 en GCP..."
echo "📅 $(date)"
echo "🐳 Container: $(hostname)"
echo "🐍 Python: $(python --version)"

# Verificar variables críticas
echo "🔍 Verificando configuración..."
[ ! -z "$API_KEY" ] && echo "✅ API_KEY configurado" || echo "⚠️ API_KEY no configurado"
[ ! -z "$BANXICO_API_TOKEN" ] && echo "✅ BANXICO_API_TOKEN configurado" || echo "⚠️ BANXICO_API_TOKEN no configurado"
[ ! -z "$FRED_API_KEY" ] && echo "✅ FRED_API_KEY configurado" || echo "⚠️ FRED_API_KEY no configurado"

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p /app/logs /app/cache /app/data/processed /app/models/production

# Descargar modelos desde GCS (público)
echo "📦 Descargando modelos desde GCS..."
cd /app/models/production
if [ ! -f "XGBoost_V2_regime_latest.pkl" ]; then
    echo "⬇️  Descargando XGBoost_V2_regime_latest.pkl..."
    wget -q "https://storage.googleapis.com/steel-rebar-models/production/XGBoost_V2_regime_latest.pkl" || \
    curl -s -o "XGBoost_V2_regime_latest.pkl" "https://storage.googleapis.com/steel-rebar-models/production/XGBoost_V2_regime_latest.pkl" || \
    echo "⚠️ No se pudo descargar XGBoost (continuando...)"
fi

if [ ! -f "MIDAS_V2_hibrida_latest.pkl" ]; then
    echo "⬇️  Descargando MIDAS_V2_hibrida_latest.pkl..."
    wget -q "https://storage.googleapis.com/steel-rebar-models/production/MIDAS_V2_hibrida_latest.pkl" || \
    curl -s -o "MIDAS_V2_hibrida_latest.pkl" "https://storage.googleapis.com/steel-rebar-models/production/MIDAS_V2_hibrida_latest.pkl" || \
    echo "⚠️ No se pudo descargar MIDAS (continuando...)"
fi

ls -lh /app/models/production/*.pkl 2>/dev/null && echo "✅ Modelos descargados" || echo "⚠️ Sin modelos"

# Descargar cache con predicción pre-calculada
echo "📦 Descargando cache con predicción..."
cd /app/cache
if [ ! -f "daily_prediction_cache.json" ]; then
    wget -q "https://storage.googleapis.com/steel-rebar-models/cache/daily_prediction_cache.json" || \
    curl -s -o "daily_prediction_cache.json" "https://storage.googleapis.com/steel-rebar-models/cache/daily_prediction_cache.json" || \
    echo "⚠️ No se pudo descargar cache"
fi
ls -lh /app/cache/*.json 2>/dev/null && echo "✅ Cache descargado" || echo "⚠️ Sin cache"

# Iniciar aplicación FastAPI
echo "🚀 Iniciando FastAPI en puerto ${PORT:-8000}..."
cd /app
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --log-level info
