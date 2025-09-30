#!/bin/bash
# Docker entrypoint script para DeAcero Steel Price Predictor V2
# Inicialización del pipeline de producción

set -e

echo "🚀 Iniciando DeAcero Steel Price Predictor V2..."
echo "📅 $(date)"
echo "🐳 Container: $(hostname)"
echo "🐍 Python: $(python --version)"

# Verificar variables de entorno críticas
echo "🔑 Verificando configuración..."

if [ -z "$BANXICO_API_TOKEN" ]; then
    echo "⚠️ BANXICO_API_TOKEN no configurado"
fi

if [ -z "$FRED_API_KEY" ]; then
    echo "⚠️ FRED_API_KEY no configurado"
fi

if [ -z "$INEGI_API_TOKEN" ]; then
    echo "⚠️ INEGI_API_TOKEN no configurado"
fi

# Verificar estructura de directorios
echo "📁 Verificando estructura de directorios..."
mkdir -p /app/data/raw /app/data/processed /app/logs /app/cache

# Verificar modelos V2
echo "🤖 Verificando modelos V2..."
if [ -d "/app/models/test" ]; then
    echo "✅ Directorio de modelos encontrado"
    ls -la /app/models/test/*.pkl 2>/dev/null || echo "⚠️ No se encontraron modelos .pkl"
else
    echo "⚠️ Directorio de modelos no encontrado - creando..."
    mkdir -p /app/models/test
fi

# Verificar conectividad (opcional)
echo "🌐 Verificando conectividad..."
curl -s --connect-timeout 5 https://api.banxico.org.mx > /dev/null && echo "✅ Banxico accesible" || echo "⚠️ Banxico no accesible"
curl -s --connect-timeout 5 https://fred.stlouisfed.org > /dev/null && echo "✅ FRED accesible" || echo "⚠️ FRED no accesible"

# Ejecutar inicialización si es necesario
if [ "$1" = "init" ]; then
    echo "🔄 Ejecutando inicialización del pipeline..."
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

# COMENTADO: No ejecutar pipeline automáticamente para evitar demoras en inicio de API
# if [ "$INIT_PIPELINE" = "true" ]; then
#     echo "🔄 Ejecutando pipeline inicial de datos..."
#     python -c "
# import asyncio
# import sys
# sys.path.append('/app')
# from src.ml_pipeline.production_pipeline import ProductionPipeline
# async def init():
#     pipeline = ProductionPipeline()
#     result = await pipeline.run_full_pipeline(force_data_refresh=True)
#     print(f'Pipeline inicial: {result[\"status\"]}')
# asyncio.run(init())
# " || echo "⚠️ Pipeline inicial falló - continuando con API"
# fi

echo "⚡ Iniciando API inmediatamente para respuesta rápida..."
echo "💡 El pipeline se ejecutará bajo demanda o mediante endpoints"

# Verificar que la aplicación puede importarse
echo "🔍 Verificando imports de la aplicación..."
python -c "
import sys
sys.path.append('/app')
try:
    from app.main import app
    print('✅ Aplicación importada correctamente')
except Exception as e:
    print(f'❌ Error importando aplicación: {e}')
    sys.exit(1)
"

echo "✅ Inicialización completada - iniciando aplicación..."
echo "🌐 API disponible en: http://localhost:8000"
echo "📚 Documentación en: http://localhost:8000/docs"

# Ejecutar comando pasado como argumento
exec "$@"
