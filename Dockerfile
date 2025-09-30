# Dockerfile para DeAcero Steel Price Predictor - Pipeline V2 de Producción
# Optimizado para predicción de precios de varilla corrugada con modelos V2

FROM python:3.11-slim

# Metadatos
LABEL maintainer="DeAcero Data Team"
LABEL version="2.0"
LABEL description="Steel Rebar Price Predictor with V2 Production Pipeline"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV TZ=America/Mexico_City

# Configuración de la aplicación
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false
ENV LOG_LEVEL=INFO

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar dependencias adicionales para producción
RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    optuna==3.6.1 \
    arch==6.3.0 \
    statsmodels==0.14.1 \
    scikit-learn==1.4.1.post1 \
    plotly==5.18.0 \
    tqdm==4.66.2 \
    fastapi==0.109.2 \
    uvicorn[standard]==0.27.1 \
    python-multipart==0.0.9 \
    python-dotenv==1.0.1

# Crear directorios necesarios
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/models/production \
    /app/models/test \
    /app/logs \
    /app/cache

# Copiar código fuente
COPY . .

# Copiar modelos entrenados si existen
COPY models/ /app/models/

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Script de inicio
COPY docker-entrypoint.sh /docker-entrypoint.sh
USER root
RUN chmod +x /docker-entrypoint.sh
USER appuser

# Comando por defecto optimizado
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30", "--access-log"]