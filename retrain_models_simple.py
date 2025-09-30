#!/usr/bin/env python3
"""
Reentrenamiento de Modelos V2 con Features CORRECTAS
Usa las variables documentadas SIN transformaciones complejas

Features según documentación:
- Metales: cobre_lme, zinc_lme, steel, aluminio_lme
- Materias primas: coking, iron  
- Macro: dxy, treasury, tasa_interes_banxico
- Riesgo: VIX, infrastructure
- Autorregresivas: precio_varilla_lme_lag_1, precio_varilla_lme_lag_20

Total: 13 features SIMPLES
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FEATURES CORRECTAS según documentación
CORRECT_FEATURES = [
    # Metales (4)
    'cobre_lme', 'zinc_lme', 'steel', 'aluminio_lme',
    # Materias Primas (2)  
    'coking', 'iron',
    # Macro/Financial (3)
    'dxy', 'treasury', 'tasa_interes_banxico',
    # Risk/Market (2)
    'VIX', 'infrastructure',
    # Autorregresivas (2)
    'precio_varilla_lme_lag_1', 'precio_varilla_lme_lag_20'
]

TARGET = 'precio_varilla_lme'

print("=" * 80)
print("🤖 REENTRENAMIENTO DE MODELOS V2 CON FEATURES CORRECTAS")
print("=" * 80)
print(f"\n📊 Features a usar ({len(CORRECT_FEATURES)}):")
for i, feat in enumerate(CORRECT_FEATURES, 1):
    print(f"  {i:2d}. {feat}")
print(f"\n🎯 Variable objetivo: {TARGET}")

# Cargar datos
print("\n📥 Cargando datos procesados...")
data_file = Path("data/processed/features_v2_latest.csv")

if not data_file.exists():
    print(f"❌ Archivo no encontrado: {data_file}")
    exit(1)

df = pd.read_csv(data_file, index_col='fecha', parse_dates=True)
print(f"✅ Datos cargados: {df.shape}")

# Verificar que tenemos todas las features necesarias
missing_features = [f for f in CORRECT_FEATURES if f not in df.columns]
if missing_features:
    print(f"\n❌ Features faltantes: {missing_features}")
    print(f"\n📋 Features disponibles en el archivo:")
    print(list(df.columns))
    exit(1)

# Verificar target
if TARGET not in df.columns:
    print(f"❌ Variable objetivo {TARGET} no encontrada")
    exit(1)

# Preparar datos
print("\n🔧 Preparando datos para entrenamiento...")
X = df[CORRECT_FEATURES].copy()
y = df[TARGET].copy()

# Eliminar NaN
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print(f"✅ Datos limpios: {X.shape[0]} filas, {X.shape[1]} features")

# Split temporal (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n📊 Split temporal:")
print(f"  Train: {X_train.shape[0]} filas ({X_train.index.min()} a {X_train.index.max()})")
print(f"  Test:  {X_test.shape[0]} filas ({X_test.index.min()} a {X_test.index.max()})")

# ============================================================================
# MODELO 1: XGBoost_V2_regime (REENTRENADO)
# ============================================================================
print("\n" + "=" * 80)
print("🤖 ENTRENANDO XGBoost_V2_regime (features simples)")
print("=" * 80)

# Scalers
scaler_X_xgb = RobustScaler()
scaler_y_xgb = RobustScaler()

X_train_scaled = scaler_X_xgb.fit_transform(X_train)
y_train_scaled = scaler_y_xgb.fit_transform(y_train.values.reshape(-1, 1)).ravel()

X_test_scaled = scaler_X_xgb.transform(X_test)
y_test_scaled = scaler_y_xgb.transform(y_test.values.reshape(-1, 1)).ravel()

# XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective='reg:squarederror'
)

print("🔄 Entrenando XGBoost...")
xgb_model.fit(X_train_scaled, y_train_scaled)

# Predicciones
y_pred_scaled = xgb_model.predict(X_test_scaled)
y_pred = scaler_y_xgb.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Métricas XGBoost:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²:   {r2:.4f}")

# Guardar modelo
models_dir = Path("models/production")
models_dir.mkdir(parents=True, exist_ok=True)

xgb_model_data = {
    'model': xgb_model,
    'scalers': {'X': scaler_X_xgb, 'y': scaler_y_xgb},
    'test_metrics': {
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    },
    'features_used': CORRECT_FEATURES,
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

xgb_file = models_dir / "XGBoost_V2_regime_latest.pkl"
with open(xgb_file, 'wb') as f:
    pickle.dump(xgb_model_data, f)

print(f"✅ Modelo guardado: {xgb_file}")

# ============================================================================
# MODELO 2: MIDAS_V2_hibrida (REENTRENADO)  
# ============================================================================
print("\n" + "=" * 80)
print("🤖 ENTRENANDO MIDAS_V2_hibrida (features simples)")
print("=" * 80)

# Scalers
scaler_X_midas = RobustScaler()
scaler_y_midas = RobustScaler()

X_train_scaled = scaler_X_midas.fit_transform(X_train)
y_train_scaled = scaler_y_midas.fit_transform(y_train.values.reshape(-1, 1)).ravel()

X_test_scaled = scaler_X_midas.transform(X_test)
y_test_scaled = scaler_y_midas.transform(y_test.values.reshape(-1, 1)).ravel()

# MIDAS simplificado (Ridge con features simples)
midas_model = Ridge(alpha=1.0, random_state=42)

print("🔄 Entrenando MIDAS...")
midas_model.fit(X_train_scaled, y_train_scaled)

# Predicciones
y_pred_scaled = midas_model.predict(X_test_scaled)
y_pred = scaler_y_midas.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Métricas MIDAS:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²:   {r2:.4f}")

# Guardar modelo
midas_model_data = {
    'model': midas_model,
    'scalers': {'X': scaler_X_midas, 'y': scaler_y_midas},
    'test_metrics': {
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    },
    'features_used': CORRECT_FEATURES,
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

midas_file = models_dir / "MIDAS_V2_hibrida_latest.pkl"
with open(midas_file, 'wb') as f:
    pickle.dump(midas_model_data, f)

print(f"✅ Modelo guardado: {midas_file}")

# ============================================================================
# GENERAR PREDICCIÓN Y ACTUALIZAR CACHE
# ============================================================================
print("\n" + "=" * 80)
print("🎯 GENERANDO PREDICCIÓN Y ACTUALIZANDO CACHE")
print("=" * 80)

# Última fila de features para predicción
X_latest = X.iloc[[-1]]

# Predicción con cada modelo
predictions = {}

# XGBoost
X_scaled = scaler_X_xgb.transform(X_latest)
y_pred_scaled = xgb_model.predict(X_scaled)
y_pred_xgb = scaler_y_xgb.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
predictions['XGBoost_V2_regime'] = float(y_pred_xgb)

# MIDAS
X_scaled = scaler_X_midas.transform(X_latest)
y_pred_scaled = midas_model.predict(X_scaled)
y_pred_midas = scaler_y_midas.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
predictions['MIDAS_V2_hibrida'] = float(y_pred_midas)

print(f"\n💰 Predicciones generadas:")
print(f"  XGBoost: ${predictions['XGBoost_V2_regime']:.2f}")
print(f"  MIDAS:   ${predictions['MIDAS_V2_hibrida']:.2f}")

# Seleccionar mejor (por R² en test)
xgb_r2 = xgb_model_data['test_metrics']['r2']
midas_r2 = midas_model_data['test_metrics']['r2']

if xgb_r2 > midas_r2:
    best_model = 'XGBoost_V2_regime'
    best_pred = predictions['XGBoost_V2_regime']
    best_conf = xgb_r2
else:
    best_model = 'MIDAS_V2_hibrida'
    best_pred = predictions['MIDAS_V2_hibrida']
    best_conf = midas_r2

print(f"\n🏆 Mejor modelo: {best_model} (R² = {best_conf:.4f})")
print(f"💰 Predicción final: ${best_pred:.2f}")

# Actualizar cache
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

cache_data = {
    "prediction": {
        "prediction_date": (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        "predicted_price_usd": best_pred,
        "currency": "USD",
        "unit": "metric ton",
        "model_confidence": best_conf,
        "timestamp": datetime.now().isoformat() + "Z",
        "best_model": best_model,
        "all_models_evaluated": 2,
        "model_metrics_used": {
            "rmse": xgb_model_data['test_metrics']['rmse'] if best_model == 'XGBoost_V2_regime' else midas_model_data['test_metrics']['rmse'],
            "mape": xgb_model_data['test_metrics']['mape'] if best_model == 'XGBoost_V2_regime' else midas_model_data['test_metrics']['mape'],
            "r2": best_conf
        }
    },
    "cached_at": datetime.now().isoformat(),
    "valid_until": (datetime.now() + pd.Timedelta(hours=24)).isoformat(),
    "cache_version": "2.0"
}

cache_file = cache_dir / "daily_prediction_cache.json"
with open(cache_file, 'w') as f:
    import json
    json.dump(cache_data, f, indent=2)

print(f"\n✅ Cache actualizado: {cache_file}")

print("\n" + "=" * 80)
print("🎉 REENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"✅ Modelos guardados en: {models_dir}")
print(f"✅ Cache actualizado en: {cache_file}")
print(f"💰 Nueva predicción: ${best_pred:.2f} ({best_model})")
print(f"📅 Válido hasta: {cache_data['valid_until']}")
