# 📊 DATASET DIARIO CONSOLIDADO - RESUMEN
## Proyecto: DeAcero Steel Price Predictor
### Fecha de Generación: 26 de Septiembre de 2025

---

## 📋 CARACTERÍSTICAS DEL DATASET

### Dimensiones
- **Total de registros**: 2,094 observaciones diarias
- **Total de variables**: 28 series temporales
- **Rango temporal**: 2020-01-02 a 2025-09-25
- **Frecuencia**: Diaria (solo días hábiles - business days)

### Variable Objetivo
- **Nombre**: `precio_varilla_lme`
- **Fuente**: London Metal Exchange (LME)
- **Completitud**: 69.0% (1,444 valores no nulos)
- **Rango**: Determina las fechas límite del dataset completo

---

## 🔧 PROCESAMIENTO APLICADO

### 1. Normalización de Fechas
- ✅ Todas las fechas convertidas a formato `YYYY-MM-DD 00:00:00`
- ✅ Eliminación de timezones
- ✅ Resolución de duplicados (ej: 00:00:00 vs 05:00:00 del mismo día)
- ✅ Agregación por promedio cuando hay múltiples valores para la misma fecha

### 2. Estrategia de Fines de Semana
- **Estrategia aplicada**: `business_days`
- **Justificación**: Los mercados financieros no operan en fines de semana
- **Caso especial - UDIS**: 
  - UDIS se actualiza todos los días (incluye fines de semana)
  - Se aplicó filtrado específico para mantener solo días hábiles
  - Registros eliminados de UDIS: 598 (todos los fines de semana)
- **Beneficios**:
  - No introduce datos artificiales
  - Mantiene la integridad de las series temporales
  - Compatible con modelos LSTM/ARIMA
  - Evita autocorrelación artificial
  - Dataset completamente homogéneo (solo días hábiles)

### 3. Truncado por Variable Objetivo
- **Fecha inicial**: 2020-01-02 (primera fecha de precio_varilla_lme)
- **Fecha final**: 2025-09-25 (última fecha de precio_varilla_lme)
- **Registros eliminados**: 2 (fuera del rango)
- **Resultado**: Dataset perfectamente alineado con la variable objetivo

### 4. Selección de Columnas de Valor
- **Prioridad 1**: `Close` (precio de cierre)
- **Prioridad 2**: `precio_cierre`
- **Prioridad 3**: `valor`
- **Conversión**: Todos los valores convertidos a numérico

---

## 📊 VARIABLES INCLUIDAS

### LME (7 series) - London Metal Exchange
1. `precio_varilla_lme` ⭐ **VARIABLE OBJETIVO**
2. `aluminio_lme`
3. `cobre_lme`
4. `zinc_lme`
5. `iron` (mineral de hierro)
6. `coking` (carbón de coque)
7. `steel` (acero ETF)

### Yahoo Finance (11 series)
8. `sp500` - S&P 500
9. `Petroleo` - Petróleo (Brent/WTI)
10. `gas_natural` - Gas Natural
11. `VIX` - Índice de Volatilidad
12. `dxy` - Dollar Index
13. `treasury` - Bonos del Tesoro 10Y
14. `commodities` - ETF de Commodities
15. `materials` - ETF de Materiales
16. `china` - ETF China
17. `emerging` - ETF Mercados Emergentes
18. `infrastructure` - ETF Infraestructura

### Banxico (4 series)
19. `tipo_cambio_usdmxn` - Tipo de Cambio USD/MXN
20. `tiie_28_dias` - TIIE 28 días
21. `udis_valor` - Valor de UDIS
22. `tasa_interes_banxico` - Tasa de Interés

### AHMSA (4 series)
23. `arcelormittal_acciones` - ArcelorMittal
24. `ternium` - Ternium México
25. `nucor_acciones` - Nucor
26. `20250926` - AHMSA precio local

### FRED (2 series)
27. `dxy_index_fred_fred` - Dollar Index (FRED)
28. `gas_natural_fred_fred` - Gas Natural (FRED)

---

## 📈 ESTADÍSTICAS DE CALIDAD

### Completitud Global
- **Porcentaje de completitud**: ~70%
- **Valores no nulos**: ~39,000
- **Valores nulos**: ~19,000

### Análisis de Gaps
- **Gaps de fin de semana**: 0 (usando business days)
- **Otros gaps**: Mínimos (días festivos)
- **Gap máximo**: 1 día

### Series Más Completas
1. Banxico: ~100% completitud
2. LME: ~69% completitud
3. Yahoo Finance: Variable según serie
4. AHMSA: ~69% completitud

---

## 🎯 USO RECOMENDADO

### Para Modelos de Pronóstico
1. **MIDAS**: Ideal para combinar con series mensuales
2. **LSTM**: Manejar gaps con masking layers
3. **XGBoost**: Usar directamente, maneja NaN
4. **ARIMA**: Considerar interpolación para series específicas

### Consideraciones Importantes
- **Retornos**: Calcular cuidadosamente entre viernes y lunes
- **Normalización**: Requerida por diferentes escalas
- **Feature Engineering**: Crear rezagos y medias móviles
- **Train/Test Split**: Usar split temporal, no aleatorio

### Archivos Generados
```
/data/processed/daily_time_series/
├── daily_series_consolidated_latest.csv     # Dataset principal
├── metadata_latest.json                     # Metadata completa
├── daily_series_consolidated_20250926_*.csv # Versión con timestamp
└── metadata_20250926_*.json                 # Metadata con timestamp
```

---

## 🔄 PRÓXIMOS PASOS

1. **Análisis Exploratorio**: Visualizar correlaciones y patrones
2. **Feature Engineering**: Crear variables derivadas
3. **Combinación con Mensuales**: Aplicar metodología MIDAS
4. **Modelado**: Implementar modelos baseline y avanzados
5. **Validación**: Backtesting con ventana móvil

---

## 📝 NOTAS TÉCNICAS

### Script de Generación
- **Archivo**: `scripts/join_daily_series.py`
- **Configuración**: `WEEKEND_STRATEGY = 'business_days'`
- **Fecha de corte**: 2025-01-01 (para series actualizadas)

### Reproducibilidad
Para regenerar el dataset:
```bash
cd deacero_steel_price_predictor
python scripts/join_daily_series.py
```

### Mantenimiento
- Actualizar semanalmente con nuevos datos
- Verificar integridad de variable objetivo
- Monitorear completitud de series críticas

---

**Última actualización**: 26/09/2025  
