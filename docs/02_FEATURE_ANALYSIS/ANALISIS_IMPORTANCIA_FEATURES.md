# 🎯 Análisis de Importancia de Features - Precio Varilla Corrugada LME

## 📌 Resumen Ejecutivo

Este documento presenta el análisis crítico de la importancia de features obtenida mediante Random Forest, revelando una dominancia problemática de variables autorregresivas sobre predictores causales fundamentales. La integración con el análisis de causalidad-correlación expone una discrepancia fundamental entre importancia estadística e importancia económica que debe resolverse para un modelado robusto.

## 🎯 Objetivo del Análisis

- **Evaluar importancia relativa** de features según Random Forest
- **Identificar sesgo hacia variables autorregresivas**
- **Reconciliar con análisis de causalidad** de Granger
- **Proponer selección óptima** de variables para predicción

## 📊 Visualización de Importancia de Features

![TOP 15 Features Más Importantes](importance.png)

## 📈 Análisis de los Resultados

### Top 15 Features por Importancia

| Ranking | Feature | Importancia | Tipo | Interpretación |
|---------|---------|-------------|------|-----------------|
| **1** | precio_varilla_lme_lag_1 | 0.246 | Autorregresiva | Valor de ayer (24.6% importancia) |
| **2** | steel_in_mxn | 0.141 | Transformada | Precio × tipo cambio |
| **3** | precio_varilla_lme_lag_10 | 0.125 | Autorregresiva | Valor hace 10 días |
| **4** | precio_varilla_lme_ma_5 | 0.113 | Media móvil | Promedio 5 días |
| **5** | precio_varilla_lme_lag_3 | 0.054 | Autorregresiva | Valor hace 3 días |
| **6** | precio_varilla_lme_ma_20 | 0.038 | Media móvil | Promedio 20 días |
| **7** | precio_varilla_lme_ma_10 | 0.038 | Media móvil | Promedio 10 días |
| **8** | precio_varilla_lme_ema_12 | 0.034 | EMA | Media exponencial 12 |
| **9** | precio_varilla_lme_ema_26 | 0.034 | EMA | Media exponencial 26 |
| **10** | precio_varilla_lme_lag_2 | 0.034 | Autorregresiva | Valor hace 2 días |
| **11** | udis_valor | 0.033 | Externa | Inflación acumulada |
| **12** | precio_varilla_lme_lag_5 | 0.027 | Autorregresiva | Valor hace 5 días |
| **13** | precio_varilla_lme_bb_lower | 0.018 | Técnica | Banda Bollinger inferior |
| **14** | nucor_acciones | 0.018 | Externa | Acción acero USA |
| **15** | Petroleo | 0.012 | Externa | Commodity energético |

## 🚨 Problema Crítico Identificado

### **Dominancia de Features Autorregresivas**

```
Features Autorregresivas/Técnicas: 12 de 15 (80%)
Features Externas Fundamentales: 3 de 15 (20%)
```

#### **Distribución de Importancia:**
- **Variables propias del precio**: 91.5% de importancia total
- **Variables externas**: 8.5% de importancia total

### **¿Por qué es esto un problema?**

1. **Overfitting temporal**: El modelo aprende a "copiar" el valor anterior
2. **Sin valor predictivo real**: lag_1 ≈ 0.99 correlación es trivial
3. **Ignora drivers fundamentales**: Iron, coking no aparecen en top 15
4. **Ilusión de precisión**: R² alto pero falla en cambios de tendencia

## 🔬 Reconciliación con Análisis de Causalidad

### **Discrepancia Fundamental**

| Variable | Importancia RF | Ranking RF | P-value Granger | Ranking Causal |
|----------|---------------|------------|-----------------|----------------|
| **lag_1** | 0.246 | 1° | N/A | N/A |
| **iron** | 0.001 | >20° | **0.0000** | **1°** |
| **coking** | <0.001 | >25° | **0.0000** | **1°** |
| **Petróleo** | 0.012 | 15° | 0.0029 | 10° |
| **udis_valor** | 0.033 | 11° | N/S | N/S |

### **Interpretación de la Discrepancia**

#### 1. **Random Forest Prefiere el Camino Fácil**
```python
# Lo que hace Random Forest:
precio_hoy ≈ precio_ayer + pequeño_ajuste
# Importancia: 24.6% para lag_1

# Lo que DEBERÍA hacer:
precio_hoy = f(iron, coking, gas_natural, mercado)
# Pero estas variables tienen < 1% importancia
```

#### 2. **Variables Causales Ignoradas**

Según el análisis de Granger, las variables con **máxima causalidad** son:

| Variable | P-value Granger | Importancia RF | Status |
|----------|-----------------|----------------|--------|
| **iron** | 0.0000 | 0.001 | ❌ Ignorada |
| **coking** | 0.0000 | <0.001 | ❌ Ignorada |
| **commodities** | 0.0005 | 0.001 | ❌ Ignorada |
| **aluminio_lme** | 0.0003 | No top 25 | ❌ Ignorada |
| **gas_natural** | 0.0013 | No top 25 | ❌ Ignorada |

## 📊 Análisis por Categorías de Features

### **Categoría 1: Features Autorregresivas (91.5% importancia)**

```python
autorregresivas = [
    'precio_varilla_lme_lag_1',    # 24.6%
    'precio_varilla_lme_lag_10',   # 12.5%
    'precio_varilla_lme_ma_5',     # 11.3%
    'precio_varilla_lme_lag_3',    # 5.4%
    # ... etc
]
# PROBLEMA: Dominan completamente el modelo
```

**Características:**
- Alta correlación con target (>0.95)
- Cero información nueva
- Inútiles para pronóstico real
- Crean ilusión de precisión

### **Categoría 2: Features Transformadas (14.1%)**

```python
transformadas = [
    'steel_in_mxn',  # 14.1% - Segunda más importante
]
# steel_in_mxn = precio_varilla × tipo_cambio
```

**Problema:** Es una transformación del target mismo, no información externa

### **Categoría 3: Features Externas (6.3%)**

```python
externas = [
    'udis_valor',      # 3.3%
    'nucor_acciones',  # 1.8%
    'Petroleo',        # 1.2%
    # Notablemente AUSENTES: iron, coking, gas_natural
]
```

**Observación crítica:** Las variables con mayor causalidad tienen menor importancia

## 💡 Explicación del Fenómeno

### **¿Por qué Random Forest falla en identificar drivers reales?**

#### 1. **Sesgo hacia Predictibilidad Inmediata**
- RF maximiza reducción de error en cada split
- lag_1 reduce error instantáneamente
- Variables causales tienen efecto retardado (20-30 días)

#### 2. **Problema de Horizonte Temporal**
```python
# RF ve esto:
precio_t = 0.99 * precio_t-1 + ruido
# R² = 0.99, Importancia lag_1 = máxima

# No ve esto:
precio_t+30 = f(iron_t, coking_t, gas_t)
# Relación real pero con rezago
```

#### 3. **Multicolinealidad con Lags**
- ma_5 ≈ promedio(lag_1, lag_2, lag_3, lag_4, lag_5)
- ema_12 ≈ versión ponderada de lags
- RF distribuye importancia entre variables redundantes

## 🎯 Estrategia de Selección Corregida

### **Modelo 1: Predictivo Genuino (Sin Autorregresión Excesiva)**

```python
# MÁXIMO 2 lags, resto variables causales
features_predictivas = [
    # Autorregresivas (máximo 20% del modelo)
    'precio_varilla_lme_lag_1',  # Solo AR(1)
    'precio_varilla_lme_volatility_20',  # Volatilidad
    
    # Causales Granger (60% del modelo)
    'iron',           # P-value = 0.0000
    'coking',         # P-value = 0.0000
    'gas_natural',    # P-value = 0.0013
    'aluminio_lme',   # P-value = 0.0003
    'commodities',    # P-value = 0.0005
    
    # Mercado (20% del modelo)
    'steel',          # Índice sectorial
    'VIX',            # Volatilidad inversa
]
```

### **Modelo 2: Explicativo (Sin Autorregresión)**

```python
# CERO lags - solo fundamentales
features_explicativas = [
    'iron', 'coking', 'gas_natural',
    'commodities', 'steel', 'aluminio_lme',
    'VIX', 'sp500', 'tipo_cambio_usdmxn'
]
# Para entender QUÉ mueve los precios
```

## 📈 Comparación: Importancia RF vs Causalidad Granger

### **Variables que DEBERÍAN ser importantes según economía:**

| Variable | Importancia RF | Importancia Económica | Acción |
|----------|---------------|----------------------|--------|
| **iron** | ❌ Casi 0% | ⭐⭐⭐⭐⭐ Máxima | ✅ INCLUIR |
| **coking** | ❌ Casi 0% | ⭐⭐⭐⭐⭐ Máxima | ✅ INCLUIR |
| **gas_natural** | ❌ No aparece | ⭐⭐⭐⭐ Alta | ✅ INCLUIR |
| **lag_1** | ⚠️ 24.6% | ⭐ Mínima | ⚠️ LIMITAR |
| **Petróleo** | 1.2% | ⭐⭐ Baja | ❌ EXCLUIR |

## 🚨 Alertas y Recomendaciones

### **Alertas Críticas:**

1. **NO confiar ciegamente en Random Forest** para selección de features
2. **Limitar severamente variables autorregresivas** (máximo 2-3)
3. **Priorizar variables con causalidad económica** sobre importancia estadística
4. **Evitar features derivadas del target** (steel_in_mxn)

### **Recomendaciones Específicas:**

#### Para Pronóstico a 1-5 días:
```python
modelo_corto_plazo = {
    'autorregresivas': ['lag_1'],  # Solo una
    'volatilidad': ['volatility_20'],
    'causales': ['iron', 'coking', 'gas_natural'],
    'mercado': ['VIX']
}
```

#### Para Pronóstico a 5-20 días:
```python
modelo_mediano_plazo = {
    'autorregresivas': [],  # Ninguna
    'causales': ['iron', 'coking', 'gas_natural', 'aluminio_lme'],
    'mercado': ['commodities', 'steel', 'VIX'],
    'macro': ['sp500', 'tasa_interes_banxico']
}
```

## 📊 Propuesta de Ponderación Manual

### **Basada en Análisis Integral:**

```python
importancia_ajustada = {
    # Tier 1: Causales Fundamentales (50%)
    'iron': 0.15,
    'coking': 0.15,
    'gas_natural': 0.10,
    'aluminio_lme': 0.10,
    
    # Tier 2: Mercado (25%)
    'commodities': 0.10,
    'steel': 0.08,
    'VIX': 0.07,
    
    # Tier 3: Autorregresivas (15%)
    'lag_1': 0.10,
    'volatility_20': 0.05,
    
    # Tier 4: Macro (10%)
    'sp500': 0.05,
    'tasa_interes_banxico': 0.05
}
```

## 🔬 Validación Propuesta

### **Test A/B de Modelos:**

```python
# Modelo A: Basado en Random Forest
modelo_rf = features_por_importancia_rf[:10]

# Modelo B: Basado en Causalidad
modelo_causal = features_por_granger[:10]

# Modelo C: Híbrido Balanceado
modelo_hibrido = combinar_rf_granger_economia()

# Evaluar en:
# 1. RMSE out-of-sample
# 2. Captura de puntos de inflexión
# 3. Estabilidad en diferentes períodos
# 4. Interpretabilidad económica
```

## ✅ Conclusiones

### **Hallazgos Principales:**

1. **Random Forest sobrevalora dramáticamente variables autorregresivas**
   - lag_1 tiene 24.6% importancia pero cero valor predictivo real
   - 12 de 15 top features son transformaciones del precio mismo

2. **Variables causales fundamentales son ignoradas**
   - Iron y coking (inputs directos) tienen <0.1% importancia
   - Contradicción total con análisis de Granger

3. **El modelo actual es un "espejo retrovisor"**
   - Excelente para "predecir" el pasado inmediato
   - Inútil para cambios de tendencia o horizontes largos

4. **Necesidad de intervención manual en selección**
   - RF no captura relaciones económicas fundamentales
   - Causalidad > Correlación > Importancia RF

### **Recomendación Final:**

> **NO usar importancia de Random Forest como único criterio de selección**
> 
> Combinar:
> - 40% peso a causalidad de Granger
> - 30% peso a teoría económica
> - 20% peso a importancia RF
> - 10% peso a estabilidad temporal

## 🚀 Próximos Pasos

1. **Re-entrenar modelo** con features seleccionadas por causalidad
2. **Comparar performance** RF puro vs Causal vs Híbrido
3. **Implementar regularización** para penalizar autorregresivas
4. **Validar en período de crisis** (2020) y burbuja (2021-2022)
5. **Documentar mejoras** en captura de cambios de tendencia

---

*Documento generado: Septiembre 2025*  
*Método: Random Forest (100 árboles)*  
*Features evaluadas: 61 totales*  
*Contraste con: Causalidad de Granger lag 30*
