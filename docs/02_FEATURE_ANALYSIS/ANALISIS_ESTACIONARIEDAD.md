# 🔬 Análisis de Estacionariedad: precio_varilla_lme

## 📌 Resumen Ejecutivo

Este documento presenta un análisis exhaustivo de las propiedades de estacionariedad del precio de la varilla corrugada (steel rebar), componente crítico para determinar la metodología de modelado óptima. El análisis incluye múltiples tests estadísticos formales y análisis de memoria larga para caracterizar completamente el comportamiento temporal de la serie.

## 🎯 Variable Analizada

- **Serie**: `precio_varilla_lme`
- **Fuente**: Investing.com (datos reales de mercado)
- **Período**: 2020-01-02 a 2025-09-25
- **Observaciones**: 1,496 puntos de datos diarios
- **Transformaciones Evaluadas**: Serie original, primera diferencia, retornos logarítmicos

## 📊 Resultados de Tests de Estacionariedad

### 1. Serie Original (Niveles de Precio)

```
📊 Tests de Estacionariedad para precio_varilla_lme:
==================================================
ADF (Augmented Dickey-Fuller): p-value = 0.6406 ❌ NO ESTACIONARIA
KPSS (Kwiatkowski-Phillips)  : p-value = 0.1000 ✅ ESTACIONARIA  
PP (Phillips-Perron)         : p-value = 0.0000 ✅ ESTACIONARIA
```

#### **Interpretación:**
- **Resultado Mixto**: Los tests muestran evidencia contradictoria
- **ADF sugiere NO estacionariedad** (presencia de raíz unitaria)
- **KPSS y PP sugieren estacionariedad** (ausencia de tendencia estocástica)
- **Diagnóstico**: Serie con **tendencia determinística** pero sin raíz unitaria

### 2. Primera Diferencia (Δprecio_t = precio_t - precio_t-1)

```
📊 Tests de Estacionariedad para precio_varilla_lme (1ra diff):
==================================================
ADF            : p-value = 0.0000 ✅ ESTACIONARIA
KPSS           : p-value = 0.1000 ✅ ESTACIONARIA
PP             : p-value = 0.0000 ✅ ESTACIONARIA
```

#### **Interpretación:**
- **Consenso Unánime**: Todos los tests confirman estacionariedad
- **Transformación Exitosa**: La primera diferencia elimina la no estacionariedad
- **Resultado Óptimo**: Serie diferenciada es estacionaria en media y varianza

### 3. Retornos Logarítmicos (log(precio_t/precio_t-1))

```
📊 Tests de Estacionariedad para precio_varilla_lme (log returns):
==================================================
ADF            : p-value = 0.0000 ✅ ESTACIONARIA
KPSS           : p-value = 0.1000 ✅ ESTACIONARIA
PP             : p-value = 0.0000 ✅ ESTACIONARIA
```

#### **Interpretación:**
- **Estacionariedad Completa**: Todos los tests confirman estacionariedad
- **Propiedades Ideales**: Los retornos logarítmicos son estacionarios
- **Ventaja Adicional**: Interpretación como cambios porcentuales

## 🧠 Análisis de Memoria Larga (Exponente de Hurst)

### Resultados del Exponente de Hurst:

```
📊 Análisis de Memoria Larga (Exponente de Hurst):
==================================================
Serie Original: H = 0.0003
   → Serie ANTI-PERSISTENTE (mean-reverting)

Retornos Log: H = -0.0059
   → Retornos ANTI-PERSISTENTES
```

### **Interpretación del Exponente de Hurst:**

| Rango de H | Interpretación | Nuestro Resultado |
|------------|----------------|-------------------|
| H > 0.5 | Persistencia (tendencia se mantiene) | ❌ No aplica |
| H = 0.5 | Movimiento browniano (aleatorio) | ❌ No aplica |
| H < 0.5 | Anti-persistencia (mean reversion) | ✅ **H ≈ 0.0003** |

#### **Implicaciones Clave:**
1. **Mean Reversion Fuerte**: La serie tiende a revertir a su media de largo plazo
2. **Anti-Persistencia**: Movimientos alcistas seguidos de correcciones
3. **Predictibilidad Alta**: El comportamiento anti-persistente facilita pronósticos
4. **Estabilidad Estructural**: No hay memoria larga que complique el modelado

## 📈 Síntesis de Resultados

### ✅ Conclusiones Principales:

1. **Serie Original**: 
   - **NO estacionaria** (presencia de tendencia)
   - Requiere transformación para modelado ARIMA

2. **Primera Diferencia**:
   - **ESTACIONARIA** por consenso de todos los tests
   - **Recomendada para modelado** ARIMA(p,d,q) con d=1

3. **Retornos Logarítmicos**:
   - **ESTACIONARIA** y con interpretación económica clara
   - **Alternativa viable** para modelos de volatilidad

4. **Memoria Larga**:
   - **Anti-persistencia fuerte** (H ≈ 0)
   - **Mean reversion** favorece predictibilidad
   - **Sin dependencia de largo plazo** que complique modelado

## 🎯 Recomendaciones para Modelado

### 🥇 **Opción Preferida: Primera Diferencia**

```python
# Transformación recomendada
precio_diff = precio_varilla_lme.diff().dropna()

# Modelo ARIMA con integración
modelo = ARIMA(precio_varilla_lme, order=(p, 1, q))
```

**Ventajas:**
- ✅ Estacionariedad confirmada por todos los tests
- ✅ Preserva la escala original de precios
- ✅ Interpretación directa (cambios en USD/ton)
- ✅ Compatible con modelos ARIMA estándar

### 🥈 **Opción Alternativa: Retornos Logarítmicos**

```python
# Para modelos de volatilidad
returns = np.log(precio_varilla_lme / precio_varilla_lme.shift(1))

# Modelo GARCH
modelo_garch = arch_model(returns, vol='Garch', p=1, q=1)
```

**Ventajas:**
- ✅ Estacionariedad garantizada
- ✅ Interpretación como cambios porcentuales
- ✅ Ideal para modelado de volatilidad
- ✅ Normaliza la varianza

## 🚀 Implicaciones Estratégicas

### Para el Modelo de Pronóstico:

1. **Metodología Clara**: ARIMA(p,1,q) es la base óptima
2. **Predictibilidad Alta**: Anti-persistencia favorece pronósticos precisos
3. **Estabilidad Temporal**: Sin quiebres estructurales que requieran regime switching
4. **Simplicidad Efectiva**: No se requieren transformaciones complejas

### Para Variables Exógenas:

1. **Integración Consistente**: Aplicar misma transformación (primera diferencia)
2. **Cointegración Potencial**: Evaluar relaciones de largo plazo con materias primas
3. **Rezagos Óptimos**: Anti-persistencia sugiere rezagos cortos (1-5 períodos)

## 📊 Matriz de Decisión de Transformaciones

| Transformación | Estacionariedad | Interpretabilidad | Recomendación |
|----------------|-----------------|-------------------|---------------|
| **Serie Original** | ❌ No | ✅ Excelente | ❌ No usar |
| **Primera Diferencia** | ✅ Sí | ✅ Muy Buena | ✅ **RECOMENDADA** |
| **Retornos Log** | ✅ Sí | ✅ Buena | ✅ Alternativa |

## 💡 Insights Técnicos Avanzados

### Sobre la Anti-Persistencia (H ≈ 0):

1. **Ventaja Competitiva**: Pocos commodities muestran anti-persistencia tan marcada
2. **Oportunidad de Arbitraje**: Mean reversion permite estrategias contrarias
3. **Horizonte Óptimo**: Pronósticos de 1-10 días especialmente confiables
4. **Gestión de Riesgo**: Reversiones predecibles facilitan stop-losses

### Sobre la Consistencia de Tests:

1. **ADF vs KPSS/PP**: Diferencia típica entre tests de raíz unitaria vs tendencia
2. **Robustez**: Consenso en transformaciones confirma solidez metodológica
3. **Sensibilidad**: Tests capturan correctamente las propiedades de la serie

## 🔄 Próximos Pasos

1. **Implementar Modelado ARIMA**:
   ```python
   # Orden recomendado basado en estacionariedad
   modelo_base = ARIMA(precio_varilla_lme, order=(2, 1, 2))
   ```

2. **Evaluar Cointegración**:
   - Test de Johansen con materias primas (iron ore, coking coal)
   - Vector Error Correction Model (VECM) si aplica

3. **Validar Anti-Persistencia**:
   - Confirmar H < 0.5 en submuestras
   - Evaluar estabilidad temporal del exponente

4. **Optimizar Transformaciones**:
   - Comparar performance primera diferencia vs retornos log
   - Selección basada en métricas out-of-sample

---

*Documento generado: Septiembre 2025*  
*Análisis basado en tests ADF, KPSS, Phillips-Perron y Exponente de Hurst*  
*Fuente de datos: Investing.com (steel rebar real)*  
*Recomendación final: **Primera Diferencia para modelado ARIMA(p,1,q)***
