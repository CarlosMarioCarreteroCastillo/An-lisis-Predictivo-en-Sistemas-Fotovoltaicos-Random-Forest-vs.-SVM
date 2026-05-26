# Examen 2 — Comparación Random Forest vs SVM  
## Sistema Fotovoltaico Instrumentado · Inteligencia Artificial

**Grupo:** [Nombres del grupo]  
**Institución:** Universidad Tecnológica de Pereira  
**Docente:** German A. Holguín L.

---

## Descripción

Este repositorio implementa la comparación entre **Random Forest** y **Support Vector Machines** aplicada al diagnóstico y predicción de un sistema fotovoltaico instrumentado bajo el paradigma de la Industria 4.0.

Se abordan dos problemas de aprendizaje supervisado:

| Problema | Variable objetivo | Modelos |
|---|---|---|
| Clasificación | `condition` (condición operativa) | RandomForestClassifier vs SVC |
| Regresión | `power_W` (potencia generada) | RandomForestRegressor vs SVR |

---

## Estructura del repositorio

```
IA_PARCIAL2/
├── data_simulation.py    # Generación del dataset sintético con física PV
├── preprocessing.py      # Preprocesamiento, escalado y prevención de fuga
├── models.py             # Entrenamiento con RandomizedSearchCV + CV=5
├── evaluation.py         # Métricas, validación cruzada y 10 figuras
├── main.py               # Pipeline completo (ejecutar este archivo)
├── requirements.txt      # Dependencias de Python
├── synthetic_pv_dataset.csv  # Dataset (generado por main.py)
├── artefactos/           # Modelos y transformadores serializados (.pkl)
└── resultados/           # Figuras generadas (.png)
```

---

## Instalación

```bash
# 1. Clonar o descomprimir el repositorio
# 2. Instalar dependencias
pip install -r requirements.txt
```

Requiere **Python ≥ 3.10**.

---

## Ejecución

```bash
# Ejecutar el pipeline completo (simulación → preprocesamiento → entrenamiento → evaluación)
python main.py
```

El script ejecuta todo de forma secuencial. Al finalizar se generan:

- `synthetic_pv_dataset.csv` — 5 000 muestras del sistema fotovoltaico
- `artefactos/` — modelos RFC, SVC, RFR, SVR + LabelEncoder + Scalers
- `resultados/` — 10 figuras PNG de análisis

---

## Figuras generadas

| Archivo | Descripción |
|---|---|
| `distribucion_clases.png` | Balance de clases del dataset |
| `confusion_matrices.png` | Matrices de confusión normalizadas |
| `clasificacion_metricas_comparativo.png` | Precision, Recall y F1 por clase |
| `feature_importance_clf.png` | Importancia de variables (RFC) |
| `regresion_comparativo.png` | Scatter real vs predicho + residuos |
| `regresion_metricas.png` | MAE, RMSE y R² comparativos |
| `feature_importance_reg.png` | Importancia de variables (RFR) |
| `validacion_cruzada.png` | Distribución de scores CV=5 |
| `sensibilidad_C.png` | Análisis de sensibilidad al hiperparámetro C |
| `efecto_normalizacion.png` | Impacto del escalado en SVM/SVR |

---

## Prevención de fuga de información

| Problema | Variable objetivo | Variables excluidas como entrada |
|---|---|---|
| Clasificación | `condition` | `condition` (es la etiqueta), `power_W` (objetivo de regresión) |
| Regresión | `power_W` | `power_W` (es la etiqueta), `condition` (codifica f\_falla), `voltage_V`, `current_A` (P ≈ V·I) |

---

## Módulos individuales

Cada módulo puede ejecutarse de forma independiente para diagnóstico:

```bash
python data_simulation.py   # Solo genera el CSV
python preprocessing.py     # Solo preprocesa y guarda artefactos
python models.py            # Solo entrena y evalúa en consola
```
