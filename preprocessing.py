# -*- coding: utf-8 -*-
"""
preprocessing.py
================
Preprocesamiento del dataset fotovoltaico sintético.
Examen 2 – Comparación Random Forest vs SVM.

Responsabilidades:
  1. Cargar y validar el CSV generado por data_simulation.py
  2. Separar variables de entrada y salida (prevención de fuga de información)
  3. Codificar etiquetas categóricas (clasificación)
  4. Dividir en conjuntos de entrenamiento / prueba (80/20, estratificado para CLF)
  5. Escalar características con StandardScaler (obligatorio para SVM/SVR)
  6. Persistir artefactos de transformación (LabelEncoder, Scalers)

══════════════════════════════════════════════════════════════════════════════
PREVENCIÓN DE FUGA DE INFORMACIÓN (Information Leakage Prevention)
══════════════════════════════════════════════════════════════════════════════

  Problema       │ Variable objetivo │ Variables EXCLUIDAS como entrada
  ───────────────┼───────────────────┼──────────────────────────────────────
  Clasificación  │ condition         │ condition  (es la etiqueta misma)
                 │                   │ power_W    (variable objetivo de regresión;
                 │                   │            su uso introduciría dependencia
                 │                   │            cruzada entre tareas)
  ───────────────┼───────────────────┼──────────────────────────────────────
  Regresión      │ power_W           │ power_W    (es la etiqueta misma)
                 │                   │ condition  (codifica implícitamente f_falla
                 │                   │            con el que se calculó power_W)
                 │                   │ voltage_V  (P ≈ V·I → predicción trivial)
                 │                   │ current_A  (P ≈ V·I → predicción trivial)

══════════════════════════════════════════════════════════════════════════════
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler

# ── Rutas ──────────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(__file__)
DATA_PATH    = os.path.join(_DIR, 'synthetic_pv_dataset.csv')
ARTIFACT_DIR = os.path.join(_DIR, 'artefactos')

# ── Variables de entrada por tarea ─────────────────────────────────────────────
FEATURES_CLF = [
    'hour',
    'day_of_year',
    'irradiance_W_m2',
    'ambient_temp_C',
    'panel_temp_C',
    'humidity_percent',
    'wind_speed_m_s',
    'voltage_V',
    'current_A',
]

FEATURES_REG = [
    'hour',
    'day_of_year',
    'irradiance_W_m2',
    'ambient_temp_C',
    'panel_temp_C',
    'humidity_percent',
    'wind_speed_m_s',
]

TARGET_CLF   = 'condition'
TARGET_REG   = 'power_W'

# ── Configuración de partición ─────────────────────────────────────────────────
TEST_SIZE    = 0.20   # 20 % para prueba, 80 % para entrenamiento
RANDOM_STATE = 42


# ── Carga y validación ─────────────────────────────────────────────────────────

def cargar_datos(ruta: str = DATA_PATH) -> pd.DataFrame:
    """
    Carga y valida el CSV del sistema fotovoltaico.

    Verifica que existan todas las columnas requeridas antes de continuar.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró '{ruta}'.\n"
            "Ejecuta primero:  python data_simulation.py"
        )
    df = pd.read_csv(ruta)
    requeridas = set(FEATURES_CLF + FEATURES_REG + [TARGET_CLF, TARGET_REG])
    faltantes  = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {sorted(faltantes)}")
    print(f"[preprocessing] Dataset cargado: {df.shape[0]} muestras × {df.shape[1]} columnas")
    return df


# ── Utilidad interna ───────────────────────────────────────────────────────────

def _split_y_escalar(X: np.ndarray, y: np.ndarray, estratificar: bool = False):
    """
    Divide el dataset en train/test y aplica StandardScaler.

    El scaler se ajusta EXCLUSIVAMENTE sobre el conjunto de entrenamiento
    y se aplica al de prueba, evitando filtración de estadísticos del test.

    Retorna
    -------
    X_train, X_test          : arrays sin escalar (para Random Forest)
    X_train_sc, X_test_sc    : arrays escalados  (para SVM / SVR)
    y_train, y_test          : etiquetas / valores objetivo
    scaler                   : StandardScaler ajustado (para persistencia)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y if estratificar else None,
    )
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler


# ── Pipeline de clasificación ──────────────────────────────────────────────────

def preparar_clasificacion(df: pd.DataFrame):
    """
    Prepara datos para clasificación multiclase de condición operativa.

    - Estratificación en la partición para mantener la distribución de clases.
    - LabelEncoder convierte etiquetas de texto a enteros.

    Retorna
    -------
    X_train, X_test          : sin escalar (Random Forest no requiere escalado)
    X_train_sc, X_test_sc    : escaladas   (SVC requiere escalado)
    y_train, y_test          : etiquetas enteras
    le                       : LabelEncoder ajustado
    scaler                   : StandardScaler ajustado sobre X_train
    feature_names            : lista de nombres de características
    """
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET_CLF].values)
    X  = df[FEATURES_CLF].values

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = \
        _split_y_escalar(X, y, estratificar=True)

    print(f"[preprocessing] Clasificación → train: {X_train.shape} | test: {X_test.shape}")
    print(f"[preprocessing] Clases: {list(le.classes_)}")
    return (X_train, X_test, X_train_sc, X_test_sc,
            y_train, y_test, le, scaler, FEATURES_CLF)


# ── Pipeline de regresión ──────────────────────────────────────────────────────

def preparar_regresion(df: pd.DataFrame):
    """
    Prepara datos para regresión continua de potencia generada.

    Variables excluidas: power_W, condition, voltage_V, current_A.
    El modelo debe predecir potencia únicamente a partir de condiciones
    ambientales, sin información eléctrica directa.

    Retorna
    -------
    X_train, X_test          : sin escalar (Random Forest)
    X_train_sc, X_test_sc    : escaladas   (SVR)
    y_train, y_test          : potencia en vatios [W]
    scaler                   : StandardScaler ajustado sobre X_train
    feature_names            : lista de nombres de características
    """
    X = df[FEATURES_REG].values
    y = df[TARGET_REG].values

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = \
        _split_y_escalar(X, y, estratificar=False)

    print(f"[preprocessing] Regresión → train: {X_train.shape} | test: {X_test.shape}")
    print(f"[preprocessing] Rango de potencia: [{y.min():.1f}, {y.max():.1f}] W")
    return (X_train, X_test, X_train_sc, X_test_sc,
            y_train, y_test, scaler, FEATURES_REG)


# ── Persistencia de artefactos ─────────────────────────────────────────────────

def guardar_artefactos(le: LabelEncoder, scaler_clf: StandardScaler,
                        scaler_reg: StandardScaler,
                        carpeta: str = ARTIFACT_DIR) -> None:
    """Persiste transformadores para uso en inferencia posterior."""
    os.makedirs(carpeta, exist_ok=True)
    joblib.dump(le,         os.path.join(carpeta, 'label_encoder.pkl'))
    joblib.dump(scaler_clf, os.path.join(carpeta, 'scaler_clf.pkl'))
    joblib.dump(scaler_reg, os.path.join(carpeta, 'scaler_reg.pkl'))
    print(f"[preprocessing] Artefactos guardados en: {carpeta}")


# ── Ejecución directa ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = cargar_datos()

    (X_tr_clf, X_te_clf, X_tr_clf_sc, X_te_clf_sc,
     y_tr_clf, y_te_clf, le, scaler_clf, _) = preparar_clasificacion(df)

    (X_tr_reg, X_te_reg, X_tr_reg_sc, X_te_reg_sc,
     y_tr_reg, y_te_reg, scaler_reg, _)     = preparar_regresion(df)

    guardar_artefactos(le, scaler_clf, scaler_reg)
