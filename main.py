# -*- coding: utf-8 -*-
"""
main.py
=======
Pipeline completo — Examen 2: Comparación Random Forest vs SVM
Sistema Fotovoltaico Instrumentado (Industria 4.0)

Flujo de ejecución:
  1. Simulación de datos con reglas físicas
  2. Preprocesamiento y prevención de fuga de información
  3. Entrenamiento con búsqueda de hiperparámetros (RandomizedSearchCV)
  4. Evaluación con métricas completas y validación cruzada (CV=5)
  5. Generación de todas las figuras de análisis
  6. Serialización de modelos y artefactos
"""

import os
import time

from data_simulation import simular_dataset
from preprocessing   import (
    cargar_datos, preparar_clasificacion, preparar_regresion,
    guardar_artefactos, ARTIFACT_DIR,
)
from models import (
    entrenar_clasificacion, entrenar_regresion,
    evaluar_clasificacion, evaluar_regresion,
    guardar_modelos,
)
from evaluation import (
    graficar_distribucion_clases,
    metricas_clasificacion,
    graficar_confusion,
    graficar_metricas_clasificacion,
    graficar_importancia_clf,
    metricas_regresion,
    graficar_regresion,
    graficar_metricas_regresion,
    graficar_importancia_reg,
    graficar_validacion_cruzada,
    graficar_sensibilidad_C,
    graficar_efecto_normalizacion,
    imprimir_resumen,
    RESULT_DIR,
)

SEP  = '═' * 60
SEP2 = '─' * 60

def paso(n: int, titulo: str) -> None:
    print(f'\n{SEP}')
    print(f' PASO {n}: {titulo}')
    print(SEP)


if __name__ == '__main__':

    t_inicio = time.time()

    # ── 1. Simulación ──────────────────────────────────────────────────────────
    paso(1, 'SIMULACIÓN DE DATOS FOTOVOLTAICOS')
    df = simular_dataset()
    print(f'\nDimensiones del dataset: {df.shape[0]} muestras × {df.shape[1]} columnas')
    print(f'Columnas: {list(df.columns)}')

    # ── 2. Preprocesamiento ────────────────────────────────────────────────────
    paso(2, 'PREPROCESAMIENTO Y PREVENCIÓN DE FUGA DE INFORMACIÓN')
    print('\n[Clasificación]')
    print('  Variables de entrada: hour, day_of_year, irradiance_W_m2,')
    print('    ambient_temp_C, panel_temp_C, humidity_percent,')
    print('    wind_speed_m_s, voltage_V, current_A')
    print('  Variable objetivo: condition')
    print('  Excluidas (leakage): condition (es la etiqueta), power_W (objetivo de regresión)')

    (X_tr_clf, X_te_clf, X_tr_clf_sc, X_te_clf_sc,
     y_tr_clf, y_te_clf, le, scaler_clf, feats_clf) = preparar_clasificacion(df)

    print('\n[Regresión]')
    print('  Variables de entrada: hour, day_of_year, irradiance_W_m2,')
    print('    ambient_temp_C, panel_temp_C, humidity_percent, wind_speed_m_s')
    print('  Variable objetivo: power_W')
    print('  Excluidas (leakage): power_W (etiqueta), condition (codifica f_falla),')
    print('    voltage_V y current_A (P ≈ V·I → predicción trivial)')

    (X_tr_reg, X_te_reg, X_tr_reg_sc, X_te_reg_sc,
     y_tr_reg, y_te_reg, scaler_reg, feats_reg) = preparar_regresion(df)

    class_names = list(le.classes_)
    guardar_artefactos(le, scaler_clf, scaler_reg)

    # ── 3. Entrenamiento ───────────────────────────────────────────────────────
    paso(3, 'ENTRENAMIENTO CON BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)')
    print('\nUsando CV=5 estratificado para clasificación, CV=5 KFold para regresión.')
    print('Kernel RBF seleccionado para SVM/SVR (ver justificación en models.py).\n')

    rfc, svc, hp_rfc, hp_svc, cv_rfc, cv_svc = \
        entrenar_clasificacion(X_tr_clf, X_tr_clf_sc, y_tr_clf, buscar_hp=True)

    rfr, svr, hp_rfr, hp_svr, cv_rfr, cv_svr = \
        entrenar_regresion(X_tr_reg, X_tr_reg_sc, y_tr_reg, buscar_hp=True)

    # ── 4. Evaluación ──────────────────────────────────────────────────────────
    paso(4, 'EVALUACIÓN EN CONJUNTO DE PRUEBA (20 %)')
    evaluar_clasificacion(rfc, svc, X_te_clf, X_te_clf_sc, y_te_clf, class_names)
    evaluar_regresion(rfr, svr, X_te_reg, X_te_reg_sc, y_te_reg)

    res_clf = metricas_clasificacion(rfc, svc, X_te_clf, X_te_clf_sc, y_te_clf, class_names)
    res_reg = metricas_regresion(rfr, svr, X_te_reg, X_te_reg_sc, y_te_reg)

    imprimir_resumen(res_clf, res_reg,
                     cv_rfc, cv_svc, cv_rfr, cv_svr,
                     hp_rfc, hp_svc, hp_rfr, hp_svr)

    # ── 5. Gráficas ────────────────────────────────────────────────────────────
    paso(5, 'GENERACIÓN DE FIGURAS')
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f'\nGuardando figuras en: {RESULT_DIR}\n')

    archivos = []
    archivos.append(graficar_distribucion_clases(df))
    archivos.append(graficar_confusion(res_clf, class_names))
    archivos.append(graficar_metricas_clasificacion(res_clf, class_names))
    archivos.append(graficar_importancia_clf(rfc, feats_clf))
    archivos.append(graficar_regresion(res_reg, y_te_reg))
    archivos.append(graficar_metricas_regresion(res_reg))
    archivos.append(graficar_importancia_reg(rfr, feats_reg))
    archivos.append(graficar_validacion_cruzada(cv_rfc, cv_svc, cv_rfr, cv_svr))
    archivos.append(graficar_sensibilidad_C(
        X_tr_clf_sc, y_tr_clf, X_tr_reg_sc, y_tr_reg))
    archivos.append(graficar_efecto_normalizacion(
        rfc, svc,
        X_tr_clf, X_te_clf, X_tr_clf_sc, X_te_clf_sc, y_tr_clf, y_te_clf,
        rfr, svr,
        X_tr_reg, X_te_reg, X_tr_reg_sc, X_te_reg_sc, y_tr_reg, y_te_reg,
    ))

    print(f'\n{SEP2}')
    print(f'Total de figuras generadas: {len(archivos)}')
    for a in archivos:
        print(f'  ✔ {os.path.basename(a)}')

    # ── 6. Persistencia ────────────────────────────────────────────────────────
    paso(6, 'SERIALIZACIÓN DE MODELOS')
    guardar_modelos(rfc, svc, rfr, svr)

    # ── Resumen final ──────────────────────────────────────────────────────────
    t_total = time.time() - t_inicio
    print(f'\n{SEP}')
    print(f' Pipeline completado en {t_total:.1f} s')
    print(f' Dataset   : synthetic_pv_dataset.csv')
    print(f' Artefactos: {ARTIFACT_DIR}')
    print(f' Resultados: {RESULT_DIR}')
    print(SEP)
