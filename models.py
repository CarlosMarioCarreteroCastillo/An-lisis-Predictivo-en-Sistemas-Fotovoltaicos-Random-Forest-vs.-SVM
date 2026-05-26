# -*- coding: utf-8 -*-
"""
models.py
=========
Entrenamiento, búsqueda de hiperparámetros y validación cruzada de modelos
para el sistema fotovoltaico.
Examen 2 – Comparación Random Forest vs SVM.

Modelos implementados:
  Clasificación : RandomForestClassifier  |  SVC  (kernel RBF)
  Regresión     : RandomForestRegressor   |  SVR  (kernel RBF)

Estrategia de hiperparámetros
──────────────────────────────
Se realiza una búsqueda con RandomizedSearchCV (5 iteraciones, CV=5)
sobre una cuadrícula reducida. El objetivo es justificar los hiperparámetros
con evidencia experimental y no solo por criterio manual.

Justificación del kernel RBF para SVM/SVR
──────────────────────────────────────────
Los datos fotovoltaicos presentan relaciones no lineales entre las variables
ambientales (irradiancia, temperatura) y las salidas (condición, potencia).
El kernel RBF (Radial Basis Function) captura estas no linealidades mediante
una función de similitud gaussiana en el espacio de características, siendo
la elección más robusta como punto de partida para datos continuos con
estructura no lineal desconocida.

Justificación de la validación cruzada (CV=5)
──────────────────────────────────────────────
La validación cruzada estratificada de 5 folds permite estimar la varianza
del desempeño sin depender de una única partición aleatoria, lo que produce
una comparación más confiable entre modelos.
"""

import os
import time
import joblib
import numpy as np
from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor
from sklearn.svm              import SVC, SVR
from sklearn.model_selection  import RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics          import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from preprocessing import (
    cargar_datos, preparar_clasificacion, preparar_regresion, ARTIFACT_DIR,
)

_DIR = os.path.dirname(__file__)

# ── Espacios de búsqueda de hiperparámetros ────────────────────────────────────
# Se definen rangos con fundamento físico/estadístico:
#   - n_estimators: más árboles → menor varianza, mayor costo computacional.
#     200-500 es suficiente para datasets de ~5 000 muestras.
#   - max_depth: None permite árboles completos; valores bajos regularizan.
#   - C (SVM): balance entre margen y errores de clasificación.
#     C alto → menos regularización. Rango logarítmico cubre varios órdenes.
#   - gamma='scale': 1/(n_features * X.var()), evita ajuste manual.
#   - epsilon (SVR): zona de insensibilidad alrededor de la predicción.

RF_CLF_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
}

SVC_GRID = {
    'C':     [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'],
}

RF_REG_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.5],
}

SVR_GRID = {
    'C':       [0.1, 1.0, 10.0, 100.0],
    'gamma':   ['scale', 'auto'],
    'epsilon': [0.1, 0.5, 1.0],
    'kernel':  ['rbf'],
}

CV_FOLDS    = 5
N_ITER      = 8    # iteraciones de RandomizedSearch (balance velocidad/exhaustividad)
RANDOM_STATE = 42


# ── Clasificación ──────────────────────────────────────────────────────────────

def entrenar_clasificacion(X_train, X_train_sc, y_train, buscar_hp: bool = True):
    """
    Entrena RandomForestClassifier y SVC con búsqueda de hiperparámetros.

    Parámetros
    ----------
    X_train, X_train_sc : datos de entrenamiento (sin/con escalar)
    y_train             : etiquetas de clase
    buscar_hp           : si True, ejecuta RandomizedSearchCV

    Retorna
    -------
    rfc, svc            : modelos entrenados con mejores hiperparámetros
    hp_rfc, hp_svc      : diccionarios con hiperparámetros seleccionados
    cv_rfc, cv_svc      : scores de validación cruzada (accuracy)
    """
    cv_strat = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Random Forest Classifier ──
    print("\n[models] Buscando hiperparámetros para RandomForestClassifier...")
    t0 = time.time()
    if buscar_hp:
        search_rfc = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            RF_CLF_GRID, n_iter=N_ITER, cv=cv_strat,
            scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_rfc.fit(X_train, y_train)
        rfc    = search_rfc.best_estimator_
        hp_rfc = search_rfc.best_params_
        print(f"  Mejores hiperparámetros RFC: {hp_rfc}")
        print(f"  Accuracy CV (búsqueda): {search_rfc.best_score_:.4f}")
    else:
        rfc    = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rfc.fit(X_train, y_train)
        hp_rfc = rfc.get_params()
    print(f"  Tiempo RFC: {time.time()-t0:.1f} s")

    # Validación cruzada final con mejores hiperparámetros
    cv_rfc = cross_val_score(rfc, X_train, y_train, cv=cv_strat,
                              scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy RFC: {cv_rfc.mean():.4f} ± {cv_rfc.std():.4f}")

    # ── SVC ──
    print("\n[models] Buscando hiperparámetros para SVC...")
    t0 = time.time()
    if buscar_hp:
        search_svc = RandomizedSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            SVC_GRID, n_iter=N_ITER, cv=cv_strat,
            scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_svc.fit(X_train_sc, y_train)
        svc    = search_svc.best_estimator_
        hp_svc = search_svc.best_params_
        print(f"  Mejores hiperparámetros SVC: {hp_svc}")
        print(f"  Accuracy CV (búsqueda): {search_svc.best_score_:.4f}")
    else:
        svc    = SVC(kernel='rbf', C=10.0, gamma='scale',
                     probability=True, random_state=RANDOM_STATE)
        svc.fit(X_train_sc, y_train)
        hp_svc = svc.get_params()
    print(f"  Tiempo SVC: {time.time()-t0:.1f} s")

    cv_svc = cross_val_score(svc, X_train_sc, y_train, cv=cv_strat,
                              scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy SVC: {cv_svc.mean():.4f} ± {cv_svc.std():.4f}")

    return rfc, svc, hp_rfc, hp_svc, cv_rfc, cv_svc


def evaluar_clasificacion(rfc, svc, X_test, X_test_sc, y_test, class_names):
    """Imprime métricas completas de clasificación en consola."""
    for nombre, modelo, X in [('Random Forest', rfc, X_test), ('SVC', svc, X_test_sc)]:
        y_pred = modelo.predict(X)
        acc    = accuracy_score(y_test, y_pred)
        print(f"\n{'─'*55}")
        print(f"[CLASIFICACIÓN] {nombre}  │  Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))


# ── Regresión ──────────────────────────────────────────────────────────────────

def entrenar_regresion(X_train, X_train_sc, y_train, buscar_hp: bool = True):
    """
    Entrena RandomForestRegressor y SVR con búsqueda de hiperparámetros.

    Parámetros
    ----------
    X_train, X_train_sc : datos de entrenamiento (sin/con escalar)
    y_train             : potencia generada [W]
    buscar_hp           : si True, ejecuta RandomizedSearchCV

    Retorna
    -------
    rfr, svr            : modelos entrenados con mejores hiperparámetros
    hp_rfr, hp_svr      : diccionarios con hiperparámetros seleccionados
    cv_rfr, cv_svr      : scores de validación cruzada (R²)
    """
    cv_kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Random Forest Regressor ──
    print("\n[models] Buscando hiperparámetros para RandomForestRegressor...")
    t0 = time.time()
    if buscar_hp:
        search_rfr = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            RF_REG_GRID, n_iter=N_ITER, cv=cv_kf,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_rfr.fit(X_train, y_train)
        rfr    = search_rfr.best_estimator_
        hp_rfr = search_rfr.best_params_
        print(f"  Mejores hiperparámetros RFR: {hp_rfr}")
        print(f"  R² CV (búsqueda): {search_rfr.best_score_:.4f}")
    else:
        rfr    = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rfr.fit(X_train, y_train)
        hp_rfr = rfr.get_params()
    print(f"  Tiempo RFR: {time.time()-t0:.1f} s")

    cv_rfr = cross_val_score(rfr, X_train, y_train, cv=cv_kf, scoring='r2', n_jobs=-1)
    print(f"  CV R² RFR: {cv_rfr.mean():.4f} ± {cv_rfr.std():.4f}")

    # ── SVR ──
    print("\n[models] Buscando hiperparámetros para SVR...")
    t0 = time.time()
    if buscar_hp:
        search_svr = RandomizedSearchCV(
            SVR(),
            SVR_GRID, n_iter=N_ITER, cv=cv_kf,
            scoring='r2', random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_svr.fit(X_train_sc, y_train)
        svr    = search_svr.best_estimator_
        hp_svr = search_svr.best_params_
        print(f"  Mejores hiperparámetros SVR: {hp_svr}")
        print(f"  R² CV (búsqueda): {search_svr.best_score_:.4f}")
    else:
        svr    = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.5)
        svr.fit(X_train_sc, y_train)
        hp_svr = svr.get_params()
    print(f"  Tiempo SVR: {time.time()-t0:.1f} s")

    cv_svr = cross_val_score(svr, X_train_sc, y_train, cv=cv_kf, scoring='r2', n_jobs=-1)
    print(f"  CV R² SVR: {cv_svr.mean():.4f} ± {cv_svr.std():.4f}")

    return rfr, svr, hp_rfr, hp_svr, cv_rfr, cv_svr


def evaluar_regresion(rfr, svr, X_test, X_test_sc, y_test):
    """Imprime métricas completas de regresión en consola."""
    for nombre, modelo, X in [('Random Forest', rfr, X_test), ('SVR', svr, X_test_sc)]:
        y_pred = modelo.predict(X)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        print(f"\n{'─'*55}")
        print(f"[REGRESIÓN] {nombre}")
        print(f"  MAE : {mae:.2f} W")
        print(f"  RMSE: {rmse:.2f} W")
        print(f"  R²  : {r2:.4f}")


# ── Persistencia ───────────────────────────────────────────────────────────────

def guardar_modelos(rfc, svc, rfr, svr, carpeta: str = ARTIFACT_DIR) -> None:
    """Serializa los cuatro modelos entrenados."""
    os.makedirs(carpeta, exist_ok=True)
    joblib.dump(rfc, os.path.join(carpeta, 'rfc.pkl'))
    joblib.dump(svc, os.path.join(carpeta, 'svc.pkl'))
    joblib.dump(rfr, os.path.join(carpeta, 'rfr.pkl'))
    joblib.dump(svr, os.path.join(carpeta, 'svr.pkl'))
    print(f"\n[models] Modelos serializados en: {carpeta}")


# ── Ejecución directa ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = cargar_datos()

    (X_tr_clf, X_te_clf, X_tr_clf_sc, X_te_clf_sc,
     y_tr_clf, y_te_clf, le, scaler_clf, _) = preparar_clasificacion(df)

    rfc, svc, hp_rfc, hp_svc, cv_rfc, cv_svc = \
        entrenar_clasificacion(X_tr_clf, X_tr_clf_sc, y_tr_clf)
    evaluar_clasificacion(rfc, svc, X_te_clf, X_te_clf_sc, y_te_clf, list(le.classes_))

    (X_tr_reg, X_te_reg, X_tr_reg_sc, X_te_reg_sc,
     y_tr_reg, y_te_reg, scaler_reg, _) = preparar_regresion(df)

    rfr, svr, hp_rfr, hp_svr, cv_rfr, cv_svr = \
        entrenar_regresion(X_tr_reg, X_tr_reg_sc, y_tr_reg)
    evaluar_regresion(rfr, svr, X_te_reg, X_te_reg_sc, y_te_reg)

    guardar_modelos(rfc, svc, rfr, svr)
