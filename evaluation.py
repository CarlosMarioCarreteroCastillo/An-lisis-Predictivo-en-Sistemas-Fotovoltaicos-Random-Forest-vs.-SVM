# -*- coding: utf-8 -*-
"""
evaluation.py
=============
Módulo de evaluación y visualización para el sistema fotovoltaico.
Examen 2 – Comparación Random Forest vs SVM.

Genera las siguientes figuras en /resultados/:
  1. distribucion_clases.png          – balance de clases del dataset
  2. confusion_matrices.png           – matrices de confusión normalizadas
  3. clasificacion_f1_comparativo.png – F1-score por clase (RF vs SVC)
  4. feature_importance_clf.png       – importancia de variables (RFC)
  5. regresion_comparativo.png        – scatter real vs predicho + residuos
  6. regresion_metricas.png           – barras comparativas MAE/RMSE/R²
  7. feature_importance_reg.png       – importancia de variables (RFR)
  8. validacion_cruzada.png           – distribución de scores CV por modelo
  9. sensibilidad_c.png               – análisis de sensibilidad al hiperparámetro C
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.svm              import SVC, SVR
from sklearn.model_selection  import cross_val_score, StratifiedKFold, KFold

# ── Configuración visual ───────────────────────────────────────────────────────
PALETTE    = {'rf': '#2E86AB', 'svm': '#E84855'}
COLOR_LIST = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#9B59B6']
FIG_DPI    = 150
STYLE      = 'seaborn-v0_8-whitegrid'
_DIR       = os.path.dirname(__file__)
RESULT_DIR = os.path.join(_DIR, 'resultados')

RANDOM_STATE = 42


def _save(fig, nombre: str) -> str:
    """Guarda figura y cierra para liberar memoria."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    ruta = os.path.join(RESULT_DIR, nombre)
    fig.savefig(ruta, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [evaluation] Guardada: {nombre}")
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUCIÓN DE CLASES
# ══════════════════════════════════════════════════════════════════════════════

def graficar_distribucion_clases(df, target_col: str = 'condition') -> str:
    """
    Visualiza el balance/desbalance de clases en el dataset.
    Relevante para interpretar correctamente el F1-score de clases minoritarias.
    """
    counts = df[target_col].value_counts()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(counts.index, counts.values, color=COLOR_LIST[:len(counts)],
                      alpha=0.87, edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, counts.values):
            pct = v / counts.sum() * 100
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                    f'{v}\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('Número de muestras', fontsize=11)
        ax.set_xlabel('Condición operativa', fontsize=11)
        ax.set_title('Distribución de Clases — Dataset Fotovoltaico Sintético',
                     fontweight='bold', fontsize=13)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.tick_params(axis='x', labelsize=9)
        fig.tight_layout()
    return _save(fig, 'distribucion_clases.png')


# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS DE CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def metricas_clasificacion(rfc, svc, X_test, X_test_sc, y_test, class_names: list) -> dict:
    """Calcula y retorna métricas de clasificación para RFC y SVC."""
    resultados = {}
    for nombre, modelo, X in [('Random Forest', rfc, X_test), ('SVC', svc, X_test_sc)]:
        y_pred = modelo.predict(X)
        resultados[nombre] = {
            'y_pred':   y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'report':   classification_report(y_test, y_pred,
                            target_names=class_names, output_dict=True),
            'cm':       confusion_matrix(y_test, y_pred),
        }
    return resultados


def graficar_confusion(resultados_clf: dict, class_names: list) -> str:
    """Matrices de confusión normalizadas por fila (proporción real)."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Matrices de Confusión — Clasificación de Condición Operativa',
                     fontsize=13, fontweight='bold')
        for ax, (nombre, res) in zip(axes, resultados_clf.items()):
            cm   = res['cm']
            cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            im   = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
            color = PALETTE['rf'] if 'Forest' in nombre else PALETTE['svm']
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel('Predicho', fontsize=10)
            ax.set_ylabel('Real', fontsize=10)
            ax.set_title(f'{nombre}\nAccuracy: {res["accuracy"]:.4f}',
                         fontsize=11, color=color, fontweight='bold')
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                            fontsize=8, color='white' if cm_n[i,j] > 0.5 else 'black')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label='Proporción')
        fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.18, wspace=0.35)
    return _save(fig, 'confusion_matrices.png')


def graficar_metricas_clasificacion(resultados_clf: dict, class_names: list) -> str:
    """Barras comparativas de Precision, Recall y F1-score por clase."""
    metricas_plot = ['precision', 'recall', 'f1-score']
    labels_plot   = ['Precision', 'Recall', 'F1-score']
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Métricas por Clase — Random Forest vs SVC', fontsize=13, fontweight='bold')
        x = np.arange(len(class_names))
        w = 0.35
        for ax, (met, lbl) in zip(axes, zip(metricas_plot, labels_plot)):
            for i, (nombre, res) in enumerate(resultados_clf.items()):
                vals  = [res['report'][c][met] for c in class_names]
                color = PALETTE['rf'] if 'Forest' in nombre else PALETTE['svm']
                bars  = ax.bar(x + i*w - w/2, vals, w, label=nombre,
                               color=color, alpha=0.85, edgecolor='white')
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                            f'{v:.2f}', ha='center', va='bottom', fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(lbl, fontsize=10)
            ax.set_ylim(0, 1.18)
            ax.set_title(lbl, fontweight='bold')
            ax.legend(fontsize=8)
        fig.tight_layout()
    return _save(fig, 'clasificacion_metricas_comparativo.png')


# ══════════════════════════════════════════════════════════════════════════════
# 3. IMPORTANCIA DE VARIABLES — CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def graficar_importancia_clf(rfc, feature_names: list) -> str:
    """
    Importancia de variables del RandomForestClassifier (Gini impurity decrease).
    Permite responder: '¿Qué variables son más informativas para identificar
    la condición operativa?'
    """
    importancias = rfc.feature_importances_
    std          = np.std([t.feature_importances_ for t in rfc.estimators_], axis=0)
    orden        = np.argsort(importancias)[::-1]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(feature_names)),
                      importancias[orden], yerr=std[orden],
                      color=PALETTE['rf'], alpha=0.85, capsize=4,
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in orden],
                            rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Importancia media (Gini)', fontsize=11)
        ax.set_title('Importancia de Variables — RFC (Clasificación)',
                     fontweight='bold', fontsize=13)
        for b, v in zip(bars, importancias[orden]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + std[orden][list(importancias[orden]).index(v)] + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, importancias.max() * 1.3)
        fig.tight_layout()
    return _save(fig, 'feature_importance_clf.png')


# ══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS DE REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def metricas_regresion(rfr, svr, X_test, X_test_sc, y_test) -> dict:
    """Calcula y retorna métricas de regresión para RFR y SVR."""
    resultados = {}
    for nombre, modelo, X in [('Random Forest', rfr, X_test), ('SVR', svr, X_test_sc)]:
        y_pred = modelo.predict(X)
        resultados[nombre] = {
            'y_pred': y_pred,
            'MAE':    mean_absolute_error(y_test, y_pred),
            'RMSE':   np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2':     r2_score(y_test, y_pred),
        }
    return resultados


def graficar_regresion(resultados_reg: dict, y_test: np.ndarray) -> str:
    """
    Gráfica de 4 paneles: scatter real vs predicho y gráfica de residuos
    para RFR y SVR, con anotación de métricas.
    """
    with plt.style.context(STYLE):
        fig  = plt.figure(figsize=(14, 10))
        gs   = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
        fig.suptitle('Evaluación de Regresión — Random Forest vs SVR',
                     fontsize=13, fontweight='bold')

        for col, (nombre, res) in enumerate(resultados_reg.items()):
            color  = PALETTE['rf'] if 'Forest' in nombre else PALETTE['svm']
            y_pred = res['y_pred']
            lims   = [min(y_test.min(), y_pred.min()) - 50,
                      max(y_test.max(), y_pred.max()) + 50]

            # Scatter real vs predicho
            ax0 = fig.add_subplot(gs[0, col])
            ax0.scatter(y_test, y_pred, alpha=0.20, s=7, color=color, rasterized=True)
            ax0.plot(lims, lims, 'k--', lw=1, label='Ideal')
            ax0.set_xlabel('Real [W]', fontsize=10)
            ax0.set_ylabel('Predicho [W]', fontsize=10)
            ax0.set_title(
                f'{nombre}\nMAE={res["MAE"]:.1f} W  |  RMSE={res["RMSE"]:.1f} W  |  R²={res["R2"]:.4f}',
                fontsize=10, color=color, fontweight='bold'
            )
            ax0.legend(fontsize=8)

            # Residuos vs predicho
            ax1 = fig.add_subplot(gs[1, col])
            residuos = y_test - y_pred
            ax1.scatter(y_pred, residuos, alpha=0.20, s=7, color=color, rasterized=True)
            ax1.axhline(0, color='black', lw=1.2, ls='--')
            ax1.set_xlabel('Predicho [W]', fontsize=10)
            ax1.set_ylabel('Residuo [W]', fontsize=10)
            ax1.set_title(f'Residuos — {nombre}', fontsize=10)
    return _save(fig, 'regresion_comparativo.png')


def graficar_metricas_regresion(resultados_reg: dict) -> str:
    """Barras comparativas de MAE, RMSE y R² para ambos modelos de regresión."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('Métricas de Regresión Comparativas — RF vs SVR',
                     fontsize=13, fontweight='bold')
        nombres = list(resultados_reg.keys())
        colors  = [PALETTE['rf'], PALETTE['svm']]
        metricas = [('MAE', 'MAE [W]'), ('RMSE', 'RMSE [W]'), ('R2', 'R²')]
        for ax, (met, label) in zip(axes, metricas):
            vals = [resultados_reg[n][met] for n in nombres]
            bars = ax.bar(nombres, vals, color=colors, alpha=0.85,
                          width=0.5, edgecolor='white')
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.03,
                        f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
            ax.set_title(label, fontweight='bold', fontsize=11)
            ax.set_ylim(0, max(vals) * 1.25)
            ax.tick_params(axis='x', labelsize=9)
        fig.tight_layout()
    return _save(fig, 'regresion_metricas.png')


# ══════════════════════════════════════════════════════════════════════════════
# 5. IMPORTANCIA DE VARIABLES — REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def graficar_importancia_reg(rfr, feature_names: list) -> str:
    """
    Importancia de variables del RandomForestRegressor.
    Permite responder: '¿Qué variables ambientales predicen mejor la potencia?'
    """
    importancias = rfr.feature_importances_
    std          = np.std([t.feature_importances_ for t in rfr.estimators_], axis=0)
    orden        = np.argsort(importancias)[::-1]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(range(len(feature_names)),
               importancias[orden], yerr=std[orden],
               color=PALETTE['rf'], alpha=0.85, capsize=4,
               edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in orden],
                            rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Importancia media (varianza reducida)', fontsize=11)
        ax.set_title('Importancia de Variables — RFR (Regresión)',
                     fontweight='bold', fontsize=13)
        for i, (idx, v) in enumerate(zip(orden, importancias[orden])):
            ax.text(i, v + std[orden][i] + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, importancias.max() * 1.35)
        fig.tight_layout()
    return _save(fig, 'feature_importance_reg.png')


# ══════════════════════════════════════════════════════════════════════════════
# 6. VALIDACIÓN CRUZADA — DISTRIBUCIÓN DE SCORES
# ══════════════════════════════════════════════════════════════════════════════

def graficar_validacion_cruzada(cv_rfc, cv_svc, cv_rfr, cv_svr) -> str:
    """
    Visualiza la distribución de scores de validación cruzada (5 folds)
    para los cuatro modelos. Permite evaluar estabilidad y varianza.
    """
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Validación Cruzada (5-Fold) — Distribución de Scores',
                     fontsize=13, fontweight='bold')

        # Clasificación
        ax = axes[0]
        data_clf = [cv_rfc, cv_svc]
        bp = ax.boxplot(data_clf, patch_artist=True, widths=0.4,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], [PALETTE['rf'], PALETTE['svm']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for i, (scores, color) in enumerate(zip(data_clf, [PALETTE['rf'], PALETTE['svm']]), 1):
            ax.scatter([i] * len(scores), scores, color=color, zorder=5, s=40, alpha=0.9)
            ax.text(i, scores.mean(), f'μ={scores.mean():.4f}\nσ={scores.std():.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Random Forest', 'SVC'], fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Clasificación', fontweight='bold', fontsize=11)
        ax.set_ylim(max(0, min(cv_rfc.min(), cv_svc.min()) - 0.05), 1.05)

        # Regresión
        ax = axes[1]
        data_reg = [cv_rfr, cv_svr]
        bp = ax.boxplot(data_reg, patch_artist=True, widths=0.4,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], [PALETTE['rf'], PALETTE['svm']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for i, (scores, color) in enumerate(zip(data_reg, [PALETTE['rf'], PALETTE['svm']]), 1):
            ax.scatter([i] * len(scores), scores, color=color, zorder=5, s=40, alpha=0.9)
            ax.text(i, scores.mean(), f'μ={scores.mean():.4f}\nσ={scores.std():.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Random Forest', 'SVR'], fontsize=10)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title('Regresión', fontweight='bold', fontsize=11)
        ax.set_ylim(max(0, min(cv_rfr.min(), cv_svr.min()) - 0.05), 1.05)

        legend_elements = [
            Line2D([0], [0], color=PALETTE['rf'],  lw=8, alpha=0.75, label='Random Forest'),
            Line2D([0], [0], color=PALETTE['svm'], lw=8, alpha=0.75, label='SVM / SVR'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, 'validacion_cruzada.png')


# ══════════════════════════════════════════════════════════════════════════════
# 7. ANÁLISIS DE SENSIBILIDAD AL HIPERPARÁMETRO C
# ══════════════════════════════════════════════════════════════════════════════

def graficar_sensibilidad_C(X_train_clf_sc, y_train_clf,
                             X_train_reg_sc, y_train_reg) -> str:
    """
    Analiza cómo varía el desempeño de SVC y SVR al cambiar el parámetro C.
    C controla el balance entre margen máximo y errores de clasificación/regresión:
      - C pequeño → mayor regularización, margen más amplio, más errores permitidos.
      - C grande  → menor regularización, ajuste más cercano a los datos de entrenamiento.
    """
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    cv_clf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_reg   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores_svc, scores_svr = [], []
    print("\n[evaluation] Análisis de sensibilidad al parámetro C...")
    for C in C_values:
        sc = cross_val_score(
            SVC(kernel='rbf', C=C, gamma='scale', random_state=RANDOM_STATE),
            X_train_clf_sc, y_train_clf, cv=cv_clf, scoring='accuracy', n_jobs=-1
        ).mean()
        sr = cross_val_score(
            SVR(kernel='rbf', C=C, gamma='scale'),
            X_train_reg_sc, y_train_reg, cv=cv_reg, scoring='r2', n_jobs=-1
        ).mean()
        scores_svc.append(sc)
        scores_svr.append(sr)
        print(f"  C={C:<8} → SVC Acc={sc:.4f} | SVR R²={sr:.4f}")

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Sensibilidad al Hiperparámetro C — SVM y SVR',
                     fontsize=13, fontweight='bold')

        for ax, scores, ylabel, titulo, color in zip(
            axes,
            [scores_svc, scores_svr],
            ['Accuracy (CV=5)', 'R² (CV=5)'],
            ['SVC — Clasificación', 'SVR — Regresión'],
            [PALETTE['svm'], PALETTE['svm']],
        ):
            ax.semilogx(C_values, scores, 'o-', color=color,
                        linewidth=2, markersize=7, markerfacecolor='white',
                        markeredgewidth=2)
            for cv, sc in zip(C_values, scores):
                ax.annotate(f'{sc:.4f}', (cv, sc),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=8)
            best_c = C_values[np.argmax(scores)]
            ax.axvline(best_c, color='gray', lw=1, ls='--', alpha=0.7)
            ax.set_xlabel('C (escala logarítmica)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(titulo, fontweight='bold', fontsize=11)
            ax.set_ylim(max(0, min(scores) - 0.05), min(1.0, max(scores) + 0.08))

        fig.tight_layout()
    return _save(fig, 'sensibilidad_C.png')


# ══════════════════════════════════════════════════════════════════════════════
# 8. EFECTO DE NORMALIZACIÓN EN SVM
# ══════════════════════════════════════════════════════════════════════════════

def graficar_efecto_normalizacion(rfc, svc,
                                   X_train_clf, X_test_clf,
                                   X_train_clf_sc, X_test_clf_sc,
                                   y_train_clf, y_test_clf,
                                   rfr, svr,
                                   X_train_reg, X_test_reg,
                                   X_train_reg_sc, X_test_reg_sc,
                                   y_train_reg, y_test_reg) -> str:
    """
    Compara el desempeño de SVC y SVR entrenados con y sin escalado.
    Demuestra por qué el escalado es obligatorio para SVM/SVR:
    las SVM son sensibles a la magnitud de las variables porque el kernel RBF
    calcula distancias en el espacio de características.
    """
    from sklearn.svm import SVC as _SVC, SVR as _SVR
    from sklearn.metrics import accuracy_score, r2_score

    # SVC sin escalado (usar los mejores parámetros ya conocidos del modelo entrenado)
    best_C_clf   = svc.C if hasattr(svc, 'C') else 10.0
    best_C_reg   = svr.C if hasattr(svr, 'C') else 10.0
    best_eps_reg = svr.epsilon if hasattr(svr, 'epsilon') else 0.5

    svc_noscale = _SVC(kernel='rbf', C=best_C_clf, gamma='scale',
                       random_state=RANDOM_STATE)
    svc_noscale.fit(X_train_clf, y_train_clf)
    acc_no  = accuracy_score(y_test_clf, svc_noscale.predict(X_test_clf))
    acc_sc  = accuracy_score(y_test_clf, svc.predict(X_test_clf_sc))

    svr_noscale = _SVR(kernel='rbf', C=best_C_reg, gamma='scale',
                       epsilon=best_eps_reg)
    svr_noscale.fit(X_train_reg, y_train_reg)
    r2_no  = r2_score(y_test_reg, svr_noscale.predict(X_test_reg))
    r2_sc  = r2_score(y_test_reg, svr.predict(X_test_reg_sc))

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Efecto de la Normalización en SVM / SVR',
                     fontsize=13, fontweight='bold')

        for ax, (sin_sc, con_sc), ylabel, titulo in zip(
            axes,
            [(acc_no, acc_sc), (r2_no, r2_sc)],
            ['Accuracy', 'R²'],
            ['SVC — Clasificación', 'SVR — Regresión'],
        ):
            cats   = ['Sin escalado', 'Con StandardScaler']
            vals   = [sin_sc, con_sc]
            colors = ['#AAAAAA', PALETTE['svm']]
            bars   = ax.bar(cats, vals, color=colors, alpha=0.85,
                            width=0.45, edgecolor='white')
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                        f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(titulo, fontweight='bold', fontsize=11)
            ax.set_ylim(0, max(vals) * 1.25)

        fig.tight_layout()
    return _save(fig, 'efecto_normalizacion.png')


# ══════════════════════════════════════════════════════════════════════════════
# 9. RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resumen(resultados_clf: dict, resultados_reg: dict,
                      cv_rfc, cv_svc, cv_rfr, cv_svr,
                      hp_rfc: dict, hp_svc: dict,
                      hp_rfr: dict, hp_svr: dict) -> None:
    """Imprime un resumen ejecutivo completo en consola."""
    sep = '═' * 60
    print(f'\n{sep}')
    print('RESUMEN EJECUTIVO — EXAMEN 2 IA')
    print(sep)

    print('\n── CLASIFICACIÓN DE CONDICIÓN OPERATIVA ──────────────────')
    for nombre, res in resultados_clf.items():
        print(f'  {nombre:<22} Accuracy (test): {res["accuracy"]:.4f}')
    print(f'\n  CV-5 RFC: {cv_rfc.mean():.4f} ± {cv_rfc.std():.4f}')
    print(f'  CV-5 SVC: {cv_svc.mean():.4f} ± {cv_svc.std():.4f}')
    print(f'\n  Mejores HP — RFC: {hp_rfc}')
    print(f'  Mejores HP — SVC: {hp_svc}')

    print('\n── REGRESIÓN DE POTENCIA GENERADA ────────────────────────')
    for nombre, res in resultados_reg.items():
        print(f'  {nombre:<22} MAE={res["MAE"]:.1f} W  '
              f'RMSE={res["RMSE"]:.1f} W  R²={res["R2"]:.4f}')
    print(f'\n  CV-5 RFR (R²): {cv_rfr.mean():.4f} ± {cv_rfr.std():.4f}')
    print(f'  CV-5 SVR (R²): {cv_svr.mean():.4f} ± {cv_svr.std():.4f}')
    print(f'\n  Mejores HP — RFR: {hp_rfr}')
    print(f'  Mejores HP — SVR: {hp_svr}')
    print(f'\n{sep}\n')
