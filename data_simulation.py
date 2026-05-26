# -*- coding: utf-8 -*-
"""
data_simulation.py
==================
Simulación de dataset sintético para un sistema fotovoltaico instrumentado.
Examen 2 – Comparación Random Forest vs SVM en problemas de ingeniería.

Modela cinco condiciones operativas con base en reglas físicas simplificadas:
  1. normal            – operación estándar
  2. sombreado_parcial – obstáculo parcial sobre paneles
  3. suciedad          – acumulación de polvo/suciedad
  4. falla_panel       – panel o string dañado
  5. falla_inversor    – inversor fuera de servicio

Ecuaciones de referencia (enunciado, sección 4):
  T_panel = T_amb + ((NOCT - 20) / 800) * G + ε_T
  P       = η · A · G_eff · [1 + γ · (T_panel - 25)] · f_falla + ε_P

Salida: synthetic_pv_dataset.csv
"""

import os
import numpy as np
import pandas as pd

# ── Semilla y tamaño ───────────────────────────────────────────────────────────
SEED      = 42
N_SAMPLES = 5_000

# ── Parámetros físicos del sistema ─────────────────────────────────────────────
ETA   = 0.18      # Eficiencia del panel (18 %)
AREA  = 20.0      # Área total del arreglo [m²]
NOCT  = 45.0      # Temperatura nominal de operación [°C]
GAMMA = -0.004    # Coeficiente de temperatura de potencia [1/°C]
G_MAX = 1_000.0   # Irradiancia máxima [W/m²]
V_OC  = 380.0     # Voltaje de circuito abierto [V]

# ── Distribución de clases ─────────────────────────────────────────────────────
CLASES      = ['normal', 'sombreado_parcial', 'suciedad', 'falla_panel', 'falla_inversor']
PROB_CLASES = [0.50,      0.20,               0.15,       0.10,          0.05]

# Factores de reducción de potencia por condición (rango uniforme)
FACTOR_FALLA = {
    'normal':            (0.95, 1.00),
    'sombreado_parcial': (0.40, 0.70),
    'suciedad':          (0.70, 0.90),
    'falla_panel':       (0.10, 0.40),
    'falla_inversor':    (0.00, 0.05),
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_pv_dataset.csv')

np.random.seed(SEED)


# ── Funciones de simulación ────────────────────────────────────────────────────

def _variables_ambientales(n: int):
    """
    Genera variables ambientales con distribuciones físicamente razonables.

    - hora:      uniforme [6, 20] h  (horas de luz solar)
    - dia:       uniforme [1, 365]   (día del año, estacionalidad)
    - temp_amb:  normal μ=25 °C, σ=6 °C, recortada [5, 45]
    - humedad:   uniforme [20, 90] %
    - viento:    exponencial λ=3 m/s, recortada [0, 15]
    """
    hora     = np.random.uniform(6.0, 20.0, n)
    dia      = np.random.randint(1, 366, n).astype(float)
    temp_amb = np.clip(np.random.normal(25.0, 6.0, n), 5.0, 45.0)
    humedad  = np.random.uniform(20.0, 90.0, n)
    viento   = np.clip(np.random.exponential(3.0, n), 0.0, 15.0)
    return hora, dia, temp_amb, humedad, viento


def _irradiancia(hora: np.ndarray, dia: np.ndarray, n: int) -> np.ndarray:
    """
    Modela la irradiancia solar con:
      - Perfil senoidal diario: máximo al mediodía, cero antes/después de luz
      - Variación estacional: ±10 % según posición en el año
      - Ruido gaussiano σ=20 W/m² (nubes, variabilidad atmosférica)
    """
    estacional = 1.0 + 0.1 * np.sin(2 * np.pi * (dia - 80) / 365)
    G = G_MAX * estacional * np.sin(np.pi * (hora - 6.0) / 14.0)
    G = np.clip(G + np.random.normal(0, 20.0, n), 0.0, G_MAX)
    return G


def _temperatura_panel(temp_amb: np.ndarray, G: np.ndarray, n: int) -> np.ndarray:
    """
    Ecuación NOCT del enunciado:
      T_panel = T_amb + ((NOCT - 20) / 800) * G + ε_T
    ε_T ~ N(0, 1.5 °C) – incertidumbre del sensor y gradiente no uniforme.
    """
    return temp_amb + ((NOCT - 20.0) / 800.0) * G + np.random.normal(0, 1.5, n)


def _condicion_y_factor(n: int):
    """
    Asigna condición operativa y factor de reducción de potencia.
    El factor se muestrea uniformemente dentro del rango físico de cada falla.
    """
    condiciones = np.random.choice(CLASES, size=n, p=PROB_CLASES)
    f_falla = np.array([
        np.random.uniform(*FACTOR_FALLA[c]) for c in condiciones
    ])
    return condiciones, f_falla


def _potencia(G: np.ndarray, T_panel: np.ndarray,
              f_falla: np.ndarray, n: int) -> np.ndarray:
    """
    Ecuación de potencia del enunciado:
      P = η · A · G_eff · [1 + γ · (T_panel - 25)] · f_falla + ε_P

    G_eff = G · U[0.97, 1.00]  (pérdidas ópticas menores)
    ε_P  ~ N(0, 10 W)
    """
    G_eff       = G * np.random.uniform(0.97, 1.00, n)
    factor_temp = np.clip(1.0 + GAMMA * (T_panel - 25.0), 0.5, 1.1)
    P = ETA * AREA * G_eff * factor_temp * f_falla
    return np.clip(P + np.random.normal(0, 10.0, n), 0.0, None)


def _voltaje_corriente(T_panel: np.ndarray, P: np.ndarray,
                       condiciones: np.ndarray, n: int):
    """
    Modela voltaje y corriente eléctrica de salida.

    Voltaje: cae linealmente con temperatura (~0.25 %/°C respecto a 25 °C).
    Falla de inversor: voltaje colapsa a valores cercanos a cero.
    Corriente: derivada de P = V·I, con ruido de medición.

    NOTA: voltage_V y current_A se incluyen en el dataset pero están
    EXCLUIDAS del problema de regresión para evitar predicción trivial
    mediante P ≈ V·I (ver preprocessing.py).
    """
    V = V_OC * (1.0 - 0.0025 * (T_panel - 25.0))
    mask_inv = condiciones == 'falla_inversor'
    V[mask_inv] *= np.random.uniform(0.0, 0.1, mask_inv.sum())
    V = np.clip(V + np.random.normal(0, 2.0, n), 0.0, V_OC * 1.1)
    I = np.clip(
        np.divide(P, V, out=np.zeros_like(P), where=V > 1.0)
        + np.random.normal(0, 0.05, n),
        0.0, None
    )
    return V, I


# ── Función principal ──────────────────────────────────────────────────────────

def simular_dataset(n: int = N_SAMPLES, ruta: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Genera el dataset sintético fotovoltaico y lo persiste como CSV.

    Parámetros
    ----------
    n    : número de muestras a generar
    ruta : ruta completa del archivo de salida

    Retorna
    -------
    pd.DataFrame con 11 columnas:
        hour, day_of_year, irradiance_W_m2, ambient_temp_C, panel_temp_C,
        humidity_percent, wind_speed_m_s, voltage_V, current_A,
        power_W, condition
    """
    hora, dia, temp_amb, humedad, viento = _variables_ambientales(n)
    G           = _irradiancia(hora, dia, n)
    T_panel     = _temperatura_panel(temp_amb, G, n)
    condiciones, f_falla = _condicion_y_factor(n)
    P           = _potencia(G, T_panel, f_falla, n)
    V, I        = _voltaje_corriente(T_panel, P, condiciones, n)

    df = pd.DataFrame({
        'hour':             np.round(hora, 2),
        'day_of_year':      dia.astype(int),
        'irradiance_W_m2':  np.round(G, 2),
        'ambient_temp_C':   np.round(temp_amb, 2),
        'panel_temp_C':     np.round(T_panel, 2),
        'humidity_percent': np.round(humedad, 2),
        'wind_speed_m_s':   np.round(viento, 2),
        'voltage_V':        np.round(V, 2),
        'current_A':        np.round(I, 3),
        'power_W':          np.round(P, 2),
        'condition':        condiciones,
    })

    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"[data_simulation] Dataset guardado: {ruta}  ({n} muestras)")
    print(f"[data_simulation] Distribución de clases:\n{df['condition'].value_counts().to_string()}")
    return df


if __name__ == '__main__':
    df = simular_dataset()
    print("\nPrimeras filas:")
    print(df.head().to_string())
    print("\nEstadísticas descriptivas:")
    print(df.describe().round(2).to_string())
