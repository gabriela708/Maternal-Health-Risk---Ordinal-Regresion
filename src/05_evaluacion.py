"""05 · Validación con la base de prueba y respuesta a la pregunta de investigación.

Responsable: Juan Esteban Silva.
Entrada:  data/processed/test.csv, test_full.csv y cadenas del MCMC (cadenas_*.npy)
Salidas:  results/tables/eval_metricas.csv, eval_umbrales.csv, eval_peso_predictivo.csv
          results/figures/eval_confusion_train.png, eval_confusion_train_full.png

Requiere haber corrido antes: python src/03_modelo_bayesiano.py
Uso: python src/05_evaluacion.py   (en Colab: %run src/05_evaluacion.py)
"""
#%%
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, recall_score)

from config import DATA_PROCESSED, FIGURES, L_NOMBRES, PREDICTORS, TABLES

# Cargar 03_modelo_bayesiano.py como módulo (no se puede importar directo porque empieza con número)
spec = importlib.util.spec_from_file_location(
    "modelo", Path(__file__).resolve().parent / "03_modelo_bayesiano.py"
)
modelo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modelo)

BURN_IN = modelo.BURN_IN

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


#%% CARGA DE MUESTRAS Y TEST
def cargar_muestras(nombre_dataset, cada=10):
    """Muestras posteriores sin burn-in, una de cada `cada` para que sea rápido."""
    cadenas = np.load(DATA_PROCESSED / f"cadenas_{nombre_dataset}.npy")
    muestras = cadenas[:, BURN_IN:, :].reshape(-1, cadenas.shape[-1])
    return muestras[::cada]


def cargar_test(archivo_test):
    test_df = pd.read_csv(DATA_PROCESSED / archivo_test)
    return test_df[PREDICTORS].values, test_df["RiskLevel_cd"].values.astype(int)


#%% PROBABILIDADES PREDICTIVAS
def calcular_probabilidades_ordinales(theta, X):
    alpha_1, alpha_2 = theta[0], theta[1]
    eta = X @ theta[2:]
    p_le_1 = expit(alpha_1 - eta)
    p_le_2 = expit(alpha_2 - eta)
    return np.column_stack([p_le_1, np.maximum(0, p_le_2 - p_le_1), 1.0 - p_le_2])


def probabilidades_predictivas(muestras, X):
    """Promedio de las probabilidades sobre todas las muestras de la posterior."""
    return np.mean([calcular_probabilidades_ordinales(t, X) for t in muestras], axis=0)


#%% REGLAS DE CLASIFICACIÓN
# Costos: fila = real (bajo, medio, alto), columna = predicho (bajo, medio, alto)
COSTOS = np.array([
    [0, 1, 2],   # real bajo:  falsa alarma (menos grave)
    [2, 0, 1],   # real medio
    [6, 3, 0],   # real alto:  predecir bajo es lo peor, medio le sigue
])


def clasificar_costo(probs, costos=COSTOS):
    """Clase con menor costo esperado (regla de decisión bayesiana)."""
    return np.argmin(probs @ costos, axis=1)


def clasificar_umbral(probs, umbral):
    """Alto si P(alto) >= umbral; si no, la más probable entre bajo y medio."""
    y_pred = np.argmax(probs[:, :2], axis=1)
    y_pred[probs[:, 2] >= umbral] = 2
    return y_pred

#   La regla principal es la de costo esperado: usa las tres probabilidades y
#   el orden de gravedad de los errores definido en pensamiento crítico.


#%% MÉTRICAS
def calcular_metricas(y_true, y_pred):
    return {
        "recall_alto_riesgo": recall_score(y_true == 2, y_pred == 2, zero_division=0),
        "recall_medio": recall_score(y_true == 1, y_pred == 1, zero_division=0),
        "recall_bajo": recall_score(y_true == 0, y_pred == 0, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "exactitud": np.mean(y_true == y_pred),
    }


def evaluar(nombre_modelo, archivo_test, etiqueta):
    muestras = cargar_muestras(nombre_modelo)
    X_test, y_test = cargar_test(archivo_test)

    # Predicción con la probabilidad promedio (regla de costo y regla MAP)
    probs = probabilidades_predictivas(muestras, X_test)
    y_pred = clasificar_costo(probs)
    puntual = calcular_metricas(y_test, y_pred)
    puntual_map = calcular_metricas(y_test, np.argmax(probs, axis=1))

    # Incertidumbre: métricas con cada muestra de la posterior
    por_muestra = pd.DataFrame([
        calcular_metricas(y_test, clasificar_costo(calcular_probabilidades_ordinales(t, X_test)))
        for t in muestras
    ])

    filas = [{
        "escenario": etiqueta, "metrica": m,
        "regla_costo": puntual[m],
        "IC_2.5%": np.percentile(por_muestra[m], 2.5),
        "IC_97.5%": np.percentile(por_muestra[m], 97.5),
        "regla_MAP": puntual_map[m],
    } for m in puntual]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
                           display_labels=L_NOMBRES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (regla de costo)\n{etiqueta}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"eval_confusion_{nombre_modelo}.png", dpi=150, bbox_inches="tight")
    plt.show()
    return pd.DataFrame(filas)


#%% SENSIBILIDAD: DISTINTOS UMBRALES PARA ALTO RIESGO
UMBRALES_PRUEBA = [0.20, 0.30, 0.35, 0.50]


def evaluar_umbrales(nombre_modelo, archivo_test, etiqueta):
    muestras = cargar_muestras(nombre_modelo)
    X_test, y_test = cargar_test(archivo_test)
    probs = probabilidades_predictivas(muestras, X_test)

    filas = []
    for umbral in UMBRALES_PRUEBA:
        m = calcular_metricas(y_test, clasificar_umbral(probs, umbral))
        filas.append({"escenario": etiqueta, "regla": f"umbral {umbral:.2f}", **m})
    filas.append({"escenario": etiqueta, "regla": "costo esperado",
                  **calcular_metricas(y_test, clasificar_costo(probs))})
    return pd.DataFrame(filas)

#   Es un análisis de sensibilidad: la regla se elige por el orden de gravedad,
#   no por el umbral que mejor resulte en test (eso sería fuga de información).


#%% PESO PREDICTIVO DE CADA VARIABLE
def peso_predictivo(nombre_modelo, etiqueta):
    betas = cargar_muestras(nombre_modelo)[:, 2:]
    return pd.DataFrame({
        "escenario": etiqueta,
        "Variable": PREDICTORS,
        "Media_beta": betas.mean(axis=0),
        "IC_2.5%": np.percentile(betas, 2.5, axis=0),
        "IC_97.5%": np.percentile(betas, 97.5, axis=0),
        "OR_por_DE": np.exp(betas.mean(axis=0)),   # cambio en odds de subir de nivel por +1 DE
        "P(beta>0)": (betas > 0).mean(axis=0),
    }).sort_values("Media_beta", ascending=False)


#%% EJECUCIÓN PARA LAS DOS BASES
if __name__ == "__main__":
    escenarios = [
        ("train", "test.csv", "Sin duplicados"),
        ("train_full", "test_full.csv", "Con duplicados"),
    ]

    metricas = pd.concat([evaluar(*e) for e in escenarios], ignore_index=True)
    metricas.to_csv(TABLES / "eval_metricas.csv", index=False)
    print("\n--- Métricas en test (regla de costo, IC 95 % y regla MAP) ---")
    print(metricas.round(3).to_string(index=False))

    umbrales = pd.concat([evaluar_umbrales(*e) for e in escenarios], ignore_index=True)
    umbrales.to_csv(TABLES / "eval_umbrales.csv", index=False)
    print("\n--- Sensibilidad a la regla de clasificación ---")
    print(umbrales.round(3).to_string(index=False))

    pesos = pd.concat([peso_predictivo(m, et) for m, _, et in escenarios], ignore_index=True)
    pesos.to_csv(TABLES / "eval_peso_predictivo.csv", index=False)
    print("\n--- Peso predictivo de las variables ---")
    print(pesos.round(3).to_string(index=False))
