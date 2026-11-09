"""04 · Diagnóstico de convergencia del MCMC.

Responsable: Juan Esteban Silva.
Entrada:  cadenas guardadas por 03_modelo_bayesiano.py (data/processed/cadenas_*.npy)
Salidas:  figuras en results/figures/ y tablas de diagnóstico en results/tables/

Requiere haber corrido antes: python src/03_modelo_bayesiano.py
Uso: python src/04_convergencia.py
"""
#%%
import importlib.util
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

from config import DATA_PROCESSED, FIGURES, PREDICTORS, TABLES

spec = importlib.util.spec_from_file_location(
    "modelo", Path(__file__).resolve().parent / "03_modelo_bayesiano.py"
)
modelo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modelo)

BURN_IN = modelo.BURN_IN
PARAM_NAMES = modelo.PARAM_NAMES
CUTPOINT_PRIOR_SD = 0.5  # valor por defecto de cutpoint_prior_sd en log_prior

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


#%% CARGA DE CADENAS
def cargar_idata(nombre_dataset):
    """Carga las cadenas, quita el burn-in y las convierte a InferenceData."""
    cadenas = np.load(DATA_PROCESSED / f"cadenas_{nombre_dataset}.npy")
    posterior = {p: cadenas[:, BURN_IN:, j] for j, p in enumerate(PARAM_NAMES)}
    return cadenas, az.from_dict(posterior=posterior)


#%% 1. TRAZAS
def graficar_trazas(idata, nombre_dataset):
    az.plot_trace(idata, var_names=PARAM_NAMES, compact=False, legend=True)
    plt.suptitle(f"Trazas y densidades posteriores ({nombre_dataset})", y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURES / f"conv_trazas_{nombre_dataset}.png", dpi=150, bbox_inches="tight")
    plt.show()

#   Se busca que las cadenas se superpongan y oscilen alrededor del mismo valor,
#   sin tendencias ni tramos donde una cadena se quede pegada.


#%% 2. AUTOCORRELACIÓN
def graficar_autocorrelacion(idata, nombre_dataset):
    az.plot_autocorr(idata, var_names=PARAM_NAMES, combined=True, max_lag=100)
    plt.suptitle(f"Autocorrelación de las cadenas ({nombre_dataset})", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / f"conv_autocorrelacion_{nombre_dataset}.png", dpi=150, bbox_inches="tight")
    plt.show()

#   La autocorrelación debe caer hacia cero a medida que aumenta el rezago; si cae
#   lento, las muestras son muy dependientes y el ESS es bajo.


#%% 3. MEDIA ACUMULADA (incluye el burn-in para justificarlo)
def graficar_media_acumulada(cadenas, nombre_dataset):
    n_iter = cadenas.shape[1]
    pasos = np.arange(1, n_iter + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for j, (ax, p) in enumerate(zip(axes.ravel(), PARAM_NAMES)):
        for c in range(cadenas.shape[0]):
            ax.plot(np.cumsum(cadenas[c, :, j]) / pasos, lw=1, label=f"Cadena {c + 1}")
        ax.axvline(BURN_IN, color="gray", ls="--", lw=1)
        ax.set_title(p)
        ax.set_xlabel("Iteración")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Media acumulada por cadena ({nombre_dataset}); línea gris = fin del burn-in")
    fig.tight_layout()
    fig.savefig(FIGURES / f"conv_media_acumulada_{nombre_dataset}.png", dpi=150, bbox_inches="tight")
    plt.show()

#   Después del burn-in la media de cada cadena debe estabilizarse en el mismo valor.


#%% 4. PRIOR VS. POSTERIOR
def graficar_prior_posterior(cadenas, nombre_dataset):
    train = pd.read_csv(DATA_PROCESSED / f"{nombre_dataset}.csv")
    escalas = pd.read_csv(DATA_PROCESSED / f"parametros_escalado_{nombre_dataset}.csv")
    medias_b, sds_b, positivo = modelo.transformed_coefficient_priors(
        escalas["DesviacionEstandar"].values
    )
    centros_a = modelo.cutpoint_prior_from_training_labels(train["RiskLevel_cd"].values)

    medias = np.r_[centros_a, medias_b]
    sds = np.r_[[CUTPOINT_PRIOR_SD] * 2, sds_b]
    truncada = np.r_[[False, False], positivo]

    muestras = cadenas[:, BURN_IN:, :].reshape(-1, cadenas.shape[-1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for j, (ax, p) in enumerate(zip(axes.ravel(), PARAM_NAMES)):
        post = muestras[:, j]
        ax.hist(post, bins=50, density=True, alpha=0.6, color="#2F79F0", label="Posterior")

        lo = min(post.min(), medias[j] - 3 * sds[j])
        hi = max(post.max(), medias[j] + 3 * sds[j])
        if truncada[j]:
            lo = 0.0
            a = (0 - medias[j]) / sds[j]
            x = np.linspace(lo, hi, 400)
            dens = truncnorm.pdf(x, a, np.inf, loc=medias[j], scale=sds[j])
        else:
            x = np.linspace(lo, hi, 400)
            dens = norm.pdf(x, medias[j], sds[j])
        ax.plot(x, dens, color="#E8743B", lw=2, label="Prior")
        ax.set_title(p)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Prior vs. posterior ({nombre_dataset})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"conv_prior_posterior_{nombre_dataset}.png", dpi=150, bbox_inches="tight")
    plt.show()

#   Si la posterior es mucho más angosta que la prior, los datos dominan; si casi
#   coinciden, la prior está determinando el resultado (revisar con Alejandra).


#%% 5. DIAGNÓSTICOS NUMÉRICOS: R-hat y ESS
def tabla_diagnosticos(idata, nombre_dataset):
    resumen = az.summary(idata, var_names=PARAM_NAMES)
    resumen.to_csv(TABLES / f"conv_diagnosticos_{nombre_dataset}.csv")
    print(f"\n--- Diagnósticos de convergencia ({nombre_dataset}) ---")
    print(resumen[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])

    ok_rhat = (resumen["r_hat"] <= 1.01).all()
    ok_ess = (resumen["ess_bulk"] >= 400).all() and (resumen["ess_tail"] >= 400).all()
    print(f"R-hat <= 1.01 en todos los parámetros: {ok_rhat}")
    print(f"ESS (bulk y tail) >= 400 en todos los parámetros: {ok_ess}")
    return resumen

#   Criterios: R-hat <= 1.01 (las cadenas llegan a la misma distribución) y
#   ESS >= 400 (suficientes muestras efectivas para resumir la posterior).


#%% EJECUCIÓN PARA LAS DOS BASES
if __name__ == "__main__":
    for nombre in ["train", "train_full"]:
        cadenas, idata = cargar_idata(nombre)
        graficar_trazas(idata, nombre)
        graficar_autocorrelacion(idata, nombre)
        graficar_media_acumulada(cadenas, nombre)
        graficar_prior_posterior(cadenas, nombre)
        tabla_diagnosticos(idata, nombre)
