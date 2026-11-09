"""03b · Figura y tabla de las priors con los hiperparámetros definitivos.

Responsable: Persona B (Alejandra).
Entrada:  data/processed/train.csv (etiquetas RiskLevel_cd) y
          data/processed/parametros_escalado_train.csv (desviaciones de train).
Salidas:  results/figures/priors_definitivas_<base>.png
          results/tables/priors_hiperparametros_<base>.csv

Los hiperparámetros se obtienen con las mismas funciones que usa la
log-posterior (transformed_coefficient_priors y
cutpoint_prior_from_training_labels), de modo que la figura coincide con lo
que realmente muestrea el MCMC. La base principal es la de train sin
duplicados; train_full se genera solo para el análisis de sensibilidad.

Uso: python src/03b_figura_priors.py
"""
import importlib.util
import inspect
from pathlib import Path

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

# Desviación de los cutpoints: se toma del valor por defecto de log_prior.
CUTPOINT_PRIOR_SD = inspect.signature(modelo.log_prior).parameters[
    "cutpoint_prior_sd"
].default

ETIQUETAS = {
    "alpha_1": r"$\alpha_1$ (bajo | medio)",
    "alpha_2": r"$\alpha_2$ (medio | alto)",
    "Age": r"$\beta_{\mathrm{Age}}$",
    "SystolicBP": r"$\beta_{\mathrm{SystolicBP}}$",
    "DiastolicBP": r"$\beta_{\mathrm{DiastolicBP}}$",
    "BS": r"$\beta_{\mathrm{BS}}$",
    "BodyTemp": r"$\beta_{\mathrm{BodyTemp}}$",
    "HeartRate": r"$\beta_{\mathrm{HeartRate}}$",
}
ORIGEN = {
    "alpha_1": "Bayes empírico (proporciones de train)",
    "alpha_2": "Bayes empírico (proporciones de train)",
    "Age": "Literatura: N(0,03; 0,02²) por año × s",
    "SystolicBP": "Literatura: N⁺(0,04; 0,02²) por mmHg × s",
    "DiastolicBP": "Literatura: N⁺(0,03; 0,02²) por mmHg × s",
    "BS": "Débil, signo positivo (guía clínica)",
    "BodyTemp": "Débil: N(0; 0,5²) por °F × s",
    "HeartRate": "Débil, centrada en cero",
}


def hiperparametros(nombre_dataset):
    """Tabla con la prior de cada parámetro en la escala del modelo."""
    train = pd.read_csv(DATA_PROCESSED / f"{nombre_dataset}.csv")
    escalas = pd.read_csv(
        DATA_PROCESSED / f"parametros_escalado_{nombre_dataset}.csv"
    ).set_index("Variable").loc[PREDICTORS, "DesviacionEstandar"]

    y = train["RiskLevel_cd"].to_numpy()
    centros = modelo.cutpoint_prior_from_training_labels(y)
    medias_b, sds_b, positivo = modelo.transformed_coefficient_priors(
        escalas.to_numpy()
    )

    filas = []
    for k, m in enumerate(centros, start=1):
        filas.append(dict(Parametro=f"alpha_{k}", Distribucion="Normal",
                          Media=m, DE=CUTPOINT_PRIOR_SD, s_train=np.nan))
    for j, var in enumerate(PREDICTORS):
        filas.append(dict(Parametro=var,
                          Distribucion="Normal truncada (>0)" if positivo[j] else "Normal",
                          Media=medias_b[j], DE=sds_b[j], s_train=escalas[var]))
    tabla = pd.DataFrame(filas)

    # Media e intervalo central del 95 % de la distribución efectiva
    # (para las truncadas, de la Normal restringida a valores positivos).
    resumen = []
    for _, r in tabla.iterrows():
        dist = _distribucion(r)
        resumen.append((dist.mean(), *dist.ppf([0.025, 0.975])))
    tabla[["Media_efectiva", "IC95_inf", "IC95_sup"]] = resumen
    tabla["Origen"] = tabla["Parametro"].map(ORIGEN)
    conteos = np.bincount(y, minlength=3)
    return tabla, conteos


def _distribucion(fila):
    if fila["Distribucion"].startswith("Normal truncada"):
        a = (0 - fila["Media"]) / fila["DE"]
        return truncnorm(a, np.inf, loc=fila["Media"], scale=fila["DE"])
    return norm(fila["Media"], fila["DE"])


def _texto_prior(fila):
    simbolo = r"\mathcal{N}^{+}" if fila["Distribucion"].startswith("Normal truncada") else r"\mathcal{N}"
    return rf"${simbolo}({fila['Media']:.2f},\ {fila['DE']:.2f}^2)$"


def graficar_priors(tabla, conteos, nombre_dataset):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (_, fila) in zip(axes.ravel(), tabla.iterrows()):
        dist = _distribucion(fila)
        truncada = fila["Distribucion"].startswith("Normal truncada")
        lo = 0.0 if truncada else fila["Media"] - 4 * fila["DE"]
        hi = fila["Media"] + 4 * fila["DE"]
        x = np.linspace(lo, hi, 500)
        y = dist.pdf(x)

        color = "#7A5195" if fila["Parametro"].startswith("alpha") else "#2F79F0"
        ax.plot(x, y, color=color, lw=2)
        dentro = (x >= fila["IC95_inf"]) & (x <= fila["IC95_sup"])
        ax.fill_between(x[dentro], y[dentro], color=color, alpha=0.18,
                        label="IC 95 %")
        ax.axvline(fila["Media_efectiva"], color=color, ls="--", lw=1)
        if not fila["Parametro"].startswith("alpha"):
            ax.axvline(0, color="0.4", lw=0.8)

        ax.set_title(f"{ETIQUETAS[fila['Parametro']]}\n{_texto_prior(fila)}",
                     fontsize=11)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.text(0.98, 0.97,
                f"media = {fila['Media_efectiva']:.2f}\n"
                f"IC95 = [{fila['IC95_inf']:.2f}, {fila['IC95_sup']:.2f}]",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                color="0.25", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    for ax in axes[0, :2]:
        ax.set_xlabel("Escala logit (umbral)", fontsize=9)
    for ax in list(axes[0, 2:]) + list(axes[1, :]):
        ax.set_xlabel("Cambio en log-odds por 1 DE", fontsize=9)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8.5)

    n = conteos.sum()
    fig.suptitle(
        f"Priors del modelo ordinal en escala estandarizada — {nombre_dataset} "
        f"(n = {n}; bajo/medio/alto = {conteos[0]}/{conteos[1]}/{conteos[2]})",
        fontsize=12,
    )
    fig.text(0.5, 0.005,
             "N⁺(μ, σ²): Normal truncada a valores positivos (μ y σ antes de truncar; "
             "por eso su media es mayor que μ). Los umbrales se restringen además a "
             "α₁ < α₂. Línea discontinua: media de la prior. Sombreado: intervalo "
             "central del 95 %.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    salida = FIGURES / f"priors_definitivas_{nombre_dataset}.png"
    fig.savefig(salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return salida


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    # Base principal (sin duplicados) y base completa para la sensibilidad.
    for nombre in ("train", "train_full"):
        tabla, conteos = hiperparametros(nombre)
        tabla.to_csv(TABLES / f"priors_hiperparametros_{nombre}.csv",
                     index=False, float_format="%.4f")
        ruta = graficar_priors(tabla, conteos, nombre)
        print(f"\n{nombre}: {ruta.name}")
        print(tabla[["Parametro", "Distribucion", "Media", "DE",
                     "IC95_inf", "IC95_sup"]].round(3).to_string(index=False))
