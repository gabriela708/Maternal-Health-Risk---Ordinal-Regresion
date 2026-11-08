"""03 · Regresión logística ordinal bayesiana: priors, likelihood y MCMC.

Responsables:
    Persona B -> funciones de likelihood, prior y posterior.
    Persona C -> algoritmo de Metropolis-Hastings (o Gibbs) y ejecución del MCMC.
Entrada:  data/processed/train.csv
Salidas:  cadenas guardadas en data/processed/ y tabla resumen en results/tables/

Uso: python src/03_modelo_bayesiano.py
"""
import numpy as np
import pandas as pd
from scipy.special import log_expit
from scipy.stats import norm

from config import DATA_PROCESSED, PREDICTORS, SEED, TABLES


# Priors definidas en unidades originales: media y desviación estándar.
RAW_COEFFICIENT_PRIORS = (
    (0.03, 0.02, False),  # Age
    (0.04, 0.02, True),   # SystolicBP
    (0.03, 0.02, True),   # DiastolicBP
    (0.00, 0.50, False),  # BodyTemp
)
N_PARAMETERS = 2 + len(PREDICTORS)


def _validated_theta(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (N_PARAMETERS,):
        raise ValueError(f"theta debe tener forma ({N_PARAMETERS},)")
    if not np.all(np.isfinite(theta)):
        return None
    return theta


def _validated_design(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != len(PREDICTORS):
        raise ValueError(f"X debe tener forma (n, {len(PREDICTORS)})")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe tener una etiqueta por fila de X")
    if not np.all(np.isfinite(X)):
        raise ValueError("X debe contener únicamente valores finitos")
    if not np.all(np.isin(y, (0, 1, 2))):
        raise ValueError("y debe contener las clases ordinales 0, 1 o 2")
    return X, y.astype(int)


def transformed_coefficient_priors(predictor_scales):
    """Devuelve hiperparámetros de las priors en la escala del modelo.

    predictor_scales debe contener las desviaciones estándar de train en el
    mismo orden que PREDICTORS. Las priors de Age, SystolicBP, DiastolicBP y
    BodyTemp se transforman desde unidades originales. BS usa una N+(0, 1) y
    HeartRate una N(0, 1) directamente en escala estandarizada.
    """
    scales = np.asarray(predictor_scales, dtype=float)
    if scales.shape != (len(PREDICTORS),):
        raise ValueError(
            f"predictor_scales debe tener forma ({len(PREDICTORS)},)"
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Las desviaciones estándar deben ser finitas y positivas")

    raw = np.asarray(RAW_COEFFICIENT_PRIORS, dtype=float)
    means = np.array([
        scales[0] * raw[0, 0],
        scales[1] * raw[1, 0],
        scales[2] * raw[2, 0],
        0.0,
        scales[4] * raw[3, 0],
        0.0,
    ])
    standard_deviations = np.array([
        scales[0] * raw[0, 1],
        scales[1] * raw[1, 1],
        scales[2] * raw[2, 1],
        1.0,
        scales[4] * raw[3, 1],
        1.0,
    ])
    positive = np.array([False, True, True, True, False, False])
    return means, standard_deviations, positive


def cutpoint_prior_from_training_labels(y_train):
    """Calcula los centros empíricos de los cutpoints usando solo etiquetas train.

    Este es un paso de empirical Bayes: se debe pasar exclusivamente y_train,
    nunca las etiquetas del conjunto de prueba.
    """
    y_train = np.asarray(y_train)
    if y_train.ndim != 1 or y_train.size == 0:
        raise ValueError("y_train debe ser un vector no vacío")
    if not np.all(np.isin(y_train, (0, 1, 2))):
        raise ValueError("y_train debe contener las clases ordinales 0, 1 o 2")

    counts = np.bincount(y_train.astype(int), minlength=3)
    if np.any(counts == 0):
        raise ValueError("y_train debe incluir observaciones de las tres clases")

    cumulative_probabilities = np.cumsum(counts)[:2] / y_train.size
    return np.log(
        cumulative_probabilities / (1.0 - cumulative_probabilities)
    )


def log_likelihood(theta, X, y):
    """Log-likelihood ordinal logit con cutpoints alpha_1 < alpha_2.

    theta = (alpha_1, alpha_2, beta_1, ..., beta_6). El modelo usa
    logit(P(Y <= k | X)) = alpha_k - X @ beta, para k = 1, 2.
    """
    theta = _validated_theta(theta)
    if theta is None:
        return -np.inf
    X, y = _validated_design(X, y)
    alpha_1, alpha_2 = theta[:2]
    if alpha_1 >= alpha_2:
        return -np.inf

    eta = X @ theta[2:]
    log_cdf_1 = log_expit(alpha_1 - eta)
    log_cdf_2 = log_expit(alpha_2 - eta)
    log_delta = np.minimum(log_cdf_1 - log_cdf_2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probabilities = np.column_stack((
            log_cdf_1,
            log_cdf_2 + np.log(-np.expm1(log_delta)),
            log_expit(eta - alpha_2),
        ))
    return float(np.sum(log_probabilities[np.arange(y.size), y]))


def log_prior(
    theta,
    predictor_scales,
    cutpoint_prior_means,
    cutpoint_prior_sd=0.5,
):
    """Log-densidad de las priors, con coeficientes en escala estandarizada.

    Los centros de los cutpoints se reciben explícitamente; obténganse con
    cutpoint_prior_from_training_labels(y_train), nunca usando y_test.
    """
    theta = _validated_theta(theta)
    if theta is None:
        return -np.inf
    alpha_1, alpha_2 = theta[:2]
    if alpha_1 >= alpha_2:
        return -np.inf

    cutpoint_means = np.asarray(cutpoint_prior_means, dtype=float)
    if cutpoint_means.shape != (2,) or not np.all(np.isfinite(cutpoint_means)):
        raise ValueError("cutpoint_prior_means debe contener dos valores finitos")
    if not np.isfinite(cutpoint_prior_sd) or cutpoint_prior_sd <= 0:
        raise ValueError("cutpoint_prior_sd debe ser finito y positivo")

    means, sds, positive = transformed_coefficient_priors(predictor_scales)
    coefficients = theta[2:]
    if np.any(positive & (coefficients <= 0)):
        return -np.inf

    total = np.sum(norm.logpdf(
        theta[:2], loc=cutpoint_means, scale=cutpoint_prior_sd
    ))
    total += np.sum(norm.logpdf(
        coefficients, loc=means, scale=sds
    ))
    # Normalize each positive-truncated Normal prior on (0, infinity).
    total -= np.sum(norm.logcdf(means[positive] / sds[positive]))
    return float(total)


def log_posterior(
    theta,
    X,
    y,
    predictor_scales,
    cutpoint_prior_means,
    cutpoint_prior_sd=0.5,
):
    """Log-posterior hasta una constante."""
    prior = log_prior(
        theta,
        predictor_scales,
        cutpoint_prior_means,
        cutpoint_prior_sd,
    )
    if not np.isfinite(prior):
        return prior
    return prior + log_likelihood(theta, X, y)




def metropolis_hastings(theta0, n_iter, X, y, predictor_scales,
                        cutpoint_prior_means, proposal_sd=0.10):
    """Metropolis-Hastings con propuesta normal simétrica (random walk).

    Devuelve la cadena (n_iter, n_parametros) y la tasa de aceptación.
    """
    theta0 = np.asarray(theta0, dtype=float)
    chain = np.zeros((n_iter, len(theta0)))
    chain[0] = theta0
    accepted = 0

    current_log_post = log_posterior(theta0, X, y, predictor_scales, cutpoint_prior_means)

    for i in range(1, n_iter):
        # 1. Proponer un candidato sumando una perturbación normal
        proposal = np.random.normal(loc=chain[i - 1], scale=proposal_sd)

        # 2. Log-posterior del candidato
        proposal_log_post = log_posterior(proposal, X, y, predictor_scales, cutpoint_prior_means)

        # 3. Razón de aceptación en escala log (la propuesta simétrica se cancela)
        log_alpha = proposal_log_post - current_log_post

        # 4. Aceptar o rechazar
        if np.log(np.random.uniform()) < log_alpha:
            chain[i] = proposal
            current_log_post = proposal_log_post
            accepted += 1
        else:
            chain[i] = chain[i - 1]

    return chain, accepted / (n_iter - 1)


# Hiperparámetros del MCMC
N_ITER = 20000
BURN_IN = 5000
N_CHAINS = 4
PARAM_NAMES = ["alpha_1", "alpha_2"] + list(PREDICTORS)


def ejecutar_pipeline_mcmc(nombre_dataset, proposal_sd):
    """Corre el MCMC para train.csv o train_full.csv y guarda cadenas y resumen."""
    print(f"\n{'=' * 40}\nEjecutando MCMC para: {nombre_dataset} (proposal_sd = {proposal_sd})\n{'=' * 40}")

    # 1. Datos de entrenamiento
    train_df = pd.read_csv(DATA_PROCESSED / f"{nombre_dataset}.csv")
    y_train = train_df["RiskLevel_cd"].values
    X_train = train_df[PREDICTORS].values

    # 2. Desviaciones de train para transformar las priors
    scales_df = pd.read_csv(DATA_PROCESSED / f"parametros_escalado_{nombre_dataset}.csv")
    predictor_scales = scales_df["DesviacionEstandar"].values

    # 3. Centros de los cutpoints (Bayes empírico, solo con y_train)
    cutpoint_prior_means = cutpoint_prior_from_training_labels(y_train)

    # 4. Varias cadenas con puntos iniciales distintos
    cadenas = []
    for c in range(N_CHAINS):
        alpha_init = cutpoint_prior_means + np.random.normal(0, 0.1, size=2)
        beta_init = 0.05 + np.random.normal(0, 0.01, size=len(PREDICTORS))
        theta0 = np.concatenate([alpha_init, beta_init])

        chain, acc_rate = metropolis_hastings(
            theta0, N_ITER, X_train, y_train,
            predictor_scales, cutpoint_prior_means, proposal_sd,
        )
        cadenas.append(chain)
        print(f"   Cadena {c + 1}: tasa de aceptación {acc_rate * 100:.2f}%")

    cadenas = np.array(cadenas)  # forma (n_chains, n_iter, n_params)
    np.save(DATA_PROCESSED / f"cadenas_{nombre_dataset}.npy", cadenas)

    # 5. Resumen de la posterior (sin burn-in, cadenas unidas)
    muestras = cadenas[:, BURN_IN:, :].reshape(-1, cadenas.shape[-1])
    resumen = pd.DataFrame({
        "Parametro": PARAM_NAMES,
        "Media": muestras.mean(axis=0),
        "Desv_Std": muestras.std(axis=0),
        "IC_2.5%": np.percentile(muestras, 2.5, axis=0),
        "IC_97.5%": np.percentile(muestras, 97.5, axis=0),
    })
    resumen.to_csv(TABLES / f"resumen_posterior_{nombre_dataset}.csv", index=False)
    print(resumen.round(3).to_string(index=False))


if __name__ == "__main__":
    np.random.seed(SEED)
    TABLES.mkdir(parents=True, exist_ok=True)

    # A. Base principal (sin duplicados)
    ejecutar_pipeline_mcmc("train", proposal_sd=0.10)

    # B. Base completa (con duplicados). Con más filas la posterior es más angosta,
    #    por eso necesita un paso más pequeño para mantener una aceptación similar.
    ejecutar_pipeline_mcmc("train_full", proposal_sd=0.06)
