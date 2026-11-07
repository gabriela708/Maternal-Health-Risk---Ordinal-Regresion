"""03 · Regresión logística ordinal bayesiana: priors, likelihood y MCMC.

Responsables:
    Persona B -> funciones de likelihood, prior y posterior.
    Persona C -> algoritmo de Metropolis-Hastings (o Gibbs) y ejecución del MCMC.
Entrada:  data/processed/train.csv
Salidas:  cadenas guardadas en data/processed/ y tabla resumen en results/tables/

Uso: python src/03_modelo_bayesiano.py
"""
import numpy as np
from scipy.special import log_expit
from scipy.stats import norm

from config import DATA_PROCESSED, PREDICTORS, TABLES


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


# ---------------------------------------------------------------- PERSONA C
def metropolis_hastings(theta0, n_iter, X, y):
    """TODO C1: implementar el algoritmo y devolver la cadena y la tasa de aceptación."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO C2: cargar train, correr varias cadenas, definir el burn-in
    #          y guardar las cadenas.
    # TODO C3: tabla resumen de la posterior (media, desviación, intervalos).
    pass
