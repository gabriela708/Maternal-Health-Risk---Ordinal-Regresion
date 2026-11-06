"""03 · Regresión logística ordinal bayesiana: priors, likelihood y MCMC.

Responsables:
    Persona B -> funciones de likelihood, prior y posterior.
    Persona C -> algoritmo de Metropolis-Hastings (o Gibbs) y ejecución del MCMC.
Entrada:  data/processed/train.csv
Salidas:  cadenas guardadas en data/processed/ y tabla resumen en results/tables/

Uso: python src/03_modelo_bayesiano.py
"""
from config import DATA_PROCESSED, TABLES

# ---------------------------------------------------------------- PERSONA B
# TODO B1: definir los hiperparámetros de cada prior (distribución, media,
#          varianza) según lo argumentado en el informe.


def log_likelihood(theta, X, y):
    """TODO B2: log-verosimilitud de la regresión logística ordinal."""
    raise NotImplementedError


def log_prior(theta):
    """TODO B3: log-densidad conjunta de las priors."""
    raise NotImplementedError


def log_posterior(theta, X, y):
    """TODO B4: log-posterior (hasta una constante)."""
    raise NotImplementedError


# ---------------------------------------------------------------- PERSONA C
def metropolis_hastings(theta0, n_iter, X, y):
    """TODO C1: implementar el algoritmo y devolver la cadena y la tasa de aceptación."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO C2: cargar train, correr varias cadenas, definir el burn-in
    #          y guardar las cadenas.
    # TODO C3: tabla resumen de la posterior (media, desviación, intervalos).
    pass
