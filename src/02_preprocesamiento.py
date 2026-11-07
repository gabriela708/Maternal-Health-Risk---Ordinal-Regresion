"""02 · Preprocesamiento y partición entrenamiento / prueba.

Entrada:  data/raw/maternal_health_risk.csv
Salidas (data/processed/):
    train.csv, test.csv                 -> base principal (sin duplicados)
    train_full.csv, test_full.csv       -> base completa (con duplicados, análisis de sensibilidad)
    parametros_escalado_train.csv       -> media y desviación de train (sin duplicados)
    parametros_escalado_train_full.csv  -> media y desviación de train (con duplicados)

Uso: python src/02_preprocesamiento.py
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_RAW, DATA_PROCESSED, SEED, LEVELS, PREDICTORS, TARGET

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

#%% CARGA
df = pd.read_csv(DATA_RAW)

#%% TRATAMIENTO DE DUPLICADOS
df_u = df.drop_duplicates().copy()
print(f"Base completa: {len(df)} filas | Sin duplicados: {len(df_u)} filas")

#   Más de la mitad de la base son registros repetidos exactos. La fuente no tiene un
#   identificador por paciente, así que no hay forma de saber si las copias son tomas de la
#   misma gestante; tratarlas como observaciones nuevas estrecharía la likelihood y causaría
#   fuga entre train y test. Se trabaja con los 452 registros únicos y la base completa
#   se conserva solo como análisis de sensibilidad.

#%% CODIFICACIÓN DE LA VARIABLE TARGET (ORDINAL)
df_u["RiskLevel_cd"] = df_u[TARGET].map(LEVELS)
df["RiskLevel_cd"] = df[TARGET].map(LEVELS)
print(df_u.groupby("RiskLevel_cd")[TARGET].value_counts())

#   low risk = 0, mid risk = 1, high risk = 2, respetando el orden 0 < 1 < 2.

#%% DATOS POCO PROBABLES O FALLAS DE DIGITACIÓN
print(df[df["HeartRate"] == 7])
df_u = df_u[df_u["HeartRate"] != 7]
df = df[df["HeartRate"] != 7]

#   La frecuencia cardiaca de 7 lpm es fisiológicamente imposible (error de registro):
#   aparece 1 vez en la base sin duplicados y 2 en la completa. Se elimina en ambas.
#   Las edades extremas y los valores altos de BS son poco probables pero no imposibles,
#   y sin criterio médico para descartarlos se conservan.
#   No hay datos faltantes en ninguna de las dos bases.

#%% PARTICIÓN / SPLIT: train y test
df_u = df_u.drop(columns=[TARGET])
df = df.drop(columns=[TARGET])

# Perfiles únicos (solo signos) y una etiqueta por perfil para estratificar
perfiles = df_u.groupby(PREDICTORS, as_index=False)["RiskLevel_cd"].max()

perfiles_train, perfiles_test = train_test_split(
    perfiles,
    test_size=0.25,
    random_state=SEED,
    stratify=perfiles["RiskLevel_cd"],
)
perfiles_train = perfiles_train[PREDICTORS]
perfiles_test = perfiles_test[PREDICTORS]

# Base principal (sin duplicados)
df_train = df_u.merge(perfiles_train, on=PREDICTORS, how="inner")
df_test = df_u.merge(perfiles_test, on=PREDICTORS, how="inner")

# Base completa (con duplicados), mismos perfiles de train y test
df_train_full = df.merge(perfiles_train, on=PREDICTORS, how="inner")
df_test_full = df.merge(perfiles_test, on=PREDICTORS, how="inner")

#%% VERIFICACIÓN
print(f"Sin duplicados -> Train: {len(df_train)} | Test: {len(df_test)} | Suma: {len(df_train) + len(df_test)} de {len(df_u)}")
print(f"Con duplicados -> Train: {len(df_train_full)} | Test: {len(df_test_full)} | Suma: {len(df_train_full) + len(df_test_full)} de {len(df)}")
print("Perfiles en ambos lados (debe ser 0):", len(perfiles_train.merge(perfiles_test, on=PREDICTORS)))
print(pd.DataFrame({
    "train": df_train["RiskLevel_cd"].value_counts(normalize=True),
    "test": df_test["RiskLevel_cd"].value_counts(normalize=True),
}).round(3))

#   Las clases no están balanceadas sin duplicados (≈ 52 / 23 / 25 %), por eso se estratifica.

#%% ESCALAMIENTO Y ESTANDARIZACIÓN (z-score con parámetros solo de train)
def escalar(train, test):
    scaler = StandardScaler()
    train, test = train.copy(), test.copy()
    train[PREDICTORS] = scaler.fit_transform(train[PREDICTORS])
    test[PREDICTORS] = scaler.transform(test[PREDICTORS])
    parametros = pd.DataFrame({
        "Variable": PREDICTORS,
        "Media": scaler.mean_,
        "DesviacionEstandar": scaler.scale_,
    })
    return train, test, parametros

df_train, df_test, df_parametros = escalar(df_train, df_test)
df_train_full, df_test_full, df_parametros_full = escalar(df_train_full, df_test_full)

#%% GUARDAR
df_train.to_csv(DATA_PROCESSED / "train.csv", index=False)
df_test.to_csv(DATA_PROCESSED / "test.csv", index=False)
df_train_full.to_csv(DATA_PROCESSED / "train_full.csv", index=False)
df_test_full.to_csv(DATA_PROCESSED / "test_full.csv", index=False)
df_parametros.to_csv(DATA_PROCESSED / "parametros_escalado_train.csv", index=False)
df_parametros_full.to_csv(DATA_PROCESSED / "parametros_escalado_train_full.csv", index=False)
print("Archivos guardados en", DATA_PROCESSED)
