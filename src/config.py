"""Configuración compartida: rutas y constantes del proyecto.

Responsable: todos (cualquier cambio se acuerda en grupo).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "maternal_health_risk.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# TODO (grupo): definir una semilla común para que los resultados sean reproducibles.
SEED = None

# TODO (grupo): listar los nombres de las variables predictoras y la variable objetivo.
PREDICTORS = []
TARGET = None

# TODO (grupo): definir la codificación ordinal de RiskLevel (qué número va con cada nivel).
LEVELS = {}
