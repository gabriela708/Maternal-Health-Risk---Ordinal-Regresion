"""Configuración compartida: rutas y constantes del proyecto.

Responsable: todos (cualquier cambio se acuerda en grupo).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "maternal_health_risk.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

SEED = 42

PREDICTORS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"

LEVELS = {"low risk": 0, "mid risk": 1, "high risk": 2}
L_NOMBRES = ["Bajo","Medio","Alto"]
ORDEN = ["low risk","mid risk","high risk"]