import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.getenv("SEMICONDUCTOR_STATE_DIR", str(ROOT / "semiconductor_state"))).resolve()
DEFAULT_DATA = ROOT / "data/semiconductor_etch_timeseries_3years.csv"
MODEL_NAME = "Semiconductor_Etch_LSTM"
ALIAS = "Production"
SEQ_LEN = 20
FEATURES = [
    "ambient_temperature_c", "ambient_humidity_pct", "cooling_water_temperature_c",
    "chamber_temperature_c", "chamber_pressure_mtorr", "cf4_flow_sccm", "o2_flow_sccm",
    "rf_power_w", "process_time_sec", "vibration_mm_s", "particle_count", "usage_count",
]


def tracking_uri():
    STATE.mkdir(parents=True, exist_ok=True)
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{STATE / 'mlflow.db'}")


def latest_data():
    uploads = list((STATE / "uploads").glob("*.csv"))
    return max(uploads, key=lambda p: p.stat().st_mtime_ns) if uploads else DEFAULT_DATA
