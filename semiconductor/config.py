"""EtchGuard 전역 설정. 운영 임계값은 근거와 함께 한곳에서 관리하고 환경변수로만 덮어쓴다."""
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.getenv("SEMICONDUCTOR_STATE_DIR", str(ROOT / "semiconductor_state"))).resolve()
LOG_DIR = Path(os.getenv("AIOPS_LOG_DIR", str(ROOT / "logs"))).resolve()
DEFAULT_DATA = ROOT / "data/semiconductor_etch_timeseries_3years.csv"

MODEL_NAME = "Semiconductor_Etch_LSTM"
ALIAS = "Production"
EXPERIMENT = "EtchGuard-AIOps"
SEQ_LEN = 20  # 최근 20사이클(30분 간격, 10시간)
FEATURES = [
    "ambient_temperature_c", "ambient_humidity_pct", "cooling_water_temperature_c",
    "chamber_temperature_c", "chamber_pressure_mtorr", "cf4_flow_sccm", "o2_flow_sccm",
    "rf_power_w", "process_time_sec", "vibration_mm_s", "particle_count", "usage_count",
]

# 위험 등급: 점수 ≥70 HIGH(점검 권고), 40~70 WARNING(관찰), <40 NORMAL
HIGH_SCORE = 70.0
WARNING_SCORE = 40.0


def _env_float(name, default):
    return float(os.getenv(name, default))


def _env_int(name, default):
    return int(os.getenv(name, default))


# 배포 게이트 (base 학습·fine-tuning 공통). HIGH 미탐은 불량 웨이퍼, 오탐은 불필요한 정비로 이어진다.
GATE_RMSE_MAX = _env_float("GATE_RMSE_MAX", 4.0)
GATE_RECALL_MIN = _env_float("GATE_RECALL_MIN", 0.80)
GATE_PRECISION_MIN = _env_float("GATE_PRECISION_MIN", 0.80)

# 성능 드리프트 판정: 12시간(24사이클) 윈도우 RMSE가 4.0을 3회 연속 초과(=36시간 지속)하면 드리프트.
# 4.0 = 배포 게이트와 같은 품질선. 근거: scripts/analyze_drift_policy.py — 2025 정상 구간 오탐 0.03회/장비·월
# (운영 목표 1회 미만), 드리프트 윈도우 판정률 76.8%(5.0은 60.5%), 탐지 지연 36시간(5.0과 동일).
DRIFT_WINDOW = _env_int("DRIFT_WINDOW", 24)
DRIFT_RMSE_THRESHOLD = _env_float("DRIFT_RMSE_THRESHOLD", 4.0)
DRIFT_CONSECUTIVE = _env_int("DRIFT_CONSECUTIVE", 3)

# 재학습(warm start fine-tuning) 정책
RETRAIN_DAYS = _env_int("RETRAIN_DAYS", 7)            # 감지 시점까지 최근 7일 × 전 장비
RETRAIN_HOLDOUT = _env_float("RETRAIN_HOLDOUT", 0.2)  # 시간순 마지막 20%를 검증으로
FINE_TUNE_EPOCHS = _env_int("FINE_TUNE_EPOCHS", 10)
FINE_TUNE_LR = _env_float("FINE_TUNE_LR", 1e-4)
RETRAIN_ASYNC = os.getenv("RETRAIN_ASYNC", "1") == "1"  # 요청 스레드와 재학습 분리


def tracking_uri():
    STATE.mkdir(parents=True, exist_ok=True)
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{STATE / 'mlflow.db'}")


def latest_data():
    uploads = list((STATE / "uploads").glob("*.csv"))
    return max(uploads, key=lambda p: p.stat().st_mtime_ns) if uploads else DEFAULT_DATA


def risk_level(score):
    return "HIGH" if score >= HIGH_SCORE else "WARNING" if score >= WARNING_SCORE else "NORMAL"
