"""EtchGuard API: 식각 장비 다음 공정(30분 후) 이상 점수 예측 서빙 + AIOps 모니터링.

uvicorn semiconductor.app:app --port 8000
환경변수: MODEL_SOURCE=local|mlflow, LOADING_MODE=lazy|eager
"""
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .config import (ROOT, STATE, LOG_DIR, FEATURES, SEQ_LEN, GATE_RMSE_MAX, GATE_RECALL_MIN,
                     GATE_PRECISION_MIN, latest_data, risk_level)
from .data import load_series, summary, validate_features, to_records
from .monitoring.data_drift import diagnose
from .monitoring.drift_detector import DriftMonitor, compute_rmse
from .monitoring.retrain_trigger import RetrainController
from .runtime import ModelManager

logger = logging.getLogger("semiconductor")
MAX_BATCH = 5000

SensorPoint = create_model(
    "SensorPoint", __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    timestamp=(datetime, ...), **{k: (float, ...) for k in FEATURES})
FeedbackPoint = create_model(
    "FeedbackPoint", __base__=SensorPoint,
    anomaly_score=(float, Field(ge=0, le=100, description="공정 후 확정된 이상 점수(정답)")))


def _check_sequence(points):
    for i, point in enumerate(points):
        validate_features([getattr(point, k) for k in FEATURES])
        if point.timestamp.tzinfo is not None:
            raise ValueError("Use timezone-naive simulation timestamps")
        if i and point.timestamp - points[i-1].timestamp != timedelta(minutes=30):
            raise ValueError("Exactly 30-minute chronological spacing is required")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equipment_id: str = Field(min_length=1, max_length=80)
    sequence: list[SensorPoint] = Field(min_length=SEQ_LEN, max_length=SEQ_LEN)

    @model_validator(mode="after")
    def validate_sequence(self):
        _check_sequence(self.sequence)
        return self


class BatchTestRequest(BaseModel):
    """확정 점수가 붙은 연속 공정 기록. 앞 SEQ_LEN개는 문맥, 이후 각 시점이 (예측, 실제) 한 쌍이 된다."""
    model_config = ConfigDict(extra="forbid")
    equipment_id: str = Field(min_length=1, max_length=80)
    records: list[FeedbackPoint] = Field(min_length=SEQ_LEN + 1, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def validate_records(self):
        _check_sequence(self.records)
        return self


class OpsMetrics:
    """운영 지표: 엔드포인트별 요청 수·에러율·지연시간(p50/p95)."""

    def __init__(self):
        self.started = time.time()
        self.requests, self.errors_4xx, self.errors_5xx = {}, {}, {}
        self.latency = {}

    def record(self, path, status, ms):
        self.requests[path] = self.requests.get(path, 0) + 1
        if 400 <= status < 500:
            self.errors_4xx[path] = self.errors_4xx.get(path, 0) + 1
        elif status >= 500:
            self.errors_5xx[path] = self.errors_5xx.get(path, 0) + 1
        self.latency.setdefault(path, deque(maxlen=1000)).append(ms)

    def snapshot(self):
        out = {}
        for path, n in self.requests.items():
            lat = np.array(self.latency[path])
            out[path] = {"requests": n, "errors_4xx": self.errors_4xx.get(path, 0),
                         "errors_5xx": self.errors_5xx.get(path, 0),
                         "error_rate_5xx": round(self.errors_5xx.get(path, 0) / n, 4),
                         "p50_ms": round(float(np.percentile(lat, 50)), 2),
                         "p95_ms": round(float(np.percentile(lat, 95)), 2)}
        return {"uptime_seconds": round(time.time() - self.started, 1), "endpoints": out}


def _configure_aiops_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    aiops = logging.getLogger("aiops")
    target = str(LOG_DIR / "aiops.log")
    if not any(getattr(h, "baseFilename", None) == target for h in aiops.handlers):
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        aiops.addHandler(handler)
    aiops.setLevel(logging.INFO)


def _append_predictions(rows):
    with open(LOG_DIR / "predictions.jsonl", "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_app():
    mode = os.getenv("LOADING_MODE", "lazy")
    if mode not in ("lazy", "eager"):
        raise ValueError("LOADING_MODE must be lazy or eager")
    _configure_aiops_log()
    manager = ModelManager(os.getenv("MODEL_SOURCE", "local"))
    monitor = DriftMonitor()
    retrain = RetrainController(manager, monitor, load_series)
    ops = OpsMetrics()
    last_batch = {}  # 장비별 마지막 배치 판정 (대시보드 표시용)

    @asynccontextmanager
    async def lifespan(app):
        start = time.perf_counter()
        if mode == "eager":
            await run_in_threadpool(manager.get)
        app.state.startup_seconds = time.perf_counter() - start
        logger.info("[%s] startup %.3fs", mode, app.state.startup_seconds)
        yield

    app = FastAPI(title="EtchGuard — 식각 장비 이상 징후 조기예측 API", version="3.0", lifespan=lifespan,
                  description="최근 20사이클(10시간) 센서 12종 → 다음 30분 이상 점수·위험 등급. "
                              "MLflow 게이트·Production 승격, RMSE 드리프트 감지, warm-start 자동 재학습.")
    app.state.manager, app.state.monitor, app.state.retrain = manager, monitor, retrain

    @app.middleware("http")
    async def measure(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        ops.record(getattr(route, "path", request.url.path), response.status_code,
                   (time.perf_counter() - start) * 1000)
        return response

    def model_or_503():
        try:
            return manager.get()
        except Exception:
            logger.exception("Model loading failed")
            raise HTTPException(503, "Model unavailable. Check training/Production registration and server logs.")

    @app.get("/health", tags=["serving"])
    def health():
        return {"status": "ok", "model_loaded": manager.model is not None,
                "loading_mode": mode, "model_source": manager.source,
                "model_version": manager.model.version if manager.model else None,
                "startup_seconds": getattr(app.state, "startup_seconds", None),
                "model_load_seconds": manager.load_seconds}

    @app.get("/ready", tags=["serving"])
    def ready():
        if manager.model is None:
            raise HTTPException(503, "Model not loaded; lazy mode loads on first prediction")
        return {"ready": True, "model_version": manager.model.version}

    @app.post("/predict", tags=["serving"])
    def predict(request: PredictRequest):
        start = time.perf_counter()
        model = model_or_503()
        try:
            score = float(model.predict([[[getattr(p, k) for k in FEATURES] for p in request.sequence]])[0])
        except Exception:
            logger.exception("Prediction failed")
            raise HTTPException(503, "Prediction failed. See server logs.")
        level = risk_level(score)
        target = request.sequence[-1].timestamp + timedelta(minutes=30)
        _append_predictions([{"logged_at": datetime.now().isoformat(timespec="seconds"), "source": "predict",
                              "equipment_id": request.equipment_id, "target_timestamp": target.isoformat(),
                              "predicted": round(score, 3), "model_version": model.version}])
        return {"equipment_id": request.equipment_id, "target_timestamp": target.isoformat(),
                "predicted_anomaly_score": round(score, 3), "risk_level": level,
                "recommended_action": {"HIGH": "장비 점검 권고 (다음 lot 투입 전 확인)",
                                       "WARNING": "추세 관찰 강화", "NORMAL": "정상 운전"}[level],
                "model_version": model.version, "model_source": manager.source,
                "latency_ms": round((time.perf_counter()-start)*1000, 2), "simulation": True}

    @app.post("/predict/batch-test", tags=["aiops"])
    def batch_test(request: BatchTestRequest):
        """확정 점수가 들어온 배치로 예측 오차를 누적 → 드리프트 판정 → 필요 시 재학습 트리거."""
        model = model_or_503()
        raw = np.array([[getattr(p, k) for k in FEATURES] for p in request.records], dtype="float32")
        x = np.stack([raw[i - SEQ_LEN:i] for i in range(SEQ_LEN, len(raw))])
        predicted = model.predict(x)
        pairs = [{"timestamp": p.timestamp.isoformat(), "predicted": round(float(s), 3),
                  "actual": round(p.anomaly_score, 3)} for p, s in zip(request.records[SEQ_LEN:], predicted)]
        _append_predictions([{"logged_at": datetime.now().isoformat(timespec="seconds"), "source": "batch-test",
                              "equipment_id": request.equipment_id, "target_timestamp": p["timestamp"],
                              "predicted": p["predicted"], "actual": p["actual"],
                              "model_version": model.version} for p in pairs])
        drift = monitor.observe(request.equipment_id, pairs)
        # 월 경계에 걸친 배치는 중앙 시점의 달을 기준 분포로 쓴다 (7/27~8/3 배치 → 7월 기준)
        diagnostics = diagnose(model.profile, request.records[len(request.records) // 2].timestamp, raw)
        decision = retrain.check_and_trigger(request.equipment_id, drift,
                                             request.records[-1].timestamp, diagnostics)
        last_batch[request.equipment_id] = {
            "batch_end": request.records[-1].timestamp.isoformat(), "status": decision["status"],
            "latest_window_rmse": drift["latest_window_rmse"], "consecutive_over": drift["consecutive_over"],
            "top_sensors": diagnostics.get("top_sensors", []), "model_version": model.version}
        return {"equipment_id": request.equipment_id, "model_version": model.version,
                "n_predictions": len(pairs), "batch_rmse": round(compute_rmse(pairs), 3),
                "predictions": pairs,
                "drift_check": {**decision, **drift, "threshold": monitor.threshold,
                                "window_cycles": monitor.window, "consecutive_required": monitor.consecutive},
                "input_diagnostics": diagnostics}

    @app.get("/monitoring/status", tags=["aiops"])
    def monitoring_status():
        return {"model_version": manager.model.version if manager.model else None,
                "gate": {"rmse_max": GATE_RMSE_MAX, "recall_min": GATE_RECALL_MIN,
                         "precision_min": GATE_PRECISION_MIN},
                "drift": monitor.snapshot(), "last_batch": dict(sorted(last_batch.items())),
                "retrain": retrain.status(), "ops": ops.snapshot()}

    @app.post("/data/upload", tags=["data"])
    async def upload(file: UploadFile = File(...)):
        raw = await file.read(64*1024*1024 + 1)
        await file.close()
        if len(raw) > 64*1024*1024:
            raise HTTPException(413, "Maximum CSV size is 64 MiB")
        from .data import read_data
        try:
            series = await run_in_threadpool(read_data, raw.decode("utf-8-sig"))
        except (UnicodeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        folder = STATE / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"etch_{uuid.uuid4().hex}.csv"
        tmp = path.with_suffix(".tmp")
        await run_in_threadpool(tmp.write_bytes, raw)
        tmp.replace(path)
        return {"filename": path.name, **summary(series), "note": "Upload stores data; train separately."}

    def series_or_400():
        try:
            return load_series()
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/data/status", tags=["data"])
    def status():
        path, series = series_or_400()
        return {"filename": path.name, **summary(series)}

    @app.get("/data/example", tags=["data"])
    def example():
        _, series = series_or_400()
        eid, g = next(iter(series.items()))
        return {"equipment_id": eid, "sequence": to_records(g, 0, SEQ_LEN, with_label=False)}

    @app.get("/data/batch", tags=["data"])
    def batch(equipment_id: str, start: datetime, cycles: int = Query(336, ge=1, le=MAX_BATCH - SEQ_LEN)):
        """시뮬레이션용: 저장소에서 start 시점부터 cycles개 (예측, 실제) 쌍을 만들 수 있는 배치를 꺼낸다."""
        _, series = series_or_400()
        if equipment_id not in series:
            raise HTTPException(404, f"Unknown equipment_id {equipment_id}")
        g = series[equipment_id]
        import bisect
        i = bisect.bisect_left(g["time"], start)
        if i < SEQ_LEN or i >= len(g["time"]):
            raise HTTPException(400, "start is outside the stored range (needs 20 prior cycles)")
        end = min(i + cycles, len(g["time"]))
        return {"equipment_id": equipment_id, "records": to_records(g, i - SEQ_LEN, end)}

    @app.get("/logs", tags=["aiops"])
    def list_logs():
        if not LOG_DIR.is_dir():
            return []
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(LOG_DIR.iterdir()) if p.is_file()]

    @app.get("/logs/{filename}", tags=["aiops"])
    def read_log(filename: str, tail: int = Query(200, ge=1, le=5000)):
        if filename != os.path.basename(filename) or filename.startswith("."):
            raise HTTPException(400, "잘못된 파일명입니다")
        path = LOG_DIR / filename
        if not path.is_file():
            raise HTTPException(404, "로그 파일을 찾을 수 없습니다")
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=tail)
        return {"name": filename, "lines": len(lines), "content": "".join(lines)}

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "semiconductor/static/index.html")

    return app


app = create_app()
