"""EtchGuard API: 식각 장비 다음 공정(30분 후) 이상 점수 예측 서빙 + AIOps 모니터링.

uvicorn semiconductor.app:app --port 8000
환경변수: MODEL_SOURCE=local|mlflow, LOADING_MODE=lazy|eager
"""
import logging
import os
import re
import threading
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

from . import demo_seed
from .config import (ROOT, STATE, LOG_DIR, FEATURES, SEQ_LEN, GATE_RMSE_MAX, GATE_RECALL_MIN,
                     GATE_PRECISION_MIN, MODEL_NAME, ALIAS, RECOMMENDED_ACTION, latest_data, risk_level,
                     tracking_uri)
from .data import load_series, summary, validate_features, to_records
from .monitoring.alerts import HighAlerts
from .monitoring.data_drift import diagnose, deviation_rank
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


class LiveStore:
    """실시간 대시보드용 메모리 버퍼.

    - 요청: 최근 3,000건의 (시각, 경로, 상태 코드, 지연 ms) → TPS·응답시간·X-view
    - 예측: 장비별 최근 500건의 위험 등급, 장비별 마지막 예측, 오늘 시간대별 예측·HIGH 건수(실시간 /predict만)
    파일에 남기지 않으므로 서버를 재시작하면 모두 비워진다.
    """
    LEVELS = ("NORMAL", "WARNING", "HIGH")
    EVENT_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)$")

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.requests = deque(maxlen=3000)
        self.levels = {}          # eid -> deque(maxlen=500) of level
        self.latest = {}          # eid -> 마지막 예측
        self.series = {}          # eid -> deque(maxlen=96) of {t, p, a}: 예측 대 확정 점수 추이(48시간)
        self.day = datetime.now().date()
        self.hour_calls = [0] * 24
        self.hour_high = [0] * 24

    def _roll_day(self, now):
        if now.date() != self.day:
            self.day, self.hour_calls, self.hour_high = now.date(), [0] * 24, [0] * 24

    def record_request(self, path, status, ms):
        with self.lock:
            self.requests.append((time.time(), path, status, ms))

    def record_predictions(self, equipment_id, rows, source, extra=None, count_today=True):
        """rows: [{"target_timestamp", "predicted", "actual"?}] 시간순. extra는 최신 예측에 덧붙일 진단 정보.
        count_today=False면 오늘 시간대별 집계에는 넣지 않는다(데모 시드가 시간대를 따로 채울 때)."""
        now = datetime.now()
        with self.lock:
            self._roll_day(now)
            bucket = self.levels.setdefault(equipment_id, deque(maxlen=500))
            trend = self.series.setdefault(equipment_id, deque(maxlen=96))
            for row in rows:
                level = risk_level(row["predicted"])
                bucket.append(level)
                trend.append({"t": row["target_timestamp"], "p": row["predicted"], "a": row.get("actual")})
                if not count_today:
                    continue
                self.hour_calls[now.hour] += 1
                if level == "HIGH":
                    self.hour_high[now.hour] += 1
            last = rows[-1]
            self.latest[equipment_id] = {"at": now.isoformat(timespec="seconds"), "source": source,
                                         "target_timestamp": last["target_timestamp"],
                                         "score": last["predicted"], "level": risk_level(last["predicted"]),
                                         "actual": last.get("actual"), **(extra or {})}

    def add_today(self, hour_calls, hour_high):
        """오늘 시간대별 예측·HIGH 건수에 더한다 (데모 시드용)."""
        with self.lock:
            self._roll_day(datetime.now())
            for h in range(24):
                self.hour_calls[h] += hour_calls[h]
                self.hour_high[h] += hour_high[h]

    def _events(self, limit=40):
        path = self.log_dir / "aiops.log"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=limit)
        events = []
        for line in lines:
            m = self.EVENT_LINE.match(line.strip())
            if not m:
                continue
            eq = re.search(r"equipment=(\S+)", m.group(3))
            events.append({"at": m.group(1), "level": m.group(2), "message": m.group(3),
                           "equipment_id": eq.group(1) if eq else None})
        return events

    @staticmethod
    def _pct(values, q):
        return round(float(np.percentile(values, q)), 1) if values else None

    def snapshot(self, window_seconds=180):
        now = time.time()
        with self.lock:
            self._roll_day(datetime.now())
            requests = list(self.requests)
            levels = {eid: list(d) for eid, d in self.levels.items()}
            latest = dict(self.latest)
            series = {eid: list(d) for eid, d in self.series.items()}
            hour_calls, hour_high = list(self.hour_calls), list(self.hour_high)
        base = int(now) - window_seconds + 1
        tps = [0] * window_seconds
        latency_buckets = [[] for _ in range(window_seconds // 5)]
        recent = []
        for ts, path, status, ms in requests:
            if ts < base:
                continue
            i = int(ts) - base
            if path in ("/predict", "/predict/batch-test"):
                tps[i] += 1
                recent.append({"ago": round(now - ts, 2), "path": path, "status": status, "ms": round(ms, 1)})
            if path == "/predict":
                latency_buckets[min(i // 5, len(latency_buckets) - 1)].append(ms)
        latency = [{"p50": self._pct(b, 50), "p95": self._pct(b, 95)} for b in latency_buckets]
        equipment = {}
        for eid in sorted(set(levels) | set(latest)):
            counts = {lv: 0 for lv in self.LEVELS}
            for lv in levels.get(eid, []):
                counts[lv] += 1
            equipment[eid] = {"counts": counts, "latest": latest.get(eid), "series": series.get(eid, [])}
        events = self._events()
        return {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "window_seconds": window_seconds,
                "tps": {"series": tps, "now": round(sum(tps[-5:]) / 5, 2), "peak": max(tps)},
                "latency": latency, "recent_requests": recent[-800:],
                "hourly": {"calls": hour_calls, "high": hour_high, "hour": datetime.now().hour},
                "equipment": equipment, "events": events,
                "event_counts": {lv: sum(1 for e in events if e["level"] == lv)
                                 for lv in ("ALERT", "WARN", "INFO", "OK", "FAIL", "ERROR")}}


def _configure_aiops_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    aiops = logging.getLogger("aiops")
    target = str(LOG_DIR / "aiops.log")
    if not any(getattr(h, "baseFilename", None) == target for h in aiops.handlers):
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        aiops.addHandler(handler)
    aiops.setLevel(logging.INFO)


def create_app():
    mode = os.getenv("LOADING_MODE", "lazy")
    if mode not in ("lazy", "eager"):
        raise ValueError("LOADING_MODE must be lazy or eager")
    _configure_aiops_log()
    manager = ModelManager(os.getenv("MODEL_SOURCE", "local"))
    monitor = DriftMonitor()
    retrain = RetrainController(manager, monitor, load_series)
    ops = OpsMetrics()
    live = LiveStore(LOG_DIR)
    last_batch = {}  # 장비별 마지막 배치 판정 (대시보드 표시용)
    alerts = HighAlerts()  # HIGH 예측 경보 (aiops.log + 선택 웹훅)
    run_metrics = {}  # MLflow run_id -> 검증 지표 (불변이라 한 번만 읽는다)

    @asynccontextmanager
    async def lifespan(app):
        start = time.perf_counter()
        if mode == "eager":
            await run_in_threadpool(manager.get)
        app.state.startup_seconds = time.perf_counter() - start
        logger.info("[%s] startup %.3fs", mode, app.state.startup_seconds)
        if demo_seed.enabled():  # 기동 시간 측정 뒤에 실행해 Eager 지표에 섞이지 않게 한다
            try:
                model = await run_in_threadpool(manager.get)
                _, series = await run_in_threadpool(load_series)
                app.state.demo_seed = await run_in_threadpool(demo_seed.seed, model, series, monitor,
                                                              live, last_batch, alerts)
                logger.info("demo seed: %s", app.state.demo_seed)
            except Exception:
                logger.exception("Demo seed failed; dashboard starts empty")
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
        path, ms = getattr(route, "path", request.url.path), (time.perf_counter() - start) * 1000
        ops.record(path, response.status_code, ms)
        live.record_request(path, response.status_code, ms)
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
        raw = [[getattr(p, k) for k in FEATURES] for p in request.sequence]
        try:
            score = float(model.predict([raw])[0])
        except Exception:
            logger.exception("Prediction failed")
            raise HTTPException(503, "Prediction failed. See server logs.")
        level = risk_level(score)
        action = RECOMMENDED_ACTION[level]
        # 점검 우선순위: 입력 창 20행이 같은 달 기준 중앙값에서 벗어난 정도(z-score) 상위 3개 센서
        deviations = deviation_rank(model.profile, request.sequence[-1].timestamp, raw)
        target = request.sequence[-1].timestamp + timedelta(minutes=30)
        live.record_predictions(request.equipment_id,
                                [{"target_timestamp": target.isoformat(), "predicted": round(score, 3)}], "predict",
                                extra={"action": action, "deviations": deviations})
        alerts.observe(request.equipment_id, score, level, target.isoformat(), action,
                       [f"{d['sensor']}({d['z']:+.2f}σ)" for d in deviations])
        return {"equipment_id": request.equipment_id, "target_timestamp": target.isoformat(),
                "predicted_anomaly_score": round(score, 3), "risk_level": level,
                "recommended_action": action, "top_deviations": deviations,
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
        drift = monitor.observe(request.equipment_id, pairs)
        # 월 경계에 걸친 배치는 중앙 시점의 달을 기준 분포로 쓴다 (7/27~8/3 배치 → 7월 기준)
        diagnostics = diagnose(model.profile, request.records[len(request.records) // 2].timestamp, raw)
        live.record_predictions(request.equipment_id,
                                [{"target_timestamp": p["timestamp"], "predicted": p["predicted"],
                                  "actual": p["actual"]} for p in pairs], "batch-test",
                                extra={"suspects": diagnostics.get("top_sensors", [])}, count_today=False)
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
                "retrain": retrain.status(), "ops": ops.snapshot(), "alerts": alerts.snapshot()}

    @app.get("/monitoring/dashboard", tags=["aiops"])
    def monitoring_dashboard():
        """실시간 대시보드용 통합 스냅샷: 최근 3분 요청·지연, 오늘 시간대별 예측·HIGH, 장비 상태, 이벤트."""
        model = manager.model
        return {**live.snapshot(),
                "model": {"version": model.version if model else None, "source": manager.source,
                          "loading_mode": mode},
                "gate": {"rmse_max": GATE_RMSE_MAX, "recall_min": GATE_RECALL_MIN,
                         "precision_min": GATE_PRECISION_MIN},
                "drift": monitor.snapshot(), "last_batch": dict(sorted(last_batch.items())),
                "retrain": retrain.status(), "ops": ops.snapshot(), "alerts": alerts.snapshot()}

    @app.get("/monitoring/models", tags=["aiops"])
    def model_history(limit: int = Query(10, ge=1, le=50)):
        """MLflow Registry 버전별 게이트 결과·검증 지표(최신순). 실패 버전도 기록으로 남는다. MODEL_SOURCE=local이면 빈 목록."""
        if manager.source != "mlflow":
            return {"source": manager.source, "production": None, "versions": []}
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(tracking_uri())
        client = MlflowClient()
        production = client.get_registered_model(MODEL_NAME).aliases.get(ALIAS)
        versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version),
                          reverse=True)[:limit]
        rows = []
        for v in versions:
            if v.run_id not in run_metrics:
                run_metrics[v.run_id] = client.get_run(v.run_id).data.metrics
            m = run_metrics[v.run_id]
            rows.append({"version": int(v.version), "mode": v.tags.get("mode", "base"),
                         "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(timespec="seconds"),
                         "gate_passed": v.tags.get("gate_passed") == "True", "production": str(v.version) == str(production),
                         "rmse": m.get("validation_rmse"), "recall": m.get("validation_recall"),
                         "precision": m.get("validation_precision"), "champion_rmse": m.get("champion_holdout_rmse"),
                         "golden_rmse": m.get("golden_normal_rmse")})
        return {"source": manager.source, "production": production,
                "gate": {"rmse_max": GATE_RMSE_MAX, "recall_min": GATE_RECALL_MIN, "precision_min": GATE_PRECISION_MIN},
                "versions": rows}

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
        return FileResponse(ROOT / "semiconductor/static/dashboard.html")

    @app.get("/console", include_in_schema=False)
    def console():
        return FileResponse(ROOT / "semiconductor/static/index.html")

    return app


app = create_app()
