import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .config import ROOT, STATE, FEATURES, SEQ_LEN, latest_data
from .data import read_data, summary, validate_features
from .runtime import ModelManager

logger = logging.getLogger("semiconductor")
SensorPoint = create_model(
    "SensorPoint", __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    timestamp=(datetime, ...), **{k: (float, ...) for k in FEATURES})


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    equipment_id: str = Field(min_length=1, max_length=80)
    sequence: list[SensorPoint] = Field(min_length=SEQ_LEN, max_length=SEQ_LEN)

    @model_validator(mode="after")
    def validate_sequence(self):
        for i, point in enumerate(self.sequence):
            validate_features([getattr(point, k) for k in FEATURES])
            if point.timestamp.tzinfo is not None:
                raise ValueError("Use timezone-naive simulation timestamps")
            if i and point.timestamp - self.sequence[i-1].timestamp != timedelta(minutes=30):
                raise ValueError("Exactly 30-minute chronological spacing is required")
        return self


def create_app():
    mode = os.getenv("LOADING_MODE", "lazy")
    if mode not in ("lazy", "eager"):
        raise ValueError("LOADING_MODE must be lazy or eager")
    manager = ModelManager(os.getenv("MODEL_SOURCE", "local"))

    @asynccontextmanager
    async def lifespan(app):
        start = time.perf_counter()
        if mode == "eager":
            await run_in_threadpool(manager.get)
        app.state.startup_seconds = time.perf_counter() - start
        yield

    app = FastAPI(title="EtchGuard — Semiconductor Day 1–2", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": manager.model is not None,
                "loading_mode": mode, "model_source": manager.source,
                "model_version": manager.model.version if manager.model else None,
                "startup_seconds": getattr(app.state, "startup_seconds", None),
                "model_load_seconds": manager.load_seconds}

    @app.get("/ready")
    def ready():
        if manager.model is None:
            raise HTTPException(503, "Model not loaded; lazy mode loads on first prediction")
        return {"ready": True, "model_version": manager.model.version}

    @app.post("/predict")
    def predict(request: PredictRequest):
        start = time.perf_counter()
        try:
            model = manager.get()
            score = float(model.predict([[[getattr(p, k) for k in FEATURES] for p in request.sequence]])[0])
        except Exception:
            logger.exception("Prediction/model loading failed")
            raise HTTPException(503, "Model unavailable. Check training/Production registration and server logs.")
        return {"equipment_id": request.equipment_id,
                "target_timestamp": (request.sequence[-1].timestamp + timedelta(minutes=30)).isoformat(),
                "predicted_anomaly_score": round(score, 3),
                "risk_level": "HIGH" if score >= 70 else "WARNING" if score >= 40 else "NORMAL",
                "defect_label": int(score >= 70), "model_version": model.version,
                "model_source": manager.source, "latency_ms": round((time.perf_counter()-start)*1000, 2),
                "simulation": True}

    @app.post("/data/upload")
    async def upload(file: UploadFile = File(...)):
        raw = await file.read(64*1024*1024 + 1)
        await file.close()
        if len(raw) > 64*1024*1024:
            raise HTTPException(413, "Maximum CSV size is 64 MiB")
        try:
            text = raw.decode("utf-8-sig")
            series = await run_in_threadpool(read_data, text)
        except (UnicodeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        folder = STATE / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"etch_{uuid.uuid4().hex}.csv"
        tmp = path.with_suffix(".tmp")
        await run_in_threadpool(tmp.write_bytes, raw)
        tmp.replace(path)
        return {"filename": path.name, **summary(series), "note": "Upload stores data; train separately."}

    @app.get("/data/status")
    def status():
        try:
            path = latest_data()
            return {"filename": path.name, **summary(read_data(path))}
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/data/example")
    def example():
        try:
            series = read_data(latest_data())
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        eid, g = next(iter(series.items()))
        return {"equipment_id": eid, "sequence": [
            {"timestamp": g["time"][i].isoformat(), **dict(zip(FEATURES, map(float, g["x"][i])))}
            for i in range(SEQ_LEN)]}

    @app.get("/")
    def home():
        return FileResponse(ROOT / "semiconductor/index.html")

    return app


app = create_app()
