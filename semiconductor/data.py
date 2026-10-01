"""Shared CSV validation and feature scaling; no TensorFlow import for lazy startup."""
import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import FEATURES, SEQ_LEN, latest_data


def read_data(source):
    text = source.read_text(encoding="utf-8-sig") if isinstance(source, Path) else source
    reader = csv.DictReader(io.StringIO(text))
    required = {*FEATURES, "timestamp", "equipment_id", "cycle_index", "anomaly_score", "is_drift"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
    series = {}
    for line, row in enumerate(reader, 2):
        try:
            eid = row["equipment_id"].strip()
            if not eid:
                raise ValueError("equipment_id is empty")
            timestamp = datetime.fromisoformat(row["timestamp"])
            if timestamp.tzinfo is not None:
                raise ValueError("Use timezone-naive simulation timestamps")
            cycle = int(row["cycle_index"])
            x = [float(row[k]) for k in FEATURES]
            y = float(row["anomaly_score"])
            drift = int(row["is_drift"])
            if not np.isfinite(x + [y]).all() or not 0 <= y <= 100 or drift not in (0, 1):
                raise ValueError("Non-finite value or invalid score/drift flag")
            validate_features(x)
            group = series.setdefault(eid, {"x": [], "y": [], "time": [], "cycle": [], "drift": []})
            if cycle < 1 or (group["time"] and (
                timestamp - group["time"][-1] != timedelta(minutes=30)
                or cycle != group["cycle"][-1] + 1
            )):
                raise ValueError("Each equipment requires consecutive cycles at 30-minute intervals")
            if group["drift"] and group["drift"][-1] > drift:
                raise ValueError("Drift region must be a suffix")
            for key, value in zip(("x", "y", "time", "cycle", "drift"), (x, y, timestamp, cycle, drift)):
                group[key].append(value)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"CSV line {line}: {exc}") from exc
    if not series or any(len(g["x"]) <= SEQ_LEN for g in series.values()):
        raise ValueError("Each equipment needs at least 21 rows")
    for g in series.values():
        g["x"] = np.asarray(g["x"], dtype="float32")
        g["y"] = np.asarray(g["y"], dtype="float32")
    return series


def validate_features(values):
    p = dict(zip(FEATURES, values))
    if not np.isfinite(values).all():
        raise ValueError("Sensors must be finite")
    if not 0 <= p["ambient_humidity_pct"] <= 100:
        raise ValueError("Humidity must be 0..100")
    for k in ("chamber_pressure_mtorr", "rf_power_w", "process_time_sec"):
        if p[k] <= 0:
            raise ValueError(f"{k} must be positive")
    for k in ("cf4_flow_sccm", "o2_flow_sccm", "vibration_mm_s", "particle_count", "usage_count"):
        if p[k] < 0:
            raise ValueError(f"{k} must be nonnegative")


def summary(series):
    return {"rows": sum(len(g["x"]) for g in series.values()),
            "equipment": {k: len(g["x"]) for k, g in series.items()},
            "start": min(g["time"][0] for g in series.values()).isoformat(),
            "end": max(g["time"][-1] for g in series.values()).isoformat(),
            "interval_minutes": 30, "features": FEATURES}


def partitions(g):
    # Dataset scenario markers determine evaluation cohorts, never model features.
    cuts = [i for i, t in enumerate(g["time"]) if t >= datetime(2025, 1, 1)]
    train_end = cuts[0] if cuts else len(g["x"])
    drift_start = next((i for i, d in enumerate(g["drift"]) if d), len(g["x"]))
    val_end = train_end + (drift_start - train_end) // 2
    if train_end <= SEQ_LEN or val_end <= train_end or drift_start <= val_end or drift_start >= len(g["x"]):
        raise ValueError("Training needs pre-2025 rows, 2025 normal validation/test, and a drift suffix")
    return {"train": (SEQ_LEN, train_end), "validation": (train_end, val_end),
            "test": (val_end, drift_start), "drift": (drift_start, len(g["x"]))}


def fit_scaler(series):
    x = np.concatenate([g["x"][:partitions(g)["train"][1]] for g in series.values()])
    return {"features": FEATURES, "sequence_length": SEQ_LEN,
            "feature_min": x.min(axis=0).tolist(), "feature_max": x.max(axis=0).tolist(),
            "target_min": 0.0, "target_max": 100.0}


def load_scaler(path):
    s = json.loads(Path(path).read_text())
    if s["features"] != FEATURES or s["sequence_length"] != SEQ_LEN:
        raise ValueError("Model feature contract mismatch")
    lo, hi = np.array(s["feature_min"]), np.array(s["feature_max"])
    if lo.shape != (len(FEATURES),) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
        raise ValueError("Invalid scaler")
    return s


def transform(x, scaler):
    lo = np.array(scaler["feature_min"], dtype="float32")
    span = np.array(scaler["feature_max"], dtype="float32") - lo
    return (np.asarray(x, dtype="float32") - lo) / np.where(span == 0, 1, span)


_series_cache = {}


def load_series(path=None):
    """업로드 저장소의 최신 CSV를 파싱해 캐시한다. 파일이 바뀌면(mtime) 다시 읽는다."""
    path = Path(path or latest_data()).resolve()
    key = (str(path), path.stat().st_mtime_ns)
    if _series_cache.get("key") != key:
        series = read_data(path)
        _series_cache.clear()
        _series_cache.update(key=key, series=series)
    return path, _series_cache["series"]


def to_records(g, start, end, with_label=True):
    """장비 시계열 구간을 API 페이로드 형식으로 변환한다."""
    rows = []
    for i in range(start, end):
        row = {"timestamp": g["time"][i].isoformat(), **dict(zip(FEATURES, map(float, g["x"][i])))}
        if with_label:
            row["anomaly_score"] = float(g["y"][i])
        rows.append(row)
    return rows


def windows(x_scaled, start, end):
    """target 인덱스 start..end-1 각각에 대해 직전 SEQ_LEN 사이클 입력을 쌓는다."""
    return np.stack([x_scaled[i - SEQ_LEN:i] for i in range(start, end)]).astype("float32")
