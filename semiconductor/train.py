"""학습 → 평가 → 배포 게이트 → MLflow 등록/Production 승격, 그리고 드리프트 후 warm-start fine-tuning.

python -m semiconductor.train              # base 학습 (로컬 번들 생성)
python -m semiconductor.train --register   # base 학습 + MLflow 기록·게이트·승격
python -m semiconductor.train --register --bundle semiconductor_state/bundles/<이름> [--rmse-gate 0]
"""
import argparse
import bisect
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import numpy as np

from .config import (STATE, ROOT, FEATURES, SEQ_LEN, MODEL_NAME, ALIAS, EXPERIMENT, HIGH_SCORE,
                     GATE_RMSE_MAX, GATE_RECALL_MIN, GATE_PRECISION_MIN, RETRAIN_DAYS, RETRAIN_HOLDOUT,
                     FINE_TUNE_EPOCHS, FINE_TUNE_LR, latest_data, tracking_uri)
from .data import read_data, summary, partitions, fit_scaler, load_scaler, transform, windows
from .monitoring.data_drift import build_reference_profile

SEED = 42
MIN_RETRAIN_SAMPLES = 200
BUNDLE_FILES = ("model.keras", "scaler.json", "reference_profile.json", "metrics.json", "example.json")


def metrics(y, p):
    actual, predicted = y >= HIGH_SCORE, p >= HIGH_SCORE
    tp, fp, fn = [int(v) for v in (np.sum(actual & predicted), np.sum(~actual & predicted), np.sum(actual & ~predicted))]
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"samples": len(y), "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "mae": float(np.mean(abs(y-p))), "precision": precision, "recall": recall,
            "f1": 2*precision*recall / max(precision+recall, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": int(np.sum(~actual & ~predicted))}


def deployment_gate(cohort, rmse_gate=GATE_RMSE_MAX, recall_gate=GATE_RECALL_MIN,
                    precision_gate=GATE_PRECISION_MIN, extra=None):
    """배포 게이트. 하나라도 실패하면 Production 별칭을 옮기지 않는다(기존 버전 유지)."""
    checks = {"finite": bool(np.isfinite([cohort["rmse"], cohort["recall"], cohort["precision"]]).all()),
              "rmse": cohort["rmse"] <= rmse_gate,
              "high_recall": cohort["recall"] >= recall_gate,
              "high_precision": cohort["precision"] >= precision_gate,
              **(extra or {})}
    return {"passed": all(checks.values()), "checks": checks,
            "thresholds": {"rmse_max": rmse_gate, "recall_min": recall_gate, "precision_min": precision_gate}}


def build_model():
    from tensorflow import keras
    return keras.Sequential([
        keras.layers.Input((SEQ_LEN, len(FEATURES))),
        keras.layers.LSTM(32, return_sequences=True), keras.layers.LSTM(16),
        keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])


def predict_scores(model, x):
    return np.clip(model.predict(x, batch_size=2048, verbose=0).ravel() * 100, 0, 100)


def dataset(series, scaler, split, batch_size):
    from tensorflow import keras
    import tensorflow as tf
    ds = None
    for g in series.values():
        start, end = partitions(g)[split]
        x = transform(g["x"][start-SEQ_LEN:end], scaler)
        part = keras.utils.timeseries_dataset_from_array(
            x[:-1], g["y"][start:end] / 100.0, sequence_length=SEQ_LEN,
            batch_size=batch_size, shuffle=False)
        ds = part if ds is None else ds.concatenate(part)
    if split == "train":
        ds = ds.shuffle(128, seed=SEED)
    return ds.prefetch(tf.data.AUTOTUNE)


def _write_pointer(bundle_name):
    # 원자적 포인터 교체: 읽는 쪽은 항상 완성된 local.json만 본다.
    pointer = STATE / f"local-{uuid.uuid4().hex}.tmp"
    pointer.write_text(json.dumps({"bundle": bundle_name}))
    pointer.replace(STATE / "local.json")


def _new_bundle_dir():
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    bundle = STATE / "bundles" / name
    bundle.mkdir(parents=True)
    return bundle


def register_bundle(bundle, result):
    """번들(모델+스케일러+기준 분포)을 하나의 MLflow 버전으로 기록하고 게이트 통과 시에만 Production으로 승격."""
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from .mlflow_model import EtchModel
    from .runtime import Bundle
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        EXPERIMENT, artifact_location=(STATE / "artifacts").as_uri())
    example = json.loads((bundle / "example.json").read_text())["sequence"]
    raw = np.array([[[point[k] for k in FEATURES] for point in example]], dtype="float32")
    out = Bundle(bundle).predict(raw)
    mode = result.get("mode", "base")
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{mode}-{bundle.name}") as run:
        mlflow.log_params({"mode": mode, "sequence_length": SEQ_LEN, "features": len(FEATURES), "seed": SEED,
                           "data_type": "synthetic", **result.get("params", {})})
        for cohort in ("validation", "test", "drift", "champion_holdout", "golden_normal", "champion_golden"):
            if cohort in result:
                mlflow.log_metrics({f"{cohort}_{k}": v for k, v in result[cohort].items()})
        mlflow.log_metric("rmse", result["validation"]["rmse"])  # 게이트 기준 지표
        mlflow.log_dict(result, "evaluation.json")
        info = mlflow.pyfunc.log_model(
            name="etch_model", python_model=EtchModel(), artifacts={"bundle": str(bundle)},
            code_paths=[str(ROOT / "semiconductor")], signature=infer_signature(raw, out),
            input_example=raw,
            pip_requirements=["mlflow==3.16.0", "tensorflow==2.21.0", "numpy==2.4.4"])
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, version.version, "gate_passed", str(result["gate"]["passed"]))
        client.set_model_version_tag(MODEL_NAME, version.version, "mode", mode)
        previous = client.get_registered_model(MODEL_NAME).aliases.get(ALIAS)
        if result["gate"]["passed"]:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        return {"run_id": run.info.run_id, "registered_version": version.version,
                "promoted": result["gate"]["passed"], "previous_production": previous}


def train_base(path, epochs, batch_size, rmse_gate):
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    series = read_data(path)
    scaler = fit_scaler(series)  # 학습 구간으로 단 한 번만 fit. 이후 재학습에서도 재사용한다.
    datasets = {s: dataset(series, scaler, s, batch_size) for s in ("train", "validation", "test", "drift")}
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    history = model.fit(datasets["train"], validation_data=datasets["validation"],
                        epochs=epochs, shuffle=False, verbose=2,
                        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    result = {"mode": "base", "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "data": summary(series), "epochs_completed": len(history.history["loss"]),
              "params": {"epochs": len(history.history["loss"]), "learning_rate": 1e-3,
                         "split": "train_2023_2024|val|test|drift"},
              "split_ranges": {}, "persistence_baseline": {}}
    for split in ("validation", "test", "drift"):
        y = np.concatenate([g["y"][slice(*partitions(g)[split])] for g in series.values()])
        pred = np.clip(model.predict(datasets[split], verbose=0).ravel()*100, 0, 100)
        result[split] = metrics(y, pred)
        previous = np.concatenate([g["y"][partitions(g)[split][0]-1:partitions(g)[split][1]-1] for g in series.values()])
        result["persistence_baseline"][split] = metrics(y, previous)
    for eid, g in series.items():
        result["split_ranges"][eid] = {s: [g["time"][a].isoformat(), g["time"][b-1].isoformat()]
                                       for s, (a, b) in partitions(g).items()}
    result["gate"] = deployment_gate(result["validation"], rmse_gate)
    bundle = _new_bundle_dir()
    model.save(bundle / "model.keras")
    (bundle / "scaler.json").write_text(json.dumps(scaler, indent=2))
    profile = build_reference_profile(series, lambda g: partitions(g)["train"][1])
    (bundle / "reference_profile.json").write_text(json.dumps(profile))
    (bundle / "metrics.json").write_text(json.dumps(result, indent=2))
    g = next(iter(series.values()))
    start = partitions(g)["test"][0]
    example = {"equipment_id": next(iter(series)), "sequence": [
        {"timestamp": g["time"][i].isoformat(), **dict(zip(FEATURES, map(float, g["x"][i])))}
        for i in range(start-SEQ_LEN, start)]}
    (bundle / "example.json").write_text(json.dumps(example, indent=2))
    _write_pointer(bundle.name)
    return bundle, result


def _recent_samples(series, scaler, until):
    """감지 시점(until)까지 최근 RETRAIN_DAYS일, 장비별 시간순 앞 80% 학습 / 뒤 20% 검증."""
    since = until - timedelta(days=RETRAIN_DAYS)
    parts = {"x_train": [], "y_train": [], "x_hold": [], "y_hold": []}
    used = {}
    for eid, g in series.items():
        start = max(bisect.bisect_right(g["time"], since), SEQ_LEN)
        end = bisect.bisect_right(g["time"], until)
        if end - start < SEQ_LEN * 3:
            continue
        cut = start + int((end - start) * (1 - RETRAIN_HOLDOUT))
        x = transform(g["x"][:end], scaler)
        parts["x_train"].append(windows(x, start, cut)); parts["y_train"].append(g["y"][start:cut])
        parts["x_hold"].append(windows(x, cut, end)); parts["y_hold"].append(g["y"][cut:end])
        used[eid] = {"train": [g["time"][start].isoformat(), g["time"][cut-1].isoformat()],
                     "holdout": [g["time"][cut].isoformat(), g["time"][end-1].isoformat()]}
    if not used:
        raise ValueError(f"No equipment has enough labeled rows in the last {RETRAIN_DAYS} days before {until}")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    return out, used


def _golden_normal(series, scaler, step=10):
    """회귀 테스트용 고정 정상 기준셋: 2025 검증 구간을 10사이클 간격으로 표본 추출."""
    xs, ys = [], []
    for g in series.values():
        try:
            start, end = partitions(g)["validation"]
        except ValueError:
            continue
        x = transform(g["x"][:end], scaler)
        idx = range(start, end, step)
        xs.append(np.stack([x[i - SEQ_LEN:i] for i in idx])); ys.append(g["y"][list(idx)])
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)


def fine_tune(series, until, base_dir, base_label, register=True):
    """Production 가중치에서 이어서(warm start) 최근 데이터로 미세조정 → 게이트 재검증 → 통과 시 승격.

    - 스케일러는 base 번들의 것을 그대로 복사한다(재-fit 금지: 같은 센서 값이 다른 숫자로 들어가면
      기존 가중치가 의미를 잃는다).
    - 게이트 = 기본 게이트(최근 홀드아웃) + 현 Production보다 나아야 함 + 고정 정상 기준셋 회귀 없음.
    """
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    base_dir = Path(base_dir)
    scaler = load_scaler(base_dir / "scaler.json")
    model = keras.models.load_model(base_dir / "model.keras", compile=False)
    data, used = _recent_samples(series, scaler, until)
    if len(data["y_train"]) < MIN_RETRAIN_SAMPLES:
        raise ValueError(f"Only {len(data['y_train'])} training samples; need ≥{MIN_RETRAIN_SAMPLES}")
    xg, yg = _golden_normal(series, scaler)
    champion_hold = metrics(data["y_hold"], predict_scores(model, data["x_hold"]))
    champion_golden = metrics(yg, predict_scores(model, xg)) if xg is not None else None

    model.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR), loss="mse")
    model.fit(data["x_train"], data["y_train"] / 100.0, epochs=FINE_TUNE_EPOCHS, batch_size=64,
              shuffle=True, verbose=0)

    result = {"mode": "fine-tune", "base": base_label, "until": until.isoformat(), "windows": used,
              "params": {"base": base_label, "epochs": FINE_TUNE_EPOCHS, "learning_rate": FINE_TUNE_LR,
                         "retrain_days": RETRAIN_DAYS, "train_samples": len(data["y_train"]),
                         "holdout_samples": len(data["y_hold"])},
              "validation": metrics(data["y_hold"], predict_scores(model, data["x_hold"])),
              "champion_holdout": champion_hold}
    extra = {"beats_champion": result["validation"]["rmse"] < champion_hold["rmse"]}
    if xg is not None:
        result["golden_normal"] = metrics(yg, predict_scores(model, xg))
        result["champion_golden"] = champion_golden
        extra["no_regression_golden"] = result["golden_normal"]["rmse"] <= GATE_RMSE_MAX
    result["gate"] = deployment_gate(result["validation"], extra=extra)

    bundle = _new_bundle_dir()
    model.save(bundle / "model.keras")
    for name in ("scaler.json", "reference_profile.json", "example.json"):
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, bundle / name)
    (bundle / "metrics.json").write_text(json.dumps(result, indent=2))
    result["bundle"] = bundle.name
    if register:
        result["registry"] = register_bundle(bundle, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--bundle", type=Path, help="이미 평가된 번들을 재학습 없이 등록")
    parser.add_argument("--rmse-gate", type=float, default=GATE_RMSE_MAX)
    args = parser.parse_args()
    if args.bundle:
        if not args.register:
            parser.error("--bundle requires --register")
        bundle = args.bundle.resolve()
        result = json.loads((bundle / "metrics.json").read_text())
        result["gate"] = deployment_gate(result["validation"], args.rmse_gate)
    else:
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch-size must be positive")
        bundle, result = train_base((args.data or latest_data()).resolve(), args.epochs, args.batch_size, args.rmse_gate)
    if args.register:
        result["registry"] = register_bundle(bundle, result)
    print(json.dumps({"bundle": str(bundle), "validation": result["validation"],
                      "test": result.get("test"), "drift": result.get("drift"), "gate": result["gate"],
                      "registry": result.get("registry")}, indent=2), flush=True)
    v = result["validation"]
    if result["gate"]["passed"]:
        print(f"[GATE PASSED] rmse={v['rmse']:.3f} ≤ {result['gate']['thresholds']['rmse_max']} "
              f"recall={v['recall']:.3f} precision={v['precision']:.3f}")
    else:
        failed = [k for k, ok in result["gate"]["checks"].items() if not ok]
        print(f"[GATE FAILED] failed={failed} rmse={v['rmse']:.3f} → 기존 Production 유지")


if __name__ == "__main__":
    main()
