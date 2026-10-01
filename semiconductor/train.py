"""python -m semiconductor.train [--register]: train, evaluate, gate, version."""
import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import numpy as np

from .config import STATE, ROOT, FEATURES, SEQ_LEN, MODEL_NAME, ALIAS, latest_data, tracking_uri
from .data import read_data, summary, partitions, fit_scaler, transform


def metrics(y, p):
    actual, predicted = y >= 70, p >= 70
    tp, fp, fn = [int(v) for v in (np.sum(actual & predicted), np.sum(~actual & predicted), np.sum(actual & ~predicted))]
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"samples": len(y), "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "mae": float(np.mean(abs(y-p))), "precision": precision, "recall": recall,
            "f1": 2*precision*recall / max(precision+recall, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": int(np.sum(~actual & ~predicted))}


def deployment_gate(result, rmse_gate=4.0, recall_gate=0.8, precision_gate=0.8):
    v = result["validation"]
    checks = {"finite": bool(np.isfinite([v["rmse"], v["recall"], v["precision"]]).all()),
              "rmse": v["rmse"] <= rmse_gate,
              "high_recall": v["recall"] >= recall_gate,
              "high_precision": v["precision"] >= precision_gate}
    return {"passed": all(checks.values()), "checks": checks,
            "thresholds": {"rmse_max": rmse_gate, "recall_min": recall_gate, "precision_min": precision_gate},
            "cohort": "validation"}


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
        ds = ds.shuffle(128, seed=42)
    return ds.prefetch(tf.data.AUTOTUNE)


def register_bundle(bundle, result):
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from .mlflow_model import EtchModel
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Semiconductor-Day2")
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        "Semiconductor-Day2", artifact_location=(STATE / "artifacts").as_uri())
    example = np.array(json.loads((bundle / "example.json").read_text())["sequence"])
    raw = np.array([[[point[k] for k in FEATURES] for point in example]], dtype="float32")
    from .runtime import Bundle
    out = Bundle(bundle).predict(raw)
    with mlflow.start_run(experiment_id=experiment_id, run_name=bundle.name) as run:
        mlflow.log_params({"epochs": result["epochs_completed"], "sequence_length": SEQ_LEN,
                           "features": len(FEATURES), "dataset_sha256": result["dataset_sha256"],
                           "seed": 42, "data_type": "synthetic", "split": "calendar_train_2023_2024"})
        for split in ("validation", "test", "drift"):
            mlflow.log_metrics({f"{split}_{k}": v for k, v in result[split].items()})
        mlflow.log_dict(result, "evaluation.json")
        info = mlflow.pyfunc.log_model(
            name="etch_model", python_model=EtchModel(), artifacts={"bundle": str(bundle)},
            code_paths=[str(ROOT / "semiconductor")], signature=infer_signature(raw, out),
            input_example=raw,
            pip_requirements=["mlflow==3.16.0", "tensorflow==2.21.0", "numpy==2.4.4"])
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, version.version, "gate_passed", str(result["gate"]["passed"]))
        previous = client.get_registered_model(MODEL_NAME).aliases.get(ALIAS)
        if result["gate"]["passed"]:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        return {"run_id": run.info.run_id, "registered_version": version.version,
                "promoted": result["gate"]["passed"], "previous_production": previous}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--bundle", type=Path, help="Register an already evaluated bundle without retraining")
    parser.add_argument("--rmse-gate", type=float, default=4.0)
    args = parser.parse_args()
    if args.bundle:
        if not args.register:
            parser.error("--bundle requires --register")
        bundle = args.bundle.resolve()
        result = json.loads((bundle / "metrics.json").read_text())
        result["gate"] = deployment_gate(result, args.rmse_gate)
    else:
        if args.epochs < 1 or args.batch_size < 1:
            parser.error("epochs and batch-size must be positive")
        import tensorflow as tf
        from tensorflow import keras
        keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        path = (args.data or latest_data()).resolve()
        series = read_data(path)
        scaler = fit_scaler(series)
        datasets = {s: dataset(series, scaler, s, args.batch_size) for s in ("train", "validation", "test", "drift")}
        model = keras.Sequential([
            keras.layers.Input((SEQ_LEN, len(FEATURES))),
            keras.layers.LSTM(32, return_sequences=True), keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        history = model.fit(datasets["train"], validation_data=datasets["validation"],
                            epochs=args.epochs, shuffle=False, verbose=2,
                            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        result = {"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "data": summary(series), "epochs_completed": len(history.history["loss"]),
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
        result["gate"] = deployment_gate(result, args.rmse_gate)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        bundle = STATE / "bundles" / name
        bundle.mkdir(parents=True)
        model.save(bundle / "model.keras")
        (bundle / "scaler.json").write_text(json.dumps(scaler, indent=2))
        (bundle / "metrics.json").write_text(json.dumps(result, indent=2))
        g = next(iter(series.values()))
        start = partitions(g)["test"][0]
        example = {"equipment_id": next(iter(series)), "sequence": [
            {"timestamp": g["time"][i].isoformat(), **dict(zip(FEATURES, map(float, g["x"][i])))}
            for i in range(start-SEQ_LEN, start)]}
        (bundle / "example.json").write_text(json.dumps(example, indent=2))
        # Atomic pointer update; existing readers keep their loaded version until restart.
        pointer = STATE / f"local-{uuid.uuid4().hex}.tmp"
        pointer.write_text(json.dumps({"bundle": name}))
        pointer.replace(STATE / "local.json")
    if args.register:
        result["registry"] = register_bundle(bundle, result)
        (bundle / "registration.json").write_text(json.dumps(result["registry"], indent=2))
    print(json.dumps({"bundle": str(bundle), "validation": result["validation"],
                      "test": result["test"], "drift": result["drift"], "gate": result["gate"],
                      "registry": result.get("registry")}, indent=2), flush=True)
    print("[GATE PASSED]" if result["gate"]["passed"] else "[GATE FAILED] existing Production preserved")


if __name__ == "__main__":
    main()
