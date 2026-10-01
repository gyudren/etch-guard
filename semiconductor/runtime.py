"""A model and its own scaler travel together for local and registry serving."""
import json
import threading
import time
from pathlib import Path

import numpy as np

from .config import STATE, MODEL_NAME, ALIAS, tracking_uri
from .data import load_scaler, transform


class Bundle:
    def __init__(self, directory, version="local"):
        from tensorflow import keras
        directory = Path(directory)
        self.scaler = load_scaler(directory / "scaler.json")
        self.model = keras.models.load_model(directory / "model.keras", compile=False)
        self.version = version

    def predict(self, raw):
        y = self.model(transform(raw, self.scaler), training=False).numpy().reshape(-1)
        return np.clip(y * (self.scaler["target_max"] - self.scaler["target_min"])
                       + self.scaler["target_min"], 0, 100)


class ModelManager:
    def __init__(self, source):
        if source not in ("local", "mlflow"):
            raise ValueError("MODEL_SOURCE must be local or mlflow")
        self.source, self.model, self.load_seconds = source, None, None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self.model is None:
                start = time.perf_counter()
                if self.source == "local":
                    pointer = json.loads((STATE / "local.json").read_text())
                    self.model = Bundle(STATE / "bundles" / pointer["bundle"], pointer["bundle"])
                else:
                    import mlflow
                    from mlflow.tracking import MlflowClient
                    mlflow.set_tracking_uri(tracking_uri())
                    v = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
                    class Registered:
                        version = f"production-v{v.version}"
                        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{v.version}")

                        def predict(self, raw):
                            return self.model.predict(np.asarray(raw, dtype="float32"))
                    self.model = Registered()
                self.load_seconds = time.perf_counter() - start
            return self.model
