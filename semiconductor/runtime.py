"""모델 로딩: 모델과 그 모델의 스케일러·기준 분포는 하나의 번들로 함께 움직인다.

MODEL_SOURCE=local  → semiconductor_state/local.json이 가리키는 번들
MODEL_SOURCE=mlflow → models:/Semiconductor_Etch_LSTM@Production (alias)
main(app.py)·train.py는 그대로 두고 이 파일만으로 소스를 바꾸는 '조립 블록' 구조다.
"""
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
        self.directory = Path(directory)
        self.scaler = load_scaler(self.directory / "scaler.json")
        self.model = keras.models.load_model(self.directory / "model.keras", compile=False)
        profile = self.directory / "reference_profile.json"
        self.profile = json.loads(profile.read_text()) if profile.exists() else None
        self.version = version

    def predict(self, raw):
        y = self.model(transform(raw, self.scaler), training=False).numpy().reshape(-1)
        return np.clip(y * (self.scaler["target_max"] - self.scaler["target_min"])
                       + self.scaler["target_min"], 0, 100)


class RegisteredModel:
    """MLflow pyfunc를 감싸되, 재학습의 출발점이 될 번들 경로와 기준 분포를 함께 노출한다."""

    def __init__(self, pyfunc_model, version_number):
        self.pyfunc = pyfunc_model
        bundle = pyfunc_model.unwrap_python_model().bundle
        self.directory, self.profile = bundle.directory, bundle.profile
        self.version = f"production-v{version_number}"

    def predict(self, raw):
        return np.asarray(self.pyfunc.predict(np.asarray(raw, dtype="float32"))).reshape(-1)


class ModelManager:
    def __init__(self, source):
        if source not in ("local", "mlflow"):
            raise ValueError("MODEL_SOURCE must be local or mlflow")
        self.source, self.model, self.load_seconds = source, None, None
        self.lock = threading.Lock()

    def _load_from_local(self):
        pointer = json.loads((STATE / "local.json").read_text())
        return Bundle(STATE / "bundles" / pointer["bundle"], f"local-{pointer['bundle']}")

    def _load_from_mlflow(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(tracking_uri())
        v = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
        return RegisteredModel(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{v.version}"), v.version)

    def _load(self):
        return self._load_from_local() if self.source == "local" else self._load_from_mlflow()

    def get(self):
        """Lazy: 첫 요청에서 로드 후 캐시. Eager는 서버 시작 시 이 함수를 한 번 호출한다."""
        with self.lock:
            if self.model is None:
                start = time.perf_counter()
                self.model = self._load()
                self.load_seconds = time.perf_counter() - start
            return self.model

    def reload(self):
        """재학습 승격 후 무중단 교체: 새 모델을 락 밖에서 먼저 로드하고 참조만 원자적으로 바꾼다.
        (캐시를 비우기만 하면 다음 요청이 로딩 시간을 떠안고, 비우지 않으면 옛 모델이 계속 응답한다.)"""
        start = time.perf_counter()
        fresh = self._load()
        with self.lock:
            previous, self.model = self.model, fresh
            self.load_seconds = time.perf_counter() - start
        return previous.version if previous else None, fresh.version
