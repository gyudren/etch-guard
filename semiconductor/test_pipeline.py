"""Day1~3 통합 테스트. 실제 상태를 건드리지 않도록 임시 디렉터리·임시 MLflow DB에서 실행한다.

python -m unittest semiconductor.test_pipeline -v      (base 학습 후 실행: 번들이 필요)
"""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SOURCE_STATE = Path(__file__).resolve().parents[1] / "semiconductor_state"
_TMP = Path(tempfile.mkdtemp(prefix="etchguard-test-"))
_BASE = max((b for b in (_SOURCE_STATE / "bundles").iterdir()
             if json.loads((b / "metrics.json").read_text()).get("mode", "base") == "base"), key=lambda b: b.name)
shutil.copytree(_BASE, _TMP / "state/bundles" / _BASE.name)  # 항상 base 모델에서 출발
(_TMP / "state/local.json").write_text(json.dumps({"bundle": _BASE.name}))
os.environ.update(SEMICONDUCTOR_STATE_DIR=str(_TMP / "state"), AIOPS_LOG_DIR=str(_TMP / "logs"),
                  MLFLOW_TRACKING_URI=f"sqlite:///{_TMP / 'state/mlflow.db'}", RETRAIN_ASYNC="0")

from fastapi.testclient import TestClient  # noqa: E402

from .config import STATE, LOG_DIR, MODEL_NAME, ALIAS, DEFAULT_DATA, tracking_uri  # noqa: E402
from .data import read_data, partitions  # noqa: E402
from .monitoring.drift_detector import DriftMonitor, compute_rmse, is_drift  # noqa: E402
from .runtime import ModelManager  # noqa: E402
from .train import deployment_gate, register_bundle  # noqa: E402


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pointer = json.loads((STATE / "local.json").read_text())
        cls.bundle = STATE / "bundles" / pointer["bundle"]
        cls.example = json.loads((cls.bundle / "example.json").read_text())
        cls.metrics = json.loads((cls.bundle / "metrics.json").read_text())
        result = dict(cls.metrics, gate=deployment_gate(cls.metrics["validation"]))
        cls.registered = register_bundle(cls.bundle, result)  # 임시 Registry에 Production v1

    def client(self, source="local", mode="lazy"):
        from .app import create_app
        with patch.dict(os.environ, {"MODEL_SOURCE": source, "LOADING_MODE": mode}):
            return TestClient(create_app())

    # ---------- Day1: 서빙·입력 검증·Lazy/Eager ----------
    def test_lazy_local_prediction_and_input_validation(self):
        with self.client() as c:
            self.assertFalse(c.get("/health").json()["model_loaded"])
            self.assertEqual(c.get("/ready").status_code, 503)
            result = c.post("/predict", json=self.example)
            self.assertEqual(result.status_code, 200, result.text)
            body = result.json()
            self.assertTrue(0 <= body["predicted_anomaly_score"] <= 100)
            self.assertIn(body["risk_level"], ("NORMAL", "WARNING", "HIGH"))
            self.assertTrue(c.get("/health").json()["model_loaded"])
            self.assertEqual(c.get("/ready").status_code, 200)
            self.assertEqual(c.get("/").status_code, 200)
            self.assertEqual(c.get("/docs").status_code, 200)
            mutations = []
            short = copy.deepcopy(self.example); short["sequence"].pop(); mutations.append(short)
            negative = copy.deepcopy(self.example); negative["sequence"][0]["chamber_pressure_mtorr"] = -1; mutations.append(negative)
            shuffled = copy.deepcopy(self.example); shuffled["sequence"].reverse(); mutations.append(shuffled)
            leaky = copy.deepcopy(self.example); leaky["sequence"][0]["anomaly_score"] = 80; mutations.append(leaky)
            for bad in mutations:
                self.assertEqual(c.post("/predict", json=bad).status_code, 422)

    def test_missing_model_is_503(self):
        with self.client() as c, patch.object(ModelManager, "get", side_effect=FileNotFoundError("expected")):
            self.assertEqual(c.post("/predict", json=self.example).status_code, 503)

    def test_upload_rejects_stock_and_invalid_csv(self):
        with self.client() as c:
            for text in ("Date,Close,Volume\n2025-01-01,123,456\n", "timestamp,equipment_id\nBAD,A\n"):
                self.assertEqual(c.post("/data/upload", files={"file": ("bad.csv", text)}).status_code, 400)

    def test_time_splits_and_no_cross_equipment(self):
        groups = read_data(DEFAULT_DATA)
        self.assertEqual(sum(len(g["x"]) for g in groups.values()), 263040)
        for g in groups.values():
            p = partitions(g)
            self.assertEqual(g["time"][p["train"][1]-1].year, 2024)
            self.assertEqual(p["validation"][1], p["test"][0])
            self.assertEqual(p["test"][1], p["drift"][0])

    # ---------- Day2: MLflow 게이트·Production ----------
    def test_eager_and_registry_parity(self):
        with self.client(mode="eager") as c:
            self.assertTrue(c.get("/health").json()["model_loaded"])
            local = c.post("/predict", json=self.example).json()
        with self.client(source="mlflow", mode="eager") as c:
            remote = c.post("/predict", json=self.example)
            self.assertEqual(remote.status_code, 200, remote.text)
            self.assertTrue(remote.json()["model_version"].startswith("production-v"))
            self.assertAlmostEqual(local["predicted_anomaly_score"], remote.json()["predicted_anomaly_score"], places=3)

    def test_nonfinite_gate_rejected(self):
        self.assertFalse(deployment_gate({"rmse": float("nan"), "recall": 1, "precision": 1})["passed"])

    def test_failed_gate_keeps_production(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(tracking_uri())
        client = MlflowClient()
        before = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
        result = dict(self.metrics, gate=deployment_gate(self.metrics["validation"], rmse_gate=0.0))
        self.assertFalse(register_bundle(self.bundle, result)["promoted"])
        self.assertEqual(before, client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)

    # ---------- Day3: 드리프트 판정 단위 테스트 ----------
    def test_compute_rmse(self):
        self.assertEqual(compute_rmse([]), 0.0)
        self.assertAlmostEqual(compute_rmse([{"predicted": 1, "actual": 4}, {"predicted": 2, "actual": 6}]), (12.5) ** 0.5)

    def test_transient_burst_does_not_trigger_but_sustained_shift_does(self):
        self.assertFalse(is_drift([9.0, 9.0], 5.0, 3))
        self.assertTrue(is_drift([1.0, 6.0, 7.0, 8.0], 5.0, 3))
        monitor = DriftMonitor(window=4, threshold=5.0, consecutive=3)
        pair = lambda i, err: {"timestamp": f"t{i}", "predicted": 50.0, "actual": 50.0 + err}
        burst = [pair(i, 9 if 4 <= i < 8 else 1) for i in range(16)]       # 윈도우 1개만 초과
        self.assertFalse(monitor.observe("E", burst)["drift"])
        shift = [pair(16 + i, 8) for i in range(12)]                       # 3개 연속 초과
        result = monitor.observe("E", shift)
        self.assertTrue(result["drift"])
        self.assertEqual(result["detected_at"], "t27")
        monitor.reset()
        self.assertEqual(monitor.snapshot()["equipment"], {})

    def test_batch_test_validation_and_log_endpoints(self):
        with self.client() as c:
            batch = c.get("/data/batch", params={"equipment_id": "ETCH-01", "start": "2025-06-01T00:00:00",
                                                 "cycles": 30}).json()
            self.assertEqual(len(batch["records"]), 50)
            no_label = copy.deepcopy(batch); no_label["records"][-1].pop("anomaly_score")
            short = copy.deepcopy(batch); short["records"] = short["records"][:20]
            for bad in (no_label, short):
                self.assertEqual(c.post("/predict/batch-test", json=bad).status_code, 422)
            ok = c.post("/predict/batch-test", json=batch)
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertEqual(ok.json()["n_predictions"], 30)
            self.assertEqual(c.get("/logs/..%2Fconfig.py").status_code, 404)
            self.assertEqual(c.get("/logs/.hidden").status_code, 400)
            self.assertEqual(c.get("/data/batch", params={"equipment_id": "NOPE", "start": "2025-06-01T00:00:00"}).status_code, 404)

    # ---------- Day3: 드리프트 → 알림 → 재학습 → 게이트 → 교체 (마지막에 실행) ----------
    def test_z_drift_triggers_retrain_and_hot_swap(self):
        with self.client(mode="eager") as c:
            before = c.get("/health").json()["model_version"]
            fetch = lambda start: c.get("/data/batch", params={"equipment_id": "ETCH-01", "start": start,
                                                               "cycles": 336}).json()
            normal = c.post("/predict/batch-test", json=fetch("2025-06-01T00:00:00")).json()
            self.assertEqual(normal["drift_check"]["status"], "ok")
            drift = c.post("/predict/batch-test", json=fetch("2025-07-20T14:00:00")).json()
            self.assertEqual(drift["drift_check"]["status"], "retrain_triggered")
            self.assertIn("rf_power_w", [s["sensor"] for s in drift["input_diagnostics"]["top_sensors"]])
            record = c.get("/monitoring/status").json()["retrain"]["last_result"]
            log = (LOG_DIR / "aiops.log").read_text()
            order = [log.index("[WARN] drift detected"), log.index("[INFO] retrain triggered")]
            self.assertEqual(order, sorted(order))
            after = c.get("/health").json()["model_version"]
            if record["promoted"]:
                self.assertIn("[OK] new_rmse=", log)
                self.assertNotEqual(before, after)
                self.assertTrue(record["checks"]["beats_champion"])
                self.assertEqual(c.get("/monitoring/status").json()["drift"]["equipment"], {})  # 윈도우 초기화
            else:
                self.assertIn("[FAIL]", log)
                self.assertEqual(before, after)
            self.assertEqual(c.get("/logs").status_code, 200)
            self.assertIn("aiops.log", [f["name"] for f in c.get("/logs").json()])


if __name__ == "__main__":
    unittest.main()
