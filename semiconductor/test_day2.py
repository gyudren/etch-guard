"""Run: python -m unittest semiconductor.test_day2 -v (after training/registration)."""
import copy
import csv
import io
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from .config import STATE, FEATURES, MODEL_NAME, ALIAS, tracking_uri
from .data import read_data, partitions
from .runtime import ModelManager
from .train import deployment_gate


class Day2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pointer = json.loads((STATE / "local.json").read_text())
        cls.bundle = STATE / "bundles" / pointer["bundle"]
        cls.example = json.loads((cls.bundle / "example.json").read_text())

    def client(self, source="local", mode="lazy"):
        from .app import create_app
        with patch.dict(os.environ, {"MODEL_SOURCE": source, "LOADING_MODE": mode}):
            return TestClient(create_app())

    def test_lazy_local_prediction_and_input_validation(self):
        with self.client() as c:
            self.assertFalse(c.get("/health").json()["model_loaded"])
            self.assertEqual(c.get("/ready").status_code, 503)
            result = c.post("/predict", json=self.example)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertTrue(c.get("/health").json()["model_loaded"])
            self.assertEqual(c.get("/ready").status_code, 200)
            self.assertGreaterEqual(result.json()["predicted_anomaly_score"], 0)
            self.assertLessEqual(result.json()["predicted_anomaly_score"], 100)
            self.assertIn("model_version", result.json())
            self.assertEqual(c.get("/").status_code, 200)
            self.assertEqual(c.get("/docs").status_code, 200)
            mutations = []
            short = copy.deepcopy(self.example); short["sequence"].pop(); mutations.append(short)
            negative = copy.deepcopy(self.example); negative["sequence"][0]["chamber_pressure_mtorr"] = -1; mutations.append(negative)
            shuffled = copy.deepcopy(self.example); shuffled["sequence"].reverse(); mutations.append(shuffled)
            leaky = copy.deepcopy(self.example); leaky["sequence"][0]["anomaly_score"] = 80; mutations.append(leaky)
            for bad in mutations:
                self.assertEqual(c.post("/predict", json=bad).status_code, 422)

    def test_eager_and_registry_parity(self):
        with self.client(mode="eager") as c:
            self.assertTrue(c.get("/health").json()["model_loaded"])
            local = c.post("/predict", json=self.example).json()
        with self.client(source="mlflow", mode="eager") as c:
            remote = c.post("/predict", json=self.example)
            self.assertEqual(remote.status_code, 200, remote.text)
            self.assertTrue(remote.json()["model_version"].startswith("production-v"))
            self.assertAlmostEqual(local["predicted_anomaly_score"], remote.json()["predicted_anomaly_score"], places=3)

    def test_missing_model_is_503(self):
        with self.client() as c, patch.object(ModelManager, "get", side_effect=FileNotFoundError("expected test failure")):
            self.assertEqual(c.post("/predict", json=self.example).status_code, 503)

    def test_upload_rejects_stock_and_invalid_csv(self):
        with self.client() as c:
            for text in ("Date,Close,Volume\n2025-01-01,123,456\n", "timestamp,equipment_id\nBAD,A\n"):
                self.assertEqual(c.post("/data/upload", files={"file": ("bad.csv", text)}).status_code, 400)

    def test_time_splits_and_no_cross_equipment(self):
        from .config import DEFAULT_DATA
        groups = read_data(DEFAULT_DATA)
        self.assertEqual(len(groups), 5)
        self.assertEqual(sum(len(g["x"]) for g in groups.values()), 263040)
        for g in groups.values():
            p = partitions(g)
            self.assertEqual(g["time"][p["train"][1]-1].year, 2024)
            self.assertEqual(p["train"][1], p["validation"][0])
            self.assertEqual(p["validation"][1], p["test"][0])
            self.assertEqual(p["test"][1], p["drift"][0])

    def test_nonfinite_gate_rejected(self):
        m = {"validation": {"rmse": float("nan"), "recall": 1, "precision": 1}}
        self.assertFalse(deployment_gate(m)["passed"])

    def test_failed_gate_keeps_production(self):
        # Exercise real registry logic without logging another large model artifact.
        from .train import register_bundle
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(tracking_uri())
        client = MlflowClient()
        before = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        result = json.loads((self.bundle / "metrics.json").read_text())
        result["gate"] = deployment_gate(result, rmse_gate=0.0)
        self.assertFalse(result["gate"]["passed"])
        registered = register_bundle(self.bundle, result)
        self.assertFalse(registered["promoted"])
        self.assertEqual(before.version, client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)


if __name__ == "__main__":
    unittest.main()
