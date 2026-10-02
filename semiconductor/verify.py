"""HTTP 스모크 검증 + Lazy/Eager 측정. 결과 요약을 화면에 출력한다(파일로 남기지 않음).

# 이미 떠 있는 서버 검증 (Docker 등)
python -m semiconductor.verify --label docker --url http://127.0.0.1:8000
# 서버 프로세스를 직접 띄워 기동 시간까지 측정 (새 프로세스 = 콜드 스타트)
python -m semiconductor.verify --spawn --mode lazy  --source local --label lazy
python -m semiconductor.verify --spawn --mode eager --source local --label eager
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .config import ROOT


def request(base, path, body=None):
    raw = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=raw,
                                 headers={"Content-Type": "application/json"} if raw else {})
    start = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        payload = json.loads(response.read())
        return {"status": response.status, "seconds": round(time.perf_counter()-start, 4), "body": payload}


def spawn(mode, source, port):
    env = dict(os.environ, LOADING_MODE=mode, MODEL_SOURCE=source, TF_CPP_MIN_LOG_LEVEL="2")
    started = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "semiconductor.app:app", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT, env=env)
    url = f"http://127.0.0.1:{port}"
    while True:
        if proc.poll() is not None:
            raise SystemExit("server exited during startup")
        try:
            urllib.request.urlopen(url + "/health", timeout=1).close()
            return proc, url, round(time.perf_counter() - started, 3)
        except OSError:
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="http")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--mode", default="lazy", choices=["lazy", "eager"])
    parser.add_argument("--source", default="local", choices=["local", "mlflow"])
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args()
    proc, evidence = None, {}
    if args.spawn:
        proc, args.url, evidence["process_start_to_first_health_seconds"] = spawn(args.mode, args.source, args.port)
    try:
        evidence["health_before"] = request(args.url, "/health")
        example = request(args.url, "/data/example")
        assert example["status"] == 200, example
        body = example["body"]
        evidence["first_predict"] = request(args.url, "/predict", body)
        evidence["second_predict"] = request(args.url, "/predict", body)
        assert evidence["first_predict"]["status"] == evidence["second_predict"]["status"] == 200
        cases = {"19_points": lambda b: b["sequence"].pop(),
                 "negative_pressure": lambda b: b["sequence"][0].update(chamber_pressure_mtorr=-1),
                 "reversed_time": lambda b: b["sequence"].reverse(),
                 "label_leak_field": lambda b: b["sequence"][0].update(anomaly_score=80)}
        evidence["invalid_inputs"] = {}
        for name, mutate in cases.items():
            bad = copy.deepcopy(body)
            mutate(bad)
            evidence["invalid_inputs"][name] = request(args.url, "/predict", bad)
            assert evidence["invalid_inputs"][name]["status"] == 422, name
        evidence["health_after"] = request(args.url, "/health")
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)
    summary = {"label": args.label, "loading_mode": evidence["health_after"]["body"]["loading_mode"],
               "model_version": evidence["health_after"]["body"]["model_version"],
               "process_start_to_first_health_s": evidence.get("process_start_to_first_health_seconds"),
               "lifespan_startup_s": round(evidence["health_before"]["body"]["startup_seconds"], 4),
               "first_predict_s": evidence["first_predict"]["seconds"],
               "second_predict_s": evidence["second_predict"]["seconds"],
               "invalid_input_statuses": {k: v["status"] for k, v in evidence["invalid_inputs"].items()}}
    evidence["summary"] = summary
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
