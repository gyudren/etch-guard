"""Day3 드리프트 시뮬레이션: 정상 → 계절 변화(여름) → 드리프트 주입 → 재학습 후 순서로 배치를 보낸다.

각 배치는 장비 1대의 7일치(336사이클) 확정 이상 점수 기록 + 앞 20사이클 문맥이다.
서버(/predict/batch-test)는 12시간 윈도우 RMSE를 누적해 3회 연속 4.0 초과 시 드리프트로 판정한다.

사전 준비: 서버 실행 (예: MODEL_SOURCE=mlflow LOADING_MODE=eager uvicorn semiconductor.app:app --port 8000)
실행:     python scripts/simulate_drift.py [--url http://127.0.0.1:8000]
"""
import argparse
import json
import time
from pathlib import Path

import requests

EQUIPMENT = ["ETCH-01", "ETCH-02", "ETCH-03", "ETCH-04", "ETCH-05"]
WEEK = 336
SCENARIO = [
    ("normal", "2025-06-01T00:00:00", "정상 운전 (2025년 6월, 학습에 쓰지 않은 기간)"),
    ("season", "2025-07-08T00:00:00", "계절 변화: 장마·고온 (재학습이 일어나면 안 됨)"),
    ("drift_injection", "2025-07-20T14:00:00", "드리프트: 소모품 노후·레시피 변경 (압력↑, RF↓, 진동↑)"),
    ("after_retrain", "2025-07-27T14:00:00", "재학습 이후 다음 주 (새 Production 성능 확인)"),
]


def fetch_batch(url, equipment_id, start):
    resp = requests.get(f"{url}/data/batch", params={"equipment_id": equipment_id, "start": start, "cycles": WEEK},
                        timeout=60)
    resp.raise_for_status()
    return resp.json()


def send_batch(url, batch, label):
    """배치를 /predict/batch-test 로 보내고 drift_check를 출력한다."""
    started = time.perf_counter()
    resp = requests.post(f"{url}/predict/batch-test", json=batch, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    check, diag = result["drift_check"], result["input_diagnostics"]
    suspects = ", ".join(f"{s['sensor']} x{s['ratio']}" for s in diag.get("top_sensors", []))
    print(f"[{label}] {batch['equipment_id']} n={result['n_predictions']} batch_rmse={result['batch_rmse']:.2f} "
          f"last_window={check['latest_window_rmse']} streak={check['consecutive_over']} "
          f"status={check['status']} model={result['model_version']} ({time.perf_counter() - started:.1f}s)"
          + (f" | suspects: {suspects}" if check["status"] != "ok" else ""))
    return {k: v for k, v in result.items() if k != "predictions"}


def wait_for_retrain(url, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = requests.get(f"{url}/monitoring/status", timeout=30).json()
        if not status["retrain"]["running"]:
            return status
        time.sleep(2)
    raise TimeoutError("retrain did not finish in time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="evidence/05_simulate_drift.json")
    parser.add_argument("--phases", default=",".join(label for label, _, _ in SCENARIO),
                        help="쉼표로 구분한 실행 단계 (기본: 전체)")
    args = parser.parse_args()
    evidence = {"health_before": requests.get(f"{args.url}/health", timeout=30).json(), "phases": []}
    print(f"[0] serving {evidence['health_before']['model_version']} ({evidence['health_before']['model_source']})")
    for label, start, description in [s for s in SCENARIO if s[0] in args.phases.split(",")]:
        print(f"\n=== {label}: {description} ===")
        phase = {"label": label, "start": start, "description": description, "results": []}
        for equipment_id in EQUIPMENT:
            phase["results"].append(send_batch(args.url, fetch_batch(args.url, equipment_id, start), label))
        status = wait_for_retrain(args.url)
        phase["retrain_after_phase"] = status["retrain"]["last_result"]
        evidence["phases"].append(phase)
        if label == "drift_injection":
            print(f"[retrain] {json.dumps(status['retrain']['last_result'], ensure_ascii=False)}")
    evidence["health_after"] = requests.get(f"{args.url}/health", timeout=30).json()
    evidence["aiops_log"] = requests.get(f"{args.url}/logs/aiops.log", timeout=30).json()["content"]
    print(f"\n[done] model {evidence['health_before']['model_version']} → {evidence['health_after']['model_version']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(f"evidence → {args.out}")


if __name__ == "__main__":
    main()
