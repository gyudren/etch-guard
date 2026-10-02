"""드리프트 판정 정책(윈도우·임계값·연속 횟수)과 계절 기준 분포를 실측으로 결정하는 분석 스크립트.

python scripts/analyze_drift_policy.py   →  정책표·PSI 결과를 화면에 출력

1) Production 후보 모델로 2025 정상 구간(검증+테스트)과 드리프트 구간을 예측한다.
2) 윈도우(12h/24h) × 임계값(4.0~6.0) × 연속 횟수(1~3)별
   - 오탐: 정상 구간에서 장비 1대·1개월당 드리프트 판정 횟수
   - 탐지율: 드리프트 구간 윈도우 중 판정 비율 / 탐지 지연(시간)
3) 입력 분포(PSI): 같은 달 기준 vs 연간 통합 기준 — 계절 오탐 비교, 드리프트 원인 센서 순위
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from semiconductor.config import DEFAULT_DATA, SEQ_LEN, STATE
from semiconductor.data import read_data, partitions, transform, windows
from semiconductor.monitoring import data_drift as dd
from semiconductor.runtime import Bundle


def predictions(series, bundle):
    out = {}
    for eid, g in series.items():
        p = partitions(g)
        x = transform(g["x"], bundle.scaler)
        start = p["validation"][0]
        pred = np.clip(bundle.model.predict(windows(x, start, len(g["x"])), batch_size=4096, verbose=0).ravel()*100, 0, 100)
        out[eid] = {"err2": (pred - g["y"][start:]) ** 2, "drift_offset": p["drift"][0] - start}
    return out


def policy_table(preds):
    rows = []
    for w in (24, 48):
        for k in (1, 2, 3):
            for t in (4.0, 4.5, 5.0, 6.0):
                alarms, days, flagged, delays = 0, 0.0, [], []
                for o in preds.values():
                    def judged(err):
                        rm = [np.sqrt(err[i:i+w].mean()) for i in range(0, len(err) - w + 1, w)]
                        streak, res = 0, []
                        for r in rm:
                            streak = streak + 1 if r > t else 0
                            res.append(streak >= k)
                        return res
                    normal, drift = judged(o["err2"][:o["drift_offset"]]), judged(o["err2"][o["drift_offset"]:])
                    alarms += sum(normal); days += len(normal) * w / 48
                    flagged.append(np.mean(drift))
                    first = next((i for i, v in enumerate(drift) if v), None)
                    delays.append(None if first is None else (first + 1) * w / 2)
                rows.append({"window_cycles": w, "window_hours": w / 2, "threshold": t, "consecutive": k,
                             "false_alarms_per_equipment_month": round(alarms / days * 30, 2),
                             "drift_windows_flagged": round(float(np.mean(flagged)), 3),
                             "detection_delay_hours": delays})
    return rows


def psi_study(series):
    train_end = lambda g: partitions(g)["train"][1]
    profile = dd.build_reference_profile(series, train_end)
    idx = dd._IDX
    pooled_rows = np.concatenate([g["x"][:train_end(g), idx] for g in series.values()])
    pooled_edges = np.quantile(pooled_rows, np.linspace(0, 1, dd.BINS + 1)[1:-1], axis=0).T
    pooled_props = dd._histogram(pooled_edges, pooled_rows)
    same, pooled, labels, tops = [], [], [], {"normal": Counter(), "drift": Counter()}
    for g in series.values():
        p = partitions(g)
        for s in range(p["test"][0], len(g["x"]) - dd.WEEK + 1, dd.WEEK):
            values = g["x"][s:s + dd.WEEK, idx]
            ref = profile["months"][str(g["time"][s + dd.WEEK // 2].month)]
            same.append(dd._psi(np.array(ref["props"]), dd._histogram(np.array(ref["edges"]), values)))
            pooled.append(dd._psi(pooled_props, dd._histogram(pooled_edges, values)))
            drift = s + dd.WEEK > p["drift"][0]
            labels.append(drift)
            diag = dd.diagnose(profile, g["time"][s + dd.WEEK // 2], g["x"][s:s + dd.WEEK])
            tops["drift" if drift else "normal"].update(t["sensor"] for t in diag["top_sensors"])
    same, pooled, labels = np.array(same), np.array(pooled), np.array(labels)
    env = [dd.DIAG_FEATURES.index(f) for f in ("ambient_temperature_c", "ambient_humidity_pct", "cooling_water_temperature_c")]
    return {"weeks": {"normal": int((~labels).sum()), "drift": int(labels.sum())},
            "environment_sensor_median_psi_normal_weeks": {
                "same_month_reference": dict(zip([dd.DIAG_FEATURES[i] for i in env], np.round(np.median(same[~labels][:, env], 0), 3).tolist())),
                "pooled_reference": dict(zip([dd.DIAG_FEATURES[i] for i in env], np.round(np.median(pooled[~labels][:, env], 0), 3).tolist()))},
            "fixed_psi_0.25_alarm_rate_normal_weeks": round(float(np.mean((same[~labels] > 0.25).any(1))), 3),
            "typical_psi": dict(zip(dd.DIAG_FEATURES, np.round(profile["typical_psi"], 3).tolist())),
            "top3_ratio_sensors": {k: v.most_common(5) for k, v in tops.items()}}


def main():
    series = read_data(DEFAULT_DATA)
    pointer = json.loads((STATE / "local.json").read_text())
    bundle = Bundle(STATE / "bundles" / pointer["bundle"])
    result = {"generated_at": datetime.now().isoformat(timespec="seconds"), "bundle": pointer["bundle"],
              "policy_table": policy_table(predictions(series, bundle)), "psi": psi_study(series)}
    chosen = next(r for r in result["policy_table"]
                  if (r["window_cycles"], r["threshold"], r["consecutive"]) == (24, 4.0, 3))
    result["chosen_policy"] = chosen
    print(f"{'W(h)':>5} {'T':>4} {'K':>2} {'FA/eq-month':>12} {'flagged':>8}  delay(h)")
    for r in result["policy_table"]:
        print(f"{r['window_hours']:>5} {r['threshold']:>4} {r['consecutive']:>2} "
              f"{r['false_alarms_per_equipment_month']:>12} {r['drift_windows_flagged']:>8}  {r['detection_delay_hours']}")
    print(json.dumps(result["psi"], indent=2, ensure_ascii=False))
    print(f"chosen: {chosen}")


if __name__ == "__main__":
    main()
