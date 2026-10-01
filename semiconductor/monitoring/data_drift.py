"""입력 분포 진단(PSI): 경보 용도가 아니라 '어느 센서부터 점검할지' 원인 후보를 순위로 보여준다.

설계 근거 (scripts/analyze_drift_policy.py 실측):
- 장비 열화·정비 주기 때문에 정상 1주에도 챔버 온도/압력/진동/파티클 PSI가 1 이상이다.
  PSI 고정 임계값(0.25)으로 경보하면 정상 주의 대부분이 경보 → 알람 피로. 그래서 경보 판정은
  예측 오차(RMSE)로만 하고, PSI는 '평상시 대비 배율(ratio)'로 바꿔 원인 센서 순위에만 쓴다.
- 계절 오탐 방지: 기준 분포는 학습 구간(2023~2024)의 '같은 달' 데이터다. 7월 데이터는 1월이 아니라
  전년도 7월과 비교한다. 연간 통합 기준을 쓰면 외기·냉각수 PSI가 정상 주에도 15~25배 커진다.
"""
import numpy as np

from ..config import FEATURES

# usage_count는 정비 때마다 0으로 돌아가는 톱니파라 1주 분포 비교가 의미 없다.
DIAG_FEATURES = [f for f in FEATURES if f != "usage_count"]
_IDX = [FEATURES.index(f) for f in DIAG_FEATURES]
BINS = 10
WEEK = 336          # 30분 × 48 × 7
MIN_ROWS = 96       # 2일 미만이면 분포 비교를 생략
SHIFT_RATIO = 3.0   # 평상시 PSI의 3배 이상이면 '평소와 다름'으로 표시


def _histogram(edges, values):
    counts = [np.bincount(np.searchsorted(edges[j], values[:, j], side="right"), minlength=BINS)
              for j in range(values.shape[1])]
    return np.stack(counts) / len(values)


def _psi(expected, actual):
    e, a = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return np.sum((a - e) * np.log(a / e), axis=1)


def build_reference_profile(series, train_end_of):
    """학습 구간을 월별로 묶어 10분위 경계·비율을 만들고, 평상시(학습 구간 주간) PSI 중앙값을 기록한다."""
    by_month = {m: [] for m in range(1, 13)}
    for g in series.values():
        for i in range(train_end_of(g)):
            by_month[g["time"][i].month].append(g["x"][i, _IDX])
    profile = {"features": DIAG_FEATURES, "months": {}}
    for month, rows in by_month.items():
        if not rows:
            continue
        values = np.asarray(rows)
        edges = np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1], axis=0).T
        profile["months"][str(month)] = {"edges": edges.tolist(), "props": _histogram(edges, values).tolist()}
    weekly = []
    for g in series.values():
        end = train_end_of(g)
        for s in range(0, end - WEEK + 1, WEEK):
            ref = profile["months"].get(str(g["time"][s + WEEK // 2].month))
            if ref:
                weekly.append(_psi(np.array(ref["props"]), _histogram(np.array(ref["edges"]), g["x"][s:s + WEEK, _IDX])))
    profile["typical_psi"] = np.median(np.array(weekly), axis=0).tolist() if weekly else [1.0] * len(DIAG_FEATURES)
    return profile


def diagnose(profile, reference_timestamp, raw_rows):
    """배치 입력(raw_rows: N×12)의 동월 기준 PSI와 평상시 대비 배율 상위 센서."""
    if profile is None or len(raw_rows) < MIN_ROWS:
        return {"available": False, "reason": f"need ≥{MIN_ROWS} rows and a reference profile"}
    month = str(reference_timestamp.month)
    ref = profile["months"].get(month)
    if ref is None:
        return {"available": False, "reason": f"no reference for month {month}"}
    values = np.asarray(raw_rows, dtype="float64")[:, _IDX]
    psi = _psi(np.array(ref["props"]), _histogram(np.array(ref["edges"]), values))
    ratio = psi / np.maximum(np.array(profile["typical_psi"]), 1e-3)
    order = np.argsort(-ratio)
    top = [{"sensor": DIAG_FEATURES[j], "psi": round(float(psi[j]), 3), "ratio": round(float(ratio[j]), 1)}
           for j in order[:3]]
    return {"available": True, "reference": f"2023~2024 같은 달({month}월)",
            "top_sensors": top, "shifted": [t["sensor"] for t in top if t["ratio"] >= SHIFT_RATIO]}
