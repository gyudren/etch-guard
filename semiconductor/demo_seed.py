"""발표·시연용 데모 시드: DEMO_SEED=1로 기동하면 대시보드를 실제 데이터로 미리 채운다.

실제 CSV의 정상 운전 구간을 실제 Production 모델로 예측해서, 서버를 막 띄운 직후에도
대시보드가 "운영 중인 화면"으로 보이게 한다. 값은 모두 진짜 예측과 진짜 확정 점수이고,
가짜인 것은 '언제'뿐이다: 기준일(DEMO_SEED_DATE)의 사이클을 오늘 같은 시각에 1:1로 대응시킨다.

채우는 것 (메모리만)
- 장비별 12h 윈도우 RMSE 이력과 상태(DriftMonitor) → 윈도우 RMSE 추이·현재 RMSE 막대
- 장비별 등급 분포·최신 예측·점검 권고·원인 후보 센서(LiveStore) → 리본·분포 막대
- 장비 타일 판정(last_batch) → EVENT 타일
- 오늘 시간대별 예측 건수·HIGH 건수: 장비 5대 × 30분마다 1건 = 시간당 10건 (실제 호출 빈도와 같다)

하지 않는 것
- 파일에는 쓰지 않는다. 재기동하면 사라지고, DEMO_SEED를 끄면 원래대로다.
- 재학습·Registry는 건드리지 않는다. 기준일은 드리프트(2025-07-20) 이전이라 DRIFT가 뜨지 않는다.
- aiops.log에는 "데모 시드를 채웠다"는 안내 한 줄만 남긴다(이벤트 목록에서 시드임을 알 수 있게).
"""
import bisect
import logging
import os
from datetime import datetime

import numpy as np

from .config import RECOMMENDED_ACTION, SEQ_LEN, risk_level
from .monitoring.data_drift import MIN_ROWS, deviation_rank, diagnose

# 2025-07-06 ~ 07-11 정상 구간: Production 모델 기준 12h 윈도우 RMSE가 모두 3.5 미만(드리프트 없음),
# 등급은 HIGH 약 20%·나머지 WARNING으로 실제 분포 그대로. 끝 시점에 ETCH-02가 HIGH(약 80점).
DEFAULT_DATE = "2025-07-11"
DEFAULT_CYCLES = 240  # 5일 = 12h 윈도우 10개 (대시보드 추이 차트는 최근 12개까지 표시)

aiops = logging.getLogger("aiops")


def enabled():
    return os.getenv("DEMO_SEED", "0") == "1"


def seed(model, series, monitor, live, last_batch, alerts=None, now=None):
    """기준일의 '지금과 같은 시각'까지 장비별 최근 cycles개를 예측해 대시보드 상태를 채운다.

    alerts(HighAlerts)를 주면 끝 시점에 HIGH인 장비는 실제 운영과 같은 경로로 HIGH 경보를 낸다."""
    now = now or datetime.now()
    day = datetime.fromisoformat(os.getenv("DEMO_SEED_DATE", DEFAULT_DATE))
    cycles = int(os.getenv("DEMO_SEED_CYCLES", DEFAULT_CYCLES))
    ref = day.replace(hour=now.hour, minute=30 if now.minute >= 30 else 0, second=0, microsecond=0)
    hour_calls, hour_high = [0] * 24, [0] * 24
    seeded = {}
    for eid, g in sorted(series.items()):
        end = bisect.bisect_right(g["time"], ref)          # ref 시각 사이클까지 포함
        start = end - cycles
        if start < SEQ_LEN:
            continue
        x = np.stack([g["x"][i - SEQ_LEN:i] for i in range(start, end)]).astype("float32")
        predicted = model.predict(x)
        times, actual = g["time"][start:end], g["y"][start:end]
        pairs = [{"timestamp": t.isoformat(), "predicted": round(float(p), 3), "actual": round(float(a), 3)}
                 for t, p, a in zip(times, predicted, actual)]
        drift = monitor.observe(eid, pairs)

        last_time = times[-1]
        level = risk_level(pairs[-1]["predicted"])
        diagnostics = diagnose(model.profile, last_time, g["x"][end - MIN_ROWS:end])
        deviations = deviation_rank(model.profile, last_time, g["x"][end - SEQ_LEN:end])
        live.record_predictions(eid, [{"target_timestamp": p["timestamp"], "predicted": p["predicted"],
                                       "actual": p["actual"]} for p in pairs], "demo-seed",
                                extra={"action": RECOMMENDED_ACTION[level], "deviations": deviations,
                                       "suspects": diagnostics.get("top_sensors", [])},
                                count_today=False)
        for t, p in zip(times, pairs):
            if t.date() == day.date():                      # 기준일 00:00 ~ ref = 오늘 00:00 ~ 지금
                hour_calls[t.hour] += 1
                hour_high[t.hour] += risk_level(p["predicted"]) == "HIGH"
        status = "drift" if drift["drift"] else "watch" if drift["consecutive_over"] else "ok"
        last_batch[eid] = {"batch_end": last_time.isoformat(), "status": status,
                           "latest_window_rmse": drift["latest_window_rmse"],
                           "consecutive_over": drift["consecutive_over"],
                           "top_sensors": diagnostics.get("top_sensors", []), "model_version": model.version}
        seeded[eid] = {"score": pairs[-1]["predicted"], "level": level, "status": status,
                       "latest_window_rmse": drift["latest_window_rmse"]}
        if alerts is not None:
            alerts.observe(eid, pairs[-1]["predicted"], level, pairs[-1]["timestamp"], RECOMMENDED_ACTION[level],
                           [f"{d['sensor']}({d['z']:+.2f}σ)" for d in deviations])
    live.add_today(hour_calls, hour_high)
    summary = ", ".join(f"{eid} {s['score']:.0f}({s['level']})" for eid, s in seeded.items())
    aiops.info(f"[INFO] demo seed loaded: {day.date()} 실제 데이터 {cycles}사이클 × {len(seeded)}대를 "
               f"{model.version}로 예측해 대시보드를 채움 (기준 시각 {ref:%H:%M}) | 최신 점수 {summary}")
    return {"reference": ref.isoformat(), "cycles": cycles, "equipment": seeded,
            "today_calls": sum(hour_calls), "today_high": sum(hour_high)}
