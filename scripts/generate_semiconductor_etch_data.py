"""반도체 식각 장비 예지보전 실습용 합성 시계열 데이터 생성기.

실제 생산 성능을 주장하기 위한 데이터가 아니라 모델 서빙/AIOps 파이프라인의
개념검증(PoC)을 위한 데이터다. 장비별 시간 순서, 점진적 열화, 돌발 이상,
정비 후 회복, 마지막 구간의 분포 변화(drift)를 재현한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


FIELDS = [
    "timestamp",
    "equipment_id",
    "cycle_index",
    "season",
    "ambient_temperature_c",
    "ambient_humidity_pct",
    "cooling_water_temperature_c",
    "chamber_temperature_c",
    "chamber_pressure_mtorr",
    "cf4_flow_sccm",
    "o2_flow_sccm",
    "rf_power_w",
    "process_time_sec",
    "vibration_mm_s",
    "particle_count",
    "usage_count",
    "maintenance",
    "anomaly_score",
    "defect_label",
    "risk_level",
    "is_drift",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def risk_level(score: float) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "WARNING"
    return "NORMAL"


def season_name(month: int) -> str:
    if month in (3, 4, 5):
        return "SPRING"
    if month in (6, 7, 8):
        return "SUMMER"
    if month in (9, 10, 11):
        return "AUTUMN"
    return "WINTER"


def generate_equipment_rows(
    equipment_no: int,
    cycles: int,
    start: datetime,
    rng: random.Random,
    interval_minutes: int,
) -> list[dict]:
    equipment_id = f"ETCH-{equipment_no:02d}"
    equipment_bias = rng.uniform(-0.6, 0.6)
    health = rng.uniform(0.01, 0.04)  # 모델 입력에 직접 포함되지 않는 잠재 열화 상태
    usage_count = 0
    burst_left = 0
    history = deque(maxlen=20)
    next_planned_maintenance = rng.randint(2800, 4200)
    rows: list[dict] = []

    for cycle in range(cycles):
        timestamp = start + timedelta(minutes=interval_minutes * cycle)
        day_of_year = timestamp.timetuple().tm_yday
        minute_of_day = timestamp.hour * 60 + timestamp.minute

        # 북반구 계절 주기 + 일교차. 여름(7~8월)에 외기/냉각수 온도가 가장 높다.
        annual_wave = math.cos(2 * math.pi * (day_of_year - 205) / 365.25)
        daily_wave = math.sin(2 * math.pi * (minute_of_day - 480) / 1440)
        ambient_temp = 14.0 + 12.0 * annual_wave + 3.2 * daily_wave + rng.gauss(0, 1.1)

        # 장마철(6월 말~7월)의 습도 상승을 가우시안 봉우리로 추가한다.
        monsoon = math.exp(-0.5 * ((day_of_year - 190) / 24.0) ** 2)
        humidity = 52.0 + 10.0 * annual_wave + 22.0 * monsoon - 2.5 * daily_wave
        humidity = clamp(humidity + rng.gauss(0, 3.5), 22.0, 95.0)
        cooling_water_temp = 19.0 + 4.8 * annual_wave + 0.7 * daily_wave + rng.gauss(0, 0.45)

        is_drift = int(cycle >= int(cycles * 0.85))
        maintenance = 0

        # 후반 15%에는 소모품 노후화/레시피 변경을 가정한 분포 변화를 주입한다.
        drift_pressure = 0.85 * is_drift
        drift_rf = -12.0 * is_drift
        drift_vibration = 0.12 * is_drift

        usage_count += 1
        health += rng.uniform(0.00018, 0.00042) * (1.0 + 0.55 * is_drift)
        health += max(0.0, rng.gauss(0.0, 0.00015))

        # 약 0.15% 확률로 5~25사이클 동안 지속되는 돌발 이상을 만든다.
        if burst_left == 0 and rng.random() < 0.0015:
            burst_left = rng.randint(5, 25)
        burst_strength = 0.0
        if burst_left > 0:
            burst_strength = 1.0 - burst_left / 35.0
            burst_left -= 1

        slow_wave = math.sin(cycle / 180.0 + equipment_no) * 0.25
        temp = 60.0 + equipment_bias + slow_wave + 2.4 * health + 2.2 * burst_strength
        pressure = 10.0 + 1.9 * health + drift_pressure + 1.8 * burst_strength
        cf4 = 100.0 - 2.8 * health - 1.7 * burst_strength + rng.gauss(0, 0.8)
        o2 = 20.0 - 0.8 * health - 0.6 * burst_strength + rng.gauss(0, 0.25)
        rf_power = 500.0 - 14.0 * health + drift_rf - 24.0 * burst_strength
        process_time = 60.0 + 2.3 * health + 2.0 * burst_strength
        vibration = 0.18 + 0.24 * health + drift_vibration + 0.38 * burst_strength
        particle_count = 7.0 + 28.0 * health + 35.0 * burst_strength

        # 계절 환경이 공정에 미치는 영향. 클린룸 제어로 영향은 완화되지만 0은 아니다.
        temp += 0.035 * (ambient_temp - 20.0) + 0.055 * (cooling_water_temp - 20.0)
        pressure += 0.006 * (humidity - 50.0)
        process_time += 0.025 * max(0.0, 15.0 - ambient_temp)  # 겨울 예열/안정화 시간
        vibration += 0.0025 * max(0.0, ambient_temp - 25.0)  # 여름 냉각 부하
        particle_count += 0.22 * max(0.0, humidity - 60.0)  # 고습 시 파티클 위험 증가

        temp += rng.gauss(0, 0.32)
        pressure += rng.gauss(0, 0.22)
        rf_power += rng.gauss(0, 2.8)
        process_time += rng.gauss(0, 0.55)
        vibration = max(0.02, vibration + rng.gauss(0, 0.025))
        particle_count = max(0, int(round(particle_count + rng.gauss(0, 3.0))))

        # 최근 20사이클의 누적 편차를 정답 생성에 사용해 시계열 모델의 이유를 만든다.
        current_stress = (
            abs(temp - 60.0) / 4.0
            + abs(pressure - 10.0) / 2.5
            + abs(rf_power - 500.0) / 35.0
            + max(0.0, vibration - 0.20) / 0.45
            + particle_count / 75.0
        )
        history.append(current_stress)
        rolling_stress = sum(history) / len(history)
        trend = 0.0 if len(history) < 10 else (sum(list(history)[-5:]) - sum(list(history)[:5])) / 5

        # 관측 센서 외 잠재 열화와 잡음도 포함해 단순 공식 복제를 방지한다.
        score = (
            7.0
            + 43.0 * health
            + 15.0 * rolling_stress
            + 7.0 * max(0.0, trend)
            + 18.0 * burst_strength
            + 5.0 * max(0.0, ambient_temp - 27.0) / 10.0
            + 4.0 * max(0.0, humidity - 70.0) / 20.0
            + rng.gauss(0, 2.2)
        )
        score = clamp(score, 0.0, 100.0)

        rows.append(
            {
                "timestamp": timestamp.isoformat(timespec="minutes"),
                "equipment_id": equipment_id,
                "cycle_index": cycle + 1,
                "season": season_name(timestamp.month),
                "ambient_temperature_c": f"{ambient_temp:.3f}",
                "ambient_humidity_pct": f"{humidity:.3f}",
                "cooling_water_temperature_c": f"{cooling_water_temp:.3f}",
                "chamber_temperature_c": f"{temp:.3f}",
                "chamber_pressure_mtorr": f"{pressure:.3f}",
                "cf4_flow_sccm": f"{cf4:.3f}",
                "o2_flow_sccm": f"{o2:.3f}",
                "rf_power_w": f"{rf_power:.3f}",
                "process_time_sec": f"{process_time:.3f}",
                "vibration_mm_s": f"{vibration:.4f}",
                "particle_count": particle_count,
                "usage_count": usage_count,
                "maintenance": maintenance,
                "anomaly_score": f"{score:.3f}",
                "defect_label": int(score >= 70),
                "risk_level": risk_level(score),
                "is_drift": is_drift,
            }
        )

        # 계획정비 또는 위험 지속 시 다음 사이클부터 장비 상태를 회복시킨다.
        if usage_count >= next_planned_maintenance or (score >= 88 and usage_count >= 900):
            rows[-1]["maintenance"] = 1
            health = rng.uniform(0.008, 0.025)
            usage_count = 0
            burst_left = 0
            history.clear()
            next_planned_maintenance = rng.randint(2800, 4200)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equipment", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=20_000)
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument("--interval-minutes", type=int, default=8)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-defect-rate",
        type=float,
        default=None,
        help="HIGH/defect_label=1의 목표 비율(예: 0.20). 미지정 시 원점수를 사용",
    )
    parser.add_argument(
        "--output",
        default="data/semiconductor_etch_timeseries.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_time = datetime.fromisoformat(args.start_date)
    if args.years is not None:
        if args.years <= 0:
            raise ValueError("--years는 1 이상이어야 합니다")
        end_time = base_time.replace(year=base_time.year + args.years)
        args.cycles = int((end_time - base_time).total_seconds() // (args.interval_minutes * 60))

    all_rows: list[dict] = []
    for equipment_no in range(1, args.equipment + 1):
        all_rows.extend(
            generate_equipment_rows(
                equipment_no,
                args.cycles,
                base_time,
                rng,
                args.interval_minutes,
            )
        )

    if args.target_defect_rate is not None:
        if not 0 < args.target_defect_rate < 1:
            raise ValueError("--target-defect-rate는 0과 1 사이여야 합니다")

        target_count = round(len(all_rows) * args.target_defect_rate)
        ranked = sorted(
            range(len(all_rows)),
            key=lambda i: float(all_rows[i]["anomaly_score"]),
            reverse=True,
        )
        defect_indices = set(ranked[:target_count])
        cutoff = float(all_rows[ranked[target_count - 1]]["anomaly_score"])
        offset = 70.0 - cutoff

        for i, row in enumerate(all_rows):
            calibrated = clamp(float(row["anomaly_score"]) + offset, 0.0, 100.0)
            is_defect = i in defect_indices
            # 반올림 경계에서도 defect_label과 HIGH가 정확히 일치하도록 보정한다.
            if is_defect:
                calibrated = max(70.0, calibrated)
            else:
                calibrated = min(69.999, calibrated)
            row["anomaly_score"] = f"{calibrated:.3f}"
            row["defect_label"] = int(is_defect)
            row["risk_level"] = risk_level(calibrated)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"generated {args.equipment * args.cycles:,} rows -> {output}")


if __name__ == "__main__":
    main()
