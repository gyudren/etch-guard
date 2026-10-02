"""성능 드리프트 판정: 예측값과 확정 이상 점수(actual)의 윈도우 RMSE.

- 윈도우: DRIFT_WINDOW(24사이클 = 12시간) 단위로 겹치지 않게 자른다.
- 판정: 윈도우 RMSE > DRIFT_RMSE_THRESHOLD(4.0)가 DRIFT_CONSECUTIVE(3)회 연속이면 드리프트.
  단발성 돌발 이상(burst)이나 정비 직후 튐은 1~2개 윈도우에서 끝나므로 재학습을 유발하지 않는다.
- 장비별로 독립 판정한다. 한 장비의 이상이 다른 장비의 윈도우를 오염시키지 않는다.
"""
import math
import threading
from collections import deque

from ..config import DRIFT_WINDOW, DRIFT_RMSE_THRESHOLD, DRIFT_CONSECUTIVE


def compute_rmse(pairs):
    """pairs: [{"predicted": float, "actual": float}, ...] → RMSE. 빈 리스트는 0.0."""
    if not pairs:
        return 0.0
    return math.sqrt(sum((p["actual"] - p["predicted"]) ** 2 for p in pairs) / len(pairs))


def is_drift(window_rmses, threshold=DRIFT_RMSE_THRESHOLD, consecutive=DRIFT_CONSECUTIVE):
    """최근 consecutive개 윈도우가 모두 임계값을 넘으면 True."""
    recent = list(window_rmses)[-consecutive:]
    return len(recent) == consecutive and all(r > threshold for r in recent)


class DriftMonitor:
    """장비별 (predicted, actual) 누적 → 윈도우 RMSE 이력 → 연속 초과 판정."""

    def __init__(self, window=DRIFT_WINDOW, threshold=DRIFT_RMSE_THRESHOLD, consecutive=DRIFT_CONSECUTIVE):
        self.window, self.threshold, self.consecutive = window, threshold, consecutive
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        """재학습 승격 후 호출: 이전 모델의 오차가 새 모델 판정에 섞이지 않도록 윈도우를 비운다."""
        with self.lock:
            self.pending = {}   # 아직 윈도우를 채우지 못한 쌍
            self.history = {}   # 완성된 윈도우 [{"end": ts, "rmse": float}]

    def observe(self, equipment_id, pairs):
        """pairs를 시간순으로 넣고, 이번 배치에서 드리프트 조건이 처음 성립한 시점을 돌려준다."""
        with self.lock:
            pending = self.pending.setdefault(equipment_id, [])
            history = self.history.setdefault(equipment_id, deque(maxlen=200))
            detected_at, detected_rmse, evaluated = None, None, 0
            for pair in pairs:
                pending.append(pair)
                if len(pending) < self.window:
                    continue
                rmse = compute_rmse(pending)
                history.append({"end": pair["timestamp"], "rmse": round(rmse, 3)})
                pending.clear()
                evaluated += 1
                if detected_at is None and is_drift([h["rmse"] for h in history], self.threshold, self.consecutive):
                    detected_at, detected_rmse = pair["timestamp"], round(rmse, 3)
            rmses = [h["rmse"] for h in history]
            return {"windows_evaluated": evaluated,
                    "latest_window_rmse": rmses[-1] if rmses else None,
                    "consecutive_over": self._streak(rmses),
                    "drift": detected_at is not None,
                    "detected_at": detected_at, "detected_window_rmse": detected_rmse}

    def _streak(self, rmses):
        streak = 0
        for r in reversed(rmses):
            if r <= self.threshold:
                break
            streak += 1
        return streak

    def snapshot(self):
        with self.lock:
            out = {}
            for eid, history in self.history.items():
                rmses = [h["rmse"] for h in history]
                out[eid] = {"windows": list(history)[-12:], "consecutive_over": self._streak(rmses),
                            "status": "DRIFT" if is_drift(rmses, self.threshold, self.consecutive)
                            else "WATCH" if rmses and rmses[-1] > self.threshold else "OK"}
            return {"policy": {"window_cycles": self.window, "window_hours": self.window / 2,
                               "rmse_threshold": self.threshold, "consecutive": self.consecutive},
                    "equipment": out}
