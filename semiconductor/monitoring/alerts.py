"""HIGH 예측 경보: 장비의 다음 공정 예측이 HIGH(70점 이상)로 들어서면 설비 엔지니어에게 알린다.

- 새로 HIGH가 되면 바로 경보, HIGH가 이어지면 ALERT_REPEAT_SEC(기본 2시간)마다 다시 알린다.
- 같은 장비는 ALERT_COOLDOWN_SEC(기본 10분) 안에 다시 울리지 않는다(알람 피로 방지).
- 전달 경로는 드리프트 경보와 같다: aiops.log + (설정 시) ALERT_WEBHOOK_URL 메신저 웹훅.
"""
import os
import threading
import time
from collections import deque
from datetime import datetime

from .retrain_trigger import notify


class HighAlerts:
    def __init__(self, cooldown=None, repeat=None):
        self.cooldown = float(os.getenv("ALERT_COOLDOWN_SEC", 600) if cooldown is None else cooldown)
        self.repeat = float(os.getenv("ALERT_REPEAT_SEC", 7200) if repeat is None else repeat)
        self.lock = threading.Lock()
        self.last_level, self.last_alert = {}, {}
        self.recent = deque(maxlen=30)
        self.day, self.today = datetime.now().date(), 0

    def observe(self, equipment_id, score, level, target, action, sensors=(), now=None):
        """예측 한 건을 보고 경보를 낼지 정한다. 경보를 냈으면 그 기록을 돌려준다."""
        now = time.time() if now is None else now
        with self.lock:
            previous = self.last_level.get(equipment_id)
            self.last_level[equipment_id] = level
            if level != "HIGH":
                return None
            since = now - self.last_alert.get(equipment_id, float("-inf"))
            if since < self.cooldown or (previous == "HIGH" and since < self.repeat):
                return None
            self.last_alert[equipment_id] = now
            stamp = datetime.fromtimestamp(now)
            if stamp.date() != self.day:
                self.day, self.today = stamp.date(), 0
            self.today += 1
            record = {"at": stamp.isoformat(timespec="seconds"), "equipment_id": equipment_id,
                      "score": round(float(score), 1), "target_timestamp": target, "action": action,
                      "sensors": list(sensors), "kind": "new" if previous != "HIGH" else "repeat"}
            self.recent.append(record)
        reason = ", ".join(sensors) if sensors else "n/a"
        notify("warning", f"[ALERT] HIGH predicted equipment={equipment_id} score={score:.1f} target={target} "
                          f"| {action} | 점검 우선순위: {reason}")
        return record

    def snapshot(self):
        with self.lock:
            if datetime.now().date() != self.day:
                self.day, self.today = datetime.now().date(), 0
            return {"today": self.today, "recent": list(self.recent),
                    "last_by_equipment": {e: datetime.fromtimestamp(t).isoformat(timespec="seconds")
                                          for e, t in self.last_alert.items()},
                    "policy": {"cooldown_seconds": self.cooldown, "repeat_seconds": self.repeat}}
