"""드리프트 감지 → 알림 → warm-start 재학습 → 게이트 재검증 → 승격·무중단 교체(또는 기존 유지).

aiops.log 기록 순서:
  [WARN] drift detected ...      설비 엔지니어 알림 (원인 후보 센서 포함)
  [INFO] retrain triggered ...   최근 7일 데이터로 Production 가중치에서 fine-tuning 시작
  [OK]   new_rmse=... promoted   게이트 통과 → 새 Production, 서빙 모델 교체, 판정 윈도우 초기화
  [FAIL] new_rmse=... gate failed → 기존 Production 유지(자동 롤백과 같은 효과)
재학습은 백그라운드 스레드에서 실행해 /predict/batch-test 응답을 막지 않고, 락으로 중복 실행을 막는다.
"""
import json
import logging
import os
import threading
import urllib.request
from datetime import datetime

from ..config import RETRAIN_ASYNC, RETRAIN_DAYS

logger = logging.getLogger("aiops")


def notify(level, message):
    """aiops.log 기록 + (선택) ALERT_WEBHOOK_URL 이 설정된 경우 메신저 웹훅 전송."""
    getattr(logger, level)(message)
    url = os.getenv("ALERT_WEBHOOK_URL")
    if url and level in ("warning", "error"):
        def send():
            try:
                req = urllib.request.Request(url, data=json.dumps({"text": f"[EtchGuard] {message}"}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).close()
            except Exception:  # 알림 실패가 서빙을 막아서는 안 된다
                logger.exception("[ERROR] alert webhook failed")
        threading.Thread(target=send, daemon=True).start()


class RetrainController:
    def __init__(self, manager, monitor, load_series):
        self.manager, self.monitor, self.load_series = manager, monitor, load_series
        self.lock = threading.Lock()
        self.state = {"running": False, "last_result": None, "history": []}

    def check_and_trigger(self, equipment_id, drift, until, diagnostics=None):
        if not drift["drift"]:
            return {"status": "watch" if drift["consecutive_over"] else "ok"}
        suspects = ", ".join(f"{s['sensor']}(x{s['ratio']})" for s in (diagnostics or {}).get("top_sensors", []))
        notify("warning", f"[WARN] drift detected equipment={equipment_id} at={drift['detected_at']} "
                          f"window_rmse={drift['detected_window_rmse']:.2f} > {self.monitor.threshold} "
                          f"x{self.monitor.consecutive} consecutive | suspect sensors: {suspects or 'n/a'} "
                          f"→ 설비 엔지니어 점검 요청")
        if not self.lock.acquire(blocking=False):
            return {"status": "retrain_in_progress"}
        self.state["running"] = True
        if RETRAIN_ASYNC:
            threading.Thread(target=self._run, args=(equipment_id, until), daemon=True).start()
            return {"status": "retrain_triggered", "background": True}
        self._run(equipment_id, until)
        return {"status": "retrain_triggered", "background": False, "result": self.state["last_result"]}

    def _run(self, equipment_id, until):
        from ..train import fine_tune
        record = {"equipment_id": equipment_id, "until": until.isoformat(),
                  "started_at": datetime.now().isoformat(timespec="seconds")}
        try:
            base = self.manager.get()
            notify("info", f"[INFO] retrain triggered (warm start from {base.version}, "
                           f"data=last {RETRAIN_DAYS}d until {until.isoformat()}, all equipment)")
            _, series = self.load_series()
            result = fine_tune(series, until, base.directory, base.version)
            v, champ = result["validation"], result["champion_holdout"]
            golden = result.get("golden_normal", {}).get("rmse")
            summary = (f"new_rmse={v['rmse']:.2f} (champion {champ['rmse']:.2f}) recall={v['recall']:.2f} "
                       f"precision={v['precision']:.2f} golden_rmse={golden if golden is None else round(golden, 2)}")
            record.update(promoted=result["gate"]["passed"], new_rmse=round(v["rmse"], 3),
                          champion_rmse=round(champ["rmse"], 3), checks=result["gate"]["checks"],
                          registered_version=result.get("registry", {}).get("registered_version"))
            if result["gate"]["passed"]:
                old, new = self.manager.reload()
                record.update(previous_version=old, new_version=new)
                notify("info", f"[OK] {summary} - production promoted: {old} → {new}, serving cache swapped")
            else:
                failed = [k for k, ok in result["gate"]["checks"].items() if not ok]
                record.update(kept_version=base.version)
                notify("warning", f"[FAIL] {summary} - gate failed {failed}, keep {base.version}")
        except Exception as exc:
            record.update(promoted=False, error=repr(exc))
            notify("error", f"[ERROR] retrain failed: {exc!r} - keep current production")
        finally:
            # 승격 여부와 무관하게 판정 윈도우를 비운다: 이전 모델의 오차로 즉시 재트리거되는 것을 막는다.
            self.monitor.reset()
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self.state["last_result"] = record
            self.state["history"] = (self.state["history"] + [record])[-20:]
            self.state["running"] = False
            self.lock.release()

    def status(self):
        return {"running": self.state["running"], "last_result": self.state["last_result"],
                "history": self.state["history"]}
