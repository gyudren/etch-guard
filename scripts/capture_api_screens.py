"""Swagger UI(/docs)에서 실제로 Try it out → Execute 한 요청·응답 화면을 캡처한다 (서브노트·기획서 증빙용).

사전 준비: pip install playwright  (Chrome이 설치되어 있으면 브라우저 추가 다운로드 불필요)
          서버 실행: MODEL_SOURCE=mlflow LOADING_MODE=eager uvicorn semiconductor.app:app --port 8000
실행:     python scripts/capture_api_screens.py [--url http://127.0.0.1:8000] [--out docs/screenshots/api]

순서: /health → /data/upload → /predict 200 → /predict 422(19개) → /predict 422(정답 필드 누설)
      → /predict/batch-test(드리프트 주입) → 재학습 완료 대기 → /monitoring/status → /logs/aiops.log
      → /predict 200(새 Production 버전)
"""
import argparse
import copy
import json
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def operation(page, method, path):
    return page.locator(f".opblock.opblock-{method}").filter(
        has=page.locator(f".opblock-summary-path[data-path='{path}']")).first


def open_try_it_out(block):
    if not block.locator(".opblock-body").count():
        block.locator(".opblock-summary").click()
    try_out = block.locator("button.try-out__btn")
    try_out.wait_for()
    if try_out.inner_text().strip() == "Try it out":
        try_out.click()
        block.locator("button.execute").wait_for()


def execute(block, status_text=None, timeout=120_000):
    block.locator("button.execute").click()
    block.locator(".live-responses-table tr.response .response-col_status").first.wait_for(timeout=timeout)
    if status_text:
        block.locator(".live-responses-table tr.response .response-col_status").first.filter(has_text=status_text).wait_for(timeout=timeout)
    block.page.wait_for_timeout(400)


def shoot(block, out, name, response_tail=False):
    """요청 입력창은 맨 위(장비 ID·첫 레코드)로, 응답은 필요하면 끝(드리프트 판정)으로 맞추고
    블록 상단부터 실제 응답 헤더까지만 잘라 저장한다 (아래 스키마 예시는 제외)."""
    page = block.page
    textarea = block.locator("textarea.body-param__text")
    if textarea.count():
        textarea.evaluate("el => el.scrollTop = 0")
    if response_tail:
        block.locator(".live-responses-table .highlight-code pre").first.evaluate(
            "el => { el.style.maxHeight = '820px'; el.scrollTop = el.scrollHeight; }")
    block.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    clip = block.evaluate("""el => {
        const top = el.getBoundingClientRect();
        const live = el.querySelector('.live-responses-table').getBoundingClientRect();
        return {x: top.left + window.scrollX, y: top.top + window.scrollY,
                width: top.width, height: live.bottom - top.top + 16};
    }""")
    path = out / f"{name}.png"
    page.screenshot(path=str(path), clip=clip, full_page=True)
    print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def wait_retrain(url, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not requests.get(f"{url}/monitoring/status", timeout=30).json()["retrain"]["running"]:
            return
        time.sleep(1)
    raise TimeoutError("retrain did not finish")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=str(ROOT / "docs/screenshots/api"))
    parser.add_argument("--csv", default=str(ROOT / "data/semiconductor_etch_timeseries_3years.csv"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    example = requests.get(f"{args.url}/data/example", timeout=60).json()
    short = copy.deepcopy(example); short["sequence"].pop()
    leak = copy.deepcopy(example); leak["sequence"][0]["anomaly_score"] = 80
    drift_batch = requests.get(f"{args.url}/data/batch", params={
        "equipment_id": "ETCH-01", "start": "2025-07-20T14:00:00", "cycles": 336}, timeout=60).json()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=2)
        page.goto(f"{args.url}/docs")
        page.locator(".opblock").first.wait_for()

        block = operation(page, "get", "/health")
        open_try_it_out(block); execute(block); shoot(block, out, "01_get_health")

        block = operation(page, "post", "/data/upload")
        open_try_it_out(block)
        block.locator("input[type=file]").set_input_files(args.csv)
        execute(block, "200"); shoot(block, out, "02_post_data_upload_200")

        block = operation(page, "post", "/predict")
        open_try_it_out(block)
        textarea = block.locator("textarea.body-param__text")
        textarea.fill(json.dumps(example, indent=2)); execute(block, "200"); shoot(block, out, "03_post_predict_200")
        textarea.fill(json.dumps(short, indent=2)); execute(block, "422"); shoot(block, out, "04_post_predict_422_19_points")
        textarea.fill(json.dumps(leak, indent=2)); execute(block, "422"); shoot(block, out, "05_post_predict_422_label_leak")

        block = operation(page, "post", "/predict/batch-test")
        open_try_it_out(block)
        block.locator("textarea.body-param__text").fill(json.dumps(drift_batch))
        execute(block, "200"); shoot(block, out, "06_post_batch_test_drift", response_tail=True)
        wait_retrain(args.url)

        block = operation(page, "get", "/monitoring/status")
        open_try_it_out(block); execute(block, "200"); shoot(block, out, "07_get_monitoring_status")

        block = operation(page, "get", "/logs/{filename}")
        open_try_it_out(block)
        block.locator("input[placeholder='filename']").fill("aiops.log")
        execute(block, "200"); shoot(block, out, "08_get_logs_aiops")

        block = operation(page, "post", "/predict")
        block.locator("textarea.body-param__text").fill(json.dumps(example, indent=2))
        execute(block, "200"); shoot(block, out, "09_post_predict_200_after_retrain")
        browser.close()


if __name__ == "__main__":
    main()
