# EtchGuard — 반도체 식각 장비 이상 징후 조기예측 · AIOps 설비 모니터링

> 판교 9반 2조 · P282 김민정 · P298 이재겸 · P287 김태동 · P289 박규리 · P280 김동욱 · P301 임동건
> 반도체 공정 장비의 시계열 센서 데이터를 활용한 이상 징후 조기예측 및 AIOps 기반 설비 모니터링 시스템

동일 장비의 **최근 20사이클(30분 간격, 10시간) × 센서 12종**으로 **다음 30분 공정의 이상 점수(0~100)** 를 예측합니다.
점수 70 이상은 HIGH(점검 권고), 40~70은 WARNING, 40 미만은 NORMAL입니다. 서빙(FastAPI) → MLOps(MLflow 게이트·Registry·Docker) →
AIOps(예측 오차 드리프트 감지 → 알림 → warm-start 재학습 → 게이트 재검증 → 무중단 교체)를 하나의 앱으로 연결했습니다.

> 데이터는 교육용 **합성 데이터**(장비 5대 × 3년, 263,040행)입니다. 계절·열화·정비·돌발 이상과 2025-07-20 이후 드리프트가 포함되어 있습니다.
> 실제 공정 성능을 보증하지 않으며 자동 장비 제어는 하지 않습니다.

## 빠른 실행

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (압축본에 CSV가 없으면) 데이터 복원 — data/README.md 참고, 원본과 SHA-256 동일
python scripts/generate_semiconductor_etch_data.py --years 3 --interval-minutes 30 --start-date 2023-01-01 --target-defect-rate 0.2 --seed 42 --output data/semiconductor_etch_timeseries_3years.csv
python -m semiconductor.train --register                      # 학습 → 게이트 → Production v1
MODEL_SOURCE=mlflow LOADING_MODE=eager uvicorn semiconductor.app:app --port 8000
python scripts/simulate_drift.py                              # (다른 터미널) 정상 → 계절 → 드리프트 → 재학습 후
```

대시보드 http://localhost:8000/ · Swagger http://localhost:8000/docs

| 단계 | 명령 |
|---|---|
| Lazy/Eager 측정 | `python -m semiconductor.verify --spawn --mode lazy --label local_lazy` (eager도 동일) |
| 게이트 실패 시연 | `python -m semiconductor.train --register --bundle semiconductor_state/bundles/<이름> --rmse-gate 0` |
| 버전 이력·롤백 | `python -m semiconductor.registry list` / `python -m semiconductor.registry rollback --version 1` |
| 드리프트 정책 근거 | `python scripts/analyze_drift_policy.py` → `evidence/04_drift_policy.json` |
| warm vs scratch | `python scripts/compare_retrain_strategy.py` → `evidence/06_retrain_strategy.json` |
| 통합 테스트 (11개) | `pip install -r semiconductor/requirements-test.txt && python -m unittest semiconductor.test_pipeline -v` |
| Docker | `docker compose up --build -d` → `python -m semiconductor.verify --label docker` |
| Swagger 요청·응답 캡처 | `pip install playwright && python scripts/capture_api_screens.py` → `docs/screenshots/api/` (Production v1 상태의 서버에서 실행) |

## 구조

```
semiconductor/
  config.py            설정·운영 임계값(게이트/드리프트/재학습) 단일 관리
  data.py              CSV 검증, 시간 분할, 스케일러(학습 구간 1회 fit), 학습·서빙 공통 transform
  train.py             base 학습·평가·게이트·MLflow 등록/승격, fine_tune(warm start)
  runtime.py           Bundle, ModelManager(_load_from_local/_load_from_mlflow, Lazy/Eager, 무중단 reload)
  mlflow_model.py      pyfunc 래퍼 (모델+스케일러+기준분포를 한 버전으로)
  app.py               FastAPI 엔드포인트, 운영 지표 미들웨어, 예측 기록
  monitoring/
    drift_detector.py  compute_rmse, is_drift, DriftMonitor(장비별 12h 윈도우 × 3회 연속)
    retrain_trigger.py check_and_trigger → [WARN]→[INFO]→[OK]/[FAIL], 백그라운드 재학습·락
    data_drift.py      같은 달 기준 PSI → 원인 후보 센서 순위(경보 아님)
  registry.py          버전 목록·롤백 CLI
  verify.py            HTTP 스모크·Lazy/Eager 측정
  static/index.html    운영 콘솔(업로드·예측·드리프트 시뮬레이션·재학습 로그)
  test_pipeline.py     Day1~3 통합 테스트 (임시 상태·임시 Registry)
scripts/  simulate_drift.py · analyze_drift_policy.py · compare_retrain_strategy.py · capture_api_screens.py · generate_semiconductor_etch_data.py
evidence/ 실행 로그·측정 JSON         docs/  기획서·스크린샷
```

## API

| Method | URL | 역할 |
|---|---|---|
| GET | `/` | 운영 콘솔 |
| GET | `/health` | 상태·모델 버전·Lazy/Eager·로딩 시간 |
| GET | `/ready` | 모델 로드 시 200, 미로드 503 |
| POST | `/predict` | `{equipment_id, sequence[20]}` → 이상 점수·위험 등급·권고·모델 버전 |
| POST | `/predict/batch-test` | `{equipment_id, records[≥21, anomaly_score 포함]}` → 예측·드리프트 판정·원인 센서 |
| GET | `/monitoring/status` | 윈도우 RMSE·판정, 재학습 이력, 엔드포인트별 p50/p95·에러율 |
| POST | `/data/upload` | CSV 검증·저장 (400: 형식 오류, 413: 64MiB 초과) |
| GET | `/data/status` · `/data/example` · `/data/batch` | 데이터 현황 · 예측 예시 · 시뮬레이션 배치 |
| GET | `/logs` · `/logs/{name}?tail=` | aiops.log·predictions.jsonl 조회 (읽기 전용) |

에러: 422(시퀀스 길이≠20, 음수 압력, 시간 역순·30분 간격 위반, `anomaly_score` 등 정답 누설 필드), 503(모델/Production 미준비).

## 운영 정책 요약

| 항목 | 기준 | 근거 |
|---|---|---|
| 배포 게이트 | 검증 RMSE ≤ 4.0, HIGH Recall ≥ 0.80, Precision ≥ 0.80 | 미탐=불량 웨이퍼 진행, 오탐=불필요 정비 |
| 재학습 게이트 | 위 기준(최근 홀드아웃) + 현 Production보다 개선 + 고정 정상셋 RMSE ≤ 4.0 | 작은 검증셋 우연 통과·망각 방지 |
| 드리프트 | 12시간 윈도우 RMSE > 5.0 이 3회 연속 | 2025 정상 구간 오탐 0건, 탐지 지연 36시간 |
| 재학습 | Production 가중치 warm start, 최근 7일 전 장비, lr 1e-4 × 10 epoch, 스케일러 재사용 | scratch 대비 정상셋 RMSE 6.43 → 3.39 |
| 실패 시 | Production 유지([FAIL] 로그), 수동 롤백 CLI | |
