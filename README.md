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

## 로컬 실행 가이드 (localhost)

저장소에 학습된 기본 모델(base 번들)과 데이터 CSV가 들어 있어서, 따로 학습하지 않아도 바로 띄울 수 있습니다. Python 3.11을 권장합니다(TensorFlow 포함, 설치에 몇 분 걸립니다).

### 1. 환경 준비 (처음 한 번)

```bash
git clone https://github.com/gyudren/etch-guard.git
cd etch-guard
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 기본 모델을 Production으로 등록 (처음 한 번, 몇 초)

```bash
python -m semiconductor.train --register --bundle semiconductor_state/bundles/20261001T002500-c775a1
python -m semiconductor.registry list    # "v1 ... ← Production" 이 보이면 완료
```

처음부터 다시 학습하려면 `python -m semiconductor.train --register`를 실행합니다(1~2분).

### 3. 서버 실행

```bash
MODEL_SOURCE=mlflow LOADING_MODE=eager python -m uvicorn semiconductor.app:app --port 8000
```

Windows PowerShell에서는 `$env:MODEL_SOURCE="mlflow"; $env:LOADING_MODE="eager"; python -m uvicorn semiconductor.app:app --port 8000`을 씁니다.
등록 없이 화면만 빨리 보려면 `python -m uvicorn semiconductor.app:app --port 8000`(로컬 번들, Lazy)으로도 뜹니다.

| 화면 | 주소 |
|---|---|
| 운영 콘솔 | http://localhost:8000/ |
| Swagger API 문서 | http://localhost:8000/docs |
| 모니터링 상태 | http://localhost:8000/monitoring/status |

`curl localhost:8000/health` 응답에 `"model_version":"production-v1"`이 보이면 정상입니다.

### 4. 시연 순서

1. 콘솔 **02** 카드에서 `예시 불러오기` → `예측`: 30분 뒤 이상 점수와 위험 등급이 나옵니다.
2. 콘솔 **03** 카드에서 `① 정상` → `② 계절 변화` → `③ 드리프트 주입` → `④ 재학습 후 다음 주` 순서로 누릅니다. 다른 터미널에서 `python scripts/simulate_drift.py`를 실행해도 같습니다.
3. 콘솔 **04** 재학습 로그에서 `[WARN] drift detected` → `[INFO] retrain triggered` → `[OK] ... production-v1 → production-v2` 순서를 확인합니다.

### 5. Docker로 실행 (선택)

```bash
docker compose up --build -d     # 처음 빌드는 몇 분 걸립니다 (빌드 중에 학습·등록까지 진행)
```

http://localhost:8000 으로 접속합니다. 8000 포트를 이미 쓰고 있으면 `ETCH_PORT=8020 docker compose up -d`로 띄우고 http://localhost:8020 으로 접속합니다.

### 6. 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| `address already in use` (포트 8000) | 이미 떠 있는 서버를 종료(`lsof -i :8000`으로 PID 확인)하거나 `--port 8010`처럼 다른 포트 사용 |
| `/predict`가 503 (mlflow 모드) | Production 모델이 없는 상태입니다. 2단계 등록을 실행 |
| 코드를 고쳤는데 화면이 그대로 | 서버를 재시작합니다(개발 중에는 `--reload` 옵션 사용) |
| 시연 후 처음 상태(v1)로 되돌리기 | `python -m semiconductor.registry rollback --version 1` 후 서버 재시작 |
| 시연 후 `semiconductor_state/local.json`이 변경됨 | 재학습 결과가 기록된 것입니다. 커밋하지 말고 `git checkout semiconductor_state/local.json`으로 되돌립니다 |
