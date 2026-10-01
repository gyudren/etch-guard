# EtchGuard — 반도체 팀 프로젝트 Day 1·2 실행 안내

P282-김민정, P298-이재겸, P287-김태동, P289-박규리, P280-김동욱, P301-임동건

`EtchGuard_Day2.zip`을 풀고 `EtchGuard_Day2` 폴더에서 아래 명령을 실행합니다.
현재 작업 폴더에서는 `project`가 같은 역할입니다. 개인 HAIC 실습은 기존
`serving_app.main:app`, 팀 반도체 실습은 `semiconductor.app:app`입니다.

Python 3.11 권장. 가상환경은 각 PC에서 새로 만드세요.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn semiconductor.app:app --port 8000
```

Windows PowerShell은 `py -3.11 -m venv .venv`와 `.venv\Scripts\Activate.ps1`을 사용합니다.
아래 환경변수 문법 `MODEL_SOURCE=mlflow ...` 대신 `$env:MODEL_SOURCE="mlflow"` 설정 후
`python -m uvicorn ...`을 실행하세요. Eager도 `$env:LOADING_MODE="eager"`로 설정합니다.

브라우저: http://localhost:8000/ · Swagger: http://localhost:8000/docs
ZIP에는 학습된 로컬 모델과 스케일러가 있어 바로 예측할 수 있습니다.
기존 서버가 8000을 사용 중이면 `--port 8001`로 실행하세요.

## 문제와 모델

설비 운영자는 센서 이상을 늦게 발견하면 검사·정비 우선순위를 놓칠 수 있습니다.
동일 장비 최근 20회(30분 간격, 10시간)의 센서 12개로 다음 30분의 이상 점수를 예측하고,
HIGH에 해당하면 엔지니어의 점검 판단을 돕는 시뮬레이션 서비스입니다.

점수 <40 NORMAL, 40~70 미만 WARNING, 70 이상 HIGH입니다. `defect_label=1`은
합성 점수 HIGH에 붙인 이름으로 실제 검사 불량 확률이 아닙니다. 자동 장비 제어나
특정 부품의 고장 원인 진단은 구현하지 않았습니다.

데이터: 2023~2025년, 장비 5대, 총 263,040행, 21개 열, 30분 간격.
계절·열화·정비·돌발 이벤트와 2025-07-20 14:00 이후의 드리프트가 포함됩니다.
전체 점수 순위를 보정하여 HIGH 비율을 20%로 맞춘 합성 벤치마크입니다. 그 보정에는
전체 기간 분포가 사용되므로 실제 현장 일반화 성능을 주장할 수 없습니다.
센서의 물리 단위·계절 영향·위험 기준은 교육용 가정이며 물리 검증 모델이 아닙니다.

## Day 1: 업로드 → 학습 → 서빙

1. 대시보드에서 `data/semiconductor_etch_timeseries_3years.csv` 업로드.
2. 다른 터미널에서 가상환경을 활성화하고 아래 명령으로 학습.

```bash
python -m semiconductor.train
```

업로드 파일이 있으면 최신 파일을 읽고, 없으면 제공된 3년 CSV를 사용합니다.
특정 파일은 `--data data/semiconductor_etch_timeseries_3years.csv`로 지정합니다.
CSV는 숫자·결측·장비별 시간 순서·30분 간격·필수 열을 검증합니다.
학습용으로는 2025년 이전 기록, 2025년 정상 기록, 마지막 드리프트 구간이 모두 필요합니다.

학습 완료 후 서버를 재시작하면 새 로컬 모델을 로드합니다. 브라우저에서
`예시 불러오기 → 다음 공정 예측`을 누르세요. 입력 19개·음수 압력·시간 역순은 422입니다.
모델 파일이나 Registry가 준비되지 않으면 `/predict`는 503을 반환합니다.

Lazy/Eager 비교는 각각 새 프로세스에서 실행합니다.

```bash
LOADING_MODE=lazy python -m uvicorn semiconductor.app:app --port 8000
# 별도 터미널
python -m semiconductor.verify --label lazy
# 서버를 Ctrl+C로 종료 후
LOADING_MODE=eager python -m uvicorn semiconductor.app:app --port 8000
# 별도 터미널
python -m semiconductor.verify --label eager
```

`/health`의 `startup_seconds`는 lifespan 준비 구간이며 Python 프로세스 전체 시작 시간이
아닙니다. `verification_*.json`에 첫/두 번째 요청 시간과 로딩 상태를 저장합니다.
검증 전에 브라우저에서 추론하면 첫 요청 측정값이 바뀝니다.

## Day 2: MLflow 기록 → 게이트 → Production → Docker

```bash
python -m semiconductor.train --register
MODEL_SOURCE=mlflow LOADING_MODE=eager python -m uvicorn semiconductor.app:app --port 8000
# 별도 터미널
python -m semiconductor.verify --label mlflow --upload
```

MLflow는 `semiconductor_state/mlflow.db` SQLite와 로컬 artifacts를 사용합니다.
실험명 `Semiconductor-Day2`, 모델명 `Semiconductor_Etch_LSTM`입니다.
모델 `.keras`, 전처리 `scaler.json`, 입력 예시, 평가 결과를 한 버전으로 묶습니다.
학습 입력과 서빙 입력은 같은 `data.transform()`을 사용합니다.

게이트: 검증 구간 RMSE ≤4점, HIGH Recall ≥0.80, Precision ≥0.80.
이는 PoC 초기 운영 기준이며 공정 공인 기준이나 수업 주가 $4의 단위가 아닙니다.
통과한 버전만 `Production` 별칭(alias)을 변경합니다. 실패한 버전도 실험/Registry에
기록하지만 기존 Production은 유지합니다. Production이 아직 없으면 MLflow 서빙은 실패합니다.
`models:/Semiconductor_Etch_LSTM@Production` 사용. Stage API가 아닌 alias 확장 방식입니다.
응답은 실제 버전 번호를 포함한 `production-vN`으로 표시합니다.
별칭을 바꾼 뒤에는 서버를 재시작해야 캐시가 새 모델을 사용합니다.

이미 학습한 번들은 재학습 없이 등록 가능:

```bash
# semiconductor_state/local.json에 적힌 bundle 이름 사용
python -m semiconductor.train --register --bundle semiconductor_state/bundles/번들이름
# 게이트 실패 시연: RMSE 임계값을 0으로 지정 (Production은 유지)
python -m semiconductor.train --register --bundle semiconductor_state/bundles/번들이름 --rmse-gate 0
```

MLflow UI는 선택 사항입니다. 학습/서빙에는 UI 서버가 필요하지 않습니다.

```bash
python -m mlflow ui --backend-store-uri sqlite:///semiconductor_state/mlflow.db --port 5002
```

Docker Desktop을 실행한 뒤:

```bash
docker compose -f compose.semiconductor.yml up --build -d
docker compose -f compose.semiconductor.yml logs -f
python -m semiconductor.verify --label docker --upload
```

Docker 빌드 중 의존성 설치와 학습·등록을 수행하며, 런타임은 단일 FastAPI 컨테이너입니다.
최초 빌드에는 수 분이 걸리고 이미지 용량이 큽니다. 로컬 MLflow DB는 컨테이너로 복사하지
않으므로 호스트 절대경로 문제를 피합니다. 앱 상태는 컨테이너 안에 있으며 재생성 시
추가 업로드/추가 학습 이력이 유지되지 않으므로 필요한 증빙을 먼저 저장하세요.
8000 충돌 시 macOS/Linux는 `ETCH_PORT=8001 docker compose -f compose.semiconductor.yml up -d`.
Windows는 `$env:ETCH_PORT="8001"`로 지정합니다.

## API 명세

| Method | 경로 | 역할 |
|---|---|---|
| GET | `/health` | 프로세스 상태, 소스·버전·Lazy/Eager·로딩시간 |
| GET | `/ready` | 모델 로드 완료 시 200, 미로드 시 503 |
| POST | `/data/upload` | multipart CSV 검증·저장, 학습은 별도 명령 |
| GET | `/data/status` | 현재 데이터 파일·행 수·장비·기간 |
| GET | `/data/example` | 바로 `/predict`에 넣을 수 있는 20개 입력 |
| POST | `/predict` | 다음 30분 이상 점수·위험 수준·모델 버전 |
| GET | `/docs` | Swagger 명세와 실행 화면 |

요청은 `equipment_id`와 `sequence`로 구성합니다. 각 시점에 `timestamp`와
센서 12개를 넣으며 20개 모두 한 장비의 기록이어야 합니다. API가 외부 장비의 실제
소속을 확인할 수는 없으므로 호출자가 equipment_id에 맞는 기록을 공급해야 합니다.
`anomaly_score`, `defect_label`, `risk_level`, `is_drift`는 입력 금지입니다.

## 평가와 한계

2023~2024 학습, 2025-01-01~04-11 06:30 검증, 04-11 07:00~07-20 13:30
최종 정상 테스트, 이후 드리프트 테스트. 장비 경계는 넘지 않으며 스케일러는 학습 구간에만 fit.
경계 이후의 예측에서 직전 구간의 과거 20개 입력을 사용하는 것은 실제 운용과 동일합니다.
최종 테스트는 EarlyStopping이나 승격 판단에 사용하지 않습니다.

| 구간 | RMSE | MAE | HIGH Precision | HIGH Recall | HIGH F1 |
|---|---:|---:|---:|---:|---:|
| 검증 | 3.164 | 2.162 | 0.876 | 0.933 | 0.904 |
| 독립 정상 테스트 | 3.083 | 2.086 | 0.932 | 0.942 | 0.937 |
| 드리프트 테스트 | 8.072 | 6.550 | 0.788 | 0.996 | 0.880 |

직전 이상 점수를 그대로 사용하는 기준 모델의 정상 테스트 RMSE는 3.736입니다.
LSTM은 이보다 약 17.5% 개선됐습니다. 다만 드리프트 구간에서는 기준 모델 RMSE 3.565보다
나쁩니다. 성능 저하를 감지해야 하는 이유이며, 아직 재학습 후 개선을 검증하지 않았습니다.
기준 모델은 직전 공정의 실제 점수가 즉시 제공된다는 가정입니다.
과거 `SEMICONDUCTOR_LSTM_RESULTS.md`의 3.248은 검증에도 사용했던 구간의 점수로,
이번 독립 테스트 결과와 구분합니다. 하드웨어/라이브러리에 따라 재학습 수치는 달라질 수 있습니다.

## 팀 검증 및 공유

```bash
python -m pip install -r semiconductor/requirements-test.txt
python -m unittest semiconductor.test_day2 -v
python -m semiconductor.package_team
```

