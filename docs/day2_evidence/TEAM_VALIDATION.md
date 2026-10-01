# Day 1·2 검증 기록

검증일: 2026-09-30. 교육용 합성 데이터 PoC이며 실제 반도체 불량 예측 성능을 보증하지 않습니다.

## 확인한 항목

- Python 3.11, 로컬 TensorFlow 2.21.0 / MLflow 3.16.0에서 학습 완료.
- 자동 통합 테스트 7개 통과: Lazy 로딩·입력 422, Eager/Registry 예측 일치,
  모델 누락 503, 잘못된 CSV 400, 시간 구간 분리, NaN 게이트 거부, 실패 시 Production 보존.
- 실제 HTTP로 263,040행 CSV 업로드 200, 정상 예측 200, 19개 입력 422 확인.
- Linux ARM64 Docker에서 의존성 설치 → LSTM 학습 → MLflow 등록 → Production 로딩 성공.
- Docker 단일 컨테이너에서 MLflow Production 모델을 이용한 HTTP 예측 및 업로드 성공.
- 로컬 Registry: 통과 버전 2를 Production으로 지정한 후 실패 버전 3을 추가해도
  Production은 2로 유지됨을 테스트. 버전 번호는 팀원의 실행 횟수에 따라 다릅니다.

## Lazy / Eager 실측

같은 로컬 모델을 각각 새 서버 프로세스에서 실행한 단일 측정입니다.
동시 Docker 빌드 등 환경 부하가 있었으므로 절대적인 속도 벤치마크가 아닙니다.

| 로컬 모드 | lifespan 준비 시간 | 첫 예측 HTTP | 두 번째 예측 HTTP |
|---|---:|---:|---:|
| Lazy | 0.000001초 미만 | 3.978초 | 0.023초 |
| Eager | 2.537초 | 0.034초 | 0.025초 |

Lazy는 첫 추론 요청에서 모델을 불러오고, Eager는 서버 시작 시 불러옵니다.
`startup_seconds`는 프로세스 실행 전체 시간이 아닌 lifespan 구간입니다.
원본 증빙은 `semiconductor_state/verification_local_lazy.json`,
`verification_local_eager.json`, `verification_docker.json`에 있습니다.

## 성능 해석

독립 정상 테스트 RMSE 3.083, HIGH Precision 0.932 / Recall 0.942.
드리프트 테스트 RMSE 8.072로 악화되었으며, 이 구간은 단순 직전값 기준 모델보다도
나쁩니다. 정상 구간 성능과 구별해 발표하세요. 자세한 분할과 수치는 `TEAM_DAY2.md` 및
번들 `metrics.json` 참조. Docker 재학습 결과는 플랫폼 차이로 소폭 다를 수 있습니다.

## 아직 확인하지 않은 범위

Windows 및 Linux x86_64에서의 실행, 다중 사용자 부하, 운영 보안, 실제 공정 데이터 성능은
검증하지 않았습니다. Day 3의 드리프트 감지 API·자동 재학습·알림은 구현 범위 밖입니다.
교수님께 제출할 팀별 실행 화면과 보고서·역할 분담은 팀이 직접 확인하고 작성해야 합니다.
