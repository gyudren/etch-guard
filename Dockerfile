# EtchGuard 단일 컨테이너: 빌드 시점에 학습 → MLflow 등록·게이트 → Production 승격까지 끝낸 자기 완결형 이미지.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR=/tmp/mpl \
    SEMICONDUCTOR_STATE_DIR=/app/semiconductor_state AIOPS_LOG_DIR=/app/logs
# 의존성을 소스보다 먼저 설치해 레이어 캐시를 활용한다 (소스만 바뀌면 재설치 생략).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/semiconductor_etch_timeseries_3years.csv data/semiconductor_etch_timeseries_3years.csv
COPY semiconductor/ semiconductor/
# 호스트의 MLflow DB(절대경로 artifact)를 복사하지 않고 컨테이너 안에서 학습·등록한다.
RUN python -m semiconductor.train --register \
 && python -c "from semiconductor.runtime import ModelManager; print(ModelManager('mlflow').get().version)"
ENV MODEL_SOURCE=mlflow LOADING_MODE=eager
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["python", "-m", "uvicorn", "semiconductor.app:app", "--host", "0.0.0.0", "--port", "8000"]
